# 1-Bit Experiments

Empirical research on 1-bit (binary weight) neural networks. Focused on what actually works, what doesn't, and why — backed by reproducible experiments on real models.

## Key Findings

### 1. Scale Personalities — Domain Specialization via Scale Tables
**Single 1-bit model, multiple personalities by swapping only the fp16 scale tables.**

Signs ({-1, +1}) stay fixed as shared routing structure. Scales (fp16, one per 128 weights) act as swappable intensity tables that specialize the model for different domains. 50 personalities cost ~58GB total vs ~20TB for 50 separate fp16 models (344x efficiency).

**Results on Bonsai 8B (native 1-bit, 3.46GB VRAM):**

| Profile | Baseline PPL | Trained PPL | Improvement |
|---------|-------------|-------------|-------------|
| tool_use | 9.33 | 6.30 | 32.6% |
| reasoning | 3.67 | 3.28 | 10.4% |
| structured | 4.96 | 3.93 | 20.9% |
| retrieval | 35.27 | 27.18 | 22.9% |
| verification | 11.40 | 8.98 | 21.2% |
| planning | 5.45 | 4.11 | 24.5% |
| compliance | 8.76 | 6.40 | 27.0% |
| creative | 22.13 | 9.90 | 55.3% |

8/8 diagonal dominance — each profile best on its own domain. Avg 26.8% PPL improvement. Validated across multiple seeds.

**Critical finding: PPL doesn't always translate to accuracy.** Reasoning profile had best math PPL (3.28) but worst GSM8K accuracy (8%, down from 20% baseline). Scales optimize for predicting common tokens, not hard answer tokens. TriviaQA accuracy DID improve (+20% relative) on structured/compliance profiles.

> [Full writeup](docs/scale-personalities.md) | [Code](experiments/scale-personalities/)

### 2. Bonsai Forensic Analysis — Reverse-Engineering Production 1-Bit Models
**Full metric extraction from PrismML's Bonsai 1.7B/4B/8B to establish ground truth for QAT. Built on Archie's (@archiexzzz) initial reverse-engineering work.**

| Metric | Value | Significance |
|--------|-------|-------------|
| bit_match | 0.70-0.79 | 21-30% of signs reconfigured from fp16 base |
| kurtosis | -1.8 to -1.9 | Bimodal weight distributions (not Gaussian) |
| ratio_med (ffn_up) | 3.53x | Scales inflated 3.5x from naive absmean |
| depth-scale correlation | 0.877 | Deeper layers get larger scales (learned, not heuristic) |
| Category hierarchy | ffn_up > gate > v > down > o > k > q | Not all layers are equal |

**Inferred 5-component recipe:** STE + per-category loss weighting + learned scales (not post-hoc absmean) + distillation + size-adaptive pressure.

> [Full writeup](docs/bonsai-forensics.md) | [Code](experiments/bonsai-forensics/)

### 3. QAT Pipeline — What Works and What Doesn't

**What works:**
- Scale learning via KL distillation: PPL 362 → 58 on Qwen3-1.7B (4 quantized layers)
- End-to-end fitness functions (model PPL, not per-layer MSE)
- Hybrid EGGROLL evolution: PPL 1.04M → 505K gradient-free on all 28 layers

**What fails:**
- Progressive quantization beyond 4 layers (gradient flow breaks at 6+)
- Per-linear MSE calibration (locally optimal solutions compose into PPL 71M)
- Adaptive precision post-hoc (PPL 36 → 90,254 when mixing fp16 into trained 1-bit)
- EGGROLL with per-layer fitness (0 flips accepted — only end-to-end PPL works)

**Universal QAT equation (derived from Bonsai forensics):**
```python
N = n_params_billions
ratio_target = 2.66 * N**(-0.25)
weight_lr_mult = 2.0 + 0.3 / N
tokens_per_param = 0.15 + 0.10 / N
scale_growth_lambda = 0.01 * ratio_target / 2.0
cat_weights = {
    "ffn_up": 3.0, "ffn_gate": 2.5, "ffn_down": 1.5,
    "attn_v": 2.0, "attn_o": 1.0, "attn_k": 0.8, "attn_q": 0.8
}
```

> [Full writeup](docs/qat-pipeline.md) | [Code](experiments/qat-pipeline/)

### 4. EFI — Expected Flip Improvement
**One forward+backward pass ranks the impact of ALL possible sign flips.**

```
EFI = sign * gradient * 2 * scale * input_magnitude
```

8 minutes for 5,000 optimized flips vs 6-12 hours for EGGROLL. Gradient-guided sign optimization that works because the sign function's derivative (via STE) tells you exactly which flips help most.

> [Full writeup](docs/efi.md) | [Code](experiments/qat-pipeline/)

### 5. Graduated Growth — Growing 1-Bit Models from Scratch
**Binary Neural Growth: start small, split overloaded weights, grow architecture from data.**

- Multi-layer 1-bit corrections compound (Wiki PPL 99 → 24)
- Conflict detection: track vote_sum AND vote_abs per weight group
  - High abs + low sum = conflicted (math vs language pulling opposite directions)
  - Layer 26: 21.6% of groups conflicted, MLP layers have most conflict
- Hidden dim 1024 optimal for correction layers

> [Full writeup](docs/graduated-growth.md) | [Code](experiments/graduated-growth/)

## Activation Probe Results (Bonsai 8B)

Where the signal actually lives in a 1-bit model:

| Layer Region | Redundant Groups | Notes |
|-------------|-----------------|-------|
| Early (0-10) | 99.6% | Uniform activations, not dead |
| Late (21-31) | 2.1% | Almost all groups active |
| ffn_up/gate | 44.7% redundant | Least redundant = most trainable |
| Overall | 52.5% | Half the model is underutilized |

**Implication:** Scale training should target ffn_up/gate in later layers. Early layers are mostly pass-through.

## Hardware & Economics

| Setup | Model | Inference | Storage |
|-------|-------|-----------|---------|
| RTX 4090 | 100B 1-bit | 77 tok/s | 12.5 GB |
| A100-80GB | 200B 1-bit | 77 tok/s | 25 GB |
| Own 4090 | 8B 1-bit | — | $0.03/1M tokens |
| Sonnet API | — | — | $15/1M tokens (270x more) |

1-bit models at 100B+ parameters fit in consumer GPU VRAM. The economics are transformative.

## Negative Results (Equally Important)

These experiments failed and the reasons are instructive:

1. **Adaptive precision post-hoc destroys 1-bit models.** Swapping individual weights to fp16 after training breaks co-adapted sign patterns. PPL 36 → 90,254. Must train WITH mixed precision from the start.

2. **Per-linear MSE composes into garbage.** Each layer individually looks good. Composed result: PPL 71 million. End-to-end fitness is the only reliable signal.

3. **PPL ≠ accuracy for reasoning.** Best math PPL (3.28) = worst GSM8K accuracy (8%). Optimizing for next-token prediction on math text improves common token prediction, not multi-step reasoning.

## Reproducing

All experiments ran on Modal (T4/A100) or local GPU (GTX 1660 Super, 6GB). Each experiment directory contains the full script with inline results and Modal deployment configs where applicable.

```bash
# Install dependencies
pip install torch transformers datasets safetensors nltk

# Run scale personality training (requires Modal account + GPU)
modal run experiments/scale-personalities/train_8profiles.py

# Run Bonsai forensic analysis (CPU-only, downloads from HuggingFace)
python experiments/bonsai-forensics/analyze.py
```

## Models Used

| Model | Source | Size (1-bit) | Notes |
|-------|--------|-------------|-------|
| Bonsai 8B | prism-ml/Bonsai-8B-unpacked | 3.46 GB | Native 1-bit, primary test bed |
| Bonsai 1.7B | prism-ml/Bonsai-1.7B | ~850 MB | Graduated growth experiments |
| Qwen3-1.7B | Qwen/Qwen3-1.7B | ~850 MB | QAT + EGGROLL experiments |
| SmolLM2-135M | HuggingFaceTB/SmolLM2-135M | ~70 MB | Fast iteration QAT |

## License

MIT
