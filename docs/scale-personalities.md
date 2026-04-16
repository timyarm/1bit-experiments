# Scale Personalities: Domain Specialization in 1-Bit Models

## The Idea

A 1-bit model's weights are signs ({-1, +1}) multiplied by learned scales (fp16, one per group of 128 weights). The signs determine the routing structure — which neurons activate for which inputs. The scales determine the intensity — how strongly each group contributes.

What if we keep the signs fixed and swap only the scale tables?

Each scale table becomes a "personality" — a domain specialization that changes how the model behaves without changing its fundamental routing structure. One set of signs supports many personalities.

## Why This Works

In a standard fp16 model, domain specialization requires a separate copy of all parameters. 50 domain experts at 8B params = 800GB of fp16 weights.

In a 1-bit model:
- **Signs** (1 bit per weight): 1GB for 8B params. Shared across all personalities.
- **Scales** (fp16 per group of 128): ~125MB per personality for 8B params.
- **50 personalities**: 1GB signs + 50 x 125MB scales = **7.25GB total**

vs 50 separate fp16 models: ~800GB. That's **110x compression** with meaningful specialization.

## Experiments

### Setup
- **Model**: Bonsai 8B (prism-ml/Bonsai-8B-unpacked), native 1-bit, 3.46GB VRAM
- **Hardware**: Modal T4 GPU (16GB VRAM)
- **Training**: SGD lr=0.01, gradient checkpointing, 10 epochs per profile
- **Method**: Freeze signs, train only group scales as nn.Parameter

### 8-Profile Training

Trained 8 domain-specific scale tables on curated data:

| Profile | Data Source | Examples |
|---------|-----------|----------|
| tool_use | glaiveai/glaive-function-calling-v2 | Function calling, API usage |
| reasoning | gsm8k + competition_math | Multi-step math, proofs |
| structured | code_search_net | SQL, APIs, structured formats |
| retrieval | natural_questions + trivia_qa | Factual recall, QA |
| verification | mbpp test cases | Bug detection, validation |
| planning | gsm8k step-by-step | Task decomposition, sequencing |
| compliance | cais/mmlu professional subsets | Precision, formal requirements |
| creative | euclaise/writingprompts | Creative writing, brainstorming |

### Results: Perplexity (8/8 Diagonal Dominance)

| Profile | Baseline PPL | Trained PPL | Improvement |
|---------|-------------|-------------|-------------|
| tool_use | 9.33 | 6.30 | **32.6%** |
| reasoning | 3.67 | 3.28 | 10.4% |
| structured | 4.96 | 3.93 | 20.9% |
| retrieval | 35.27 | 27.18 | 22.9% |
| verification | 11.40 | 8.98 | 21.2% |
| planning | 5.45 | 4.11 | 24.5% |
| compliance | 8.76 | 6.40 | 27.0% |
| creative | 22.13 | 9.90 | **55.3%** |

Every profile achieves its best PPL on its own domain. Average improvement: 26.8%.

### Results: Accuracy (The Surprise)

| Scales | GSM8K | MMLU | MBPP | TriviaQA |
|--------|-------|------|------|----------|
| original | 20% | 4% | 5% | 20% |
| tool_use | 20% | 4% | 5% | 20% |
| reasoning | **8%** | 4% | 5% | 20% |
| structured | 20% | 4% | 5% | **24%** |
| compliance | 12% | 4% | 5% | **24%** |

**PPL does not always translate to accuracy.** The reasoning profile had the best math PPL (3.28) but the worst GSM8K accuracy (8%). Scale training optimized for predicting common math tokens ("=", "+", "step 1") but disrupted the precise multi-step chains needed for correct final answers.

TriviaQA accuracy DID improve on structured and compliance profiles (+20% relative). Scales change real behavior — the effect is just not always what PPL predicts.

### 3-Way Validation (Cross-Domain Independence)

Tested whether math scales help code and vice versa:

| Table | Math PPL | Lang PPL | Code PPL |
|-------|----------|----------|----------|
| Original | 4.91 | 29.75 | 2.19 |
| Math | **4.39** | 29.59 | 2.30 |
| Lang | 4.83 | **26.65** | 2.19 |
| Code | 5.01 | 30.91 | **1.39** |

Math and code don't help each other. Code scales HURT math (4.91 → 5.01). Math scales HURT code (2.19 → 2.30). Each personality learns specific sub-skills, not general improvement.

Reproduced across seeds (42, 123). The diagonal dominance is real.

### Activation Probe: Where the Signal Lives

Probed which weight groups actually respond to scale training:

| Layer Region | Redundant Groups | Interpretation |
|-------------|-----------------|----------------|
| Early (0-10) | 99.6% | Mostly pass-through, uniform activations |
| Late (21-31) | 2.1% | Almost all groups actively differentiating |
| ffn_up/gate | 44.7% redundant | **Least redundant = most trainable** |
| Overall | 52.5% | Half the model is underutilized |

**Implication**: Scale training should target ffn_up and ffn_gate in later layers. Training scales on early layers is wasted compute — they're all doing the same thing regardless of domain.

## Key Takeaways

1. **Scale tables are a viable personality mechanism.** 26.8% avg PPL improvement with 8 distinct domain specializations on a single 1-bit model.

2. **PPL is misleading for reasoning tasks.** Always validate with downstream accuracy, not just perplexity.

3. **Domains are independent.** Math doesn't help code and vice versa. Scale tables learn specific sub-skills.

4. **Signal concentrates in later layers, ffn_up/gate.** Early layers are mostly uniform. Target training accordingly.

5. **52% of weight groups are underutilized.** Massive room for improvement via sign optimization (EFI) on the idle groups.

## Follow-up: Bonsai 1.7B v2 Recipe (Local GTX 1660 Super, 2026-04-16)

Ported the 8B recipe to Bonsai 1.7B and trained three profiles (math, knowledge, code) locally. Added literature-backed improvements over v1:

- Token-weighted loss (Rho-1): train only on top 60% hardest tokens
- Elastic band regularization: MSE penalty to original scales (lambda=0.1)
- AdamW lr=1e-4 (100x lower than the v1 SGD 0.01)
- Longer sequences (256 vs 64), mixed data (70% domain + 30% diverse), 3 epochs

### PPL: 3/3 diagonal dominance reproduces

| Scales    | math PPL  | knowledge PPL | code PPL |
|-----------|-----------|---------------|----------|
| baseline  | 6.06      | 30.69         | 17.95    |
| math      | **4.14**  | 36.37         | 18.47    |
| knowledge | 6.29      | **17.96**     | 18.92    |
| code      | 5.94      | 32.69         | **15.04** |

### Accuracy: mostly regressions, one real signal (150q per benchmark)

| Scales    | TriviaQA | ARC-Easy  | HellaSwag |
|-----------|----------|-----------|-----------|
| baseline  | **9.3%** | **64.7%** | 35.3%     |
| math      | 6.0%     | 26.0%     | **42.0%** |
| knowledge | 7.3%     | 62.7%     | 34.7%     |
| code      | 6.7%     | 64.7%     | 30.0%     |

**Deltas from baseline:**
- math: −3.3% TriviaQA, **−38.7% ARC-Easy**, **+6.7% HellaSwag**
- knowledge: −2.0%, −2.0%, −0.7%
- code: −2.7%, +0.0%, −5.3%

### What this says, honestly

1. **A prior 100-question TriviaQA result (knowledge 9% vs baseline 7%) did not hold at 150 questions.** Baseline rose to 9.3%, knowledge fell to 7.3%. Sample size mattered; small-n "wins" are noise.
2. **Math scales show a real HellaSwag gain (+6.7%) but catastrophically forget ARC-Easy (−38.7%).** Classic specialization-vs-generalization collapse: the scales push the distribution hard toward numerical/procedural tokens, which breaks multiple-choice letter selection.
3. **PPL diagonal dominance does not imply accuracy improvement.** This reproduces the 8B finding.
4. **Knowledge and code are within noise of baseline on all three benchmarks.** The v2 recipe as implemented is not sufficient to move accuracy on these tasks at this model scale.

### Open hypotheses to test next

- KL distillation from the fp16 teacher (NVIDIA QAD) instead of token-weighted CE on labels
- Larger sample sizes (500+) to separate signal from noise
- Target only ffn_up/ffn_gate in late layers (per activation probe) instead of training all scales
- Tighter elastic band (lambda=0.3–0.5) to constrain catastrophic forgetting on math

## Reproduce

```bash
# Requires Modal account with GPU access
modal run experiments/scale-personalities/train_8profiles.py
modal run experiments/scale-personalities/validate_3way.py
modal run experiments/scale-personalities/activation_probe.py

# Local Bonsai 1.7B v2 (single ~6GB GPU)
python experiments/scale-personalities/scale_v2_proper.py
python experiments/scale-personalities/benchmarks.py
# Results: experiments/scale-personalities/bonsai1b_v2_multibench.json
```
