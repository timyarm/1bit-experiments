# Bonsai Forensic Analysis: Reverse-Engineering Production 1-Bit Models

## Motivation

PrismML released Bonsai — a family of native 1-bit models (1.7B, 4B, 8B) that achieve surprisingly good quality. No training recipe was published. Archie (@archiexzzz) began reverse-engineering the weight structure to understand how they were made. We extended that analysis, extracted 6 forensic metrics, and used the findings to establish ground-truth targets for our own QAT pipeline.

## Method

Extracted 6 metrics per weight group across all linear layers, compared against the fp16 base models (Qwen, Llama) to measure how far Bonsai diverged from naive quantization.

## Metrics

### 1. bit_match (Sign Agreement with Base)
How many weight signs match the fp16 original after `sign(fp16_weight)`.

**Finding: 0.70-0.79 across all models.**

This means 21-30% of signs were actively reconfigured during training. Not a small perturbation — nearly a third of the routing structure was changed from the fp16 starting point.

### 2. kurtosis (Weight Distribution Shape)
Kurtosis of the scale values within each group.

**Finding: -1.8 to -1.9 (bimodal).**

This is NOT the Gaussian distribution (kurtosis ~0) you'd get from naive quantization. The bimodal pattern means scales cluster at two distinct magnitudes within each group — some weights are "loud" and some are "quiet." This emerges from training, not from any post-hoc normalization.

### 3. ratio_med (Scale Inflation)
Ratio of trained group scale to naive absmean of the fp16 weights.

**Finding: Category-dependent inflation.**

| Layer Type | ratio_med (1.7B) | ratio_med (8B) |
|-----------|------------------|----------------|
| ffn_up | 3.53x | ~3.2x |
| attn_v | 3.30x | ~3.0x |
| attn_k | 2.00x | ~1.8x |
| ffn_down | ~2.5x | ~2.3x |

Scales are inflated 2-3.5x from what you'd get by just computing absmean of the fp16 weights. This proves scales are LEARNED, not computed post-hoc.

### 4. depth_corr (Depth-Scale Correlation)
Correlation between layer index and scale magnitude.

**Finding: 0.877 for ffn_up.**

Deeper layers get larger scales. This is a strong, consistent pattern across all model sizes. It's the signature of joint optimization — the training process learned that later layers need stronger signals.

### 5. Category Hierarchy
Ranking of layer types by scale magnitude and training intensity.

**Finding: ffn_up > ffn_gate > attn_v > ffn_down > attn_o > attn_k > attn_q > token_embd**

This hierarchy is consistent across model sizes. ffn_up layers are always the most aggressively scaled. Attention Q projections are the least modified. This tells us where quantization pressure is highest and where training effort should concentrate.

### 6. Size-Adaptive Patterns
How metrics change across 1.7B → 4B → 8B.

**Finding**: Smaller models show more aggressive reconfiguration. bit_match is lower (more flips), ratio_med is higher (more inflation), kurtosis is more extreme. The training process applies stronger pressure on smaller models to compensate for fewer parameters.

## Inferred Recipe

Based on all metrics, we infer Bonsai was trained with 5 components:

1. **STE (Straight-Through Estimator)**: Gradients flow through the sign function. Standard for binary weight training.

2. **Per-category quantization loss**: Different layer types get different loss weights matching the observed hierarchy (ffn_up highest, attn_q lowest).

3. **Learned per-group scales**: Scales are nn.Parameters optimized jointly with weights. NOT computed post-hoc via absmean.

4. **Distillation from fp16 reference**: KL divergence or similar loss against the original fp16 model outputs.

5. **Size-adaptive optimization pressure**: Smaller models get stronger quantization alignment loss to compensate for capacity constraints.

## Universal QAT Equation

From the forensic data, we derived a parametric recipe:

```python
N = n_params_billions
ratio_target = 2.66 * N**(-0.25)        # scale inflation target
weight_lr_mult = 2.0 + 0.3 / N          # weight learning rate multiplier
tokens_per_param = 0.15 + 0.10 / N      # training data budget
scale_growth_lambda = 0.01 * ratio_target / 2.0  # scale growth pressure
cat_weights = {
    "ffn_up": 3.0, "ffn_gate": 2.5, "ffn_down": 1.5,
    "attn_v": 2.0, "attn_o": 1.0, "attn_k": 0.8, "attn_q": 0.8
}
```

## Reproduce

```bash
# Downloads Bonsai from HuggingFace, extracts all 6 metrics
python experiments/bonsai-forensics/analyze.py
```
