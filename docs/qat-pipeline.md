# QAT Pipeline: What Works and What Doesn't for 1-Bit Training

## Overview

We tested every major approach to training 1-bit models. Most failed. The failures are as informative as the successes — they reveal fundamental constraints about how binary weight networks learn.

## The One Rule That Predicts Everything

**End-to-end fitness is the only reliable signal.**

If your loss function evaluates the full model's output (PPL on real text, KL divergence against a teacher), it works. If your loss function evaluates individual layers in isolation (per-linear MSE reconstruction), it fails catastrophically — even when each layer individually looks perfect.

This is the single most important finding across all QAT experiments.

## Approaches That Work

### 1. Scale Learning via KL Distillation

**Model**: Qwen3-1.7B, 4 quantized layers (24-27)
**Method**: Freeze signs at sign(fp16_weight), learn scales via KL divergence against fp16 teacher

```python
# Core loop
teacher_logits = fp16_model(input_ids)  # cached, no grad
student_logits = quantized_model(input_ids)  # scales as nn.Parameter
loss = kl_divergence(student_logits, teacher_logits, temperature=2.0)
loss.backward()  # gradients flow to scales only
optimizer.step()  # AdamW lr=5e-5, cosine schedule
```

**Results**:
- FP16 baseline: PPL 34.22
- Naive PTQ (sign + absmean scales): PPL 362.35
- After 3 epochs scale learning: **PPL 57.88** (6.3x recovery)
- Model correctly generates: "The capital of France is Paris"

**Why it works**: KL loss measures what actually matters — how close the quantized model's output distribution is to the teacher's. Scales adjust to minimize real output divergence, not proxy metrics.

**Forensic metrics after training**:
- kurtosis: -1.6 to -1.7 (approaching Bonsai's -1.8/-1.9)
- ratio_med: ~1.17 (still below Bonsai's 3.53 — need sign flips to unlock higher ratios)
- bit_match: 1.000 (no sign flips yet, scales-only training)

### 2. Hybrid EGGROLL Evolution (Gradient-Free Sign Optimization)

**Model**: Qwen3-1.7B, all 28 layers quantized
**Method**: Random sign mutations evaluated by end-to-end PPL

Based on NVIDIA's EGGROLL paper — evolutionary strategies where mutations are sign flips on binary weights.

**Results**:
- Starting PPL: 1.04M (all 28 layers naively quantized)
- After 26 minutes: **PPL 505K** (monotonic improvement)
- Sign improvements accepted: 155
- Total signs flipped: 1.1M

**Why it works**: Each mutation is evaluated by running actual text through the full model. The fitness signal is real — not a per-layer approximation. Even though progress is slow (PPL still very high), every accepted mutation provably helps.

**Key insight**: EGGROLL with per-layer MSE fitness accepted 0 mutations. Same algorithm with end-to-end PPL fitness accepted 155. The fitness function is everything.

### 3. QAT Recipe v11 (SmolLM2-135M)

**Model**: SmolLM2-135M (fast iteration on T4)
**Method**: Full BitLinear STE training with per-category loss weighting

```python
# Derived from Bonsai forensic analysis
cat_weights = {
    "ffn_up": 3.0, "ffn_gate": 2.5, "ffn_down": 1.5,
    "attn_v": 2.0, "attn_o": 1.0, "attn_k": 0.8, "attn_q": 0.8
}
```

**Results at step 1500**:
- Kurtosis: ALL 6 categories hit target (-1.74 to -1.93 vs Bonsai's -1.8/-1.9)
- ratio_med: ALL 6 hit target (3.2 to 5.0 vs Bonsai's ~3.5)
- bit_match: 3/6 in range (0.70-0.79), attn_q/k overshooting at 0.62-0.63
- depth_corr: 0.43 for ffn_up (below Bonsai's 0.88 — needs longer training)
- PPL: 273 (vs 14 for fp32 — the gap remains an open question)

**What this proved**: The forensic metrics CAN be reproduced. Kurtosis and ratio_med match Bonsai exactly. The remaining gap is in depth correlation and absolute PPL.

## Approaches That Fail

### 1. Progressive Quantization Beyond 4 Layers

**Setup**: Quantize layers incrementally — start with last 4, add more
**Result**: Works at 4 layers. Collapses at 6+.

**Root cause**: KL gradients can't flow through 6+ consecutive quantized layers. The compounding error from STE approximations in each layer accumulates until gradients carry no useful signal.

**Implication**: For full-model quantization, you need either layer-wise calibration (GPTQ-style) or gradient-free methods (EGGROLL). End-to-end gradient flow through many quantized layers is not viable.

### 2. Per-Linear MSE Calibration (GPTQ-Style, All 28 Layers)

**Setup**: For each linear layer independently, minimize MSE between quantized output and fp16 output
**Result**: Each layer individually achieves excellent reconstruction. Composed model: **PPL 71 million.**

This is the most important negative result. Per-layer optimization finds solutions that are locally optimal but globally catastrophic. Layer N's output distribution shifts slightly. Layer N+1 was calibrated assuming Layer N's original distribution. The error compounds through 28 layers into complete garbage.

**Why GPTQ works for 4-bit but not 1-bit**: At 4-bit, per-layer reconstruction error is tiny — the compounding is manageable. At 1-bit, each layer introduces substantial error that downstream layers can't absorb.

### 3. Adaptive Precision Post-Hoc

**Setup**: Take a trained 1-bit model, selectively promote some weights to fp16
**Result**: PPL 36 (pure 1-bit) → 1,256 (29% fp16) → 90,254 (67% fp16)

Adding fp16 precision AFTER training makes the model WORSE. The 1-bit weights co-adapted as a complete system. Each sign assumes all other signs are binary. Introducing continuous values into the system creates signal mismatches that the model never learned to handle.

**Implication**: Mixed precision must be trained from the start. You cannot post-hoc upgrade a 1-bit model to mixed precision.

### 4. EGGROLL with Per-Layer Fitness

**Setup**: Same evolutionary sign flipping, but fitness = MSE reconstruction per layer
**Result**: 0 mutations accepted. Zero improvement.

The per-layer MSE landscape for sign flips is essentially flat. Flipping one sign in a layer of millions changes the per-layer MSE by less than noise. The signal only exists at the model output level.

## The Universal QAT Equation

Derived from Bonsai forensic analysis, parameterized by model size:

```python
N = n_params_billions

# How much to inflate scales beyond naive absmean
ratio_target = 2.66 * N**(-0.25)

# Learning rate multiplier for weight updates
weight_lr_mult = 2.0 + 0.3 / N

# Training data budget (tokens per parameter)
tokens_per_param = 0.15 + 0.10 / N

# Scale growth regularization
scale_growth_lambda = 0.01 * ratio_target / 2.0

# Per-category loss weighting (from forensic hierarchy)
cat_weights = {
    "ffn_up": 3.0, "ffn_gate": 2.5, "ffn_down": 1.5,
    "attn_v": 2.0, "attn_o": 1.0, "attn_k": 0.8, "attn_q": 0.8
}
```

## Open Questions

1. **PPL 273 vs 14**: Is the 20x gap at 135M params fundamental to 1-bit, or is it a training issue? Bonsai achieves much better ratios at larger scale.

2. **Depth correlation**: We hit 0.43 vs Bonsai's 0.88. Longer training, or a missing ingredient?

3. **Optimal pipeline**: Scale learning (KL) + EGGROLL (signs) in sequence? Or a joint approach?

## Reproduce

```bash
# Scale learning on Qwen3-1.7B (Modal, requires A100)
modal run experiments/qat-pipeline/qwen_qat.py

# Full QAT on SmolLM2-135M (Modal, T4 sufficient)
modal run experiments/qat-pipeline/e2e_qat.py

# GPTQ-style layerwise (demonstrates the failure mode)
modal run experiments/qat-pipeline/layerwise_calibration.py
```
