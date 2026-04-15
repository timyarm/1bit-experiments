# Graduated Growth: Growing 1-Bit Models From Scratch

## Concept

Instead of quantizing a large fp16 model down to 1-bit, grow a 1-bit model upward. Start with a small binary network. When a weight group becomes overloaded (conflicting gradients from different domains), split it into two groups. The architecture grows from data pressure, not design decisions.

## Binary Neural Growth (BNG)

### How Splitting Works

Each weight group tracks two signals:
- **vote_sum**: net direction of gradient updates (positive = wants to be +1, negative = wants to be -1)
- **vote_abs**: total magnitude of gradient updates (how much pressure this weight is under)

A weight with high vote_abs but low vote_sum is **conflicted** — different inputs want it flipped in different directions. This is the signal for splitting.

When a group splits:
1. Clone the current signs
2. One copy keeps the original signs
3. The other copy starts accumulating independent EFI-guided flips
4. A learned gate routes inputs to the appropriate copy

### Results (Bonsai 1.7B)

Multi-layer 1-bit correction blocks applied in parallel:

| Config | Wiki PPL | Eval Score |
|--------|----------|-----------|
| Baseline (no corrections) | 99 | 7/17 |
| 4 layers × 1M params | **24** | **11/17** |

4x PPL improvement and 57% accuracy improvement from additive 1-bit corrections.

### What We Learned About Growth

**Hidden dim 1024 is optimal** for correction layers. Smaller dims (256, 512) underfit. Larger dims (2048+) cause training instability without careful init scaling (sqrt(256/hidden)).

**Corrections compound across layers.** Each correction layer independently improves output. 4 layers of 1M params each beat 1 layer of 4M params — depth matters more than width for corrections.

## Conflict Detection

### The Metric

For each weight group of 128 signs:

```python
vote_sum = sum(gradients)      # net direction
vote_abs = sum(abs(gradients)) # total pressure

conflict_ratio = 1.0 - abs(vote_sum) / (vote_abs + epsilon)
# 0.0 = unanimous (all gradients agree)
# 1.0 = perfectly conflicted (equal push both ways)
```

### Findings (Bonsai 1.7B, Layer 26)

| Category | % of Groups |
|----------|------------|
| Conflicted (ratio > 0.5) | 21.6% |
| Settled (ratio < 0.1) | 0.1% |
| Neutral | 75.3% |

**MLP layers (gate/up/down) have the most conflict.** Attention layers have less. This aligns with the activation probe findings — MLP layers are doing more domain-specific computation.

If we split all conflicted groups: **21.7M targeted parameters** (vs 117M random corrections). Conflict detection tells you exactly where the model needs more capacity.

### Full Cycle Test (Grow → Consolidate)

```
Grow:         5/9 → 9/9 (perfect score)
Consolidate:  9/9 → 4/9 (lost math knowledge)
Wiki PPL:     20.14 → 19.24 (improved despite score drop)
```

Consolidation was too aggressive (30 steps, 0.5% flip rate). The math-specific corrections were pruned because they looked like noise to the consolidation pass. Needs domain-aware consolidation that preserves specialized corrections.

## Native 1-Bit Kernel

Built a kernel for efficient 1-bit operations:

```python
class NativeBitLinear:
    packed_signs: int32    # 1 bit per weight, 32 weights per int
    group_scales: fp16     # 1 per 128 weights, nn.Parameter
    flip_votes:   int8     # per-group vote accumulator
```

**Memory**: 3.27GB for 1.7B model (vs 7GB for STE wrapper)
**Group-level votes**: 128x memory savings (0.4MB vs 50MB for per-weight tracking)

### Results

| Method | Eval Score |
|--------|-----------|
| Single-layer flip training | 7/9 |
| Group-level votes | 5/8 |
| Full graduated growth | 9/9 |

## What Didn't Work

- **Deep-copying Bonsai layers → NaN**: The copied weights interact badly with the existing layer norms
- **Full decoder layers as corrections → instability**: Too many parameters, training diverges
- **SwiGLU FFN > 256 hidden without init scaling → corrupts hidden states**: Must scale init by sqrt(256/hidden)
- **Unfreezing downstream layers → NaN**: Gradients through quantized layers are too noisy
- **STE-wrapping full model → OOM**: STE requires fp32 shadow weights, doubles memory

## Implications for 1-Bit Model Design

1. **Don't start big and compress. Start small and grow.** The splitting mechanism naturally discovers where the model needs capacity.

2. **Conflict detection is a free signal.** You're already computing gradients. Tracking vote_sum and vote_abs adds negligible cost and tells you exactly where to invest parameters.

3. **Depth > width for binary corrections.** Multiple thin correction layers beat one wide layer. Binary weights benefit from sequential refinement.

4. **MLP layers are the bottleneck.** Attention layers route information. MLP layers transform it. Transformation is where 1-bit conflicts arise.

## Reproduce

```bash
modal run experiments/graduated-growth/sequential_distill.py
modal run experiments/graduated-growth/combined_test.py
```
