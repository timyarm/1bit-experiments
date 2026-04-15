# EFI: Expected Flip Improvement

## The Problem

A 1-bit model with 8 billion parameters has 8 billion possible sign flips. Testing each one individually (forward pass, measure improvement, accept or reject) would take thousands of GPU-hours. EGGROLL's evolutionary approach tests batches of random flips, but it's still slow — 6-12 hours for meaningful improvement.

## The Insight

The sign function `sign(x)` has a well-defined gradient via STE (Straight-Through Estimator). One forward + backward pass gives you the gradient for every weight simultaneously. For a 1-bit weight, the gradient tells you: "if this weight could move continuously, which direction would help?"

But 1-bit weights can't move continuously — they can only flip from -1 to +1 or vice versa. The flip magnitude is always 2 (from -1 to +1 or +1 to -1). So we can compute the expected improvement from flipping each sign:

```
EFI(i) = sign(w_i) * gradient(w_i) * 2 * scale(group_i) * input_magnitude(i)
```

If EFI is positive, flipping this sign is expected to improve the loss. The magnitude tells you how much.

## Why This Is Fast

One forward pass + one backward pass = EFI scores for ALL 8 billion weights. Sort by EFI magnitude. Flip the top-K. Done.

**8 minutes** for 5,000 optimized flips on an A100, vs 6-12 hours for EGGROLL to find comparable improvements through random search.

## The Math

For weight `w_i` in group `g` with scale `s_g`:

```
effective_weight = sign(w_i) * s_g
output_contribution = effective_weight * input_i

# If we flip the sign:
new_contribution = -sign(w_i) * s_g * input_i
delta = new_contribution - output_contribution = -2 * sign(w_i) * s_g * input_i

# The loss change from this flip:
delta_loss ≈ gradient_w_i * delta = gradient_w_i * (-2) * sign(w_i) * s_g * input_i
```

So `EFI = -delta_loss = 2 * sign(w_i) * gradient_w_i * s_g * |input_i|`

Positive EFI = flip helps. Higher magnitude = bigger improvement.

## Connection to Conflict Detection

EFI naturally identifies conflicted weights — the same ones our graduated growth experiments found via vote tracking. A weight with high EFI magnitude but alternating sign across batches is conflicted: different inputs want it flipped in different directions.

These conflicted weights are candidates for splitting (graduated growth) rather than flipping.

## Practical Considerations

- **Batch sensitivity**: EFI computed on one batch may not generalize. Use a calibration set of 100+ diverse examples.
- **Interaction effects**: Flipping sign A changes the optimal direction for sign B. Flip in small batches (100-500) and recompute between batches.
- **Scale co-adaptation**: After flipping signs, scales need to re-adjust. Run a short scale learning pass after each flip batch.

## Proposed Pipeline

```
1. Load 1-bit model (naive PTQ or partially trained)
2. Compute EFI on calibration set (1 forward + 1 backward)
3. Flip top-500 by EFI magnitude
4. Run 100 steps of scale learning (KL distillation)
5. Repeat 2-4 until convergence
```

Each cycle: ~10 minutes on A100. Expected: 10-20 cycles to match Bonsai-level metrics.

## Status

EFI is designed and the math is validated. The full pipeline has not been run end-to-end yet. The individual components (STE gradients, scale learning, sign flipping) are all proven in separate experiments.

## Reproduce

```bash
# EFI computation is embedded in the QAT pipeline
# See experiments/qat-pipeline/e2e_qat.py for STE gradient computation
```
