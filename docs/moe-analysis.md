# MoE Expert Independence: Why Delta Encoding Fails for 1-Bit

## Hypothesis

Mixture-of-Experts models have multiple expert FFN blocks per layer. If experts share a common "base" with small differences, we could encode them as: one set of shared 1-bit signs + small delta masks per expert. This would be dramatically more efficient than storing full independent copies.

## Experiment

**Model**: GLM-4.7-Flash (30B total, ~2.87B per expert, 64 experts, 47 layers)

Measured pairwise sign agreement between all experts within each layer. If experts share structure, we'd expect 70-90% sign agreement. If they're independent, we'd expect ~50% (random).

## Results

| Metric | Value |
|--------|-------|
| Average pairwise sign agreement | **50.03%** |
| Consensus signs vs any single expert | 55% |
| Variance across 64 experts | < 0.1% |
| Best-case pair agreement | ~52% |

**50.03% — statistically indistinguishable from random coin flips.**

Every expert pair in every layer showed the same pattern. No expert is more similar to any other expert than to random chance. The 55% consensus-vs-individual number is expected: majority vote of 64 random binary vectors will agree with any individual vector ~55% of the time (binomial statistics).

## What This Means

1. **Delta encoding is useless.** The delta between any two experts is as large as a full copy. You save nothing.

2. **Consensus signs are meaningless.** Taking the majority vote across experts gives you a model that's 55% similar to every expert and optimal for none of them.

3. **Experts are fully independent models.** The MoE training process did not discover shared structure — it trained 64 completely independent FFN blocks that happen to share the same input/output interface.

4. **Filtering by weight magnitude doesn't help.** We tried only comparing the highest-magnitude weights (most "confident" signs). Agreement stayed at ~50%. The independence is not a noise effect — it's structural.

## Why This Matters for 1-Bit MoE

If you want to build a 1-bit MoE model, you have two options:

**Option A: Store all experts independently.** 64 experts × full 1-bit copy each. No compression from shared structure because there is no shared structure.

**Option B: Dense distillation.** Instead of converting the MoE directly, distill the MoE's outputs into a dense 1-bit student. The student learns to approximate the MoE's routing behavior through a single set of signs + scale personalities (see [scale-personalities.md](scale-personalities.md)).

We chose Option B for subsequent work.

## The Broader Lesson

MoE architectures appear to reuse structure (same layer positions, same hidden dims, same activation functions) but the learned weights are fully independent. This is consistent with lottery ticket hypothesis — each expert finds its own sparse subnetwork within the same architecture, and these subnetworks don't overlap.

For 1-bit quantization specifically: don't assume weight sharing exists just because the architecture has repeated structure. Measure it. In our case, 50% agreement = zero sharing.

## Reproduce

```bash
# Downloads GLM-4.7-Flash (~59GB), computes pairwise sign agreement
# WARNING: Requires significant disk space and download time
python experiments/moe-analysis/expert_sign_agreement.py
```
