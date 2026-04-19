# Dense vs MoE at 1-Bit: A Theoretical Analysis

**Status:** theoretical analysis derived from scale personality findings. No MoE experiments have been run. Claims are clearly marked as measured vs. inferred.

---

## The question

Mixture-of-Experts (MoE) exists to solve a specific problem: how do you get specialist capacity without activating all parameters every forward pass? Mixtral 8×7B has 47B total parameters but activates ~13B per token — you get specialist depth at ~28% of the activation cost.

The question this analysis asks: **does 1-bit change the calculus that makes MoE worth the complexity?**

---

## What MoE is actually solving

At FP16, the storage problem is acute. A dense 47B model requires ~94GB — well beyond a single GPU. MoE solves this by making most of the parameters inactive per token, enabling larger total capacity at bounded activation cost. The sparsity isn't a feature in itself; it's the mechanism that makes very large models deployable.

MoE introduces real costs:
- **Routing overhead:** a learned router must classify every token at every MoE layer
- **Expert collapse:** routing can become imbalanced, with some experts overloaded and others idle
- **Load balancing loss:** an auxiliary loss term is required to prevent collapse, which interferes with the primary objective
- **Communication overhead:** in distributed settings, token routing across devices adds latency
- **Training complexity:** MoE models are harder to train stably than dense equivalents

These costs are worth paying at FP16 because the alternative — a dense model of equivalent total capacity — doesn't fit in reasonable hardware.

---

## What 1-bit changes

1-bit quantization stores weights as packed binary values (`{-1, +1}`) plus one fp16 scale per 128-weight group. The measured storage for a production 1-bit 8B model (PrismML Bonsai 8B) is **1.15GB** — roughly 14× smaller than the FP16 equivalent.

Extrapolating linearly:

| Model size | FP16 storage | 1-bit storage |
|---|---|---|
| 8B | ~16GB | ~1.15GB |
| 47B (Mixtral-scale) | ~94GB | ~6.8GB |
| 100B | ~200GB | ~14.4GB |
| 400B | ~800GB | ~57GB |

The implication: **1-bit solves the storage problem that MoE was designed to work around.** A 1-bit dense 47B model fits on a single consumer GPU with room to spare. The primary motivation for MoE — making large total capacity deployable — largely disappears below ~100-200B parameters.

---

## Where scale personalities fit

The scale personality mechanism (*measured*, not inferred) shows that fp16 scale tables (~0.8% of parameter bytes) carry real domain specialization over frozen 1-bit signs. Key findings:

- 28% GSM8K on a 1.7B model vs 5.3% baseline — a 5.3× lift from training only the scales
- 8/8 diagonal dominance across 8 profiles at 8B — each profile best on its own domain
- Router compounds above any single profile (70% ARC-Easy vs 64.7% baseline)
- LoRA at 3× the parameter count matches but doesn't exceed scale accuracy
- Signs are committed and sufficient — scales are the optimal continuous degree of freedom

Scale personalities are essentially **post-training specialization at near-zero storage cost**. Per-profile overhead is ~22MB on a ~1.15GB backbone.

---

## Dense wins below ~100-200B

At scales where 1-bit dense fits in reasonable hardware, dense + scale personalities dominates MoE on nearly every axis:

| Property | 1-bit dense + scales | 1-bit MoE |
|---|---|---|
| Storage | 1.15GB/8B (linear) | Similar (same bit count) |
| Activation cost | Full model every token | Sparse (2-4/N experts) |
| Routing overhead | None | Router at every MoE layer |
| Expert collapse | N/A | Requires auxiliary loss |
| Scale personality surface | Full 196 weight groups | Per-expert subset |
| Specialization cost | 22MB per personality | 22MB × N experts |
| Inspectability | One scale table | N expert scale tables |
| Training complexity | Low | High |

Below ~100B, the activation cost difference between dense and MoE shrinks in practical terms because both fit comfortably in VRAM. Running a 1-bit dense 47B model at full activation costs less in absolute terms than running a FP16 MoE with equivalent total parameters at sparse activation — the denominator has changed.

---

## MoE becomes relevant again above ~100-200B

There's a crossover point where even 1-bit dense stops fitting in single-GPU VRAM. A 1-bit dense 400B model is ~57GB — still multi-GPU at current hardware, and activation cost scales with total parameters regardless of storage format.

Above this threshold, 1-bit MoE becomes relevant again, but for a different reason than it is today: not storage, but **activation cost at extreme scale**. A 1-bit MoE 400B activating 2 of 8 experts per token runs at ~50B-parameter activation cost — that's the meaningful efficiency.

The architecture that emerges at that scale: **1-bit MoE with per-expert scale tables**. Each expert carries its own scale personality. This creates two independent specialization axes:

1. **Which expert fires** — learned at training time, determined by the router
2. **How intense each expert's specialization is** — configurable post-deployment via scale tables

If expert routing already handles domain-level selection, per-expert scale tables could handle finer-grained configuration: a single "math" expert with separate scale tables for competition math vs. applied math vs. symbolic reasoning. The router picks the expert; the scale table configures it.

---

## The compounding hypothesis

The scale router experiments (*measured*) showed that soft blending four scale tables produced ARC-Easy performance above any single profile — emergent compounding, not selection. The working hypothesis is that scale space is a continuous manifold where interior points can outperform any learned endpoint.

In a 1-bit MoE, this effect could compound at two levels:
1. **Expert level:** the router soft-blends expert activations (this is what MoE already does)
2. **Scale level:** scale personalities soft-blend within each expert (this is what we've validated on dense)

Two independent blending mechanisms operating at different granularities. Whether they interact additively, multiplicatively, or cancel is unknown — and a clean empirical test.

---

## What would need to be measured

The analysis above is derived from scale personality findings on dense models. The following experiments would license or falsify the MoE claims:

1. **1-bit MoE baseline.** Apply Bonsai-style QAT to a MoE architecture (Mixtral 8×7B is the natural target — public, well-documented). Measure whether the 1-bit training recipe that works for dense transfers to MoE experts without collapse.

2. **Per-expert scale personalities.** Train separate scale tables for each expert on domain data. Test whether individual expert scales show diagonal dominance the way dense model scales do.

3. **Scale headroom in pre-specialized experts.** The key unknown: does MoE expert pre-specialization reduce the room for scale personality training? An expert already trained on code has lower entropy in its scale distribution — less for post-hoc scale training to move. Measure: compare scale personality lift on a fresh (non-specialized) expert vs. a fully trained MoE expert.

4. **Dense vs MoE at matched parameter count.** At 47B total parameters, compare: 1-bit dense + global scale personalities vs. 1-bit MoE 8×7B + per-expert scale personalities. Activation cost differs; total specialization capacity may not.

5. **Crossover point empirically.** The ~100-200B estimate for where MoE becomes worthwhile at 1-bit is derived from storage math, not from activation cost measurements on real hardware. The actual crossover depends on memory bandwidth, not just VRAM capacity.

---

## Summary

**The core claim** (derived from measured results, not itself measured): 1-bit quantization removes the primary motivation for MoE at scales below ~100-200B. Dense 1-bit + scale personalities is simpler, equally capable, and more inspectable than 1-bit MoE in this range.

**Above ~100-200B**, 1-bit MoE becomes relevant again for activation cost reasons, and per-expert scale tables create a two-level specialization architecture that has no analog in either dense 1-bit or FP16 MoE alone.

**The honest boundary:** this is architectural reasoning from first principles plus the scale personality mechanism. None of the MoE-specific claims have been tested. The crossover point is an estimate. The compounding hypothesis is a hypothesis. What's solid is the foundation: the mechanism is real, the storage math is measured, and the logic follows from both.
