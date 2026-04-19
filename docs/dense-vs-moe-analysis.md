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

## At low-trillion scale — what opens up

*Everything in this section is theoretical extension of the validated mechanism. No 1T 1-bit model exists. The path there requires 8B v2 → 70B+ 1-bit → MoE. Estimated 12-18 months from the experimental infrastructure existing.*

### Scale table economics change character

At 1.7B a personality table is 22MB — negligible. At 1T it's ~15.6GB. The "near-zero storage" framing changes:

| Scale | Per-personality table | 50 personalities | 50 FP16 models |
|---|---|---|---|
| 1.7B | 22MB | 1.1GB | 170GB |
| 8B | ~100MB | 5GB | 800GB |
| 1T | ~15.6GB | 780GB | 100TB |

Still 128× cheaper than FP16 at 1T — decisively so. But the absolute size matters for deployment: 780GB of scale tables is a real infrastructure consideration, not a footnote.

The natural response is hierarchical: rather than one flat scale table per personality, split into global (domain), per-expert (sub-domain), and per-layer-depth (task type) tables. Each level is a fraction of the full 15.6GB. Three levels at ~0.8% overhead each gives three independent configuration axes at ~5GB total overhead per personality instead of 15.6GB.

### Expert depth changes what scale profiles can do

At 1T MoE with 8 experts, each expert is ~125B parameters. A 125B math expert is not a shallow domain specialist — it's a model with genuine deep capability, trained routing structure, and rich sign patterns in its domain. Scale profiles on top of a 125B expert operate on a fundamentally richer sign skeleton than scale profiles on a 1.7B or 8B backbone.

Concretely: the Exp 25 finding — scales encode intensity, signs encode precision — has a different resolution at 1T MoE. A dedicated 125B safety expert has enough sign structure to discriminate finely between harmful and benign inputs, not just amplify general caution. The precision problem that requires EFI sign surgery at 1.7B is solved at 1T MoE by the expert's own depth. Scale floor becomes: safety expert activation probability ≥ FLOOR. The FLOOR mechanism is identical; the expert behind it is vastly more capable.

### Three-level personality hierarchy

At 1T MoE the scale personality architecture can operate at three granularities simultaneously:

```
Global scale table       → broad domain (science, medicine, law, safety)
  Per-expert scale table → sub-domain (oncology within medicine)
    Layer-depth table    → task type (diagnosis vs. research vs. treatment)
```

Each level requires training a separate scale snapshot and a router weight. At 1T there are enough scale groups at each granularity (~60M per layer for a 1T model) that all three levels carry independent signal. At 1.7B the total scale count (~11M) is too shallow to meaningfully differentiate three levels — the signal bleeds between levels.

### The deployment economics at trillion scale

One trillion-parameter training run. Everything after — deployment configuration, domain specialization, policy tuning, safety floor — via scale tables.

For an organization running hundreds of deployment variants, the difference between "retrain per variant" and "swap scale table per variant" is the difference between millions of dollars per variant and thousands. The trillion-parameter sign structure is shared infrastructure; the scale tables are configuration. This separation of concerns doesn't exist in FP16 architectures — every fine-tuned variant is a full copy of the model.

The auditable safety property scales directly: a safety floor is a single readable number regardless of model size. A 1T model with a floor=0.3 safety configuration is as inspectable as a 1.7B model with the same floor. The scale table that encodes it is larger (15.6GB vs 22MB) but structurally identical and equally readable.

### Dynamic scale tables — a new capability at scale

At 1T, a lightweight network that generates scale tables conditioned on input becomes practical. Instead of a static 15.6GB personality table, a 100M-param "scale network" produces a scale table per request — dynamic personality selection rather than fixed global configuration.

This is the continuous-manifold hypothesis from the blend experiments taken to its logical endpoint: rather than manually choosing blend points (α=0.7 math + 0.3 knowledge), a small network learns the optimal blend for each input. The router already does this for expert selection; a scale network does it for intensity configuration within each expert.

At 1.7B this would be overhead-heavy relative to the model. At 1T, 100M params is 0.01% of the model — negligible inference cost for the capability it provides.

---

## Summary

**The core claim** (derived from measured results, not itself measured): 1-bit quantization removes the primary motivation for MoE at scales below ~100-200B. Dense 1-bit + scale personalities is simpler, equally capable, and more inspectable than 1-bit MoE in this range.

**Above ~100-200B**, 1-bit MoE becomes relevant again for activation cost reasons, and per-expert scale tables create a two-level specialization architecture that has no analog in either dense 1-bit or FP16 MoE alone.

**At low-trillion scale**, the architecture matures: expert depth solves the precision problems that sign surgery addresses at small scale, three-level personality hierarchies become practical, and dynamic scale generation via a lightweight network becomes feasible. The trillion-parameter training run becomes shared infrastructure; scale tables become the configuration layer above it.

**The honest boundary:** this is architectural reasoning from first principles plus the scale personality mechanism. None of the MoE-specific or trillion-scale claims have been tested. The crossover point is an estimate. What's solid is the foundation: the mechanism is real at 1.7B and 8B, the storage math is measured, and the architectural logic follows from both.
