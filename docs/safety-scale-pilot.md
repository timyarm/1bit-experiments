# Safety-Scale Pilot — Plan

**Status:** proposed. Not yet run. Written before execution so predictions are pre-registered and not reverse-engineered from results.

## The question

The scale-personalities mechanism validates that fp16 scale tables (~0.8% of parameter bytes) carry domain-level distribution shift: math, knowledge, code profiles each move their own benchmark. The open question is whether scales can also carry **policy-level preference** — specifically, safety behavior (refusal of harmful prompts, appropriate responses on benign prompts).

Two possible outcomes, both informative:

1. **Scales can encode policy.** Safety training moves a safety benchmark (XSTest) monotonically with the safety-scale weight, same way math training moved GSM8K. If true → modular safety architecture becomes a real thing: the same 1-bit backbone can be configured at deploy time with a fp16 scale table that sets the safety level, auditable by reading the router weight.
2. **Scales cannot encode policy.** Safety requires sign flips (policy lives in the combinatorial structure of which weights are positive), not just scale reweighting. If true → this is a meaningful bound on scale-personality theory. It says domain is encodable as intensity but policy requires structure. Useful negative result.

Either outcome advances the research. The null result is not a failure mode of the pilot.

## Architecture — floor + boost

We *don't* make safety a single routed-to slot. The failure mode of classifier-gated safety is: router misclassifies "how do I make TNT" as "math", safety never engages. That's unacceptable for a safety claim.

Architecture proposal:

```
router_output = softmax(router(embeddings))          # 5-way over profiles
safety_weight = max(router_output[safety], FLOOR)    # clamp to floor
other_weights = rescale rest so sum = 1 - safety_weight
blended_scales = sum over profiles (weight * profile_scales)
```

Properties:
- **Classifier failure is not catastrophic.** Even if the router puts 0 weight on safety, the floor clamps safety_weight ≥ FLOOR. Harmful inputs still get the safety dose.
- **Classifier success still matters.** On a detected harmful prompt the router can route safety up to ~60%+, amplifying the effect.
- **Legible.** Maps to constitutional AI's "pervasive preference" framing more cleanly than a routed slot.

## The core measurement: Pareto curve

Sweep FLOOR in {0%, 10%, 20%, 30%, 40%, 50%}. At each level, eval both a safety benchmark and a capability benchmark. Plot the tradeoff curve.

Predictions (pre-registered):

- **FLOOR = 0%:** baseline safety, full capability.
- **FLOOR increasing:** safety metric improves monotonically *if* scales can encode policy. Capability metric degrades as the safety profile dilutes task-specialized profiles.
- **Sweet spot (if it exists):** some FLOOR value where safety is materially improved (e.g., +15pp on XSTest) while capability drops by less than noise (< 5pp on GSM8K/MMLU).
- **Null outcome:** XSTest flat across all FLOOR values. Would mean the scale table can't move the refusal distribution, and safety lives in the signs.

## Datasets

### Training the safety scale

[**Anthropic/hh-rlhf**](https://huggingface.co/datasets/Anthropic/hh-rlhf), `harmless-base` subset. Take the "chosen" (preferred, harmless) responses as the training distribution. ~40K examples available; we'll use 150 to match v2 recipe protocol (and train on 300 and 500 for the data-efficiency comparison if we have budget).

Format to match v2 recipe:
```
Human: {prompt}
Assistant: {chosen_harmless_response}
```

Why HH-RLHF and not a dedicated refusal set: HH-RLHF is the same data Anthropic trained on, so the distributional signal is clean, and using their public data is more legible than using a third-party refusal set.

### Eval

Safety: [**XSTest**](https://huggingface.co/datasets/Paul/XSTest) (Röttger et al. 2023). 250 prompts: 200 "unsafe" (should refuse or respond safely) + 50 "safe-but-superficially-concerning" (should NOT refuse). Measures both under-refusal (capability-destroying safety) and over-refusal (annoying safety). n=250 full, fits our protocol.

Capability (to measure the cost): re-use GSM8K n=150 + MMLU-Knowledge n=144 from the existing eval harness. Same protocol as the scale-personalities headline results, so safety-floor numbers are directly comparable to the baseline/math/knowledge numbers already in the repo.

## Protocol

1. **Train safety scale** using the v2 recipe (AdamW 1e-4, Rho-1 token weighting, elastic band reg λ=0.1, seq_len 256, 3 epochs) on 150 HH-RLHF harmless-base examples. Snapshot the trained scale table same way the math/knowledge/code profiles are snapshotted.
2. **Standalone check.** Eval safety-scale-only (no router, no blending) on XSTest. Does it move? If it can't move safety at all in the isolated case, the router pilot won't either — fail fast.
3. **Retrain router** to 5-way (baseline, math, knowledge, code, safety) with the domain-CE head extended to 5 classes. Use the same training mix (GSM8K/TriviaQA/CodeSearchNet/WikiText) + HH-RLHF harmless data.
4. **Sweep FLOOR ∈ {0, 0.1, 0.2, 0.3, 0.4, 0.5}.** At each floor: patch GGUF with the floor-constrained blend for a single representative input type (or rerun router inference per-example and aggregate — the cheap version patches with each floor's mean routing over eval set), eval XSTest + GSM8K + MMLU.
5. **Plot Pareto.** X-axis: GSM8K accuracy. Y-axis: XSTest (safe-refusal rate — want this high). Points colored by FLOOR value. The shape of this curve is the core finding.

## Success criteria

- **Minimum viable result:** XSTest changes monotonically with FLOOR. Proves scales carry at least some policy signal. Publishable as "scale personalities extend to policy domains."
- **Strong result:** there exists a FLOOR value where XSTest improves ≥ 15pp vs baseline while GSM8K/MMLU drop less than noise at n=150 (< 5pp). A real Pareto-frontier point.
- **Null result:** XSTest flat ± noise regardless of FLOOR. Publishable as "scale-only training cannot encode refusal policy; signs must flip. Useful bound on the scale-personality hypothesis."

## Compute estimate

- Safety scale training: ~30 min (GTX 1660 Super, matches other profiles)
- Router retrain with 5th class: ~45 min (slightly larger router, more training mix)
- Sweep eval: 6 floor levels × (XSTest 250 + GSM8K 150 + MMLU 150) × ~2s per eval via llama.cpp ≈ 3.5 hr
- Total wall clock: ~5 hr on GPU box

Fits in a single focused session on the local GPU; no cloud needed.

## Bait questions we've considered

- **"Why not just do constitutional AI fine-tuning?"** That requires sign flips. This pilot is specifically testing whether fp16 scales alone — ~0.8% of parameter bytes — can carry safety policy. Complementary, not competing.
- **"Is XSTest enough?"** n=250, public, measures both over- and under-refusal. Standard-ish for a pilot. If the pilot shows signal, AdvBench (520 adversarial prompts) is the natural follow-up at A100 scale.
- **"What if safety scale tanks all capability?"** That's what the floor+boost architecture and the FLOOR sweep are designed to measure — the tradeoff curve IS the finding, whether or not there's a sweet spot.
- **"Are the HH-RLHF chosen responses diverse enough to teach safety?"** Unknown — that's part of what this tests. If the standalone check (step 2) fails, we can fall back to a dedicated refusal-tuning set before moving to the router sweep.

## Why this matters for the research artifact

The scale-personalities track has clean results on three positive profiles (math ✓, knowledge ✓, code ✗ for data reasons) plus an anti-forgetting router. Safety as the fourth positive profile — or as a clean negative result bounding the theory — is the test that shows whether the mechanism generalizes from domain to policy. "Here is the curve we measured" beats "here is what we think the mechanism predicts" by a significant margin.

## Not in scope

- Red-teaming / adversarial evaluation. That's the follow-up after the Pareto curve exists.
- Comparison against a full SFT baseline on HH-RLHF. Matched-param LoRA comparison is already queued for A100 generally; extending it to safety is a separate run.
- Multi-turn safety (conversational context). Single-prompt eval only.
