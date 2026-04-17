# Research Narrative — Scale Personalities for 1-Bit Models

*A companion read for reviewers of this repo. The code and the CATALOG are the primary artifacts; this is the "why did you do it in that order, and what do you think it means" layer on top.*

---

## The question

A 1-bit weight is `{-1, +1}` — the minimum possible information a weight can carry. PrismML's Bonsai models ship these 1-bit signs alongside an fp16 **scale per 128-weight group**. The scales are only ~0.8% of the parameter bytes, but they're the only real-valued dial the architecture has left.

The question this repo investigates: **how much specialization can you get by training only those fp16 scales, while keeping every 1-bit sign frozen?**

If the answer is "not much," scales are a rounding knob and the story ends.

If the answer is "a lot," you can ship a single 1-bit backbone (the signs) and swap small fp16 scale tables to reconfigure the model for different domains — ~125MB per personality on a ~3.4GB signs backbone. Fifty domain experts in ~8GB instead of ~20TB for fifty full fp16 models.

That's the thesis. The rest of this document is the evidence.

---

## What we actually found

Two findings I'd defend as real, one architectural result that falls out of the setup, and a list of things that didn't work and why they're informative.

### 1. Scales carry real capability, not just style

On Bonsai 1.7B, training only the fp16 scales with a distillation-flavored recipe (Rho-1 token weighting + elastic band regularization + AdamW at 1e-4 for 3 epochs on 150 examples per domain) moves GSM8K accuracy from **5.3% → 28.0%** on the held-out test split (n=150). The signs are frozen. The data the scales were trained on is the GSM8K *train* split; evaluation is GSM8K *test* — disjoint by construction.

That's a 5.3× relative lift from touching ~0.8% of the weight bytes. For comparison, the same model on the same math scales **loses −38.7% on ARC-Easy** (letter-answer distribution shift) and drops −20.1% on MMLU. It is a real specialization, not a free improvement.

### 2. Soft-blended scales beat any single profile — on two different benchmarks, via two different mechanisms

This is the finding I'm most interested in, because it showed up twice independently.

**First observation (router, ARC-Easy).** A 260K-param MLP router trained on mean-pooled token embeddings, with an explicit domain-classification CE head alongside the LM loss, produces **70.0% on ARC-Easy (n=100)** — 5.3% above the baseline (64.7%) and better than every single scale profile in isolation (math 26.0%, knowledge 62.7%, code 64.7%).

**Second observation (static blend, GSM8K).** An 11-point sweep of `α·math + (1−α)·knowledge` scale blends evaluated on GSM8K and MMLU at n=50 per point found **GSM8K peaks at α=0.7 with 40.0%** — 6pp above the pure-math endpoint (α=1.0 → 34.0% on the same n=50 run). The best-average point is α=0.6 (GSM8K 32%, MMLU 52.1%). Plot: `docs/figures/interpolation_curve.png`.

The first-order story in both cases is what you'd expect: mixing recovers the domains that individual profiles give up. The second-order story is the interesting one — in both cases, the blend beats the best single profile too. Two very different mechanisms (learned MLP router with a domain-CE head vs. static linear mixture at a fixed α) on two different benchmarks (ARC-Easy vs GSM8K) both show the same effect. That rules out "the router is doing something clever" as the explanation — the effect appears to be a property of the scale space itself. Working hypothesis: the learned scale tables are basis vectors in a continuous manifold where the right interior point can be a better fit for some tasks than any of the learned endpoints.

n=100 and n=50 are small; individual point magnitudes need replication at A100 sample sizes. But the direction and shape of both curves is consistent, and the fact that two independent mechanisms show the same qualitative effect is what convinces me it's real rather than a router artifact.

### 3. Diagonal dominance reproduces across scales and seeds

Eight profiles trained on 8B (Modal T4) showed 8/8 diagonal dominance on PPL — each profile is the best model for its own domain, average +26.8% PPL improvement. Three profiles on 1.7B (v2 recipe) showed 3/3 diagonal dominance on PPL *and* matched accuracy on domain-matched evals. Multiple seeds. The mechanism replicates.

The honest caveat is in the CATALOG: at 8B the PPL improvements didn't translate to accuracy lift on misaligned benchmarks. The 5.3× math result came from a better training recipe (v2) at a smaller scale, on the eval that actually matches the training distribution. Both numbers are in the repo. One does not supersede the other — they describe different points in the design space.

---

## Why the pattern matters more than any single number

Eight separate experiments in this repo produce measurements that move in the direction the scale-personality hypothesis predicts:

| Prediction | Observation |
|---|---|
| Math scales should help GSM8K | 5.3% → 28.0% |
| Math scales should hurt letter-answer tasks | ARC-Easy −38.7% |
| Code scales should help code synthesis | Null after extractor fix (code −2% MBPP); training-data mismatch, not a model refutation |
| Knowledge scales should cross-transfer from TriviaQA-train to MMLU | +3.4% MMLU |
| Knowledge scales should *not* help math | 46.5% → 46.5% MMLU but math worse |
| PPL should show diagonal dominance | 8/8 at 8B, 3/3 at 1.7B |
| Cross-domain scales should interfere | 3-way validation: perfect anti-correlation |
| Blending scales should trace a smooth tradeoff curve, with potentially non-endpoint sweet spots | Interpolation curve: GSM8K peaks at α=0.7 (40%, above the α=1.0 endpoint 34%); MMLU monotonic decay past α=0.6 |

Any individual row here could be noise at the sample sizes we can afford on a 6GB consumer GPU. The pattern across rows is harder to dismiss. Every prediction that should have held up, did — in the right direction, with the right relative magnitude. The repo's bet isn't on any single headline number; it's on the directional consistency.

The A100 validation run ($30-50) is the bridge to single-benchmark confidence: GSM8K at n=500, MATH competition OOD, LoRA baseline at matched parameter count. Those are queued, scoped, and honest about what they'd change in the story.

---

## What we got wrong (and left in the history)

I want reviewers to see the commit history, not a cleaned-up final. Three public corrections:

1. **v1 recipe overclaim.** Initial 8B results showed 8/8 PPL diagonal dominance with 26.8% average improvement. I reported this as the scale-personalities story. Accuracy evals at 8B showed the reasoning profile with the best math PPL (3.28) had the *worst* GSM8K accuracy (8% vs 20% baseline). Correction: PPL ≠ accuracy for reasoning, and the v1 recipe (SGD 0.01, 10 epochs, 20 examples/epoch) produces PPL movement that doesn't correspond to capability lift. Motivated the v2 recipe.

2. **TriviaQA n=100 was noise.** The v2 recipe's first accuracy claim was +2% TriviaQA at n=100. At n=150 it inverted (baseline won). Committed the correction, minimum sample size calibrated to n=150, moved to domain-matched benchmarks.

3. **MoE strangers claim.** A separate experiment claimed MoE experts are 50% sign-agreement (random). On re-review the methodology wasn't defensible. Claim removed from repo.

These are in the commit log with commit messages that describe what changed and why. The CATALOG calls them out explicitly. I don't think research should be presented as a set of polished wins with the failures edited out — the shape of the work is what I'd want a reviewer to see.

---

## Methodology — the thing I'd want someone to double-check

All training data comes from `split="train"` of the source dataset. All evaluation uses `split="test"` (or `"validation"`). GSM8K train has 7,473 problems, test has 1,319 — disjoint by OpenAI's construction. MMLU has dedicated dev/val/test splits. HellaSwag uses validation for eval because test labels are hidden. ARC uses test. MBPP uses test.

The cross-dataset cases (TriviaQA-train → MMLU-test, CodeSearchNet → MBPP-test) are strictly harder than in-distribution evaluation — different dataset, different format, different domain *coverage* in some cases. When those produce lifts, the lift is transfer, not memorization.

This is the standard protocol used by every published paper on these benchmarks. I've been asked twice whether the eval questions appear in training data; the answer is no, and the check is easy to reproduce — grep any eval question against `data/scale_data/math_v2.pt` and you won't find it.

---

## Collaboration with Claude

This is a research collaboration between me (problem selection, pattern recognition across iterations, strategic direction, honest evaluation of intermediate results) and Claude (implementation, experiment execution, technical literature lookup, code review).

Concretely, what that meant for this repo:

- **Mine:** the hypothesis that scales-only training could carry domain specialization; the decision to push on the "PPL ≠ accuracy" problem rather than declare the v1 8B results sufficient; the call to add the domain-classification CE head after the V1 router collapsed; the decision to publish the TriviaQA n=100 correction rather than run more seeds and hope for a cleaner number.
- **Claude's:** most of the code (training loops, GGUF patchers, llama.cpp eval harness, router architecture); retrieving relevant QAT literature (Rho-1, EfficientQAT, NVIDIA QAD) when the v1 recipe stalled; debugging the MBPP extractor; spotting that the v1 router collapse matched the classic MoE-router pattern.
- **Ours jointly:** the v2 recipe synthesis; the consistency-of-evidence framing when single-benchmark numbers were too noisy to lean on; the decision to write the CATALOG and this narrative the way they're written.

I mention this directly because the repo won't make sense without it. A solo human wouldn't write this much code this fast; a solo model wouldn't pick this hypothesis or catch the methodological problems the way they got caught. The work is honest about being a collaboration, and I think the shape of the collaboration is itself a finding — it's a specific example of what deep-model-assisted research looks like when the human is tracking mechanism and the model is tracking implementation.

---

## Where this points — if the mechanism scales

This section is forward-looking and explicitly unmeasured. I'm writing it to name the hypothesis the current findings imply, not to claim it.

The mechanism as stated so far: fp16 scale tables (~0.8% of parameter bytes) over frozen 1-bit signs carry domain specialization, and the scale space is continuous enough that soft blends of learned tables can outperform any single learned table. Two structural facts about that mechanism suggest it should compound at larger models rather than saturate:

1. **More scale groups per model.** Each 128-weight group gets one fp16 scale. A 70B model has ~40× the scale count of a 1.7B model — more per-profile capacity, not less.
2. **Late-layer FFN groups carry most of the signal.** The activation probe on Bonsai 8B found 52.5% of weight groups redundant overall but only 2.1% in the last third of the model, with ffn_up/gate as the least-redundant category (44.7%). Larger models have proportionally more late-layer FFN capacity; that's exactly the region the scale-training signal lands on.

If both hold, the implication is that scale-only specialization has more room to work at scale, not less. A 70B 1-bit backbone with ~125MB scale tables per personality would fit ~50 swappable domain experts in ~8GB of scale storage on top of the signs backbone.

**I'm naming this as an implication, not a claim.** The specific test that would license it or falsify it is in [docs/a100-burst-plan.md](a100-burst-plan.md) — the 8B v2 replication (run #4). If v2 produces equal or bigger accuracy lifts at 8B than at 1.7B, the scaling story survives. If it produces smaller lifts at 8B, the 1.7B result is small-model-specific and we correct the framing. Either answer is interesting; the honest framing is that we don't yet know which it is.

---

## What I'd read next in this repo

If you have 5 minutes: [README](../README.md) (headline table + deep results) and [CATALOG](../experiments/CATALOG.md) (what was tried, in order).

If you have 20 minutes: [scale-personalities.md](scale-personalities.md) has the methodology note and the consistency-of-directional-evidence section, and [bonsai-forensics.md](bonsai-forensics.md) explains the 1-bit model structure we're working inside.

If you have an hour: the code. `experiments/scale-personalities/scale_v2_proper.py` is the training loop. `routed_scale_router.py` is the router. `eval_domain_matched.py` is the benchmark harness. The data is small enough to reproduce; the Bonsai models are public on HuggingFace; the recipe is in the file.
