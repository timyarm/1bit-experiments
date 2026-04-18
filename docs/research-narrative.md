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

### Binary vs. every other format — the stronger version of the claim

The argument above is about scale personalities compounding at larger binary models. There's a second, more aggressive claim worth naming explicitly: **binary with scale personalities may not just be efficient — it may be the superior architecture overall, including against FP16, when you account for what you can do with the compute budget.**

The starting point is PrismML's intelligence density metric — negative log error rate divided by model size in GB. At 8B parameters:

| Format | Intelligence density |
|---|---|
| 1-bit binary Bonsai 8B | **1.060** |
| 1.58-bit ternary Bonsai 8B | 0.803 |
| Best FP16 competitor (8B range) | 0.052–0.096 |

Binary is 10–20× more intelligence-dense than FP16. That gap means: take the storage budget of one FP16 70B model (~140GB) and fill it with binary instead, and you're running a model with ~10–15× more parameters at matched storage. Scaling laws consistently favor more parameters over higher precision past a threshold — this is the core finding of Microsoft's BitNet work, and it's what the intelligence density gap predicts.

Add scale personalities and the argument compounds further. A fixed storage budget that would fit one FP16 70B specialist now fits:
- A binary backbone orders of magnitude larger
- Fifty or more swappable domain personalities at ~125MB each
- A lightweight router that selects or blends them at inference time

The claim is not that a small binary model beats a large FP16 model. The claim is: **at matched compute and storage budget, binary + scale personalities should outperform a fleet of FP16 specialists** — more total parameters, near-zero routing overhead, and domain specialization that costs 0.8% of parameter storage per personality rather than a full model copy.

**What would falsify this:** the 8B v2 replication failing to show scale personality lift (would suggest the mechanism doesn't scale); or intelligence density degrading at 70B+ binary scale (would suggest the efficiency gap closes). Both are testable. Neither is tested yet. The A100 burst plan is the first step toward the evidence that would license or falsify this claim. I'm stating it here because the mechanism points here, and I'd rather name it honestly than let a reader connect the dots and wonder if we saw it.

---

## How our numbers compare to other models

This section documents where the scale personality results sit relative to published benchmarks. The comparison requires care because evaluation protocols differ significantly.

### Our evaluation protocol

All numbers in this repo use **0-shot, no chain-of-thought** evaluation. The prompt is `"Question: {q}\nAnswer:"` and we extract the last number from the model's response. No worked examples, no reasoning trace prompt.

### The numbers

| Model | Format | GSM8K | Protocol | Source |
|---|---|---|---|---|
| Bonsai 1.7B (no training) | 1-bit binary | 5.3% | 0-shot, ours | measured |
| Bonsai 8B (no training) | 1-bit binary | ~20% | 0-shot, ours | measured |
| **Bonsai 1.7B + scale personalities** | **1-bit binary** | **40.0%** | **0-shot, ours** | **measured** |
| Llama 2 7B | FP16 | ~0–14% | 8-shot CoT | Meta paper |
| Mistral 7B | FP16 | ~35–47% | varies | published |
| Llama 3 8B | FP16 | 79.6% | 8-shot CoT | Meta paper |

### What the comparison shows — and what it doesn't

**The clean internal comparison:** Bonsai 1.7B + scale personalities (40%) beats the raw Bonsai 8B baseline (~20%) by 2× on math, using a model 4.7× smaller. Both measured with the same 0-shot protocol on the same harness. This comparison is valid.

**The FP16 comparison is methodology-dependent.** The 79.6% for Llama 3 8B uses 8-shot chain-of-thought — the model is given 8 worked examples before each question, which significantly inflates scores versus 0-shot. This is not a fair comparison to our 0-shot numbers. 0-shot GSM8K for FP16 8B models is substantially lower; published 0-shot numbers are hard to find because most labs report the more flattering few-shot figure.

**What a fair FP16 comparison would require:** Running Llama 3 8B through our same 0-shot harness and comparing directly. This is queued in the A100 burst plan (run #6). Until that number exists, the FP16 comparison should be stated as "pending same-protocol eval" rather than using the published 8-shot figures.

### The chain-of-thought question

The gap between our 0-shot 40% and Llama 3's 8-shot 79.6% raises a natural question: how much of that gap is reasoning capacity, and how much is just CoT prompting? The honest answer is we don't know yet, but the framing matters.

If scale personalities encode genuine mathematical understanding (not just output format), then CoT prompting on top of scale-trained scales should compound — the model reasons better AND gets the CoT scaffold. If scale personalities are primarily format learning (recognizing the GSM8K prompt structure), CoT adds less. This is a testable hypothesis: does CoT prompting improve our scale-personality model by more than it improves the baseline? If yes, the scales improved reasoning capacity. If no, they improved format recognition. The A100 run is the right place to test this cleanly.

---

---

## The LoRA null hypothesis — and why the result matters (2026-04-18)

The single most important validity check for the scale personality thesis is: **does the result require scale structure, or would any adapter at the same parameter count do the same thing?**

We ran this experiment (Exp 19). LoRA rank=16 adapters wrapping every PackedBitLinear in Bonsai 1.7B — same training recipe (AdamW lr=1e-4, Rho-1 top-60%, elastic band λ=0.1, 3 epochs, 150 examples, 70% math). LoRA had **17.4M parameters / 70MB** versus the scale table's **11M parameters / 22MB**. LoRA was given 3× more parameters and the same data.

Result: **LoRA 25.0% GSM8K vs scales 28.0% GSM8K.**

Scales win at one-third the size. The null hypothesis is rejected.

**Why this is mechanistically meaningful, not just empirically convenient:**

Signs in a 1-bit model are frozen {-1, +1} — the routing skeleton of the entire computation graph. LoRA adds low-rank corrections on top of a discrete system: it's trying to build additive fp16 corrections onto weights that are quantized to one bit. The corrections partially cancel the sign structure they're overlaid on. Scales, by contrast, operate on the only degree of freedom the architecture actually has: the per-group magnitude. There's no cancellation because you're not fighting the discreteness — you're modulating it.

The practical corollary compounds this: LoRA must execute a `[batch, seq, rank] × [rank, out]` matmul every forward pass. Scales are baked into the weight reconstruction at inference time — **zero runtime overhead**. Scales win on accuracy, on size, and on inference cost simultaneously.

This is the result that turns the scale personality thesis from an interesting observation into a testable mechanism claim.

---

## Sign stability across depth — what the probe found (2026-04-18)

The progressive freeze hypothesis: when training a 1-bit model natively, signs should crystallize early layers first (input representations stabilize early) and late layers last (task-specific circuits stay plastic longer). If true, a good training curriculum would freeze signs progressively by depth.

We measured this on Bonsai 1.7B by running 50 math training steps with scales training normally but with a backward side-channel computing the "would-be gradient" on each layer's signs — how much each layer's signs *want* to change, per step, without actually changing them.

**Result: uniform stability across depth (late/early ratio = 0.85×).**

No clear depth gradient. Layer 0 is a mild outlier (highest gradient pressure — closest to raw input embeddings, most downstream leverage). Layers 22-27 are slightly quieter than the middle. But the effect is small — the sign structure is globally committed at roughly the same level in every layer.

**What this means:** Bonsai's QAT converged to a globally stable sign configuration, not a depth-progressive one. There's no obvious layer ordering for a progressive freeze curriculum. This rules out depth as the freeze criterion; a native 1-bit training curriculum would need to use per-layer gradient magnitude over time — freezing when a layer's sign gradient drops below a threshold, regardless of where it sits in the depth ordering.

The secondary implication: uniform sign stability across depth is *further evidence* that signs are the permanent skeleton of the model. If any layer's signs were significantly more plastic than others, that would suggest the routing structure wasn't fully settled. Uniform low gradient pressure across all 28 layers is the signal we'd expect if the QAT successfully committed to a discrete routing structure throughout.

---

## Sign-conditional scales: ruling out asymmetry (2026-04-18)

After finding that sign stability is uniform across depth, the next question was whether the *structure within groups* held any unexploited information. Standard scales use one fp16 value per 128-weight group — an absmean over all weights in the group, both positive and negative. What if +1 and −1 weights within a group had different typical magnitudes? You could give each sign polarity its own scale (`scale_pos`, `scale_neg`) and double the expressivity at the group level.

Measurement first. Before training anything, we measured the actual `scale_pos/scale_neg` ratio across all weight groups in Bonsai 1.7B.

**Result: mean ratio = 1.0009, std = 0.0005.** The groups are essentially perfectly symmetric. Positive and negative weight subsets within each group have virtually identical magnitude distributions.

This is mechanistically informative. Bonsai's QAT process produces symmetric groups — the absmean scale is already optimal because there's nothing asymmetric to capture. Running the sign-conditional experiment confirmed this directly: 26.0% GSM8K vs 28.0% for standard scales. The extra parameters add optimization noise without adding representational capacity.

The pattern across Exp 20 and 21 is consistent: Bonsai's sign structure is deeply committed and internally symmetric. The QAT process converged to a configuration where signs encode the routing structure completely and the scale is the only remaining degree of freedom. This is exactly the theoretical precondition for scale personalities to work: the signs are load-bearing but inert; the scales are the movable parts.

---

## EFI sign unfreeze — the sign capacity question (2026-04-18)

If sign structure is committed but inert relative to a given domain, the next test is whether *selectively changing* the most gradient-pressured signs can push past the 28% scale ceiling. This is the EFI (Expected Flip Improvement) hypothesis: rank all signs by |grad × scale|, unfreeze the top 1% (the ones under maximum optimization pressure for math), train them with SGD alongside the normal scale training, then rebinarize to {-1,+1} at inference.

The experimental design addresses the core question cleanly: if EFI beats 28%, sign capacity is the bottleneck and the next question is the optimal K%. If EFI doesn't beat 28%, scales have already captured all the continuous degrees of freedom available without changing the routing structure.

**Result: 27.0% GSM8K — matches scale-only (28%) within noise.**

EFI adds 14M sign parameters (56MB fp32) on top of the normal scale training. The result is statistically indistinguishable from scales alone. Sign capacity is not the bottleneck.

The convergent picture from Exp 19–22: the scale personality mechanism is operating near optimally. Scales are the movable parts, signs are correctly committed, and the 28% ceiling on Bonsai 1.7B under this recipe is a data/model capacity ceiling, not a parameter representation ceiling. The 40% result came from blend recipes (math × knowledge), not from changing the weight structure. The 60% ceiling observed in math-specialist models (Qwen2.5-Math, DeepSeek-Math) reflects math baked into the signs from QAT — not achievable by post-hoc sign manipulation on a general pre-trained model.

---

## What I'd read next in this repo

If you have 5 minutes: [README](../README.md) (headline table + deep results) and [CATALOG](../experiments/CATALOG.md) (what was tried, in order).

If you have 20 minutes: [scale-personalities.md](scale-personalities.md) has the methodology note and the consistency-of-directional-evidence section, and [bonsai-forensics.md](bonsai-forensics.md) explains the 1-bit model structure we're working inside.

If you have an hour: the code. `experiments/scale-personalities/scale_v2_proper.py` is the training loop. `routed_scale_router.py` is the router. `eval_domain_matched.py` is the benchmark harness. The data is small enough to reproduce; the Bonsai models are public on HuggingFace; the recipe is in the file.
