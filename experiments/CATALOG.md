# Experiment Catalog

Chronological log of every experiment in this repo: what was tried, what was expected, what happened, and what was concluded. Includes failed attempts — the negative results shape the track as much as the wins.

Format per entry: **Date · Experiment · Expectation · Outcome · Conclusion.**

---

## April 2026

### 2026-04-11 · 8-Profile Training on Bonsai 8B (v1 recipe, Modal T4)

**Expectation:** Train 8 domain-specific scale tables on Bonsai 8B (tool_use, reasoning, structured, retrieval, verification, planning, compliance, creative). PPL should drop for each profile on its own domain. Accuracy should follow PPL.

**Setup:** SGD lr=0.01, 10 epochs, 20 examples per epoch per profile, custom NativeBitLinear with scales as `nn.Parameter`, gradient checkpointing.

**Outcome:** 8/8 diagonal dominance on PPL (every profile best on its own domain, avg +26.8% improvement). **Accuracy did NOT follow PPL.** Reasoning profile had best math PPL (3.28) but worst GSM8K (8% vs 20% baseline). TriviaQA DID improve for structured/compliance profiles (+20% relative) but the accuracy story was messier than PPL suggested.

**Conclusion:** PPL is a necessary but not sufficient signal. Scales learn token-level distribution shifts that help PPL but can hurt reasoning. Need a better training recipe for accuracy lift.

---

### 2026-04-11 · 3-Way Cross-Domain Validation

**Expectation:** Test whether math scales help code and vice versa. If scales are truly domain-specific, cross-domain help should be minimal or negative.

**Setup:** Train math / lang / code scales on 1.7B. Eval all 3 on math, lang, code PPL.

**Outcome:** Perfect diagonal dominance. Math scales hurt code (2.19 → 2.30 PPL). Code scales hurt math (4.91 → 5.01 PPL). Each personality learns *specific* sub-skills, not general improvement.

**Conclusion:** Scale personalities are genuinely domain-specialized. No free lunch — specializing one direction costs you elsewhere. Motivated the routing experiments.

---

### 2026-04-11 · Activation Probe on Bonsai 8B

**Expectation:** Identify which weight groups actually respond to scale training. Hypothesis: not all layers equal.

**Outcome:**
- Early layers (0-10): 99.6% redundant (uniform activations)
- Late layers (21-31): 2.1% redundant
- ffn_up / ffn_gate: 44.7% redundant (least redundant = most trainable)
- Overall: 52.5% of groups underutilized

**Conclusion:** Scale training should target ffn_up/gate in late layers. Half the model is currently wasted compute during scale training. Open follow-up: targeted training (Tier B).

---

### 2026-04-15 · Initial v1 Port to Bonsai 1.7B (Local GTX 1660 Super)

**Expectation:** Replicate 8B results at 1.7B on local hardware (6GB VRAM). Used original SGD recipe.

**Outcome:** PPL diagonal dominance reproduced (3/3). **Accuracy: flat across TriviaQA, ARC-Easy, HellaSwag.** The "PPL ≠ accuracy" pattern persisted.

**Conclusion:** Recipe change needed, not just scale-up-to-8B-solves-it. Dispatched literature review agent (Rho-1, EfficientQAT, NVIDIA QAD). Found: we were training with wrong LR (100× too high), wrong seq length (4× too short), no token weighting, no regularization.

---

### 2026-04-16 · v2 Recipe — First Accuracy Lift (Local, 1.7B)

**Expectation:** Literature-informed recipe might finally produce accuracy lift. Apply Rho-1 token weighting + elastic band regularization + AdamW 1e-4 + seq 256 + mixed data + 3 epochs.

**Outcome:** PPL diagonal dominance: 3/3 again, cleaner.
- TriviaQA at n=100: knowledge scales 9% vs baseline 7% (+2% absolute). **First claimed accuracy improvement.**

**Conclusion:** Reported as a win. [**Later corrected — see next entry.**]

---

### 2026-04-16 · Multi-Benchmark Eval at n=150 (Sample-Size Correction)

**Expectation:** Validate the TriviaQA +2% result at larger sample size and add ARC-Easy / HellaSwag.

**Outcome:** **The TriviaQA win inverted at n=150.** Baseline 9.3%, knowledge 7.3%. The +2% at n=100 was noise. Math scales showed +6.7% on HellaSwag but **−38.7% catastrophic forgetting on ARC-Easy**. Knowledge and code were flat within noise.

**Conclusion:** Public correction committed. Calibrated minimum sample size to n=150. Identified the catastrophic-forgetting pattern (math distribution shift crushes letter-answer tokens). Started framing scales as "distribution shift, not capability" — temporarily.

---

### 2026-04-16 · Domain-Matched Evals (GSM8K / MMLU / MBPP)

**Expectation:** The "scales are just style" framing was suspicious — earlier benchmarks were all misaligned to profile domains. Test each profile on its intended domain.

**Setup:** Math on GSM8K (test split, held-out from training), knowledge on MMLU-Knowledge subjects, code on MBPP. n=150 / 144 / 100 respectively.

**Outcome:**
- Math → GSM8K: **5.3% → 28.0% (+22.7% absolute, 5.3× relative)** 🎯
- Knowledge → MMLU: 43.1% → 46.5% (+3.4%) — cross-dataset transfer
- Code → MBPP: 0.0% across all four models (baseline included) — **extraction/eval bug**
- Math → MMLU: −20.1% (catastrophic forgetting, same pattern as ARC-Easy)

**Conclusion:** Reversed the style-only read. Scales produce real in-distribution capability lift on matched tasks. Catastrophic forgetting is predictable (letter-answer-token collapse) and motivates routing. MBPP 0% needs diagnosis.

---

### 2026-04-16 · Per-Sequence Scale Router V1 (LM loss only)

**Expectation:** Router trained on LM loss should learn to pick the right profile for each input, stacking gains while side-stepping forgetting.

**Outcome:** Training ran but router collapsed to near-uniform routing regardless of input domain. Classic MoE router collapse — LM loss alone gave insufficient signal to separate domains.

**Conclusion:** Abandoned after epoch 2. Added explicit domain-classification CE head for V2.

---

### 2026-04-16 · Per-Sequence Scale Router V2 (LM + domain CE)

**Expectation:** Added CE loss with known-domain labels alongside LM loss. Should break the collapse and produce diagonal routing.

**Setup:** 480 examples (120 per domain), 3 epochs, ~30 min training + ~1 hr eval on local GPU.

**Outcome:**
- Training converged: LM 3.01 → 2.79, domain CE **1.69 → 0.96** (43% drop)
- Routing: 3/4 domains diagonal-dominant (math→math, knowledge→knowledge, general→baseline); code was three-way tied
- **ARC-Easy: 64.7% baseline → 70.0% router (+5.3%)** — beats every single profile
- TriviaQA: −3.3% vs baseline (soft blending hurts when one profile is correct)
- HellaSwag: near-flat
- **Catastrophic forgetting eliminated.** Math-alone: ARC 26.0%. Routed: 70.0%.

**Conclusion:** Router works as mechanism-level safety rail AND produces emergent compounding on ARC-Easy. Major paper-worthy finding: soft blending of scale tables can exceed any individual table.

---

### 2026-04-17 · MBPP Extractor Diagnosis (Fixing the 0% Hole)

**Expectation:** Either Bonsai 1.7B genuinely can't synthesize Python, or the code-extraction regex is broken.

**Setup:** Diagnostic script that dumps raw model output + extracted code + test result for 10 examples.

**Outcome:** Bug found. The prompt ended with an opening ` ```python ` fence, so the model's response started with actual code and terminated with a closing ` ``` `. My extractor was taking `parts[1]` (content after the closing fence, i.e., model's self-commentary) instead of `parts[0]` (the code). **With fixed extractor: 5/10 passing on 10-example diagnostic.** Full n=100 eval in flight.

**Conclusion:** Extraction bug, not model capability. Code profile can finally be measured.

---

### 2026-04-17 · MBPP Full Re-Eval with Fixed Extractor (n=100)

**Expectation:** Code profile should beat baseline on MBPP now that the extractor isn't throwing the answer away. Math should be flat or slightly better (code-like reasoning leaks). Knowledge should be flat.

**Setup:** Same eval harness as diagnostic but at full n=100 across all four profiles (baseline, math, knowledge, code). Single run, seed-fixed order.

**Outcome:**
- baseline: 24.0% (24/100)
- math:     26.0% (+2.0%)
- knowledge: 24.0% (flat)
- **code:   22.0% (−2.0%)** — null/slightly negative

**Conclusion:** Honest null result for the code profile on MBPP. Two likely causes, both instructive: (1) **training-distribution mismatch** — CodeSearchNet Python is GitHub function bodies (long, doc-stringed, library-style), MBPP is tight task-description → small-function synthesis; the distribution shift the scales learned doesn't help on the eval format; (2) **tokens that matter are already shared** — unlike GSM8K where math-distribution shift is a big lever, most MBPP correctness tokens are generic Python syntax the baseline already handles. Either way: code profile v2 recipe needs reworked training data (e.g. MBPP/HumanEval-style short-prompt + solution pairs) rather than more epochs on CodeSearchNet. Filed as a negative result, not a failure.

---

### 2026-04-17 · Scale Interpolation Curve (math ↔ knowledge, 11-alpha sweep)

**Expectation:** If the scale space is a continuous manifold, blending math and knowledge scales at intermediate ratios should produce intermediate behavior — gradual tradeoff along a smooth curve. A weaker form of the router's emergent-compounding finding: can a static blend at the right α also exceed either endpoint?

**Setup:** 11 alphas in [0.0, 1.0] step 0.1. Blended scales = α·math + (1−α)·knowledge. Patched GGUF at each α, eval GSM8K (n=50) + MMLU-Knowledge (n=50). 26 min wall clock on GPU box.

**Outcome:**

| α | GSM8K | MMLU |
|---|---|---|
| 0.0 (pure knowledge) | 6.0% | 56.2% |
| 0.2 | 8.0% | 54.2% |
| 0.4 | 20.0% | 56.2% |
| 0.5 | 28.0% | 50.0% |
| 0.6 | 32.0% | 52.1% |
| **0.7** | **40.0%** | 41.7% |
| 0.8 | 28.0% | 33.3% |
| 0.9 | 28.0% | 20.8% |
| 1.0 (pure math) | 34.0% | 18.8% |

Best-average point: α = 0.6 (GSM8K 32%, MMLU 52.1%). Plot: [docs/figures/interpolation_curve.png](../docs/figures/interpolation_curve.png).

**Key findings:**

1. **GSM8K peaks at α = 0.7 (40%) — above pure math's 34% endpoint.** Soft blending two scale profiles produces a better math model than either in isolation. This replicates the emergent-compounding pattern first seen with the router on ARC-Easy (router 70% vs best single profile 64.7%), but now in a static 2-profile blend. Two independent observations of the same effect on different benchmarks.

2. **MMLU is remarkably stable up to α = 0.4 (stays at 54-56%)**, then collapses past α = 0.6. Suggests an asymmetry: moderate math-scale contamination doesn't hurt knowledge recall much, but knowledge-scale dilution of math scales hits steeply past ~60% math weight.

3. **α = 0.4-0.6 is a clear Pareto sweet spot.** GSM8K jumps 3-5× vs pure knowledge while MMLU stays within 4 pts of pure-knowledge peak. A practical deployment configuration for a math-capable model that retains knowledge performance.

**Caveats:** n=50 per point per benchmark is small. Individual α differences may be within noise. The *shape* of the curve is robust — monotonic rise on GSM8K, inflection at α ≈ 0.6 on MMLU — but specific point values need replication at larger n.

**Conclusion:** Supports the continuous-scale-manifold hypothesis. Two independent observations of emergent compounding (router ARC-Easy +5.3%, interpolation GSM8K +6pp above pure-math endpoint) on very different mechanisms (learned MLP router vs static linear blend) suggest this isn't a router artifact — it's a property of the scale space itself. The A100 re-run at n=400+ is the obvious follow-up.

---

### 2026-04-17 · Data Efficiency Curve — math scales at n ∈ {10, 30, 300}

**Expectation:** Probe whether the 150-example training set used for the headline GSM8K 5.3× result is data-starved or saturated. Train math scales on progressively larger datasets {10, 30, 100, 300} and measure GSM8K n=100 at each. If the curve is still rising at 300 → n=150 is under-trained. If it plateaus earlier → 150 is past the elbow.

**Setup:** v2 recipe (AdamW 1e-4, Rho-1 token weighting, elastic band reg, seq 256, 3 epochs), identical to the scale_v2_proper.py run that produced the 28% headline, varying only the number of training examples. Each n trained as an isolated python subprocess after an earlier "CUDA error: unknown error" from in-process back-to-back trainings turned out to be residual GPU state. **n=100 crashed at first backward() across three attempts** (fresh subprocess each time), each with the same identical shuffled mix (seed=42, 70 math + 30 diverse) — specific data shuffle triggers a reproducible CUDA kernel issue on the GTX 1660 Super. n=10, 30, 300 all succeeded on the same hardware with the same script; the failure is specific to the n=100 mix. Left as a documented gap rather than forcing a different seed (would break apples-to-apples with the other points).

**Outcome:**

| n | GSM8K | delta vs baseline |
|---|---|---|
| 0 (baseline) | 5.0% | — |
| 10 | 19.0% | +14.0% |
| 30 | **29.0%** (peak) | +24.0% |
| 100 | (crashed, retrying) | |
| 150 (headline, prior run) | 28.0% | +22.7% |
| 300 | 24.0% | +19.0% |

Training loss at n=300 was still decreasing monotonically (epoch 1: 2.85 → epoch 2: 2.05 → epoch 3: 1.61), but GSM8K went down. Textbook overfitting signature. Plot: [docs/figures/data_efficiency_curve.png](../docs/figures/data_efficiency_curve.png).

**Key findings (honest version, accounting for n=100 eval σ ≈ 4.5pp):**

1. **~30 examples is enough to extract most of the available signal.** The curve jumps 5% → 19% → 29% from n=0 → 10 → 30, then plateaus. n=150 (28%) and n=30 (29%) are statistically indistinguishable at eval-n=100; the point estimate peak at n=30 is within noise but *consistent* with the plateau.
2. **n=300 shows meaningful degradation.** The 5pp drop from peak (29% → 24%) is ~1σ below peak and comes with monotonically-decreasing training loss — the classic overfitting signature. The extra training data is actively hurting on the held-out split.
3. **The 5.3× headline was not data-limited.** We could have gotten the same number with 5× less training data.

**What this says about the mechanism.** Scale-only training of 11M fp16 parameters on a frozen 1.7B 1-bit backbone *converges* on a tiny training set — on the order of 30 examples for the math capacity envelope of this model with frozen signs. This is consistent with the architectural claim that scales carry domain-level distribution shift (a small number of examples suffices to identify the shift), not complex new capability (which would need more data). It's also consistent with the interpolation finding that the math-scale endpoint maps to a specific region in the scale manifold — once you're in that region, more training pushes you past it into overfitting.

**Caveat:** All at eval n=100 with single seed. The specific "n=30 beats n=150" claim is not statistically defensible. The "plateau around 28-29%, degradation at 300" pattern is what the data supports.

---

---

## Experiment 14 — Per-layer α blend sweep (2026-04-17)

**Question:** Does math specialization live in early or late layers? If late layers carry signal (as the activation probe implied), giving late layers math scales should beat giving early layers math scales.

**Setup:** 6 profiles × (GSM8K n=50 + MMLU n=50). No new training — existing math/knowledge scale tables patched per-layer. Bonsai 1.7B local GPU. `eval_layerwise_blend.py`.

| Profile | GSM8K | MMLU | Notes |
|---|---|---|---|
| flat_0.7 (control) | 40.0% | 41.7% | Wins on GSM8K |
| linear_late (knowledge early, math late) | 2.0% | 56.2% | Math collapses completely |
| linear_early (math early, knowledge late) | 28.0% | 33.3% | Math survives |
| step_late | 2.0% | 58.3% | Confirms linear_late |
| step_early | 32.0% | 27.1% | Confirms linear_early |
| ffn_late (ffn_up/gate top half = math) | 22.0% | 50.0% | Partial fix — softens but doesn't solve |

**Result: Math specialization lives in EARLY layers, not late layers.** Giving late layers math scales destroys GSM8K (2%). Giving early layers math scales preserves it (28-32%). The flat α=0.7 still wins because it doesn't force hard layer boundaries. Late layers carry knowledge/general capability — MMLU spikes to 56-58% when late layers get knowledge scales.

**Contradiction with activation probe:** The probe found late layers are less redundant (2.1% vs 99.6% early), which suggested late layers carry signal. Correct reading: late layers are specialized for *general reasoning and knowledge retrieval*, not domain-specific math encoding. Early layers handle domain recognition (what kind of problem is this?); late layers handle domain completion (how to answer it). The probe measured redundancy, not domain-specificity.

**Implication for #12:** Original plan was to train ffn_up/gate in late layers. This finding corrects that — train ffn_up/gate in *early* layers for math. Direct course correction before burning a training run.

**Implication for safety pilot:** Safety behavior is likely a late-layer phenomenon (how to complete a response, not what kind of problem it is). Safety scale pilot should target late layers.

---

## Experiment 15 — Asymmetric blend sweep (2026-04-17)

**Question:** Can a graduated asymmetric blend (early layers higher α, late layers lower α) beat flat_0.7 on both GSM8K and MMLU simultaneously?

**Setup:** `eval_asymmetric_blend.py`. Two sweeps: (1) alpha grid — early_α ∈ {0.8, 0.9, 1.0} × late_α ∈ {0.3, 0.4, 0.5}, split fixed at layer 12. (2) Crossover sweep — early=0.9, late=0.4, split ∈ {6, 8, 10, 14, 16, 18}. 16 total evals. No training. 85 min.

**Key results:**

| Profile | GSM8K | MMLU | vs flat_0.7 |
|---|---|---|---|
| flat_0.7 (control) | 40.0% | 41.7% | baseline |
| e0.8_l0.5_s12 | 42.0% | 35.4% | +2pp math, -6pp knowledge |
| e0.8_l0.4_s12 | 36.0% | 43.8% | -4pp math, +2pp knowledge |
| e0.9_l0.4_s6 | 34.0% | 45.8% | best MMLU ever, GSM8K collapses |
| e0.9+ | 26-38% | 22-31% | both collapse |

**Result: flat_0.7 is Pareto optimal.** No static asymmetric pattern beats it on both benchmarks simultaneously. Higher early α (0.9+) collapses both. The best MMLU ever (45.8% at s6) comes at the cost of GSM8K dropping to 34%.

**Mechanistic interpretation:** Depth structure is real but soft — the math/knowledge encoding is interleaved across layers in a way that hard splits disrupt. Flat α=0.7 approximates the optimal smooth per-layer gradient better than any fixed step or ramp pattern. A *learned* per-layer α (training, not blending) would find the true optimum.

---

## Experiment 16 — Early-layer targeted math scale training (2026-04-17)

**Question:** If early layers carry math, does training only early-layer scales match the full-model 28% GSM8K with fewer trainable parameters?

**Setup:** `train_early_layer_math.py`. Two conditions vs full-model baseline (28% GSM8K). Same v2 recipe throughout (AdamW lr=1e-4, Rho-1 token weighting, elastic band λ=0.1, 3 epochs, 150 examples). Each condition run in its own subprocess (CUDA state isolation).

| Condition | GSM8K | Params | % of total | vs full-model |
|---|---|---|---|---|
| full-model v2 (baseline) | 28.0% | 11.0M | 100% | — |
| early_all (layers 0-11) | 21.0% | 4.7M | 42.9% | −7pp |
| early_ffn (ffn_up/gate, layers 0-11) | 22.0% | 2.4M | 21.4% | −6pp |

**Result: Early layers are the primary site but not the complete story.** Training early layers only gets 75-79% of the full lift (21-22% vs 28%) with 21-43% of the params. Late layers contribute the remaining ~6-7pp.

**Two specific findings:**
1. `early_ffn` (21.4% of params) matches `early_all` (42.9%) within 1pp. ffn_up/gate captures nearly all early-layer math signal — consistent with activation probe (ffn_up/gate least redundant). Attention weights in early layers contribute almost nothing to math specialization.
2. The efficiency ratio: 21.4% of params → 79% of the math lift. Strong signal even without full match.

**Three-experiment coherent picture:**
- Exp 14 (blend): Math scales in late layers only → 2% GSM8K (catastrophic)
- Exp 16 (train): Math scales in early layers only → 21-22% GSM8K (partial)
- Full model: Both → 28% GSM8K

**Conclusion:** Math specialization is early-heavy but distributed. The mechanism requires both regions — early layers encode domain recognition, late layers complete the lift. Neither alone is sufficient; together they produce the full 5.3×.

---

---

## Experiment 17a — Two-region cooperative scale training (2026-04-17)

**Question:** If early layers encode math and late layers encode knowledge, can we train each region on its natural domain and combine the scale tables to exceed the full-model 28% GSM8K baseline while recovering MMLU?

**Setup:** `train_two_region_coop.py` + `eval_two_region_combine.py`. Region A: early-ffn (mlp.gate_proj/up_proj, layers 0-11) trained on GSM8K math. Region B: late-ffn (layers 12-23) trained on TriviaQA knowledge. Combined GGUF patches early-ffn from A and late-ffn from B, baseline everywhere else. Due to WSL2 WDDM GPU driver issue, reused Exp 16 `early_math_early_ffn_scales.pt` for Region A (same recipe, same data). `eval GSM8K n=100 + MMLU n=50`.

| Profile | GSM8K | MMLU |
|---|---|---|
| early_ffn only (Exp 16) | 22.0% | — |
| full math training | 28.0% | — |
| flat_0.7 blend (no training) | 40.0% | 41.7% |
| **two-region coop (this run)** | **25.0%** | **43.8%** |

**Result:** Did not beat full-model math on GSM8K (-3pp), but **MMLU 43.8% exceeded the flat_0.7 blend (41.7%)** — first time a trained model beat the untrained blend on knowledge. The cooperative mechanism works: regions don't interfere, they add. The GSM8K shortfall is explained by replacing late-layer math contribution (~6pp) with knowledge training rather than adding knowledge on top.

---

## Experiment 17b — Late-knowledge overlay on full math scales (2026-04-17)

**Question:** What if we start from the complete math-trained model (28% GSM8K) and only replace late-ffn with knowledge scales — adding knowledge on top of math rather than substituting?

**Setup:** `eval_late_knowledge_overlay.py`. Base: `math_scales_v2.pt` (all layers math-trained). Overlay: replace late-ffn (layers 12-23, gate/up) with `knowledge_scales_v2.pt`. No training — pure scale arithmetic + GGUF patch.

| Profile | GSM8K | MMLU |
|---|---|---|
| full math (base) | 28.0% | — |
| **late-knowledge overlay** | **31.0%** | **20.8%** |
| flat_0.7 blend | 40.0% | 41.7% |

**Result: GSM8K +3pp above full-model math baseline — broke the 28% ceiling.** Knowledge-trained late-ffn scales help math, not hurt it. Hypothesis: knowledge-trained late-ffn is better at retrieving arithmetic facts stored in those layers than math-overtrained scales. However, MMLU collapsed to 20.8% — math dominance in early layers suppresses knowledge retrieval globally regardless of late-layer content.

---

## Experiment 18 — Late-layer α sweep (2026-04-17)

**Question:** Is there a blend ratio for late-ffn (between full math and pure knowledge) that gives both high GSM8K AND recovers MMLU?

**Setup:** `eval_late_alpha_sweep.py`. Early layers locked at full math (α=1.0). Sweep `α_late ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}` for late-ffn only. `math_scales_v2` base + `knowledge_scales_v2` blend component.

| α_late | GSM8K | MMLU |
|---|---|---|
| 0.0 (pure knowledge) | 31.0% | 20.8% |
| 0.1 | 29.0% | 20.8% |
| 0.2 | 29.0% | 20.8% |
| 0.3 | 30.0% | 20.8% |
| 0.5 | 29.0% | 20.8% |
| 0.7 | 28.0% | 18.8% |
| 1.0 (full math) | 28.0% | — |

**Result: MMLU is locked at 20.8% across every α_late value.** No late-layer blend ratio recovers MMLU — early-layer math dominance suppresses knowledge retrieval globally, regardless of what happens in late layers. The Pareto frontier for balancing both benchmarks requires softening ALL layers (flat_0.7), not just the late ones.

**Three-experiment conclusion (17a + 17b + 18):** The scale space is holistic, not modular. You cannot carve it up regionally and get the same benefits as a global soft blend. flat_0.7 wins because it distributes the math-knowledge tradeoff across every layer. Hard regional assignments (train one region for one domain) lose the cross-layer synergy that produces the 40% GSM8K peak. Local experiments on depth structure are now exhausted.

---

### In Flight / Next

- **Router interpretability.** Load saved router, analyze routing decisions per token type.
- **Tighter elastic band sweep (λ=0.3, 0.5).** Does stronger regularization shrink the math-MMLU forgetting tradeoff?

### Queued for A100 Session (~$30-50)

- Math GSM8K at n=500 + MATH competition OOD (the single most important validation run)
- LoRA baseline comparison at matched parameter count
- Per-token router V3 (current V2 is sequence-level only due to local VRAM)
- 8B v2 recipe replication
- EFI sign unfreeze sweep: top {0.1%, 0.5%, 1%, 2%} signs unfrozen, test if ~29% ceiling is sign-capacity bound
- Full ablations

---

## Meta-Notes on the Process

**Three corrections were made publicly during this work:**

1. **V1 recipe ≠ v2 recipe conflation.** Initially reported 8B PPL-only results as a complete scale-personalities story; had to correct that PPL improvements didn't imply accuracy.
2. **100q TriviaQA "win" was noise.** Reported at 100q, inverted at 150q. Corrected with a "Sample-Size Correction" commit.
3. **MoE strangers finding.** Earlier work claimed MoE experts were 50% sign-agreement (random). On re-review we were not confident in the methodology and removed the claim from the repo.

Seeing these in the commit history is part of the point. The research is honest about what we know, what we don't, and what we got wrong.

---

## Experiment 19 — LoRA Baseline: Validity Check (2026-04-18)

**Hypothesis:** A LoRA adapter at matched (or larger) parameter count will equal or exceed scale personality gains on GSM8K. If true, scales win only on deployment overhead, not on mechanism.

**Setup:**
- Model: Bonsai 1.7B (PackedBitLinear, signs frozen)
- LoRA rank=16, alpha=16 → 17.4M params / 70MB fp32 (vs 11M params / 22MB fp16 for scales)
- Same recipe as v2: AdamW lr=1e-4, Rho-1 top-60% token weighting, elastic band λ=0.1, 3 epochs, 150 examples (70% GSM8K / 30% wikitext), seq_len=128
- Eval: GSM8K test n=100, 0-shot, greedy, max_new_tokens=200
- LoRA forward: `F.linear(F.linear(x, lora_A), lora_B) * scaling` — never materializes full [out_f, in_f] weight matrix

**Results:**

| Method | GSM8K | Size |
|--------|-------|------|
| No training (baseline) | 5.3% | — |
| LoRA rank=16 | **25.0%** | 70MB |
| Scale math only (Exp 7) | **28.0%** | 22MB |
| Scale blend flat_0.7 (Exp 10) | **40.0%** | ~0MB extra |

**Verdict: Scales beat LoRA despite LoRA having 3× more parameters.**

**Conclusion:** Scale tables are not just a parameter-efficient trick — they are the right inductive bias for 1-bit models. Signs encode frozen routing structure; scales are the only continuous degrees of freedom the model has. LoRA tries to add low-rank corrections on top of a discrete system and loses. Scales win on accuracy (+3pp over LoRA), on size (22MB vs 70MB), AND on inference overhead (zero — scales replace at load time, LoRA must execute A/B multiply every forward pass). The scale personality thesis is now validated against its primary null hypothesis.

---

## Experiment 20 — Sign Stability Probe (2026-04-18)

**Hypothesis:** Sign pressure (|grad_sign| per layer) increases with transformer depth — early layers crystallize first, late layers remain plastic longest. If true, progressive freeze (freeze early layers first during QAT) is the right curriculum for native 1-bit training.

**Setup:**
- Model: Bonsai 1.7B (PackedBitLinear, signs frozen as fp16 buffers)
- Side-channel measurement: custom `_BitLinearFn.backward` captures `|grad_w| × |scale_expanded|` per layer without making signs trainable (zero VRAM overhead)
- 50 training steps on GSM8K math examples (only scales train)
- Per-transformer-layer mean |sign gradient| grouped by depth (0–27)

**Results:**
- Late/Early ratio: **0.85×** — signs are uniformly crystallized across all 28 layers
- No progressive freeze gradient by depth
- All layers show similar sign pressure throughout training

**Verdict: UNIFORM — no depth trend. Progressive freeze hypothesis not supported by gradient signal.**

**Conclusion:** Bonsai's pre-training already crystallized signs uniformly at all depths — there is no natural early→late ordering to exploit. Progressive freeze by depth would be arbitrary, not data-driven. The hypothesis was reasonable (deep layers are usually more task-specific in FP16 models) but doesn't hold for binary models where the sign structure is locked in pre-training. If progressive freeze is ever revisited, it would need a per-layer gradient criterion during training from scratch, not a depth heuristic.

---

## Experiment 21 — Sign-Conditional Scales (2026-04-18)

**Hypothesis:** Each weight group has a separate scale for +1 signs and −1 signs (`scale_pos_g`, `scale_neg_g`). Standard scales use a single absmean per group. If +/− subsets have meaningfully different magnitude distributions, asymmetric scales could express more fine-grained structure and improve accuracy.

**Setup:**
- Model: Bonsai 1.7B
- 2× scales per group (scale_pos, scale_neg), initialized from actual pos/neg absmeans
- Custom `_SignCondBitLinearFn` — recomputes `pos_mask = (signs > 0)` in backward (not saved in ctx, saves ~1.6GB VRAM)
- Elastic band regularization toward init for both scale_pos and scale_neg
- Same v2 recipe: AdamW lr=1e-4, 3 epochs, 150 examples, top-60% token weighting
- Asymmetry analysis: `scale_pos / scale_neg` ratio per group

**Results:**

| Method | GSM8K | Params |
|--------|-------|--------|
| No training (baseline) | 5.3% | — |
| Standard scales (math, Exp 7) | **28.0%** | 11M / 22MB |
| Sign-conditional scales | **26.0%** | 22M / 44MB |

**Key measurement:** `scale_pos/scale_neg` ratio = **1.0009 (std=0.0005)** — groups are essentially perfectly symmetric. The +1 and −1 weight subsets within each group have nearly identical magnitude distributions.

**Verdict: BELOW standard scales (Δ=−2.0%) — extra expressivity not useful.**

**Conclusion:** The symmetry measurement explains the null result mechanistically. Bonsai's QAT process produces near-perfectly symmetric weight groups — the sign-conditional hypothesis assumed asymmetry that doesn't exist. Doubling scale parameters adds optimization noise (more parameters to navigate) without adding representational capacity (because the asymmetry the extra params would capture is ~0). This rules out fine-grained sign-level magnitude differentiation as a direction. Standard one-scale-per-128-group appears to be the right granularity for this model family.

---

## Experiment 22 — EFI Sign Unfreeze (2026-04-18)

**Hypothesis:** Scale training has saturated the continuous parameter space of the frozen sign structure. If the routing structure itself (signs) is the bottleneck, selectively unfreezing the top 1% of signs under maximum gradient pressure and training them with SGD should push past the 28% GSM8K ceiling.

**Setup:**
- Model: Bonsai 1.7B (PackedBitLinear)
- Ranking pass: 20 backward passes, accumulate |grad_scale| per group (group-level, not per-sign — 25MB vs 3.2GB)
- Top 1% of groups unfrozen → 14,092,800 signs (56MB fp32) as nn.Parameters
- Dual optimizer: AdamW lr=1e-4 for scales, SGD lr=1e-3 for sign params
- Same v2 recipe: 3 epochs, 150 examples, top-60% token weighting, elastic band λ=0.1
- After training: signs rebinarized to {-1,+1} → zero inference overhead
- Eval: GSM8K test n=100, 0-shot, greedy

**Results:**

| Method | GSM8K | Params | Notes |
|--------|-------|--------|-------|
| No training (baseline) | 5.3% | — | — |
| LoRA rank=16 (Exp 19) | 25.0% | 70MB | — |
| EFI K=1% signs | **27.0%** | ~100MB train | sign flips + scales |
| Standard scales (Exp 7) | **28.0%** | 22MB | scales only |
| Scale blend flat_0.7 (Exp 10) | **40.0%** | ~0MB extra | blend |

**Implementation notes:**
- Three OOM variants fixed: (1) lm_head not frozen before ranking → 32000×2048 grad OOM; (2) per-sign accumulation storing 3.2GB → replaced with per-group 25MB; (3) Python loop over 820M items → `np.argpartition`
- Sign flip rate: measured after training to confirm signs actually changed polarity

**Verdict: MATCHES scale-only within noise (Δ=−1.0%) — sign capacity is not the bottleneck.**

**Conclusion:** Unfreezing 1% of signs and training them alongside scales produces 27.0% — statistically indistinguishable from scale-only at 28.0%. This rules out sign-routing capacity as the bottleneck for the 28% ceiling on Bonsai 1.7B.

The convergent finding across Exp 19 (LoRA), 20 (sign stability), 21 (sign-conditional), and 22 (EFI): the scale personality mechanism is already near-optimal for this model and data regime. Scales are the only movable part; the signs are correctly committed; the ceiling is set by the model's capacity and the training data, not by the parameter representation.

The path past 28% is: (a) blend recipes (40% confirmed in Exp 10 via knowledge+math blend), (b) more/better training data, or (c) models with math pre-baked into signs from QAT (Qwen2.5-Math, DeepSeek-Math). Sign manipulation post-hoc doesn't help.

---

## Experiment 23 — Targeted Late-Layer Scale Training (2026-04-19)

**Hypothesis:** Per Bonsai forensics, ffn_up and gate_proj in late transformer layers (20-27) have the highest gradient pressure and depth-scale correlation (0.877). If these layers carry disproportionate adaptive impact, training only their scales (~10% of total scale params = 2.2MB) should approach the full-model result (28% GSM8K, 22MB).

**Setup:**
- Model: Bonsai 1.7B (PackedBitLinear)
- Targeted layers: 20-27 (late 8 of 28)
- Targeted types: `mlp.up_proj`, `mlp.gate_proj` (top 2 per forensics category hierarchy)
- Frozen: all other scales (attention Q/K/V/O, ffn_down, early/mid layers)
- Same v2 recipe: AdamW lr=1e-4, elastic band λ=0.1, 3 epochs, 150 examples, top-60% token weighting
- Eval: GSM8K test n=100, 0-shot, greedy

**Results:**

| Method | GSM8K | Scale params |
|--------|-------|--------------|
| No training (baseline) | 5.3% | 0 |
| Targeted late ffn_up+gate (Exp 23) | **18.0%** | ~10% (2.2MB) |
| Full scale training (Exp 7) | **28.0%** | 100% (22MB) |
| Scale blend flat_0.7 (Exp 10) | **40.0%** | 100% + 22MB blend |

**Verdict: BELOW full training (Δ=−10.0%) — late ffn_up+gate insufficient alone.**

**Conclusion:** Targeting the highest-forensics-pressure layers gets +12.7pp over baseline (5.3%→18.0%) but leaves −10pp on the table vs full training. Math adaptation is distributed across all layer depths and all projection types — it cannot be concentrated in the late ffn_up/gate alone.

This bounds the "targeted adaptation" hypothesis: the forensics correctly identify which layers are *most* impactful per parameter, but "most impactful" is not "sufficient." The full 22MB scale table is load-bearing for math; a 2.2MB targeted subset captures only ~65% of the total lift. If compact scale tables are a deployment goal, the tradeoff is approximately 2.2MB → 18% vs 22MB → 28%.

The distribution of adaptive signal across all layers also makes sense mechanistically: math reasoning requires vocabulary lookup (early embedding layers), multi-step attention patterns (mid layers), and output projection (late layers) to all be tuned together. Domain adaptation is inherently holistic.

**Implementation note:** The `is_targeted()` function checks layer index AND projection type string, enabling arbitrary layer/type subset experiments without modifying the training loop.

---

## Experiment 24 — STE Sign QAT K=15% (2026-04-19)

**Hypothesis:** Exp 22 (EFI) was methodologically broken — greedy sign swapping without STE meant the optimizer didn't know about the binarization boundary, making the null result inconclusive. A proper straight-through estimator (STE) with fp32 sign parameters initialized to ±0.1 (close to flip boundary), optimized with Adam, should give a clean answer on sign capacity.

**Setup:**
- Model: Bonsai 1.7B (PackedBitLinear fp16 base + fp32 patches for selected K%)
- Ranking: scale gradient proxy (20 backward passes, |grad_scale| per group)
- K=15%: top 1.65M groups → 211M signs unfrozen as fp32 patches (846MB)
- STE: forward uses `sign(patch)`, backward passes gradient through as identity
- Optimizer: Adam lr=1e-3, no gradient clipping (SGD with clip_norm=1.0 in prior run neutered gradients to ~7e-7 per param — 0 flips resulted)
- Init: patch_signs = ±0.1 (not ±1.0 — close to boundary so gradients cross zero)
- After training: commit_patches() writes binarized values to fp16 buffer; eval runs without index_put overhead
- Eval: GSM8K test n=100, 0-shot, greedy

**Results:**

| Method | GSM8K | Sign flips | Notes |
|--------|-------|-----------|-------|
| No training (baseline) | ~17% | — | HF unpacked eval harness |
| Math scales only | 23% | 0 | n=200 validated |
| STE sign QAT K=15% | **22%** | 52,101 (0.004%) | clean null |

**Implementation notes:**
- Prior attempt (same session) had patch_signs initialized to ±1.0 with SGD + clip_norm=1.0: clip_grad_norm_ across 211M params → each param grad ~7e-7 → 0 flips, loss stuck at 4.015. Fixed by switching to Adam (per-param normalization) and ±0.1 init.
- Loss decreased (4.015 → 3.958 → 3.422 → 3.238) confirming optimizer actually worked
- 52,101 flips = 0.004% of all signs; 0.025% of unfrozen signs — extremely few

**Verdict: NULL/REGRESSION (Δ=−1%) — signs have no additional capacity beyond scales at K=15%.**

**Conclusion:** This is the first methodologically clean test of sign capacity. Adam + ±0.1 init + STE = optimizer works (loss drops, some flips happen), yet accuracy is flat or regresses vs scale-only. Unlike Exp 22 (broken mechanics, inconclusive), this result cleanly answers the question: at K=15% with a proper optimizer, sign flips do not unlock additional math capability. Signs are correctly committed from QAT. Scales are the complete and sufficient continuous degree of freedom for post-hoc domain adaptation.

---

## T4 Burst Validation (2026-04-19) — n=200 corrections

Large-n re-validation of headline results using Modal T4 GPUs. Three concurrent functions: scale math v2, LoRA rank=16, STE sign QAT.

**Corrected numbers (all n=200 unless noted):**

| Experiment | n=100 (prior) | n=200 (validated) | Change |
|---|---|---|---|
| Math scales only | 28% | **23%** | −5pp |
| LoRA rank=16 | 25% | **25%** | stable |
| Sign QAT K=15% (STE) | — | **22%** (n=100) | new |
| Blend α=0.7 math+knowledge | 40% (n=50–100) | **30.5%** | −9.5pp |

**Key findings:**
1. Math-only scales: 23% is the validated number. 28% was small-n variance at n=100.
2. LoRA vs scales: equivalent accuracy at n=200 (25% vs 23%). Earlier "scales win" claim was marginal and doesn't hold at larger n. Scales still win on size (22MB vs 70MB) and zero inference overhead.
3. Blend compounding confirmed: blend adds +7.5pp over math-only (23%→30.5%). The compounding mechanism is real; the 40% number was not.
4. Sign QAT: clean null. 52K flips, loss decreases, no accuracy gain.

This is the third major correction in the repo (after TriviaQA n=100 noise and v1 PPL overclaim). Each correction is in the commit history.
