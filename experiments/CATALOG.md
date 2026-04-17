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

### In Flight / Next

- **Scale interpolation curve.** 11 alpha values blending math↔knowledge scales, plot GSM8K/MMLU tradeoff. Tests whether the personality space is a continuous manifold with sweet spots.
- **Data efficiency curve.** Train math scales on {10, 30, 100, 300} examples, plot GSM8K vs data size. Tests the "data-starved at 150 examples" hypothesis.
- **Router interpretability.** Load saved router, analyze routing decisions per token type.
- **Tighter elastic band sweep (λ=0.3, 0.5).** Does stronger regularization shrink the math-MMLU forgetting tradeoff?

### Queued for A100 Session (~$30-50)

- Math GSM8K at n=500 + MATH competition OOD (the single most important validation run)
- LoRA baseline comparison at matched parameter count
- Per-token router V3 (current V2 is sequence-level only due to local VRAM)
- 8B v2 recipe replication
- Full ablations

---

## Meta-Notes on the Process

**Three corrections were made publicly during this work:**

1. **V1 recipe ≠ v2 recipe conflation.** Initially reported 8B PPL-only results as a complete scale-personalities story; had to correct that PPL improvements didn't imply accuracy.
2. **100q TriviaQA "win" was noise.** Reported at 100q, inverted at 150q. Corrected with a "Sample-Size Correction" commit.
3. **MoE strangers finding.** Earlier work claimed MoE experts were 50% sign-agreement (random). On re-review we were not confident in the methodology and removed the claim from the repo.

Seeing these in the commit history is part of the point. The research is honest about what we know, what we don't, and what we got wrong.
