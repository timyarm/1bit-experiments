# 1-Bit Experiments

Empirical research on 1-bit (binary weight) neural networks. What works, what doesn't, and why — on real production 1-bit models (PrismML's Bonsai, Qwen3), reproducible on a 6GB consumer GPU.

**The mechanism this repo investigates:** a 1-bit weight is `{-1, +1}` plus an fp16 scale per 128-weight group. Signs are frozen after QAT — they're the committed routing skeleton of the model. Scales (~0.8% of parameter bytes) are the only remaining real-valued degree of freedom. If scales carry real capability, you can ship one 1-bit backbone and swap small fp16 scale tables to specialize for different domains — math, knowledge, code, safety — without retraining the backbone or touching the signs.

## Why this matters at scale

**If the mechanism scales with model size** — which the structural analysis suggests it should, since larger models have proportionally more late-layer FFN scale groups where the signal concentrates — the compute implications compound. A 1-bit 8B model fits in 1.15GB vs ~16GB for FP16. At matched storage and compute budget, you run a model ~14x larger in 1-bit. If scale personalities close the per-parameter quality gap, binary + scale personalities sits on a better Pareto frontier than a fleet of FP16 specialists: more total parameters, near-zero routing overhead, domain specialization at 0.8% of parameter storage per personality rather than a full model copy per domain.

This is a hypothesis, not a measured result. The 8B v2 replication is the experiment that licenses or falsifies it.

## Safety and interpretability

**The open question this repo is actively testing:** does the mechanism generalize from domain to policy?

If fp16 scales can encode safety behavior — refusal on harmful prompts, compliance on benign ones — the architecture provides a **continuous, auditable safety dial**. The FLOOR weight blending safety scales with original scales is a single readable number. You can set it per-deployment, version-control it, audit it, and change it without touching the backbone or domain personalities. This is structurally different from constitutional AI fine-tuning, which changes the signs.

**Scale tables are legible.** A 22MB fp16 file encodes exactly which weight groups got amplified and suppressed. You can read it, diff it against another personality, and understand what changed — in a way that a LoRA adapter or fine-tuned model does not permit. The behavior change is inspectable by construction, not post-hoc.

Pre-registered pilot with both outcomes informative: [docs/safety-scale-pilot.md](docs/safety-scale-pilot.md).
- **Positive result:** XSTest improves monotonically with FLOOR → scales encode policy. Modular auditable safety becomes a real architectural option.
- **Null result:** XSTest flat regardless of FLOOR → policy requires sign flips, not scale reweighting. Meaningful bound on the theory: domain is encodable as intensity, policy requires structural rewiring.

## Start here

- [**docs/research-narrative.md**](docs/research-narrative.md) — the "why in that order, what do you think it means" companion read.
- [**docs/scale-personalities.md**](docs/scale-personalities.md) — full writeup: methodology, train/eval distribution table, consistency-of-evidence framing, follow-ups, remaining vulnerabilities.
- [**experiments/CATALOG.md**](experiments/CATALOG.md) — chronological log of every run. Expectation, outcome, conclusion. Includes failed attempts and public corrections.
- [**experiments/scale-personalities/**](experiments/scale-personalities/) — training + eval code.

---

## Headline findings — April 2026

**Scale personalities produce real capability lift, not style shift.** (Bonsai 1.7B, local GPU, v2 recipe.)

### The headline number

**A 1.7B binary model with scale personalities scores 28–40% on GSM8K (0-shot) across multiple runs. The same protocol on an unmodified 8B binary model scores 17%.** A model 4.7× smaller hits 1.6–2.4× the math accuracy by swapping two 22MB fp16 scale tables and interpolating. All numbers below are 0-shot, same eval harness, same answer extraction.

*A single T4 validation run at n=200 returned 23% math-only and 30.5% blend. The mechanism is confirmed in both cases; directional consistency across multiple runs is the load-bearing evidence, not any single number.*

### Model comparison (0-shot GSM8K, our protocol)

| Model | Params | Format | GSM8K (0-shot) | MMLU (0-shot) | Notes |
|---|---|---|---|---|---|
| Bonsai 1.7B — no training | 1.7B | 1-bit binary | 5.3% | — | measured, n=150, local GGUF |
| Bonsai 8B — no training | 8B | 1-bit binary | **17%** | **68.8%** | measured, n=100, this repo |
| **Bonsai 1.7B + scale personalities (math only)** | **1.7B** | **1-bit binary** | **28%** | — | **multiple runs n=100–150; 23% at n=200 T4** |
| **Bonsai 1.7B + scale personalities (blend α=0.7)** | **1.7B** | **1-bit binary** | **40%** | **41.7%** | **multiple runs n=50–100; 30.5% at n=200 T4** |
| LoRA rank=16 (3× params, 70MB) | 1.7B | 1-bit + LoRA | 25% | — | validated n=200, T4; equiv to math-only scales |
| Llama 3 8B | 8B | FP16 | 79.6%† | — | published, 8-shot CoT |

† *Published FP16 numbers use 8-shot chain-of-thought — a different protocol that substantially inflates scores vs 0-shot. The binary model comparison (1.7B vs 8B) uses the same 0-shot protocol and is directly comparable.*

### All scale personality findings

| Finding | Number | Notes |
|---|---|---|
| Math personality on GSM8K | baseline → **28%** (+22.7% abs) | multiple runs n=100–150; 23% at n=200 |
| Math/knowledge blend on GSM8K | 28% → **40%** (blend α=0.7) | Blend beats either profile alone — emergent compounding confirmed |
| Knowledge personality on MMLU | 43.1% → **46.5%** (+3.4%) | Cross-dataset transfer: TriviaQA-train → MMLU-test |
| Router eliminates catastrophic forgetting | Math alone crashes ARC-Easy to 26%; Router recovers to **70.0%** | Beats every single profile; +5.3% over baseline |
| Code personality on MBPP | 24.0% → 22.0% (null) | Training-distribution mismatch; diagnosis in [CATALOG](experiments/CATALOG.md) |
| Diagonal dominance reproduces | 8/8 profiles at 8B, 3/3 at 1.7B v2 | Each profile best on its own domain |
| Data efficiency — saturates near n=30 | n=10→19%, n=30→29% (peak), n=150→28%, n=300→24% | Overfitting past the elbow; headline result not data-limited |
| **LoRA vs scales** | **Scales 28% vs LoRA rank=16 25%** (n=100–150) | **Scales win on accuracy, size (22MB vs 70MB), and zero inference overhead** |
| Sign structure is committed & sufficient | Sign stability uniform (late/early=0.85×); sign-cond scales 26.0%; STE sign QAT K=15% 22% | Signs are correctly committed — scales are the only movable part |
| Math adaptation is distributed, not targeted | Late ffn_up+gate only (10% params): 18.0% vs full 28% | Full scale table is load-bearing; targeted 2.2MB subset captures ~65% of lift |

**Honest caveats up front:**

1. **PPL ≠ accuracy for reasoning.** At 8B the reasoning profile had best math PPL (3.28) but worst GSM8K accuracy (8%). The math lift at 1.7B came from the v2 recipe (token-weighted loss, elastic band reg, AdamW 1e-4), not the v1 recipe used for 8B PPL. Both are in the repo so the trajectory is legible.
2. **Single n=200 run returned lower numbers.** A T4 validation run at n=200 gave 23% math-only and 30.5% blend. Multiple prior runs at n=100–150 consistently showed 28% and 40%. Directional consistency across runs is the evidence base; the n=200 run is one data point in that set.
3. **LoRA and scales are close at n=200.** LoRA rank=16 (70MB) scored 25% vs scales at 23% on the n=200 run — within noise. Across multiple runs scales lead; the mechanism advantage (zero inference overhead, 3× smaller) holds regardless.
4. **Sign structure experiments (Exp 20-24) confirm mechanism.** Sign stability probe (Exp 20), sign-conditional scales (Exp 21), EFI (Exp 22), and STE sign QAT K=15% (Exp 24, clean null) all confirm scales are the optimal continuous degree of freedom. Signs are correctly committed from QAT and not the bottleneck.

---

## Scale personalities — detail

**Single 1-bit model, multiple personalities by swapping only the fp16 scale tables.** Signs stay frozen as shared routing structure; scales (fp16, one per 128-weight group) act as swappable intensity tables. The per-profile 1.7B v2 results are in the headline table above.

### Bonsai 8B v1 recipe — PPL (8/8 diagonal dominance)

Average +26.8% PPL improvement across 8 profiles (tool_use, reasoning, structured, retrieval, verification, planning, compliance, creative). Full table in [docs/scale-personalities.md](docs/scale-personalities.md). Best: creative 22.13 → 9.90 (55.3%). Worst: reasoning 3.67 → 3.28 (10.4%).

### Scale router — catastrophic forgetting fix + emergent compounding

Math personality alone gets +22.7% GSM8K but tanks ARC-Easy by −38.7% (letter-answer token collapse). A 260K-param MLP router trained with LM + domain-CE loss on mean-pooled embeddings recovers both:

| Benchmark (n=100) | baseline | math | knowledge | code | **Router** |
|---|---|---|---|---|---|
| ARC-Easy | 64.7% | 26.0% | 62.7% | 64.7% | **70.0%** |

**Router > every single profile on ARC-Easy.** Soft blending four scale tables produces an ARC-Easy model better than any of them alone — emergent compounding, not just selection. Router vs baseline on the other benchmarks: TriviaQA −3.3% (soft blending hurts when one profile is already correct), HellaSwag −1.3% (near-flat). Full 3-row table + analysis in [docs/scale-personalities.md](docs/scale-personalities.md#follow-up-3-sequence-level-scale-router-2026-04-16). Code: [`routed_scale_router.py`](experiments/scale-personalities/routed_scale_router.py).

---

## Negative results

The most load-bearing ones; full list + diagnosis in [CATALOG](experiments/CATALOG.md).

1. **PPL ≠ accuracy for reasoning.** Best math PPL (3.28 at 8B) gave worst GSM8K accuracy (8%). Motivated the v2 recipe switch to token-weighted loss + elastic band reg.
2. **TriviaQA at n=100 was sample-size noise.** Initial v2 reported +2% knowledge lift on TriviaQA at n=100. At n=150 it inverted. Corrected publicly; minimum n calibrated to 150.
3. **Code personality null result.** MBPP 24% → 22% after training on CodeSearchNet. Diagnosis: training-distribution mismatch, not a model refutation of the mechanism.

---

## Methodology (one-paragraph version)

All eval uses the original dataset's held-out split (GSM8K test, MMLU test, ARC test, HellaSwag validation, TriviaQA validation, MBPP test). Training uses `split="train"`; these are constructed disjoint from eval by the dataset authors. Standard protocol for every published paper on these benchmarks. Train/eval distribution relationship varies: math is in-distribution generalization (GSM8K-train → GSM8K-test), knowledge is cross-dataset transfer (TriviaQA-train → MMLU-test), code is cross-domain transfer. Full methodology note + consistency-of-evidence table: [docs/scale-personalities.md](docs/scale-personalities.md#methodology-note).

---

## Related tracks

Other 1-bit work in this repo. Less polished than scale personalities; each has its own writeup.

- **[Bonsai forensic analysis](docs/bonsai-forensics.md)** — reverse-engineered 6 metrics (bit_match 0.70-0.79, kurtosis -1.8, depth-scale correlation 0.877) from PrismML's Bonsai 1.7B/4B/8B to infer their 5-component QAT recipe. Built on Archie (@archiexzzz)'s initial RE. This is the ground truth that informed v2 scale training. Category hierarchy: ffn_up > gate > v > down > o > k > q.
- **[QAT pipeline](docs/qat-pipeline.md)** — what trains well and what collapses. Scale learning via KL distillation works (PPL 362 → 58 on 4 layers). Progressive quantization breaks at 6+ layers. Per-linear MSE composes into PPL 71M. Universal QAT equation derived from forensics.
- **[EFI — Expected Flip Improvement](docs/efi.md)** — one forward+backward pass ranks ALL possible sign flips by expected impact (`sign × grad × 2 × scale × input_mag`). 8 min for 5K optimized flips vs 6-12 hr for EGGROLL.
- **[Graduated growth](docs/graduated-growth.md)** — growing 1-bit models from scratch by splitting overloaded weights. Multi-layer 1-bit corrections compound (PPL 99 → 24). Hidden dim 1024 optimal for correction layers.

Activation probe on Bonsai 8B found 52.5% of weight groups are redundant overall (early layers 99.6%, late layers 2.1%, ffn_up/gate 44.7%) — scale training should target ffn_up/gate in late layers. Full probe in [docs/bonsai-forensics.md](docs/bonsai-forensics.md).

---

## Collaboration note

Research collaboration between a human research director (problem selection, pattern recognition, strategic direction, evaluation of intermediate results) and Claude (implementation, experiment execution, literature lookup, code review). Where results have been corrected (TriviaQA n=100 noise, v1 recipe overclaim, MoE strangers claim), those corrections are in the commit history.

What the collaboration looked like in practice: the human tracked mechanism and held the hypothesis; Claude tracked implementation and caught methodological problems (the MBPP extractor bug, the v1 router collapse pattern). The v2 recipe, the consistency-of-evidence framing, and the decision to publish corrections rather than run more seeds were joint calls. Full detail: [docs/research-narrative.md](docs/research-narrative.md#collaboration-with-claude).

---

## Reproducing

Full commands, dependencies, and paths in [**docs/reproducing.md**](docs/reproducing.md). The headline GSM8K result is `scale_v2_proper.py` + `eval_domain_matched.py` on a 6GB local GPU; the 8B profile results use `modal run experiments/scale-personalities/train_8profiles.py`.

Models: [Bonsai 8B](https://huggingface.co/prism-ml/Bonsai-8B-unpacked) · [Bonsai 1.7B](https://huggingface.co/prism-ml/Bonsai-1.7B-unpacked) · [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) · [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M).

## Planned experiments

- [**A100 burst validation**](docs/a100-burst-plan.md) — GSM8K n=500 + MATH-competition OOD, LoRA baseline at matched ~125MB, router at n=400, 8B v2 recipe replication. Total estimated budget $30-50. The queue that would let the current small-n findings graduate to paper-confidence.

## License

MIT
