# 1-Bit Experiments

Empirical research on 1-bit (binary weight) neural networks. What works, what doesn't, and why — on real production 1-bit models (PrismML's Bonsai, Qwen3), reproducible on a 6GB consumer GPU.

**The main thesis this repo investigates:** a 1-bit weight is `{-1, +1}` plus an fp16 scale per 128-weight group. If the scales (only ~0.8% of the parameter bytes) carry real capability, you can ship one 1-bit backbone and swap small fp16 scale tables to specialize for different domains. Fifty experts in ~8GB instead of ~20TB.

## Headline findings — April 2026

**Scale personalities produce real capability lift, not style shift.** (Bonsai 1.7B, local GPU, v2 recipe.)

| Finding | Number | Notes |
|---|---|---|
| Math personality on GSM8K | 5.3% → **28.0%** (+22.7% abs, **5.3× rel**) | n=150 held-out test split; stat-sig (z ≈ 5+); single seed at 1.7B v2, replication queued |
| Knowledge personality on MMLU | 43.1% → **46.5%** (+3.4%) | Cross-dataset transfer (TriviaQA-train → MMLU-Knowledge test), n=144 |
| Code personality on MBPP | 24.0% → 22.0% (−2.0%, null) | Training-distribution mismatch after extractor fix; diagnosis in [CATALOG](experiments/CATALOG.md) |
| Router eliminates catastrophic forgetting | Math-alone crashes ARC-Easy to 26%; Router recovers to **70.0%** — beats every single profile | Directional win over baseline (+5.3%, n=100) is not stat-sig at this n; anti-forgetting mechanism is the robust claim |
| Emergent compounding in the scale manifold | α=0.7 static math/knowledge blend → **GSM8K 40.0%** (pure math endpoint: 34.0%) | Second observation of blend > single profile, now on a static linear mix instead of a learned router. n=50 per alpha |
| Data efficiency — saturated near n=30 | Curve: n=10→19%, n=30→29% (peak), n=150→28% (headline), n=300→24% (overfitting) | Eval n=100; point estimates noisy, but the peak-then-decline pattern is consistent with overfitting past the elbow. Plot in [docs/figures](docs/figures/data_efficiency_curve.png) |
| Diagonal dominance reproduces | 8/8 profiles at 8B (multi-seed), 3/3 at 1.7B v2 (single seed) | 1.7B replication queued |

**Honest caveats up front:**

1. **PPL ≠ accuracy for reasoning.** At 8B the reasoning profile had best math PPL (3.28) but worst GSM8K accuracy (8%). The math 5.3× at 1.7B came from the v2 recipe (token-weighted loss, elastic band reg, AdamW 1e-4), not the v1 recipe used for 8B PPL. Both are in the repo so the trajectory is legible.
2. **Sample sizes reflect a 6GB consumer GPU.** Headline benchmarks are n=100–150. Individual-row statistical significance is mixed: math GSM8K +22.7% is solidly significant (z ≈ 5+); the +5.3% ARC-Easy delta is directional, not significant at n=100. The strongest evidence is the pattern *across* rows (see [consistency of evidence](docs/scale-personalities.md#consistency-of-directional-evidence)), not any single number. A100 re-runs at n=400+ are queued.
3. **No LoRA baseline at matched parameter count yet.** The natural first question — "how does scale-only training compare to a LoRA adapter at ~125MB?" — is queued for the A100 burst. Calling it out because its absence would otherwise read as avoidance rather than budget.

## Start here

- [**docs/research-narrative.md**](docs/research-narrative.md) — the "why in that order, what do you think it means" companion read.
- [**docs/scale-personalities.md**](docs/scale-personalities.md) — full writeup: methodology, train/eval distribution table, consistency-of-evidence framing, follow-ups, remaining vulnerabilities.
- [**experiments/CATALOG.md**](experiments/CATALOG.md) — chronological log of every run. Expectation, outcome, conclusion. Includes failed attempts and public corrections.
- [**experiments/scale-personalities/**](experiments/scale-personalities/) — training + eval code.

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

## Related tracks

Other 1-bit work in this repo. Less polished than scale personalities; each has its own writeup.

- **[Bonsai forensic analysis](docs/bonsai-forensics.md)** — reverse-engineered 6 metrics (bit_match 0.70-0.79, kurtosis -1.8, depth-scale correlation 0.877) from PrismML's Bonsai 1.7B/4B/8B to infer their 5-component QAT recipe. Built on Archie (@archiexzzz)'s initial RE. This is the ground truth that informed v2 scale training. Category hierarchy: ffn_up > gate > v > down > o > k > q.
- **[QAT pipeline](docs/qat-pipeline.md)** — what trains well and what collapses. Scale learning via KL distillation works (PPL 362 → 58 on 4 layers). Progressive quantization breaks at 6+ layers. Per-linear MSE composes into PPL 71M. Universal QAT equation derived from forensics.
- **[EFI — Expected Flip Improvement](docs/efi.md)** — one forward+backward pass ranks ALL possible sign flips by expected impact (`sign × grad × 2 × scale × input_mag`). 8 min for 5K optimized flips vs 6-12 hr for EGGROLL.
- **[Graduated growth](docs/graduated-growth.md)** — growing 1-bit models from scratch by splitting overloaded weights. Multi-layer 1-bit corrections compound (PPL 99 → 24). Hidden dim 1024 optimal for correction layers.

Activation probe on Bonsai 8B found 52.5% of weight groups are redundant overall (early layers 99.6%, late layers 2.1%, ffn_up/gate 44.7%) — scale training should target ffn_up/gate in late layers. Full probe in [docs/bonsai-forensics.md](docs/bonsai-forensics.md).

---

## Methodology (one-paragraph version)

All eval uses the original dataset's held-out split (GSM8K test, MMLU test, ARC test, HellaSwag validation, TriviaQA validation, MBPP test). Training uses `split="train"`; these are constructed disjoint from eval by the dataset authors. Standard protocol for every published paper on these benchmarks. Train/eval distribution relationship varies: math is in-distribution generalization (GSM8K-train → GSM8K-test), knowledge is cross-dataset transfer (TriviaQA-train → MMLU-test), code is cross-domain transfer. Full methodology note + consistency-of-evidence table: [docs/scale-personalities.md](docs/scale-personalities.md#methodology-note).

## Negative results

The two most load-bearing ones; full list + diagnosis in [CATALOG](experiments/CATALOG.md).

1. **PPL ≠ accuracy for reasoning.** Best math PPL (3.28 at 8B) gave worst GSM8K accuracy (8%). Motivated the v2 recipe switch to token-weighted loss + elastic band reg, which finally produced the 5.3× lift.
2. **TriviaQA at n=100 was sample-size noise.** Initial v2 reported +2% knowledge lift on TriviaQA at n=100. At n=150 it inverted. Corrected publicly; minimum n calibrated to 150.

---

## Reproducing

Full commands, dependencies, and paths in [**docs/reproducing.md**](docs/reproducing.md). The headline GSM8K 5.3× result is `scale_v2_proper.py` + `eval_domain_matched.py` on a 6GB local GPU; the 8B profile results use `modal run experiments/scale-personalities/train_8profiles.py`.

Models: [Bonsai 8B](https://huggingface.co/prism-ml/Bonsai-8B-unpacked) · [Bonsai 1.7B](https://huggingface.co/prism-ml/Bonsai-1.7B-unpacked) · [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) · [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M).

## Planned experiments

- [**Safety-scale pilot**](docs/safety-scale-pilot.md) — does the scale-personality mechanism extend from domain (math/code/knowledge) to *policy* (refusal/safety behavior)? Pre-registered plan for a floor+boost architecture with a Pareto sweep of safety-floor weight vs capability. Both outcomes informative — null result bounds the theory; positive result gives modular auditable safety. ~5hr local GPU, not yet executed.
- [**A100 burst validation**](docs/a100-burst-plan.md) — GSM8K n=500 + MATH-competition OOD, LoRA baseline at matched ~125MB, router at n=400, 8B v2 recipe replication. Total estimated budget $30-50. The queue that would let the current small-n findings graduate to paper-confidence.

## Collaboration note

Research collaboration between a human research director (problem selection, pattern recognition, strategic direction, evaluation of intermediate results) and Claude (implementation, experiment execution, literature lookup, code review). Where results have been corrected (TriviaQA n=100 noise, v1 recipe overclaim, MoE strangers claim), those corrections are in the commit history. Full detail: [docs/research-narrative.md](docs/research-narrative.md#collaboration-with-claude).

## License

MIT
