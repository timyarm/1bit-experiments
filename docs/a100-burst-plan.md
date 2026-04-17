# A100 Burst — Validation Run Plan

**Status:** scoped, not yet executed. Budget estimate: $30-50 on Modal A100-80GB.

The work in this repo has been done on a 6GB consumer GPU (GTX 1660 Super) with small-n evaluations (n=50-150). Directional patterns are clean; individual magnitudes need replication at paper-quality n. This doc is the specific set of A100 runs that would close the biggest remaining validation gaps.

Queued in priority order, with the rationale for each.

## 1. GSM8K at n=500 + MATH-competition OOD

**Why first.** The 5.3× headline number is the single most consequential claim in the repo. At n=150 it's solidly statistically significant (z ≈ 5+), but a reviewer looking at the rank of the finding wants to see n ≥ 500 on the primary eval, and an OOD math benchmark that shows the lift isn't GSM8K-format-specific.

**Setup.** Re-train the math profile on Bonsai 1.7B with the v2 recipe (same one that produced 28.0% at n=150) using three seeds. Evaluate each on:
- GSM8K test n=500 (~3-4× our current number)
- MATH competition, Level 1-3 subset, n=200 (never seen in training — strict OOD)

**What it proves.** If GSM8K n=500 holds at 25-30% and MATH shows any lift at all (even +2pp), the scales-carry-capability claim survives scale-up. If MATH is flat, the claim survives as "in-distribution generalization" but doesn't extend to out-of-format math reasoning. Either is a clean, publishable read.

**Budget.** ~1hr training × 3 seeds + ~2hr eval = ~5 A100-hrs. Call it $10.

## 2. LoRA baseline at matched parameter count

**Why second.** This is the question I expect every reviewer to ask. The math scale table is ~125MB of fp16. The obvious baseline: a LoRA adapter of roughly the same size. If LoRA does better, scale-only is just "LoRA with a name change." If LoRA does worse or similar, scale-only has independent merit.

**Setup.** Train a LoRA adapter on Bonsai 1.7B at rank chosen so total adapter size ≈ 125MB. Same training data (GSM8K train, 150 examples), same AdamW 1e-4, same 3 epochs. Eval GSM8K test n=500, MATH-competition n=200.

**What it proves.**
- **LoRA > scales on GSM8K, ≤ scales on MATH-OOD:** scales learn better generalization on the underlying reasoning distribution even if LoRA memorizes the training distribution more aggressively. This is a strong, nuanced finding.
- **LoRA > scales on both:** scale-only training is LoRA-ish, we should rewrite framing to reflect that.
- **Scales > LoRA on GSM8K:** surprising but would make the case that freezing signs and only moving scales is a genuinely different inductive bias.
- **Comparable on both:** negative result, scale-only training is a valid but not-superior approach.

**Budget.** ~1hr train + ~2hr eval = ~3 A100-hrs. Call it $6.

## 3. Router at n=400 per benchmark + domain-matched evals

**Why third.** The +5.3% ARC-Easy headline is at n=100 and not statistically significant (advisor caught this). The mechanism-level claim (router recovers from catastrophic forgetting) is robust, but the specific point estimate needs bigger n. Also: the router was never evaluated on GSM8K or MMLU — a fair criticism is that we sold the router win on ARC-Easy without testing whether it preserves the math profile's big wins on math benchmarks.

**Setup.** Re-train the V2 router on Bonsai 1.7B (same architecture, same loss). Eval on:
- ARC-Easy n=400
- TriviaQA n=400
- HellaSwag n=400
- GSM8K n=400 (new for router)
- MMLU-Knowledge n=300 (new for router)
- XSTest n=250 (new for router, and groundwork for the safety pilot)

**What it proves.** Either the router's ARC-Easy win is robust at n=400, or it shrinks toward the 95% CI lower bound (−7.6%). And the new columns answer the fair critique: does the router protect the math lift on math tasks? If router-GSM8K ≥ 20%, most of the math profile's 28% survived soft routing — that's the full anti-forgetting story. If it drops to baseline 5%, the router is trading math for ARC and we should say that.

**Budget.** ~30min retrain + ~3hr eval = ~3.5 A100-hrs. Call it $7.

## 4. 8B v2 recipe replication

**Why fourth.** The 8B numbers in the repo are v1 recipe (PPL diagonal dominance 8/8, but accuracy didn't translate). The 1.7B numbers are v2 (accuracy lifted). We've never run the v2 recipe at 8B. Open question: does v2's accuracy lift scale to 8B, or does 1.7B have some property that makes scale-only training work particularly well there?

**Setup.** Three profiles (math, knowledge, code) on Bonsai 8B with the v2 recipe. Same hyperparameters as the 1.7B run — AdamW 1e-4, Rho-1 token weighting, elastic band reg, seq 256, 3 epochs, 150 examples per profile. Eval domain-matched (GSM8K n=300, MMLU n=200, MBPP n=200 with the fixed extractor).

**What it proves.** Scaling behavior. If 8B shows similar or bigger lifts than 1.7B, the mechanism isn't small-model-specific — it scales. If 8B shows smaller or null lift, there's something about the 1.7B regime that makes scale-only training particularly effective, which is itself a finding worth understanding.

**Budget.** ~3hr training (3 profiles × ~1hr on 8B) + ~2hr eval = ~5 A100-hrs. Call it $10.

## 5. Data efficiency curve at 1.7B at n=500 (if not already done locally)

**Why fifth.** The local data-efficiency run at {10, 30, 100, 300} examples on n=100 GSM8K probes whether 150 is saturated. If the local run shows the curve still rising at 300, A100 should extend to {300, 1000, 3000} on GSM8K n=400. If the local run shows a plateau before 150, this is skippable.

**Decision point after local run finishes.**

**Budget.** ~2hr train + ~1hr eval = ~3 A100-hrs, $6.

## 6. FP16 8B 0-shot baseline (same protocol as ours)

**Why sixth.** Every published GSM8K number for FP16 8B models uses 8-shot chain-of-thought, which is not comparable to our 0-shot eval. Before claiming "1.7B binary + scale personalities beats 8B FP16 on math," we need Llama 3 8B (or equivalent) run through our exact same harness: 0-shot, same prompt format, same answer extraction.

**Setup.** Load Llama 3 8B (or Llama 3.1 8B) via llama.cpp GGUF. Run our standard eval: GSM8K n=500 (same as run #1), MMLU n=300. Same `"Question: {q}\nAnswer:"` prompt, same number extraction.

**What it proves.** Either our 0-shot 40% on 1.7B binary beats 0-shot FP16 8B (strong claim, directly licenses the "smaller binary beats larger FP16" headline), or FP16 8B 0-shot is higher (expected — but we now know the gap and can frame it honestly). Either way the comparison is on a fair ruler.

**Budget.** ~1.5hr eval = ~3 A100-hrs. Call it $6.

## 7. CoT-data math personality (does scale training encode reasoning or format?)

**Why seventh.** Our math scales were trained on GSM8K data which includes the answer format but not full verbose chain-of-thought. The open question: are the scales encoding genuine mathematical reasoning capacity, or primarily learning the output format? Training on CoT-rich data (explicit step-by-step solutions) and evaluating 0-shot would test whether the scales can bake in reasoning style rather than just answer format. If CoT-trained scales produce CoT-style output without prompting, it's weight-level reasoning, not format learning.

**Setup.** Train math scales on GSM8K train with full CoT solutions (already in the dataset's "answer" field). Same v2 recipe. Eval 0-shot on GSM8K test n=500. Compare to standard math scales (same eval). Also compare CoT prompting on baseline vs CoT prompting on scale-trained model — if scales compound with CoT prompting, reasoning capacity is real.

**Budget.** ~1hr train + ~2hr eval = ~3 A100-hrs. Call it $6.

## Total budget

| Run | A100-hrs | Est $ |
|---|---|---|
| 1. GSM8K n=500 + MATH OOD, 3 seeds | 5 | $10 |
| 2. LoRA matched baseline | 3 | $6 |
| 3. Router n=400 + domain evals | 3.5 | $7 |
| 4. 8B v2 replication | 5 | $10 |
| 5. Data efficiency (conditional) | 3 | $6 |
| 6. FP16 8B 0-shot baseline (fair comparison) | 3 | $6 |
| 7. CoT-data math personality | 3 | $6 |
| **Total** | **~25.5** | **~$51** |

Modal A100-80GB is ~$2/hr effective. Fits cleanly in the $30-50 budget.

## What comes out of this

A clean "measured at paper-quality n on A100" table that either replicates the current small-n findings at full confidence or surfaces the places where they don't survive scale-up. Both are wins: replication strengthens the thesis; a failed replication tells us we were riding small-n noise on some specific claim and lets us correct it in the repo before submission.

Explicitly *not* in this plan: the safety-scale pilot (separate doc, runs locally first to establish the mechanism before burning A100 budget).

## Order-of-operations dependency

1 → 2 → 3 can run in parallel on three A100 slots (if we grab 3 at once) or sequential over ~2 sessions. 4 depends on 1/2/3 only loosely (if v2 collapses at 8B we'd still want to know). 5 is conditional on the local data-efficiency outcome.

Minimum viable A100 session: just #1 and #2 — the two most consequential — for ~$16 total. Everything else improves the research but isn't load-bearing for the core claims.
