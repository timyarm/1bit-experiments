# Reproducing

All experiments run on either Modal (T4/A100) or a local GPU (GTX 1660 Super, 6GB). Each experiment directory under `experiments/` has the full script with inline results.

## Dependencies

```bash
pip install torch transformers datasets safetensors nltk gguf
```

For llama.cpp-based evaluation (used by `eval_domain_matched.py`, `eval_interpolation.py`, `eval_mbpp_fixed.py`), you also need a local llama.cpp build and a GGUF of the target model. The eval harness expects `~/freigent/apps/trucksim/data/llm/Bonsai-1.7B.gguf`; edit `llama_fast_eval.py` if your layout differs.

## Scale personalities — Bonsai 1.7B v2 (local, single ~6GB GPU)

Produces the headline GSM8K 5.3× result.

```bash
# Train the math / knowledge / code scale profiles (~30 min each)
python experiments/scale-personalities/scale_v2_proper.py

# Domain-matched evaluation: GSM8K / MMLU / MBPP across 4 profiles
python experiments/scale-personalities/eval_domain_matched.py

# Extra: interpolation curve (11-alpha math↔knowledge blend)
python experiments/scale-personalities/eval_interpolation.py

# Extra: data efficiency curve (math scales at n={10, 30, 100, 300})
python experiments/scale-personalities/eval_data_efficiency.py

# Plot the curves after the above finish
python experiments/scale-personalities/plot_results.py all
```

## Scale router — Bonsai 1.7B V2 (local, ~6GB GPU)

Requires the trained scale tables from `scale_v2_proper.py`.

```bash
python experiments/scale-personalities/routed_scale_router.py
```

## Scale personalities — Bonsai 8B (Modal T4, ~2-3 hr)

Produces the 8/8 diagonal-dominance PPL results.

```bash
modal run experiments/scale-personalities/train_8profiles.py
modal run experiments/scale-personalities/validate_3way.py
modal run experiments/scale-personalities/activation_probe.py
```

## Bonsai forensic analysis (CPU-only)

Downloads the unpacked Bonsai models from HuggingFace and extracts the 6 forensic metrics that informed the v2 QAT recipe.

```bash
python experiments/bonsai-forensics/analyze.py
```

## Diagnostic: fixed MBPP extractor

The initial MBPP run returned 0% across all profiles due to an extraction bug. This diagnostic reproduces the diagnosis on n=10 and outputs raw model generations + extracted code + test result.

```bash
python experiments/scale-personalities/diag_mbpp.py
python experiments/scale-personalities/eval_mbpp_fixed.py  # full n=100
```

## Notes on local GPU reproduction

- 6GB VRAM fits Bonsai 1.7B with `gradient_checkpointing_enable()` and seq_len ≤ 256. Larger sequences will OOM.
- Training runs write scale checkpoints to `checkpoints/{domain}_scales_v2.pt` and patched GGUFs to `~/freigent/apps/trucksim/data/llm/Bonsai-1.7B-{domain}-v2.gguf`.
- Eval results JSON land in `checkpoints/` alongside the scales; plot scripts read from there.
