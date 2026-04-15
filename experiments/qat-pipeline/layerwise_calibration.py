"""Layer-wise binary optimizer — reproduce Bonsai's 1-bit recipe.

Instead of STE (which creates train/eval gap and mode collapse), this does
GPTQ-style layer-wise reconstruction in exact binary space:

For each linear layer:
  1. Capture input activations X from calibration data through teacher
  2. Compute teacher output Y = W_teacher @ X
  3. Find binary weights (signs * scales) that minimize ||W_binary @ X - Y||²
  4. Use greedy sign-flipping: start from teacher signs, flip ones that help most
  5. Jointly optimize scales per group
  6. Propagate through optimized layer to get inputs for next layer

This stays in exact binary space at all times — no train/eval gap.

Validates against Bonsai-1.7B forensic measurements we confirmed:
  - bit_match: 0.70 (ffn_up) to 0.77 (attn_v)
  - ratio_med: 1.6 (attn_q) to 2.95 (ffn_up)
  - kurtosis: -1.6 to -1.94
  - depth-scale correlation: 0.88 (ffn_up)
"""
import modal
import json

app = modal.App("1bit-layerwise")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.51.0",
        "accelerate>=1.5.0",
        "huggingface_hub>=0.28.0",
        "datasets>=3.0.0",
        "scipy>=1.12.0",
    )
)

vol = modal.Volume.from_name("1bit-checkpoints", create_if_missing=True)

GROUP_SIZE = 128
N_CALIB_SAMPLES = 256
MAX_SEQ_LEN = 512
# Sign-flip iterations per layer (more = better reconstruction but slower)
FLIP_ITERATIONS = 3
# Batch size for activation capture (memory limited)
CALIB_BATCH = 4


def classify_layer(name: str) -> str:
    if "embed" in name or "lm_head" in name:
        return "token_embd"
    if "q_proj" in name: return "attn_q"
    if "k_proj" in name: return "attn_k"
    if "v_proj" in name: return "attn_v"
    if "o_proj" in name: return "attn_o"
    if "up_proj" in name: return "ffn_up"
    if "gate_proj" in name: return "ffn_gate"
    if "down_proj" in name: return "ffn_down"
    return "other"


def get_layer_depth(name: str) -> int:
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return -1


@app.function(
    image=image,
    gpu="A10G:1",  # 24GB — enough for 1.7B + activations
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=32768,
    volumes={"/data": vol},
)
def run_layerwise_optimization():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import os
    import gc
    import numpy as np
    from scipy import stats as scipy_stats
    from collections import defaultdict

    hf_token = os.environ.get("HF_TOKEN")
    device = "cuda"
    model_id = "Qwen/Qwen3-1.7B"

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset

    # ══════════════════════════════════════════════════════════
    # Step 1: Load calibration data
    # ══════════════════════════════════════════════════════════
    print("=" * 70)
    print("Step 1: Loading calibration data...")
    print("=" * 70, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Mix of datasets for diversity (same principle as our QAT runs)
    calib_texts = []

    # SlimOrca — reasoning (uses "conversations" field with role/value dicts)
    print("  Loading SlimOrca...", flush=True)
    orca = load_dataset("Open-Orca/SlimOrca", split="train", streaming=True)
    count = 0
    for row in orca:
        convs = row.get("conversations", [])
        # Extract assistant responses
        text = " ".join(c.get("value", "") for c in convs if c.get("from") == "gpt")
        if len(text) > 100:
            calib_texts.append(text[:2000])
            count += 1
            if count >= 180:
                break
    print(f"    Got {count} SlimOrca samples", flush=True)

    # Alpaca — instruction following
    print("  Loading Alpaca...", flush=True)
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    alpaca_shuf = alpaca.shuffle(seed=42)
    count = 0
    for row in alpaca_shuf:
        text = (row.get("instruction", "") + " " + row.get("output", "")).strip()
        if len(text) > 100:
            calib_texts.append(text[:2000])
            count += 1
            if count >= 76:
                break
    print(f"    Got {count} Alpaca samples", flush=True)

    print(f"  Total calibration samples: {len(calib_texts)}", flush=True)

    # Tokenize
    print("  Tokenizing...", flush=True)
    calib_tokens = []
    for text in calib_texts:
        toks = tokenizer(text, return_tensors="pt", truncation=True,
                         max_length=MAX_SEQ_LEN, padding=False)
        calib_tokens.append(toks["input_ids"].squeeze(0))

    # ══════════════════════════════════════════════════════════
    # Step 2: Load teacher model
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Step 2: Loading Qwen3-1.7B teacher...")
    print("=" * 70, flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cpu", token=hf_token,
    )
    model.eval()
    print(f"  Model loaded. Layers: {len(model.model.layers)}", flush=True)

    n_layers = len(model.model.layers)

    # ══════════════════════════════════════════════════════════
    # Step 3: Capture per-linear-layer input activations via hooks
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Step 3: Capturing per-layer activations via hooks")
    print("=" * 70, flush=True)

    model.to(device)

    # We'll capture inputs to every Linear layer via forward hooks
    # Process calibration data in batches, accumulating activations
    linear_inputs = defaultdict(list)  # name -> list of input tensors
    hooks = []

    def make_hook(name):
        def hook_fn(module, args, output):
            # args[0] is the input tensor
            inp = args[0].detach().cpu()
            linear_inputs[name].append(inp)
        return hook_fn

    # Register hooks on all Linear layers we want to optimize
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.numel() >= GROUP_SIZE:
            cat = classify_layer(name)
            if cat != "other" and cat != "token_embd":
                hooks.append(module.register_forward_hook(make_hook(name)))

    print(f"  Registered {len(hooks)} hooks on linear layers", flush=True)

    # Run calibration data through the FULL teacher model
    print("  Running calibration data through teacher...", flush=True)
    n_processed = 0
    with torch.no_grad():
        for i in range(0, len(calib_tokens), CALIB_BATCH):
            batch_tokens = calib_tokens[i:i+CALIB_BATCH]
            max_len = max(t.shape[0] for t in batch_tokens)
            padded = torch.full((len(batch_tokens), max_len), tokenizer.pad_token_id, dtype=torch.long)
            attn_mask = torch.zeros(len(batch_tokens), max_len, dtype=torch.long)
            for j, t in enumerate(batch_tokens):
                padded[j, :t.shape[0]] = t
                attn_mask[j, :t.shape[0]] = 1

            padded = padded.to(device)
            attn_mask = attn_mask.to(device)

            # Full forward pass — hooks capture all linear inputs
            model(padded, attention_mask=attn_mask)
            n_processed += len(batch_tokens)
            if n_processed % 32 == 0:
                print(f"    Processed {n_processed}/{len(calib_tokens)} samples...", flush=True)

    # Remove hooks
    for h in hooks:
        h.remove()

    print(f"  Captured activations for {len(linear_inputs)} linear layers", flush=True)
    for name in list(linear_inputs.keys())[:3]:
        acts = linear_inputs[name]
        total_tokens = sum(a.reshape(-1, a.shape[-1]).shape[0] for a in acts)
        print(f"    {name}: {len(acts)} batches, ~{total_tokens} tokens", flush=True)

    # Store optimized weights
    optimized_layers = {}
    layer_metrics = {}

    def optimize_linear_layer(name, linear, input_acts):
        """Optimize a single linear layer to binary weights.

        Algorithm:
        1. Naive PTQ init (teacher signs + mean-abs scales)
        2. Iterative: compute vectorized MSE deltas, flip top-K signs
        3. After each flip round: recompute scales from weight projection
           (NOT activation-based least-squares which requires residuals)
        4. Final: coordinate-descent scale refinement using activations
        """
        cat = classify_layer(name)
        W = linear.weight.data.float()
        out_dim, in_dim = W.shape
        gs = GROUP_SIZE

        # Collect activations
        X_list = []
        for acts in input_acts[:64]:
            X_list.append(acts.reshape(-1, acts.shape[-1]))
        X = torch.cat(X_list, dim=0).float()
        if X.shape[0] > 4096:
            X = X[torch.randperm(X.shape[0])[:4096]]
        n_tokens = X.shape[0]

        Y_teacher = X @ W.t()

        # Pad input dim
        pad_in = (gs - in_dim % gs) % gs
        if pad_in > 0:
            W_padded = F.pad(W, (0, pad_in))
            X_padded = F.pad(X, (0, pad_in))
        else:
            W_padded = W
            X_padded = X
        in_padded = W_padded.shape[1]
        n_grp = in_padded // gs

        W_grouped = W_padded.reshape(out_dim, n_grp, gs)
        X_grouped = X_padded.reshape(n_tokens, n_grp, gs)

        # ── Init: naive PTQ ──
        signs = W_grouped.sign()
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        # Weight-projection scale: s = mean(W * signs) per group
        # When signs=sign(W), this = mean(|W|). After flips, accounts for changes.
        scales = (W_grouped * signs).mean(dim=2).abs().clamp(min=1e-7)

        def build_W():
            return (signs * scales.unsqueeze(2)).reshape(out_dim, in_padded)[:, :in_dim]

        def eval_mse():
            return ((Y_teacher - X @ build_W().t()) ** 2).mean().item()

        initial_mse = eval_mse()

        # ── Sign optimization ──
        for iteration in range(FLIP_ITERATIONS):
            mse_before = eval_mse()
            W_bin = build_W()
            error = (X @ W_bin.t()) - Y_teacher

            # Vectorized MSE delta for every element
            X_flat = X_grouped.reshape(n_tokens, -1)
            dot = (error.t() @ X_flat).reshape(out_dim, n_grp, gs)
            x_norms = (X_grouped ** 2).sum(dim=0)

            s_exp = scales.unsqueeze(2)
            mse_delta = (1.0 / n_tokens) * (
                -4.0 * signs * s_exp * dot + 4.0 * s_exp**2 * x_norms.unsqueeze(0)
            )
            if pad_in > 0:
                mse_delta[:, -1, -pad_in:] = float('inf')

            # Flip top 5% most beneficial (conservative)
            max_flips = max(1, int(0.05 * signs.numel()))
            n_beneficial = (mse_delta < 0).sum().item()
            n_to_flip = min(n_beneficial, max_flips)

            if n_to_flip > 0:
                _, top_idx = mse_delta.reshape(-1).topk(n_to_flip, largest=False)
                flat_s = signs.reshape(-1)
                flat_s[top_idx] = -flat_s[top_idx]
                signs = flat_s.reshape(out_dim, n_grp, gs)

            # Recompute scales from weight projection (stable, no activation dependency)
            scales = (W_grouped * signs).mean(dim=2).abs().clamp(min=1e-7)

            mse_after = eval_mse()
            print(f"      iter {iteration}: MSE {mse_before:.6f} → {mse_after:.6f} "
                  f"(flipped {n_to_flip}/{n_beneficial})", flush=True)

        # ── Final scale refinement via coordinate descent ──
        # For each group g, compute optimal scale using residual from other groups
        print(f"      scale refinement...", end="", flush=True)
        for cd_iter in range(3):
            for g in range(n_grp):
                X_g = X_grouped[:, g, :]     # [tokens, gs]
                signs_g = signs[:, g, :]     # [out, gs]

                # Residual = Y_teacher - contribution of all OTHER groups
                W_current = build_W()
                # Contribution of group g: signs_g * scale_g @ X_g.T
                contrib_g = (signs_g * scales[:, g:g+1]) @ X_g.t()  # [out, tokens]
                Y_other = (X @ W_current.t()).t() - contrib_g  # [out, tokens]
                residual = Y_teacher.t() - Y_other  # [out, tokens] — what group g needs to explain

                # basis = signs_g @ X_g.T → [out, tokens]
                basis = signs_g @ X_g.t()
                numer = (basis * residual).sum(dim=1)  # [out]
                denom = (basis ** 2).sum(dim=1).clamp(min=1e-10)
                scales[:, g] = (numer / denom).clamp(min=1e-7)

        final_mse = eval_mse()
        print(f" done. Final MSE: {final_mse:.6f}", flush=True)

        # ── Metrics ──
        teacher_signs = W_grouped.sign()
        teacher_signs = torch.where(teacher_signs == 0, torch.ones_like(teacher_signs), teacher_signs)
        bit_match = (signs == teacher_signs).float().mean().item()

        teacher_scales = W_grouped.abs().mean(dim=2)
        ratio_med = (scales / teacher_scales.clamp(min=1e-8)).flatten().median().item()

        W_final = build_W()
        kurt = float(scipy_stats.kurtosis(W_final.reshape(-1).numpy(), fisher=True))

        metrics = {
            "category": cat,
            "bit_match": round(bit_match, 4),
            "ratio_med": round(ratio_med, 4),
            "kurtosis": round(kurt, 4),
            "initial_mse": round(initial_mse, 6),
            "final_mse": round(final_mse, 6),
            "mse_reduction": round(1.0 - final_mse / max(initial_mse, 1e-10), 4),
            "scale_mean": round(scales.mean().item(), 6),
        }

        return W_final.half(), signs.cpu(), scales.cpu(), metrics

    # ══════════════════════════════════════════════════════════
    # Process each linear layer using captured activations
    # ══════════════════════════════════════════════════════════

    # Move model to CPU to free GPU memory for optimization
    model.cpu()
    torch.cuda.empty_cache()

    for full_name in sorted(linear_inputs.keys()):
        # Find the module
        module = model
        for part in full_name.split("."):
            module = getattr(module, part)

        cat = classify_layer(full_name)
        depth = get_layer_depth(full_name)
        print(f"\n  Optimizing {full_name} ({cat}, {list(module.weight.shape)})...", flush=True)

        W_binary, signs, scales, metrics = optimize_linear_layer(
            full_name, module, linear_inputs[full_name]
        )

        # Replace weight in-place (for later inference)
        module.weight.data = W_binary.cpu()

        optimized_layers[full_name] = {
            "signs": signs.cpu(),
            "scales": scales.cpu(),
        }
        layer_metrics[full_name] = metrics
        layer_metrics[full_name]["depth"] = depth

        print(f"    → bit_match={metrics['bit_match']:.4f}, ratio_med={metrics['ratio_med']:.4f}, "
              f"kurtosis={metrics['kurtosis']:.4f}, MSE: {metrics['initial_mse']:.6f}→{metrics['final_mse']:.6f} "
              f"({metrics['mse_reduction']*100:.1f}% reduction)", flush=True)

    # Free activation memory
    del linear_inputs
    gc.collect()

    # ══════════════════════════════════════════════════════════
    # Per-category summary
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PER-CATEGORY SUMMARY")
    print("=" * 70)

    cat_agg = defaultdict(lambda: {"bit_match": [], "ratio_med": [], "kurtosis": [], "mse_reduction": []})
    for name, m in layer_metrics.items():
        cat_agg[m["category"]]["bit_match"].append(m["bit_match"])
        cat_agg[m["category"]]["ratio_med"].append(m["ratio_med"])
        cat_agg[m["category"]]["kurtosis"].append(m["kurtosis"])
        cat_agg[m["category"]]["mse_reduction"].append(m["mse_reduction"])

    print(f"\n{'Category':<12} {'N':>3} {'bit_match':>10} {'ratio_med':>10} {'kurtosis':>10} {'MSE_red%':>10}")
    print("-" * 60)

    bonsai_targets = {
        "ffn_up": (0.70, 2.96), "ffn_gate": (0.70, 2.65), "attn_v": (0.77, 2.71),
        "ffn_down": (0.74, 2.08), "attn_o": (0.74, 2.32), "attn_k": (0.71, 1.83),
        "attn_q": (0.71, 1.64),
    }

    cat_order = ["ffn_up", "ffn_gate", "attn_v", "ffn_down", "attn_o", "attn_k", "attn_q"]
    our_summary = {}
    for cat in cat_order:
        if cat not in cat_agg:
            continue
        a = cat_agg[cat]
        bm = np.median(a["bit_match"])
        rm = np.median(a["ratio_med"])
        ku = np.median(a["kurtosis"])
        mr = np.median(a["mse_reduction"]) * 100

        target_bm, target_rm = bonsai_targets.get(cat, (0, 0))
        bm_match = "✓" if abs(bm - target_bm) < 0.05 else "✗"
        rm_match = "✓" if abs(rm - target_rm) < 0.5 else "✗"

        print(f"{cat:<12} {len(a['bit_match']):>3} {bm:>10.4f}{bm_match} {rm:>10.4f}{rm_match} "
              f"{ku:>10.4f} {mr:>10.1f}%")

        our_summary[cat] = {
            "bit_match": round(float(bm), 4),
            "ratio_med": round(float(rm), 4),
            "kurtosis": round(float(ku), 4),
            "mse_reduction_pct": round(float(mr), 1),
        }

    # ══════════════════════════════════════════════════════════
    # Evaluate: generate text with the binary model
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EVALUATION: Text generation with binary model")
    print("=" * 70, flush=True)

    model.to(device)
    model.eval()

    test_prompts = [
        "What is the capital of France?",
        "Explain gravity in one sentence.",
        "1 + 1 =",
        "The color of the sky is",
        "List three animals:",
        "A truck driver should check their mirrors because",
        "The speed limit in a school zone is typically",
    ]

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=60, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = output[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen, skip_special_tokens=True)
        print(f"  Q: {prompt}")
        print(f"  A: {text[:200]}", flush=True)

    # ══════════════════════════════════════════════════════════
    # Save checkpoint
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Saving checkpoint...")
    print("=" * 70, flush=True)

    ckpt_path = "/data/qwen17b_bonsai_layerwise.pt"
    torch.save({
        "optimized_layers": {k: {"signs": v["signs"], "scales": v["scales"]}
                             for k, v in optimized_layers.items()},
        "metrics": layer_metrics,
        "summary": our_summary,
        "config": {
            "model": model_id,
            "group_size": GROUP_SIZE,
            "n_calib_samples": N_CALIB_SAMPLES,
            "flip_iterations": FLIP_ITERATIONS,
        },
    }, ckpt_path)
    print(f"  Saved to {ckpt_path}", flush=True)

    # Also save results JSON
    results_path = "/data/qwen17b_bonsai_layerwise_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "per_layer": layer_metrics,
            "per_category": our_summary,
            "config": {
                "model": model_id,
                "group_size": GROUP_SIZE,
                "n_calib_samples": len(calib_texts),
                "flip_iterations": FLIP_ITERATIONS,
            },
        }, f, indent=2)
    print(f"  Results saved to {results_path}", flush=True)
    vol.commit()

    print(f"\n{'=' * 70}")
    print("LAYER-WISE OPTIMIZATION COMPLETE")
    print(f"{'=' * 70}", flush=True)

    return {"per_category": our_summary, "per_layer": layer_metrics}


@app.local_entrypoint()
def main():
    print("Bonsai Layer-wise Binary Optimizer — Qwen3-1.7B\n")
    result = run_layerwise_optimization.remote()
    if result and "per_category" in result:
        print("\n\nFINAL SUMMARY:")
        for cat, m in result["per_category"].items():
            print(f"  {cat}: bit_match={m['bit_match']:.4f}, ratio_med={m['ratio_med']:.4f}, "
                  f"kurtosis={m['kurtosis']:.4f}")
