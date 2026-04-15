"""GPTQ-style 1-bit quantization for Qwen3-1.7B.

Proper column-by-column quantization with Hessian error compensation.
This is the industry-standard approach adapted for 1-bit (binary signs + per-group scales).

For each linear layer:
1. Compute Hessian H = X^T X / n (input covariance)
2. Process columns left-to-right in groups of GROUP_SIZE
3. For each group: quantize to binary, compute error, distribute error
   to remaining columns via H^-1 (Cholesky-based for stability)

This naturally produces the Bonsai-like metrics:
- Sign flips (21-30%) from error compensation modifying future columns
- Scale inflation from accumulated error redistribution
- Depth-scale correlation from deeper layers having larger activations
"""
import modal
import json

app = modal.App("1bit-gptq")

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
CALIB_BATCH = 8
MAX_SEQ_LEN = 512
DAMPENING = 0.01  # Hessian diagonal dampening for numerical stability


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
    gpu="A10G:1",
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=32768,
    volumes={"/data": vol},
)
def run_gptq_1bit():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import os
    import gc
    import math
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

    calib_texts = []

    # SlimOrca (conversations format)
    print("  Loading SlimOrca...", flush=True)
    orca = load_dataset("Open-Orca/SlimOrca", split="train", streaming=True)
    count = 0
    for row in orca:
        convs = row.get("conversations", [])
        text = " ".join(c.get("value", "") for c in convs if c.get("from") == "gpt")
        if len(text) > 100:
            calib_texts.append(text[:2000])
            count += 1
            if count >= 180:
                break
    print(f"    Got {count} SlimOrca", flush=True)

    # Alpaca
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
    print(f"    Got {count} Alpaca", flush=True)
    print(f"  Total: {len(calib_texts)} samples", flush=True)

    # Tokenize
    calib_tokens = []
    for text in calib_texts:
        toks = tokenizer(text, return_tensors="pt", truncation=True,
                         max_length=MAX_SEQ_LEN, padding=False)
        calib_tokens.append(toks["input_ids"].squeeze(0))

    # ══════════════════════════════════════════════════════════
    # Step 2: Load model
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Step 2: Loading Qwen3-1.7B...")
    print("=" * 70, flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cpu", token=hf_token,
    )
    model.eval()
    n_layers = len(model.model.layers)
    print(f"  Loaded. {n_layers} layers.", flush=True)

    # ══════════════════════════════════════════════════════════
    # Step 3: Capture per-linear-layer input activations
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Step 3: Capturing activations via hooks...")
    print("=" * 70, flush=True)

    model.to(device)

    # Collect Hessians directly (not raw activations — saves memory)
    # H = X^T X / n for each linear layer
    hessians = {}
    sample_counts = defaultdict(int)
    hooks = []

    def make_hook(name):
        def hook_fn(module, args, output):
            inp = args[0].detach().float()  # [batch, seq, in_dim]
            inp_2d = inp.reshape(-1, inp.shape[-1])  # [tokens, in_dim]
            n = inp_2d.shape[0]
            # Accumulate H = X^T X
            if name not in hessians:
                hessians[name] = torch.zeros(inp_2d.shape[1], inp_2d.shape[1], device=device)
            hessians[name].addmm_(inp_2d.t(), inp_2d, alpha=1.0)
            sample_counts[name] += n
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.numel() >= GROUP_SIZE:
            cat = classify_layer(name)
            if cat != "other" and cat != "token_embd":
                hooks.append(module.register_forward_hook(make_hook(name)))

    print(f"  {len(hooks)} hooks registered", flush=True)

    n_processed = 0
    with torch.no_grad():
        for i in range(0, len(calib_tokens), CALIB_BATCH):
            batch = calib_tokens[i:i+CALIB_BATCH]
            max_len = max(t.shape[0] for t in batch)
            padded = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
            attn_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
            for j, t in enumerate(batch):
                padded[j, :t.shape[0]] = t
                attn_mask[j, :t.shape[0]] = 1
            model(padded.to(device), attention_mask=attn_mask.to(device))
            n_processed += len(batch)
            if n_processed % 64 == 0:
                print(f"    {n_processed}/{len(calib_tokens)}...", flush=True)

    for h in hooks:
        h.remove()

    # Normalize Hessians
    for name in hessians:
        hessians[name] /= sample_counts[name]

    print(f"  Computed {len(hessians)} Hessians", flush=True)

    # ══════════════════════════════════════════════════════════
    # Step 4: GPTQ 1-bit quantization
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Step 4: GPTQ 1-bit quantization")
    print("=" * 70, flush=True)

    layer_metrics = {}
    optimized_layers = {}

    for name in sorted(hessians.keys()):
        # Get module
        module = model
        for part in name.split("."):
            module = getattr(module, part)

        cat = classify_layer(name)
        depth = get_layer_depth(name)
        W = module.weight.data.float()  # [out_dim, in_dim]
        out_dim, in_dim = W.shape
        gs = GROUP_SIZE

        H = hessians[name]  # [in_dim, in_dim]

        # Add dampening to diagonal for stability
        damp = DAMPENING * H.diag().mean()
        H.diagonal().add_(damp)

        # Cholesky decomposition of H (for efficient column updates)
        try:
            L = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(L)
        except Exception as e:
            print(f"  {name}: Cholesky failed ({e}), using diagonal approx", flush=True)
            H_inv = torch.diag(1.0 / H.diag().clamp(min=1e-8))

        # Work on a copy of W
        W_quant = W.clone()  # This will be modified column-by-column
        W_binary = torch.zeros_like(W)  # Final binary output

        # Process columns in groups
        n_groups = (in_dim + gs - 1) // gs

        for g in range(n_groups):
            col_start = g * gs
            col_end = min(col_start + gs, in_dim)
            cols = slice(col_start, col_end)
            actual_gs = col_end - col_start

            # Current weights for this group (may have been modified by error compensation)
            w_group = W_quant[:, cols]  # [out_dim, actual_gs]

            # Quantize: binary signs + per-group scale
            signs = w_group.sign()
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            scale = w_group.abs().mean(dim=1, keepdim=True)  # [out_dim, 1] per-row scale
            w_binary = signs * scale

            W_binary[:, cols] = w_binary

            # Quantization error
            quant_error = w_group - w_binary  # [out_dim, actual_gs]

            # Distribute error to remaining columns using Hessian inverse
            if col_end < in_dim:
                # H_inv block: how much each remaining column should adjust
                # for the error in the current columns
                H_qq = H_inv[cols, cols]  # [actual_gs, actual_gs]
                H_qr = H_inv[cols, col_end:]  # [actual_gs, remaining]

                # Error compensation: remaining cols -= error @ H_qq^-1 @ H_qr
                # This is the GPTQ update rule
                try:
                    H_qq_inv = torch.linalg.inv(H_qq)
                    update = quant_error @ H_qq_inv @ H_qr  # [out_dim, remaining]
                except Exception:
                    # Fallback: diagonal approximation
                    diag_inv = 1.0 / H_qq.diag().clamp(min=1e-8)
                    update = quant_error * diag_inv.unsqueeze(0) @ H_qr

                W_quant[:, col_end:] -= update

        # ── Compute metrics ──
        # Reshape for group-level analysis (groups along columns per row)
        pad_in = (gs - in_dim % gs) % gs
        if pad_in > 0:
            W_b_padded = F.pad(W_binary, (0, pad_in))
            W_orig_padded = F.pad(W, (0, pad_in))
        else:
            W_b_padded = W_binary
            W_orig_padded = W
        in_pad = W_b_padded.shape[1]
        n_grp = in_pad // gs

        W_b_grouped = W_b_padded.reshape(out_dim, n_grp, gs)
        W_o_grouped = W_orig_padded.reshape(out_dim, n_grp, gs)

        b_signs = W_b_grouped.sign()
        b_signs = torch.where(b_signs == 0, torch.ones_like(b_signs), b_signs)
        o_signs = W_o_grouped.sign()
        o_signs = torch.where(o_signs == 0, torch.ones_like(o_signs), o_signs)

        bit_match = (b_signs == o_signs).float().mean().item()

        b_scales = W_b_grouped.abs().mean(dim=2)
        o_scales = W_o_grouped.abs().mean(dim=2)
        ratio_med = (b_scales / o_scales.clamp(min=1e-8)).flatten().median().item()

        kurt = float(scipy_stats.kurtosis(W_binary.cpu().reshape(-1).numpy(), fisher=True))

        # Replace model weight
        module.weight.data = W_binary.half()

        metrics = {
            "category": cat,
            "depth": depth,
            "bit_match": round(bit_match, 4),
            "ratio_med": round(ratio_med, 4),
            "kurtosis": round(kurt, 4),
            "scale_mean": round(b_scales.mean().item(), 6),
        }
        layer_metrics[name] = metrics

        optimized_layers[name] = {
            "signs": b_signs.cpu(),
            "scales": b_scales.cpu(),
        }

        print(f"  {name} ({cat}): bit_match={bit_match:.4f}, ratio_med={ratio_med:.4f}, "
              f"kurtosis={kurt:.4f}", flush=True)

    # Free Hessians
    del hessians
    gc.collect()
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════
    # Per-category summary
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PER-CATEGORY SUMMARY")
    print("=" * 70)

    cat_agg = defaultdict(lambda: {"bit_match": [], "ratio_med": [], "kurtosis": []})
    for m in layer_metrics.values():
        cat_agg[m["category"]]["bit_match"].append(m["bit_match"])
        cat_agg[m["category"]]["ratio_med"].append(m["ratio_med"])
        cat_agg[m["category"]]["kurtosis"].append(m["kurtosis"])

    bonsai_targets = {
        "ffn_up": (0.70, 2.96, -1.94), "ffn_gate": (0.70, 2.65, -1.89),
        "attn_v": (0.77, 2.71, -1.94), "ffn_down": (0.74, 2.08, -1.93),
        "attn_o": (0.74, 2.32, -1.77), "attn_k": (0.71, 1.83, -1.81),
        "attn_q": (0.71, 1.64, -1.65),
    }

    print(f"\n{'Cat':<10} {'N':>3} {'bit_match':>10} {'B_target':>10} {'ratio_med':>10} "
          f"{'B_target':>10} {'kurtosis':>10} {'B_target':>10}")
    print("-" * 80)

    cat_order = ["ffn_up", "ffn_gate", "attn_v", "ffn_down", "attn_o", "attn_k", "attn_q"]
    our_summary = {}
    for cat in cat_order:
        if cat not in cat_agg:
            continue
        a = cat_agg[cat]
        bm = np.median(a["bit_match"])
        rm = np.median(a["ratio_med"])
        ku = np.median(a["kurtosis"])
        t_bm, t_rm, t_ku = bonsai_targets.get(cat, (0, 0, 0))

        print(f"{cat:<10} {len(a['bit_match']):>3} {bm:>10.4f} {t_bm:>10.2f} "
              f"{rm:>10.4f} {t_rm:>10.2f} {ku:>10.4f} {t_ku:>10.2f}")

        our_summary[cat] = {
            "bit_match": round(float(bm), 4),
            "ratio_med": round(float(rm), 4),
            "kurtosis": round(float(ku), 4),
        }

    # ══════════════════════════════════════════════════════════
    # Evaluate: generate text
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EVALUATION: Text generation")
    print("=" * 70, flush=True)

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
    # Save results
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Saving results...")
    print("=" * 70, flush=True)

    ckpt_path = "/data/qwen17b_gptq1bit.pt"
    torch.save({
        "optimized_layers": {k: {"signs": v["signs"], "scales": v["scales"]}
                             for k, v in optimized_layers.items()},
        "metrics": layer_metrics,
        "summary": our_summary,
        "config": {"model": model_id, "group_size": GROUP_SIZE, "dampening": DAMPENING},
    }, ckpt_path)

    results_path = "/data/qwen17b_gptq1bit_results.json"
    with open(results_path, "w") as f:
        json.dump({"per_layer": layer_metrics, "per_category": our_summary}, f, indent=2)

    vol.commit()
    print(f"  Saved to {ckpt_path} and {results_path}", flush=True)

    print(f"\n{'=' * 70}")
    print("GPTQ 1-BIT QUANTIZATION COMPLETE")
    print(f"{'=' * 70}", flush=True)

    return {"per_category": our_summary, "per_layer": layer_metrics}


@app.local_entrypoint()
def main():
    print("GPTQ 1-bit Quantization — Qwen3-1.7B\n")
    result = run_gptq_1bit.remote()
    if result and "per_category" in result:
        print("\n\nFINAL SUMMARY:")
        for cat, m in result["per_category"].items():
            print(f"  {cat}: bit_match={m['bit_match']:.4f}, "
                  f"ratio_med={m['ratio_med']:.4f}, kurtosis={m['kurtosis']:.4f}")
