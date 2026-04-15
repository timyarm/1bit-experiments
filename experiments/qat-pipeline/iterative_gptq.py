"""Iterative GPTQ with joint sign+scale optimization for 1-bit Qwen3-1.7B.

Algorithm per layer:
  1. Capture input activations via forward hook
  2. Initialize from GPTQ signs + mean-abs scales
  3. Iterate N rounds:
     a. Compute residual: R = W_teacher @ X - (signs * scale) @ X
     b. For each element: compute improvement from flipping its sign
     c. Flip top-K% most beneficial signs
     d. Re-optimize scales via activation-weighted least-squares
  4. Scales inflate naturally to 2-3x as signs diverge from teacher

Key insight: activation-weighted scale optimization forces scales UP when signs
differ from teacher — this is where Bonsai's 2-3x ratio_med comes from.
"""
import modal
import json

app = modal.App("1bit-iterative-gptq")

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
MAX_SEQ_LEN = 512
N_CALIB = 256
N_ITERS = 20         # sign-only iterations per layer (scales fixed)
FLIP_FRAC = 0.01     # flip top 1% per iteration (smaller batches = less interaction noise)


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


@app.function(
    image=image,
    gpu="A10G:1",
    timeout=10800,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=49152,
    volumes={"/data": vol},
)
def run_iterative_gptq():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import os
    import gc
    import numpy as np
    from scipy import stats as scipy_stats
    from collections import defaultdict, OrderedDict
    import time

    hf_token = os.environ.get("HF_TOKEN")
    device = "cuda"
    model_id = "Qwen/Qwen3-1.7B"

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset

    # ==================================================================
    # Step 1: Load calibration data
    # ==================================================================
    print("=" * 60)
    print("Step 1: Loading calibration data...")
    print("=" * 60, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    calib_texts = []

    print("  Loading SlimOrca...", flush=True)
    orca = load_dataset("Open-Orca/SlimOrca", split="train", streaming=True)
    count = 0
    for row in orca:
        convs = row.get("conversations", [])
        text = " ".join(c.get("value", "") for c in convs if c.get("from") == "gpt")
        if len(text) > 100:
            calib_texts.append(text[:2000])
            count += 1
            if count >= 200:
                break
    print(f"    Got {count} SlimOrca", flush=True)

    print("  Loading Alpaca...", flush=True)
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    alpaca_shuf = alpaca.shuffle(seed=42)
    count = 0
    for row in alpaca_shuf:
        text = (row.get("instruction", "") + " " + row.get("output", "")).strip()
        if len(text) > 100:
            calib_texts.append(text[:2000])
            count += 1
            if count >= 56:
                break
    print(f"    Got {count} Alpaca", flush=True)
    print(f"  Total: {len(calib_texts)} samples", flush=True)

    calib_tokens = []
    for text in calib_texts:
        toks = tokenizer(text, return_tensors="pt", truncation=True,
                         max_length=MAX_SEQ_LEN, padding=False)
        calib_tokens.append(toks["input_ids"].squeeze(0))

    # ==================================================================
    # Step 2: Load model and capture activations per linear layer
    # ==================================================================
    print("\n" + "=" * 60)
    print("Step 2: Loading model and capturing activations...")
    print("=" * 60, flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map=device, token=hf_token,
    )
    model.eval()

    # Register hooks on all target linear layers to capture inputs
    layer_inputs = {}   # name -> list of input tensors
    hooks = []

    target_layers = {}  # name -> module
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        cat = classify_layer(name)
        if cat in ("other", "token_embd"):
            continue
        target_layers[name] = module

    print(f"  Target layers: {len(target_layers)}", flush=True)

    def make_hook(layer_name):
        def hook_fn(module, input, output):
            x = input[0].detach().float()  # [batch, seq, in_dim]
            # Flatten batch and seq dims
            x_flat = x.reshape(-1, x.shape[-1])  # [tokens, in_dim]
            if layer_name not in layer_inputs:
                layer_inputs[layer_name] = []
            layer_inputs[layer_name].append(x_flat.cpu())
        return hook_fn

    for name, module in target_layers.items():
        h = module.register_forward_hook(make_hook(name))
        hooks.append(h)

    # Run calibration data through model
    print("  Running calibration forward passes...", flush=True)
    n_tokens = 0
    with torch.no_grad():
        for i, tok in enumerate(calib_tokens):
            input_ids = tok.unsqueeze(0).to(device)
            model(input_ids)
            n_tokens += tok.shape[0]
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(calib_tokens)} samples, {n_tokens} tokens", flush=True)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Concatenate activations per layer, subsample if too many tokens
    MAX_TOKENS = 4096  # limit for memory
    print(f"  Total tokens captured: {n_tokens}", flush=True)
    for name in layer_inputs:
        all_x = torch.cat(layer_inputs[name], dim=0)  # [total_tokens, in_dim]
        if all_x.shape[0] > MAX_TOKENS:
            indices = torch.randperm(all_x.shape[0])[:MAX_TOKENS]
            all_x = all_x[indices]
        layer_inputs[name] = all_x
        if name == list(layer_inputs.keys())[0]:
            print(f"  Activations shape per layer: {all_x.shape}", flush=True)

    # Save teacher weights before freeing model
    teacher_weights = {}
    for name, module in target_layers.items():
        teacher_weights[name] = module.weight.data.float().cpu()

    # Free model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("  Model freed, teacher weights saved", flush=True)

    # ==================================================================
    # Step 3: Iterative sign+scale optimization per layer
    # ==================================================================
    print("\n" + "=" * 60)
    print(f"Step 3: Iterative sign+scale optimization ({N_ITERS} iters/layer)")
    print("=" * 60, flush=True)

    optimized_layers = OrderedDict()
    cat_metrics = defaultdict(lambda: {"bit_match": [], "ratio_med": [], "kurtosis": []})

    t_start = time.time()

    for layer_idx, (name, W_teacher) in enumerate(teacher_weights.items()):
        cat = classify_layer(name)
        out_dim, in_dim = W_teacher.shape

        # Get activations for this layer
        X = layer_inputs[name].to(device)  # [n_tokens, in_dim]
        W = W_teacher.to(device)  # [out_dim, in_dim]
        n_tokens_layer = X.shape[0]

        # Pad input dim to multiple of GROUP_SIZE
        pad = (GROUP_SIZE - in_dim % GROUP_SIZE) % GROUP_SIZE
        if pad > 0:
            W_padded = F.pad(W, (0, pad))
            X_padded = F.pad(X, (0, pad))
        else:
            W_padded = W
            X_padded = X

        in_padded = W_padded.shape[1]
        n_groups = in_padded // GROUP_SIZE

        # Reshape to groups: W_g [out_dim, n_groups, GROUP_SIZE]
        W_g = W_padded.reshape(out_dim, n_groups, GROUP_SIZE)

        # Initialize signs from teacher
        signs = W_g.sign()
        signs[signs == 0] = 1.0

        # Initialize scales from mean-abs (FIXED during sign optimization)
        init_scales = W_g.abs().mean(dim=2)  # [out_dim, n_groups]
        scales = init_scales.clone()

        # Teacher output: Y_t = W @ X^T  [out_dim, n_tokens]
        Y_teacher = W @ X.T  # Use unpadded for teacher output

        # Precompute ||X_col||^2 (constant across iterations)
        X_col_norm_sq = (X_padded ** 2).sum(dim=0)  # [in_padded]
        X_col_norm_sq_g = X_col_norm_sq.reshape(n_groups, GROUP_SIZE)

        # Compute initial MSE
        Q_init = (signs * scales.unsqueeze(2)).reshape(out_dim, in_padded)[:, :in_dim]
        Y_init = Q_init @ X.T
        mse_init = ((Y_teacher - Y_init) ** 2).mean().item()
        best_mse = mse_init
        total_flips = 0

        # ============================================================
        # Phase 1: Sign-only optimization (scales FIXED at mean-abs)
        # ============================================================
        for iteration in range(N_ITERS):
            # Current quantized output with FIXED scales
            Q = (signs * scales.unsqueeze(2)).reshape(out_dim, in_padded)
            Y_quant = Q[:, :in_dim] @ X.T
            R = Y_teacher - Y_quant

            # Sign flip improvement (same formula, but scales won't change)
            RX = R @ X_padded
            RX_g = RX.reshape(out_dim, n_groups, GROUP_SIZE)

            s_expanded = scales.unsqueeze(2)
            improvement = (
                -4.0 * s_expanded * signs * RX_g
                - 4.0 * s_expanded ** 2 * X_col_norm_sq_g.unsqueeze(0)
            )

            # Flip top FLIP_FRAC of positive-improvement elements
            positive_mask = improvement > 0
            n_positive = positive_mask.sum().item()

            if n_positive > 0:
                n_to_flip = max(1, int(FLIP_FRAC * signs.numel()))
                n_to_flip = min(n_to_flip, n_positive)

                flat_improvement = improvement.reshape(-1)
                topk_vals, topk_idx = flat_improvement.topk(n_to_flip)

                actual_flips = (topk_vals > 0).sum().item()
                if actual_flips > 0:
                    flip_mask = torch.zeros_like(flat_improvement, dtype=torch.bool)
                    flip_mask[topk_idx[:actual_flips]] = True
                    flip_mask = flip_mask.reshape(signs.shape)
                    signs[flip_mask] *= -1
                    total_flips += actual_flips

            # Verify MSE decreased
            Q_check = (signs * scales.unsqueeze(2)).reshape(out_dim, in_padded)[:, :in_dim]
            Y_check = Q_check @ X.T
            mse_now = ((Y_teacher - Y_check) ** 2).mean().item()

            if mse_now > best_mse * 1.01:
                # Revert flips if MSE increased (shouldn't happen with fixed scales)
                if actual_flips > 0:
                    signs[flip_mask] *= -1
                    total_flips -= actual_flips
                break
            best_mse = mse_now

        mse_after_signs = best_mse

        # ============================================================
        # Phase 2: Final scale optimization (per-group least-squares,
        #           clamped to prevent shrinkage below 0.8x mean-abs)
        # ============================================================
        X_grouped = X_padded.T.reshape(n_groups, GROUP_SIZE, n_tokens_layer)
        new_scales = torch.zeros_like(scales)
        for g in range(n_groups):
            signed_basis = signs[:, g, :] @ X_grouped[g]  # [out_dim, n_tokens]
            teacher_g = W_g[:, g, :] @ X_grouped[g]       # [out_dim, n_tokens]

            numerator = (teacher_g * signed_basis).sum(dim=1)
            denominator = (signed_basis ** 2).sum(dim=1).clamp(min=1e-10)
            opt_scale = (numerator / denominator).clamp(min=1e-8)

            # Clamp: scale >= 0.8 * init to prevent shrinkage
            min_scale = 0.8 * init_scales[:, g]
            new_scales[:, g] = torch.max(opt_scale, min_scale)

        scales = new_scales

        # Final MSE after scale optimization
        Q_final = (signs * scales.unsqueeze(2)).reshape(out_dim, in_padded)[:, :in_dim]
        Y_final = Q_final @ X.T
        mse_new = ((Y_teacher - Y_final) ** 2).mean().item()

        # Final metrics
        teacher_signs = W_g.sign()
        teacher_signs[teacher_signs == 0] = 1.0
        bm = (signs == teacher_signs).float().mean().item()
        teacher_scale = W_g.abs().mean(dim=2).clamp(min=1e-8)
        rm = (scales / teacher_scale).flatten().median().item()

        # Kurtosis of baked weights
        baked = (signs * scales.unsqueeze(2)).reshape(out_dim, in_padded)[:, :in_dim]
        kurt = float(scipy_stats.kurtosis(baked.cpu().reshape(-1).numpy(), fisher=True))

        cat_metrics[cat]["bit_match"].append(bm)
        cat_metrics[cat]["ratio_med"].append(rm)
        cat_metrics[cat]["kurtosis"].append(kurt)

        # Store result
        optimized_layers[name] = {
            "signs": signs.cpu(),
            "scales": scales.cpu(),
        }

        # Free GPU memory for this layer
        del X, W, W_padded, X_padded, W_g, signs, scales, Y_teacher, init_scales
        del Q_init, Y_init, improvement, X_col_norm_sq, X_col_norm_sq_g
        del X_grouped, new_scales, Q_final, Y_final, baked
        torch.cuda.empty_cache()

        if (layer_idx + 1) % 20 == 0 or layer_idx < 3:
            elapsed = time.time() - t_start
            eta = elapsed / (layer_idx + 1) * (len(teacher_weights) - layer_idx - 1)
            flip_pct = (1 - bm) * 100
            print(f"  [{layer_idx+1}/{len(teacher_weights)}] {name}: "
                  f"bm={bm:.4f}({flip_pct:.1f}%flip) rm={rm:.4f} ku={kurt:.4f} "
                  f"mse {mse_init:.6f}->{mse_after_signs:.6f}->{mse_new:.6f} "
                  f"flips={total_flips} ({elapsed:.0f}s, ~{eta:.0f}s ETA)", flush=True)

    total_time = time.time() - t_start
    print(f"\n  Total optimization time: {total_time:.0f}s", flush=True)

    # ==================================================================
    # Step 4: Print forensic summary
    # ==================================================================
    print("\n" + "=" * 60)
    print("Step 4: Forensic summary")
    print("=" * 60, flush=True)

    bonsai_tgt = {
        "ffn_up": (0.70, 2.96, -1.94), "ffn_gate": (0.70, 2.65, -1.89),
        "attn_v": (0.77, 2.71, -1.94), "ffn_down": (0.74, 2.08, -1.93),
        "attn_o": (0.74, 2.32, -1.77), "attn_k": (0.71, 1.83, -1.81),
        "attn_q": (0.71, 1.64, -1.65),
    }

    print(f"  {'Cat':<10} {'bit_match':>9} {'tgt':>6} {'ratio_med':>10} {'tgt':>6} {'kurtosis':>9} {'tgt':>6}")
    print("  " + "-" * 58)

    final_summary = {}
    for cat in ["ffn_up", "ffn_gate", "attn_v", "ffn_down", "attn_o", "attn_k", "attn_q"]:
        if cat not in cat_metrics:
            continue
        bm = np.median(cat_metrics[cat]["bit_match"])
        rm = np.median(cat_metrics[cat]["ratio_med"])
        ku = np.median(cat_metrics[cat]["kurtosis"])
        tb, tr, tk = bonsai_tgt.get(cat, (0, 0, 0))
        print(f"  {cat:<10} {bm:>9.4f} {tb:>6.2f} {rm:>10.4f} {tr:>6.2f} {ku:>9.4f} {tk:>6.2f}")
        final_summary[cat] = {
            "bit_match": round(float(bm), 4),
            "ratio_med": round(float(rm), 4),
            "kurtosis": round(float(ku), 4),
        }

    # ==================================================================
    # Step 5: Rebuild model with optimized binary weights and test
    # ==================================================================
    print("\n" + "=" * 60)
    print("Step 5: Rebuild model and test generation")
    print("=" * 60, flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map=device, token=hf_token,
    )

    # Replace weights with baked binary
    replaced = 0
    for name, opt in optimized_layers.items():
        signs = opt["signs"]
        scales = opt["scales"]
        out_dim = signs.shape[0]
        in_padded = signs.shape[1] * GROUP_SIZE
        in_dim = teacher_weights[name].shape[1]

        baked = (signs * scales.unsqueeze(2)).reshape(out_dim, in_padded)[:, :in_dim]

        # Navigate to the module and replace weight
        parts = name.split(".")
        module = model
        for p in parts:
            module = getattr(module, p)
        module.weight.data.copy_(baked.to(module.weight.dtype).to(module.weight.device))
        replaced += 1

    print(f"  Replaced {replaced} layers with binary weights", flush=True)

    # Generate text
    print("\n  TEXT GENERATION (baked binary):")
    prompts = [
        "What is the capital of France?",
        "Explain gravity in one sentence.",
        "1 + 1 =",
        "The color of the sky is",
        "List three animals:",
        "A truck driver should check their mirrors because",
    ]

    model.eval()
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=60, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen, skip_special_tokens=True)
        print(f"    Q: {prompt}")
        print(f"    A: {text[:200]}", flush=True)

    # Save checkpoint
    print("\n  Saving checkpoint...", flush=True)
    torch.save({
        "optimized_layers": {k: {
            "signs": v["signs"],
            "scales": v["scales"],
        } for k, v in optimized_layers.items()},
        "config": {
            "model": model_id,
            "group_size": GROUP_SIZE,
            "n_iters": N_ITERS,
            "flip_frac": FLIP_FRAC,
            "n_calib": len(calib_texts),
        },
    }, "/data/qwen17b_iterative_gptq.pt")

    results = {
        "final_summary": final_summary,
        "config": {
            "model": model_id, "group_size": GROUP_SIZE,
            "n_iters": N_ITERS, "flip_frac": FLIP_FRAC,
            "n_calib": len(calib_texts),
        },
    }
    with open("/data/qwen17b_iterative_gptq_results.json", "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()

    print(f"\n{'=' * 60}")
    print("ITERATIVE GPTQ COMPLETE")
    print(f"{'=' * 60}", flush=True)

    return results


@app.local_entrypoint()
def main():
    print("Iterative GPTQ -- Qwen3-1.7B\n")
    result = run_iterative_gptq.remote()
    if result and "final_summary" in result:
        print("\n\nFINAL:")
        for cat, m in result["final_summary"].items():
            print(f"  {cat}: bm={m['bit_match']:.4f}, rm={m['ratio_med']:.4f}, ku={m['kurtosis']:.4f}")
