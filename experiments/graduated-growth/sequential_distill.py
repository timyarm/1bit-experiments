"""Sequential block-wise 1-bit distillation for Qwen3-1.7B.

Key insight: Bonsai quantized Qwen to 1-bit by distilling one block at a time,
not all at once. When only ONE block has STE, the gradient signal is strong
(flows through 23 fp16 blocks). Our all-at-once STE failed because gradient
was diluted across 24 STE bottlenecks.

Algorithm:
  For each transformer block (0..23):
    1. Replace its 7 linear layers with BitLinear (sign*scale via STE)
    2. Keep all other blocks at fp16 (or frozen 1-bit if already processed)
    3. Run KL distillation vs fp16 teacher for N steps
    4. Bake binary weights, freeze, move to next block
"""
import modal
import json

app = modal.App("1bit-sequential-distill")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.51.0",
        "accelerate>=1.5.0",
        "huggingface_hub>=0.28.0",
        "datasets>=3.0.0",
        "scipy>=1.12.0",
        "bitsandbytes>=0.45.0",
    )
)

vol = modal.Volume.from_name("1bit-checkpoints", create_if_missing=True)

GROUP_SIZE = 128
STEPS_PER_BLOCK = 80        # more steps at lower LR
BATCH_SIZE = 4              # samples per step
LR_WEIGHT = 3e-3            # balanced: ~15% flips, stable through all blocks
LR_SCALE = 6e-3             # scale LR
N_PASSES = 1                # single pass
MAX_SEQ_LEN = 512
N_CALIB = 256
TEMPERATURE = 2.0           # KL distillation temperature
TOP_K_LOGITS = 1024         # cache top-K teacher logits per position


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


class BitLinear(object):
    """Wraps a nn.Linear with 1-bit STE.

    We store latent_weight (continuous) and scale (per-group).
    Forward: output = (sign(latent) * scale) @ input
    Backward: identity STE (gradient flows through sign as if it were identity)
    """
    pass  # Defined inside the Modal function to avoid import issues


@app.function(
    image=image,
    gpu="A10G:1",
    timeout=14400,  # 4 hours
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=49152,
    volumes={"/data": vol},
)
def run_sequential_distill():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import os
    import gc
    import numpy as np
    from scipy import stats as scipy_stats
    from collections import defaultdict
    import time
    import random

    hf_token = os.environ.get("HF_TOKEN")
    device = "cuda"
    model_id = "Qwen/Qwen3-1.7B"

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset

    # ==================================================================
    # BitLinear module with identity STE
    # ==================================================================
    class BitLinear(nn.Module):
        """1-bit linear with identity STE and per-group scales."""

        def __init__(self, original: nn.Linear, group_size: int = GROUP_SIZE):
            super().__init__()
            w = original.weight.data.float()
            out_dim, in_dim = w.shape
            self.in_features = in_dim
            self.out_features = out_dim
            self.group_size = group_size

            # Pad to multiple of group_size
            self.pad = (group_size - in_dim % group_size) % group_size
            if self.pad > 0:
                w = F.pad(w, (0, self.pad))
            in_padded = w.shape[1]
            n_groups = in_padded // group_size

            # Reshape to groups
            w_g = w.reshape(out_dim, n_groups, group_size)

            # Initialize latent weights = teacher weights (continuous)
            # These will be quantized via sign() in forward
            self.latent_weight = nn.Parameter(w_g.clone())

            # Initialize scales = mean-abs per group
            init_scales = w_g.abs().mean(dim=2)  # [out_dim, n_groups]
            self.scale = nn.Parameter(init_scales)

            # Store teacher signs for forensic comparison
            teacher_signs = w_g.sign()
            teacher_signs[teacher_signs == 0] = 1.0
            self.register_buffer("teacher_signs", teacher_signs)
            self.register_buffer("teacher_scale", init_scales.clone())

            self.has_bias = original.bias is not None
            if self.has_bias:
                self.bias = nn.Parameter(original.bias.data.clone())
            else:
                self.bias = None

        def forward(self, x):
            # Identity STE: forward uses sign(), backward passes gradient through
            signs = self.latent_weight.sign()
            signs = signs + (signs == 0).float()  # handle exact zeros

            # STE: detach sign from graph, add back latent for gradient flow
            signs_ste = signs.detach() + self.latent_weight - self.latent_weight.detach()

            # Quantized weight: sign * |scale|
            scale_abs = self.scale.abs().unsqueeze(2)  # [out, n_groups, 1]
            w_q = (signs_ste * scale_abs).reshape(self.out_features, -1)

            # Trim padding
            if self.pad > 0:
                w_q = w_q[:, :self.in_features]

            out = F.linear(x.float(), w_q, self.bias)
            return out.to(x.dtype)

        def bake(self):
            """Return baked binary weight tensor."""
            with torch.no_grad():
                signs = self.latent_weight.sign()
                signs[signs == 0] = 1.0
                scale_abs = self.scale.abs().unsqueeze(2)
                w_q = (signs * scale_abs).reshape(self.out_features, -1)
                if self.pad > 0:
                    w_q = w_q[:, :self.in_features]
                return w_q

        def forensics(self):
            """Compute bit_match, ratio_med, kurtosis vs teacher."""
            with torch.no_grad():
                signs = self.latent_weight.sign()
                signs[signs == 0] = 1.0
                bm = (signs == self.teacher_signs).float().mean().item()
                rm = (self.scale.abs() / self.teacher_scale.clamp(min=1e-8)).flatten().median().item()
                baked = self.bake()
                kurt = float(scipy_stats.kurtosis(
                    baked.cpu().reshape(-1).numpy(), fisher=True
                ))
                return bm, rm, kurt

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

    # Tokenize all calibration data
    calib_ids = []
    for text in calib_texts:
        toks = tokenizer(text, return_tensors="pt", truncation=True,
                         max_length=MAX_SEQ_LEN, padding=False)
        calib_ids.append(toks["input_ids"].squeeze(0))

    # ==================================================================
    # Step 2: Cache teacher logits (top-K per position)
    # ==================================================================
    print("\n" + "=" * 60)
    print("Step 2: Caching teacher logits (top-K)...")
    print("=" * 60, flush=True)

    teacher = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device, token=hf_token,
    )
    teacher.eval()

    # Cache teacher logits for all calibration samples
    teacher_cache = []  # list of (input_ids, topk_vals, topk_ids)
    t0 = time.time()
    with torch.no_grad():
        for i, ids in enumerate(calib_ids):
            input_ids = ids.unsqueeze(0).to(device)
            logits = teacher(input_ids).logits[0]  # [seq_len, vocab]
            topk_v, topk_i = logits.float().topk(TOP_K_LOGITS, dim=-1)
            teacher_cache.append((ids, topk_v.cpu(), topk_i.cpu()))
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(calib_ids)} cached ({time.time()-t0:.0f}s)", flush=True)

    print(f"  Teacher caching done in {time.time()-t0:.0f}s", flush=True)

    # Keep teacher on CPU for reference, free GPU
    teacher_state = {k: v.cpu() for k, v in teacher.state_dict().items()}
    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    # ==================================================================
    # Step 3: Load student model
    # ==================================================================
    print("\n" + "=" * 60)
    print("Step 3: Loading student model...")
    print("=" * 60, flush=True)

    student = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device, token=hf_token,
    )

    # Count transformer blocks
    n_blocks = len(student.model.layers)
    print(f"  Model has {n_blocks} transformer blocks", flush=True)

    # Identify all linear layers per block
    block_linears = {}
    for block_idx in range(n_blocks):
        block = student.model.layers[block_idx]
        linears = {}
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear):
                cat = classify_layer(name)
                if cat not in ("other", "token_embd"):
                    linears[name] = (module, cat)
        block_linears[block_idx] = linears
        if block_idx == 0:
            print(f"  Block 0 linears: {list(linears.keys())}", flush=True)

    # ==================================================================
    # Step 4: Sequential block-wise distillation (multi-pass)
    # ==================================================================
    print("\n" + "=" * 60)
    print(f"Step 4: Sequential distillation ({STEPS_PER_BLOCK} steps/block, {N_PASSES} passes)")
    print("=" * 60, flush=True)

    cat_metrics = defaultdict(lambda: {"bit_match": [], "ratio_med": [], "kurtosis": []})

    # Store original teacher weights for forensic comparison (signs + scales)
    # These are computed from the ORIGINAL fp16 weights, not from baked binary
    teacher_forensic = {}
    for block_idx in range(n_blocks):
        block = student.model.layers[block_idx]
        for name, (orig_module, cat) in block_linears[block_idx].items():
            w = orig_module.weight.data.float()
            out_dim, in_dim = w.shape
            pad = (GROUP_SIZE - in_dim % GROUP_SIZE) % GROUP_SIZE
            if pad > 0:
                w = F.pad(w, (0, pad))
            w_g = w.reshape(out_dim, -1, GROUP_SIZE)
            t_signs = w_g.sign()
            t_signs[t_signs == 0] = 1.0
            t_scale = w_g.abs().mean(dim=2)
            teacher_forensic[(block_idx, name)] = (t_signs.cpu(), t_scale.cpu())

    total_t0 = time.time()
    global_step = 0

    for pass_idx in range(N_PASSES):
        print(f"\n  === PASS {pass_idx + 1}/{N_PASSES} ===", flush=True)

        # Reduce LR on pass 2+ (fine-tuning)
        base_lr_weight = LR_WEIGHT * (0.3 if pass_idx > 0 else 1.0)
        base_lr_scale = LR_SCALE * (0.3 if pass_idx > 0 else 1.0)

        for block_idx in range(n_blocks):
            # Uniform LR — 3e-3 is stable enough without depth decay
            pass_lr_weight = base_lr_weight
            pass_lr_scale = base_lr_scale
            block = student.model.layers[block_idx]
            block_t0 = time.time()

            # --- Get current linear modules in this block ---
            current_linears = {}
            for name, (_, cat) in block_linears[block_idx].items():
                parts = name.split(".")
                module = block
                for p in parts:
                    module = getattr(module, p)
                current_linears[name] = (module, cat)

            # --- Replace with BitLinear ---
            bit_modules = {}
            for name, (curr_module, cat) in current_linears.items():
                parts = name.split(".")
                parent = block
                for p in parts[:-1]:
                    parent = getattr(parent, p)

                bit = BitLinear(curr_module).to(device)
                # Override teacher_signs/teacher_scale with ORIGINAL teacher values
                t_signs, t_scale = teacher_forensic[(block_idx, name)]
                bit.teacher_signs = t_signs.to(device)
                bit.teacher_scale = t_scale.to(device)
                setattr(parent, parts[-1], bit)
                bit_modules[name] = (bit, cat)

            # --- Freeze everything except this block's BitLinear ---
            for param in student.parameters():
                param.requires_grad = False

            weight_params = []
            scale_params = []
            for name, (bit, cat) in bit_modules.items():
                bit.latent_weight.requires_grad = True
                bit.scale.requires_grad = True
                weight_params.append(bit.latent_weight)
                scale_params.append(bit.scale)

            optimizer = torch.optim.AdamW([
                {"params": weight_params, "lr": pass_lr_weight},
                {"params": scale_params, "lr": pass_lr_scale},
            ], weight_decay=0.0)

            # Cosine LR schedule within block
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=STEPS_PER_BLOCK, eta_min=pass_lr_weight * 0.1
            )

            # --- Training loop ---
            student.train()
            student.gradient_checkpointing_enable()

            losses = []
            for step in range(STEPS_PER_BLOCK):
                batch_indices = random.sample(range(len(teacher_cache)),
                                              min(BATCH_SIZE, len(teacher_cache)))
                total_loss = 0.0
                optimizer.zero_grad()

                for idx in batch_indices:
                    input_ids, topk_v, topk_i = teacher_cache[idx]
                    input_ids = input_ids.unsqueeze(0).to(device)
                    student_logits = student(input_ids).logits[0]
                    seq_len = student_logits.shape[0]
                    topk_v_dev = topk_v[:seq_len].to(device)
                    topk_i_dev = topk_i[:seq_len].to(device)
                    teacher_probs_topk = F.softmax(topk_v_dev / TEMPERATURE, dim=-1)
                    student_log_probs_full = F.log_softmax(
                        student_logits.float() / TEMPERATURE, dim=-1)
                    student_log_probs_topk = student_log_probs_full.gather(1, topk_i_dev)
                    kl = F.kl_div(student_log_probs_topk, teacher_probs_topk,
                                  reduction="sum", log_target=False)
                    loss = kl * (TEMPERATURE ** 2) / seq_len
                    # Clamp loss to prevent NaN cascade from accumulated error
                    loss = loss.clamp(max=50.0)
                    loss.backward()
                    total_loss += loss.item()

                # Skip step if NaN detected
                has_nan = any(
                    p.grad is not None and torch.isnan(p.grad).any()
                    for p in weight_params + scale_params
                )
                if has_nan or total_loss != total_loss:  # NaN check
                    optimizer.zero_grad()
                    losses.append(float('nan'))
                    continue

                # Gradient clipping to prevent explosions
                torch.nn.utils.clip_grad_norm_(
                    weight_params + scale_params, max_norm=0.5)

                optimizer.step()
                scheduler.step()

                # Clamp latent weights
                with torch.no_grad():
                    for name, (bit, cat) in bit_modules.items():
                        bit.latent_weight.data.clamp_(-1.0, 1.0)

                avg_loss = total_loss / len(batch_indices)
                losses.append(avg_loss)
                global_step += 1

                if (step + 1) % 50 == 0 or step == 0:
                    bm_list, rm_list = [], []
                    for name, (bit, cat) in bit_modules.items():
                        bm, rm, ku = bit.forensics()
                        bm_list.append(bm)
                        rm_list.append(rm)
                    avg_bm = np.mean(bm_list)
                    avg_rm = np.mean(rm_list)
                    print(f"  P{pass_idx+1} B{block_idx:2d} s{step+1:3d}/{STEPS_PER_BLOCK}: "
                          f"loss={avg_loss:.4f} bm={avg_bm:.4f}({(1-avg_bm)*100:.1f}%flip) "
                          f"rm={avg_rm:.4f}", flush=True)

            student.gradient_checkpointing_disable()

            # --- Bake and freeze ---
            # Clear metrics for this pass (only keep latest pass)
            if pass_idx == N_PASSES - 1:
                for name, (bit, cat) in bit_modules.items():
                    bm, rm, ku = bit.forensics()
                    cat_metrics[cat]["bit_match"].append(bm)
                    cat_metrics[cat]["ratio_med"].append(rm)
                    cat_metrics[cat]["kurtosis"].append(ku)

            for name, (bit, cat) in bit_modules.items():
                baked_w = bit.bake().to(torch.float16)
                parts = name.split(".")
                parent = block
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                new_linear = nn.Linear(bit.in_features, bit.out_features,
                                       bias=bit.has_bias, dtype=torch.float16,
                                       device=device)
                new_linear.weight.data.copy_(baked_w)
                if bit.has_bias:
                    new_linear.bias.data.copy_(bit.bias.data.to(torch.float16))
                new_linear.weight.requires_grad = False
                if new_linear.bias is not None:
                    new_linear.bias.requires_grad = False
                setattr(parent, parts[-1], new_linear)

            del optimizer, scheduler, bit_modules, weight_params, scale_params
            gc.collect()
            torch.cuda.empty_cache()

            elapsed = time.time() - total_t0
            total_blocks_done = pass_idx * n_blocks + block_idx + 1
            total_blocks = N_PASSES * n_blocks
            eta = elapsed / total_blocks_done * (total_blocks - total_blocks_done)
            block_time = time.time() - block_t0

            print(f"  P{pass_idx+1} B{block_idx:2d} DONE: {block_time:.0f}s, "
                  f"loss {losses[0]:.3f}->{losses[-1]:.3f}, "
                  f"{elapsed:.0f}s elapsed, ~{eta:.0f}s ETA", flush=True)

    total_time = time.time() - total_t0
    print(f"\n  Total distillation time: {total_time:.0f}s ({global_step} steps)", flush=True)

    # ==================================================================
    # Step 5: Forensic summary
    # ==================================================================
    print("\n" + "=" * 60)
    print("Step 5: Forensic summary")
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
    # Step 6: Test generation
    # ==================================================================
    print("\n" + "=" * 60)
    print("Step 6: Test generation")
    print("=" * 60, flush=True)

    prompts = [
        "What is the capital of France?",
        "Explain gravity in one sentence.",
        "1 + 1 =",
        "The color of the sky is",
        "List three animals:",
        "A truck driver should check their mirrors because",
    ]

    student.eval()
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = student.generate(
                **inputs, max_new_tokens=60, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen, skip_special_tokens=True)
        print(f"    Q: {prompt}")
        print(f"    A: {text[:200]}", flush=True)

    # ==================================================================
    # Save results
    # ==================================================================
    print("\n  Saving results...", flush=True)

    results = {
        "final_summary": final_summary,
        "config": {
            "model": model_id,
            "group_size": GROUP_SIZE,
            "steps_per_block": STEPS_PER_BLOCK,
            "lr_weight": LR_WEIGHT,
            "lr_scale": LR_SCALE,
            "temperature": TEMPERATURE,
            "n_calib": len(calib_texts),
            "n_blocks": n_blocks,
            "total_time_s": total_time,
        },
    }
    with open("/data/qwen17b_sequential_distill_results.json", "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()

    print(f"\n{'=' * 60}")
    print("SEQUENTIAL DISTILLATION COMPLETE")
    print(f"{'=' * 60}", flush=True)

    return results


@app.local_entrypoint()
def main():
    print("Sequential Block-wise 1-bit Distillation -- Qwen3-1.7B\n")
    result = run_sequential_distill.remote()
    if result and "final_summary" in result:
        print("\n\nFINAL:")
        for cat, m in result["final_summary"].items():
            print(f"  {cat}: bm={m['bit_match']:.4f}, rm={m['ratio_med']:.4f}, ku={m['kurtosis']:.4f}")
