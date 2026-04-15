"""Quick test: GPTQ warm start → sequential block-wise STE distillation.

Hypothesis: GPTQ gives ~10% flips (Hessian), STE adds ~8% more (gradient).
Combined should reach ~18-20% — much closer to Bonsai's 25-30%.

Fast test: only first 4 blocks, 50 steps each. ~10-15 min on A10G.
"""
import modal
import json

app = modal.App("1bit-graduated-growth")

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
TEST_BLOCKS = 4             # only first N blocks for quick test
STEPS_PER_BLOCK = 50
BATCH_SIZE = 4
LR_WEIGHT = 1e-3
LR_SCALE = 3e-3
MAX_SEQ_LEN = 512
TEMPERATURE = 2.0
TOP_K_LOGITS = 1024
N_CALIB = 128               # fewer samples for speed


def classify_layer(name: str) -> str:
    if "embed" in name or "lm_head" in name: return "token_embd"
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
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=32768,
    volumes={"/data": vol},
)
def run_combined_test():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import os, gc, time, random
    import numpy as np
    from scipy import stats as scipy_stats
    from collections import defaultdict

    hf_token = os.environ.get("HF_TOKEN")
    device = "cuda"
    model_id = "Qwen/Qwen3-1.7B"

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset

    # ==================================================================
    # BitLinear with identity STE
    # ==================================================================
    class BitLinear(nn.Module):
        def __init__(self, weight_data, group_size=GROUP_SIZE):
            super().__init__()
            w = weight_data.float()
            out_dim, in_dim = w.shape
            self.in_features = in_dim
            self.out_features = out_dim
            self.group_size = group_size
            self.pad = (group_size - in_dim % group_size) % group_size
            if self.pad > 0:
                w = F.pad(w, (0, self.pad))
            w_g = w.reshape(out_dim, -1, group_size)
            self.latent_weight = nn.Parameter(w_g.clone())
            init_scales = w_g.abs().mean(dim=2)
            self.scale = nn.Parameter(init_scales)
            teacher_signs = w_g.sign()
            teacher_signs[teacher_signs == 0] = 1.0
            self.register_buffer("teacher_signs", teacher_signs)
            self.register_buffer("teacher_scale", init_scales.clone())
            self.bias = None

        def forward(self, x):
            signs = self.latent_weight.sign()
            signs = signs + (signs == 0).float()
            signs_ste = signs.detach() + self.latent_weight - self.latent_weight.detach()
            scale_abs = self.scale.abs().unsqueeze(2)
            w_q = (signs_ste * scale_abs).reshape(self.out_features, -1)
            if self.pad > 0:
                w_q = w_q[:, :self.in_features]
            return F.linear(x.float(), w_q).to(x.dtype)

        def bake(self):
            with torch.no_grad():
                signs = self.latent_weight.sign()
                signs[signs == 0] = 1.0
                w_q = (signs * self.scale.abs().unsqueeze(2)).reshape(self.out_features, -1)
                if self.pad > 0:
                    w_q = w_q[:, :self.in_features]
                return w_q

        def forensics(self):
            with torch.no_grad():
                signs = self.latent_weight.sign()
                signs[signs == 0] = 1.0
                bm = (signs == self.teacher_signs).float().mean().item()
                rm = (self.scale.abs() / self.teacher_scale.clamp(min=1e-8)).flatten().median().item()
                baked = self.bake()
                kurt = float(scipy_stats.kurtosis(baked.cpu().reshape(-1).numpy(), fisher=True))
                return bm, rm, kurt

    # ==================================================================
    # Step 1: Load calibration data (small set for speed)
    # ==================================================================
    print("=" * 60)
    print("Step 1: Loading calibration data...")
    print("=" * 60, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    calib_texts = []
    orca = load_dataset("Open-Orca/SlimOrca", split="train", streaming=True)
    count = 0
    for row in orca:
        convs = row.get("conversations", [])
        text = " ".join(c.get("value", "") for c in convs if c.get("from") == "gpt")
        if len(text) > 100:
            calib_texts.append(text[:2000])
            count += 1
            if count >= N_CALIB:
                break
    print(f"  Got {count} calibration samples", flush=True)

    calib_ids = []
    for text in calib_texts:
        toks = tokenizer(text, return_tensors="pt", truncation=True,
                         max_length=MAX_SEQ_LEN, padding=False)
        calib_ids.append(toks["input_ids"].squeeze(0))

    # ==================================================================
    # Step 2: Load model, run GPTQ per-layer on test blocks
    # ==================================================================
    print("\n" + "=" * 60)
    print(f"Step 2: GPTQ initialization (first {TEST_BLOCKS} blocks)...")
    print("=" * 60, flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device, token=hf_token,
    )
    model.eval()

    n_blocks = len(model.model.layers)
    print(f"  Model has {n_blocks} blocks, testing first {TEST_BLOCKS}", flush=True)

    # Capture activations for target layers in test blocks
    layer_inputs = {}
    hooks = []
    target_layers = {}

    for block_idx in range(TEST_BLOCKS):
        block = model.model.layers[block_idx]
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear):
                cat = classify_layer(name)
                if cat not in ("other", "token_embd"):
                    full_name = f"model.layers.{block_idx}.{name}"
                    target_layers[full_name] = module

    print(f"  Target layers: {len(target_layers)}", flush=True)

    def make_hook(layer_name):
        def hook_fn(module, input, output):
            x = input[0].detach().float().reshape(-1, input[0].shape[-1])
            if layer_name not in layer_inputs:
                layer_inputs[layer_name] = []
            layer_inputs[layer_name].append(x.cpu())
        return hook_fn

    for name, module in target_layers.items():
        hooks.append(module.register_forward_hook(make_hook(name)))

    print("  Running calibration forward passes...", flush=True)
    with torch.no_grad():
        for i, tok in enumerate(calib_ids):
            model(tok.unsqueeze(0).to(device))
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(calib_ids)}", flush=True)

    for h in hooks:
        h.remove()

    MAX_TOKENS = 4096
    for name in layer_inputs:
        all_x = torch.cat(layer_inputs[name], dim=0)
        if all_x.shape[0] > MAX_TOKENS:
            all_x = all_x[torch.randperm(all_x.shape[0])[:MAX_TOKENS]]
        layer_inputs[name] = all_x

    # Run GPTQ per layer: column-by-column with Hessian error compensation
    print("\n  Running GPTQ column-by-column quantization...", flush=True)

    gptq_weights = {}  # full_name -> (signs_g, scales_g, teacher_signs, teacher_scale)

    for layer_name, module in target_layers.items():
        W = module.weight.data.float().cpu()
        X = layer_inputs[layer_name].to(device)
        out_dim, in_dim = W.shape

        # Pad
        pad = (GROUP_SIZE - in_dim % GROUP_SIZE) % GROUP_SIZE
        if pad > 0:
            W_padded = F.pad(W, (0, pad))
            X_padded = F.pad(X, (0, pad))
        else:
            W_padded = W
            X_padded = X

        in_padded = W_padded.shape[1]
        n_groups = in_padded // GROUP_SIZE

        W_dev = W_padded.to(device)

        # Teacher signs and scales (for forensic comparison)
        W_g = W_padded.reshape(out_dim, n_groups, GROUP_SIZE)
        teacher_signs = W_g.sign()
        teacher_signs[teacher_signs == 0] = 1.0
        teacher_scale = W_g.abs().mean(dim=2)

        # GPTQ with proper Hessian damping
        # Compute full Hessian: H = X^T @ X / n_tokens
        n_tok = X_padded.shape[0]
        H = (X_padded.T @ X_padded) / n_tok  # [in_padded, in_padded]

        # Damping for numerical stability
        damp = 0.01 * torch.diag(H).mean()
        H.diagonal().add_(damp)

        # Cholesky for stable column-by-column processing
        # We only need the diagonal of H_inv and the off-diagonal for compensation
        # Use simplified: process group-by-group instead of column-by-column
        W_work = W_padded.to(device).clone()

        # Process group by group (more stable than column-by-column)
        for g in range(n_groups):
            g_start = g * GROUP_SIZE
            g_end = g_start + GROUP_SIZE

            # Group Hessian block
            H_g = H[g_start:g_end, g_start:g_end]

            # Group weights
            W_g_block = W_work[:, g_start:g_end]  # [out_dim, GROUP_SIZE]

            # Per-group scale = mean-abs
            group_scale = W_g_block.abs().mean(dim=1).clamp(min=1e-8)  # [out_dim]

            # Quantize: sign * scale
            Q_g = W_g_block.sign() * group_scale.unsqueeze(1)
            Q_g[W_g_block == 0] = 0  # handle exact zeros

            # Error
            err_g = W_g_block - Q_g  # [out_dim, GROUP_SIZE]

            # Distribute error to NEXT group only (limited lookahead)
            if g < n_groups - 1:
                next_start = g_end
                next_end = next_start + GROUP_SIZE
                # Cross-group Hessian: H[current_group, next_group]
                H_cross = H[g_start:g_end, next_start:next_end]
                H_next_diag = H[next_start:next_end, next_start:next_end].diagonal().clamp(min=1e-10)
                # Compensation: err @ H_cross / diag(H_next)
                comp = err_g @ H_cross / H_next_diag.unsqueeze(0)
                # Conservative clamp
                max_comp = W_work[:, next_start:next_end].abs().mean() * 0.1
                comp = comp.clamp(-max_comp, max_comp)
                W_work[:, next_start:next_end] += comp

            W_work[:, g_start:g_end] = Q_g

        # Extract signs and scales from quantized weights
        W_q_g = W_work.reshape(out_dim, n_groups, GROUP_SIZE)
        signs_g = W_q_g.sign()
        signs_g[signs_g == 0] = 1.0
        scales_g = torch.zeros(out_dim, n_groups, device=device)
        for g in range(n_groups):
            g_start = g * GROUP_SIZE
            g_end = g_start + GROUP_SIZE
            scales_g[:, g] = W_work[:, g_start:g_end].abs().mean(dim=1).clamp(min=1e-8)

        # Compute bit_match vs teacher
        bm = (signs_g == teacher_signs.to(device)).float().mean().item()
        rm = (scales_g / teacher_scale.to(device).clamp(min=1e-8)).flatten().median().item()

        gptq_weights[layer_name] = {
            "signs_g": signs_g.cpu(),
            "scales_g": scales_g.cpu(),
            "teacher_signs": teacher_signs.cpu(),
            "teacher_scale": teacher_scale.cpu(),
            "in_features": in_dim,
            "out_features": out_dim,
            "pad": pad,
        }

        cat = classify_layer(layer_name)
        if layer_name == list(target_layers.keys())[0] or "up_proj" in layer_name:
            print(f"    {layer_name}: bm={bm:.4f}({(1-bm)*100:.1f}%flip) rm={rm:.4f}", flush=True)

    # Summarize GPTQ results
    gptq_flips = []
    for name, data in gptq_weights.items():
        bm = (data["signs_g"] == data["teacher_signs"]).float().mean().item()
        gptq_flips.append(1 - bm)
    avg_gptq_flip = np.mean(gptq_flips) * 100
    print(f"\n  GPTQ average: {avg_gptq_flip:.1f}% sign flips", flush=True)

    # ==================================================================
    # Step 3: Cache teacher logits
    # ==================================================================
    print("\n" + "=" * 60)
    print("Step 3: Caching teacher logits...")
    print("=" * 60, flush=True)

    teacher_cache = []
    t0 = time.time()
    with torch.no_grad():
        for i, ids in enumerate(calib_ids):
            logits = model(ids.unsqueeze(0).to(device)).logits[0]
            topk_v, topk_i = logits.float().topk(TOP_K_LOGITS, dim=-1)
            teacher_cache.append((ids, topk_v.cpu(), topk_i.cpu()))
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(calib_ids)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  Done in {time.time()-t0:.0f}s", flush=True)

    # ==================================================================
    # Step 4: Initialize student from GPTQ weights, run STE distillation
    # ==================================================================
    print("\n" + "=" * 60)
    print(f"Step 4: STE distillation on GPTQ-initialized blocks ({STEPS_PER_BLOCK} steps)")
    print("=" * 60, flush=True)

    # First, inject GPTQ weights into model as the starting point
    # We'll convert GPTQ signs+scales into continuous weights for BitLinear init
    for layer_name, data in gptq_weights.items():
        signs_g = data["signs_g"]
        scales_g = data["scales_g"]
        out_dim = data["out_features"]
        in_dim = data["in_features"]

        # Bake GPTQ binary weights
        baked = (signs_g * scales_g.unsqueeze(2)).reshape(out_dim, -1)[:, :in_dim]

        # Navigate to module and replace weight
        parts = layer_name.split(".")
        module = model
        for p in parts:
            module = getattr(module, p)
        module.weight.data.copy_(baked.to(module.weight.dtype).to(module.weight.device))

    print("  Injected GPTQ weights into model", flush=True)

    # Now run sequential STE distillation on test blocks
    cat_metrics_gptq_only = defaultdict(list)  # before STE
    cat_metrics_combined = defaultdict(list)    # after STE

    # Record GPTQ-only metrics
    for layer_name, data in gptq_weights.items():
        cat = classify_layer(layer_name)
        bm = (data["signs_g"] == data["teacher_signs"]).float().mean().item()
        rm = (data["scales_g"] / data["teacher_scale"].clamp(min=1e-8)).flatten().median().item()
        cat_metrics_gptq_only[cat].append(bm)

    total_t0 = time.time()

    for block_idx in range(TEST_BLOCKS):
        block = model.model.layers[block_idx]
        block_t0 = time.time()

        # Find linear modules in this block
        bit_modules = {}
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear):
                cat = classify_layer(name)
                if cat in ("other", "token_embd"):
                    continue
                full_name = f"model.layers.{block_idx}.{name}"

                # Create BitLinear from GPTQ-initialized weights
                parts = name.split(".")
                parent = block
                for p in parts[:-1]:
                    parent = getattr(parent, p)

                bit = BitLinear(module.weight.data).to(device)
                # Override teacher references with ORIGINAL teacher values
                data = gptq_weights[full_name]
                bit.teacher_signs = data["teacher_signs"].to(device)
                bit.teacher_scale = data["teacher_scale"].to(device)
                setattr(parent, parts[-1], bit)
                bit_modules[name] = (bit, cat)

        # Freeze all, unfreeze this block's BitLinear
        for param in model.parameters():
            param.requires_grad = False

        weight_params, scale_params = [], []
        for name, (bit, cat) in bit_modules.items():
            bit.latent_weight.requires_grad = True
            bit.scale.requires_grad = True
            weight_params.append(bit.latent_weight)
            scale_params.append(bit.scale)

        optimizer = torch.optim.AdamW([
            {"params": weight_params, "lr": LR_WEIGHT},
            {"params": scale_params, "lr": LR_SCALE},
        ], weight_decay=0.0)

        model.train()
        model.gradient_checkpointing_enable()

        for step in range(STEPS_PER_BLOCK):
            batch_indices = random.sample(range(len(teacher_cache)),
                                          min(BATCH_SIZE, len(teacher_cache)))
            total_loss = 0.0
            optimizer.zero_grad()

            for idx in batch_indices:
                input_ids, topk_v, topk_i = teacher_cache[idx]
                input_ids = input_ids.unsqueeze(0).to(device)
                student_logits = model(input_ids).logits[0]
                seq_len = student_logits.shape[0]
                topk_v_dev = topk_v[:seq_len].to(device)
                topk_i_dev = topk_i[:seq_len].to(device)
                teacher_probs = F.softmax(topk_v_dev / TEMPERATURE, dim=-1)
                student_lp = F.log_softmax(student_logits.float() / TEMPERATURE, dim=-1)
                student_lp_topk = student_lp.gather(1, topk_i_dev)
                kl = F.kl_div(student_lp_topk, teacher_probs, reduction="sum", log_target=False)
                loss = kl * (TEMPERATURE ** 2) / seq_len
                loss.backward()
                total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(weight_params + scale_params, max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                for name, (bit, cat) in bit_modules.items():
                    bit.latent_weight.data.clamp_(-1.0, 1.0)

            avg_loss = total_loss / len(batch_indices)

            if step == 0 or (step + 1) % 25 == 0:
                bm_list, rm_list = [], []
                for name, (bit, cat) in bit_modules.items():
                    bm, rm, ku = bit.forensics()
                    bm_list.append(bm)
                    rm_list.append(rm)
                print(f"  B{block_idx} s{step+1:3d}/{STEPS_PER_BLOCK}: "
                      f"loss={avg_loss:.4f} bm={np.mean(bm_list):.4f}"
                      f"({(1-np.mean(bm_list))*100:.1f}%flip) "
                      f"rm={np.mean(rm_list):.4f}", flush=True)

        model.gradient_checkpointing_disable()

        # Record combined metrics and bake
        for name, (bit, cat) in bit_modules.items():
            bm, rm, ku = bit.forensics()
            cat_metrics_combined[cat].append(bm)

            baked_w = bit.bake().to(torch.float16)
            parts = name.split(".")
            parent = block
            for p in parts[:-1]:
                parent = getattr(parent, p)
            new_linear = nn.Linear(bit.in_features, bit.out_features,
                                   bias=False, dtype=torch.float16, device=device)
            new_linear.weight.data.copy_(baked_w)
            new_linear.weight.requires_grad = False
            setattr(parent, parts[-1], new_linear)

        del optimizer, bit_modules, weight_params, scale_params
        gc.collect()
        torch.cuda.empty_cache()

        elapsed = time.time() - total_t0
        print(f"  B{block_idx} DONE ({time.time()-block_t0:.0f}s, "
              f"{elapsed:.0f}s total)", flush=True)

    # ==================================================================
    # Step 5: Compare GPTQ-only vs Combined
    # ==================================================================
    print("\n" + "=" * 60)
    print("Step 5: GPTQ-only vs GPTQ+STE comparison")
    print("=" * 60, flush=True)

    bonsai_tgt = {
        "ffn_up": 0.70, "ffn_gate": 0.70, "attn_v": 0.77,
        "ffn_down": 0.74, "attn_o": 0.74, "attn_k": 0.71, "attn_q": 0.71,
    }

    print(f"  {'Cat':<10} {'GPTQ_bm':>8} {'flip%':>6} {'Comb_bm':>8} {'flip%':>6} {'target':>7}")
    print("  " + "-" * 50)

    results = {}
    for cat in ["ffn_up", "ffn_gate", "attn_v", "ffn_down", "attn_o", "attn_k", "attn_q"]:
        gptq_bm = np.median(cat_metrics_gptq_only.get(cat, [1.0]))
        comb_bm = np.median(cat_metrics_combined.get(cat, [1.0]))
        tgt = bonsai_tgt.get(cat, 0.70)
        gptq_flip = (1 - gptq_bm) * 100
        comb_flip = (1 - comb_bm) * 100
        print(f"  {cat:<10} {gptq_bm:>8.4f} {gptq_flip:>5.1f}% {comb_bm:>8.4f} {comb_flip:>5.1f}% {tgt:>7.2f}")
        results[cat] = {
            "gptq_flip_pct": round(gptq_flip, 2),
            "combined_flip_pct": round(comb_flip, 2),
            "target_bm": tgt,
        }

    # Save results
    with open("/data/combined_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()

    print(f"\n{'=' * 60}")
    print("COMBINED TEST COMPLETE")
    print(f"{'=' * 60}", flush=True)

    return results


@app.local_entrypoint()
def main():
    print("Combined GPTQ+STE Test -- Qwen3-1.7B (first 4 blocks)\n")
    result = run_combined_test.remote()
    if result:
        print("\n\nRESULTS:")
        for cat, m in result.items():
            print(f"  {cat}: GPTQ {m['gptq_flip_pct']:.1f}% → Combined {m['combined_flip_pct']:.1f}% (target: {(1-m['target_bm'])*100:.0f}%)")
