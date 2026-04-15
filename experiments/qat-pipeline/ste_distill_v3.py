"""GPTQ-initialized STE distillation for 1-bit Qwen3-1.7B (v3).

Key fixes from v2: latent weights initialized near sign boundary (+-INIT_MAG)
instead of at teacher magnitude (~0.02). Makes sign flipping 20x easier.
Higher weight LR (3e-3) and tighter clamping (1.1x scale).
"""
import modal
import json

app = modal.App("1bit-ste-distill-v3")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.51.0",
        "accelerate>=1.5.0",
        "huggingface_hub>=0.28.0",
        "datasets>=3.0.0",
        "bitsandbytes>=0.45.0",
        "scipy>=1.12.0",
    )
)

vol = modal.Volume.from_name("1bit-checkpoints", create_if_missing=True)

# Config
GROUP_SIZE = 128
MAX_SEQ_LEN = 512
TOP_K_LOGITS = 1024
SIGN_EPOCHS = 15
SCALE_EPOCHS = 5
WEIGHT_LR = 5e-4        # Conservative — flip ~20-30% signs over 15 epochs
SCALE_LR = 3e-4
GRAD_CLIP = 1.0
CLAMP_RATIO = 1.3
MAX_SCALE_RATIO = 4.0
INIT_MAG = 0.01          # Near sign boundary but not too close


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
    memory=32768,
    volumes={"/data": vol},
)
def run_ste_distillation():
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

    # ==================================================================
    # BitLinear module with Identity STE
    # ==================================================================
    class BitLinear(nn.Module):
        def __init__(self, in_features, out_features, bias=False, group_size=GROUP_SIZE):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.group_size = group_size
            self.weight = nn.Parameter(torch.zeros(out_features, in_features))
            pad_in = (group_size - in_features % group_size) % group_size
            n_groups = (in_features + pad_in) // group_size
            self.scale = nn.Parameter(torch.ones(out_features, n_groups))
            self.pad_in = pad_in
            self._baked = False
            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.bias = None

        def _quantize_ste(self, w):
            gs = self.group_size
            out_dim, in_dim = w.shape
            if self.pad_in > 0:
                w_padded = F.pad(w, (0, self.pad_in))
            else:
                w_padded = w
            in_padded = w_padded.shape[1]
            w_grouped = w_padded.reshape(out_dim, -1, gs)
            signs = w_grouped.sign()
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            binary_w = signs.detach() + w_grouped - w_grouped.detach()
            scale = self.scale.abs().clamp(min=1e-6).unsqueeze(2)
            quantized = binary_w * scale
            return quantized.reshape(out_dim, in_padded)[:, :in_dim]

        def forward(self, x):
            if self._baked:
                return F.linear(x.to(self.weight.dtype), self.weight, self.bias).to(x.dtype)
            w_q = self._quantize_ste(self.weight)
            return F.linear(x.float(), w_q, self.bias).to(x.dtype)

        def bake(self):
            with torch.no_grad():
                w = self.weight
                gs = self.group_size
                out_dim, in_dim = w.shape
                if self.pad_in > 0:
                    w_padded = F.pad(w, (0, self.pad_in))
                else:
                    w_padded = w
                in_padded = w_padded.shape[1]
                w_grouped = w_padded.reshape(out_dim, -1, gs)
                signs = w_grouped.sign()
                signs = torch.where(signs == 0, torch.ones_like(signs), signs)
                scale = self.scale.abs().clamp(min=1e-6).unsqueeze(2)
                baked = (signs * scale).reshape(out_dim, in_padded)[:, :in_dim]
                self.weight.data.copy_(baked)
            self._baked = True

        def clamp_weights(self):
            with torch.no_grad():
                gs = self.group_size
                out_dim, in_dim = self.weight.shape
                if self.pad_in > 0:
                    w_padded = F.pad(self.weight, (0, self.pad_in))
                else:
                    w_padded = self.weight.data.clone()
                in_padded = w_padded.shape[1]
                w_grouped = w_padded.reshape(out_dim, -1, gs)
                s = self.scale.abs().clamp(min=1e-6).unsqueeze(2) * CLAMP_RATIO
                w_grouped.clamp_(-s, s)
                self.weight.data.copy_(w_grouped.reshape(out_dim, in_padded)[:, :in_dim])

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
            if count >= 50:
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
            if count >= 14:
                break
    print(f"    Got {count} Alpaca", flush=True)
    print(f"  Total: {len(calib_texts)} samples", flush=True)

    calib_tokens = []
    for text in calib_texts:
        toks = tokenizer(text, return_tensors="pt", truncation=True,
                         max_length=MAX_SEQ_LEN, padding=False)
        calib_tokens.append(toks["input_ids"].squeeze(0))

    # ==================================================================
    # Step 2: Cache teacher logits (top-K)
    # ==================================================================
    print("\n" + "=" * 60)
    print("Step 2: Caching teacher logits...")
    print("=" * 60, flush=True)

    teacher = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map=device, token=hf_token,
    )
    teacher.eval()

    cached_logits = []
    with torch.no_grad():
        for i, tok in enumerate(calib_tokens):
            input_ids = tok.unsqueeze(0).to(device)
            outputs = teacher(input_ids)
            logits = outputs.logits[0].float()
            topk_vals, topk_ids = logits.topk(TOP_K_LOGITS, dim=-1)
            cached_logits.append({
                "input_ids": tok,
                "topk_vals": topk_vals.cpu().half(),
                "topk_ids": topk_ids.cpu(),
            })
            if (i + 1) % 50 == 0:
                print(f"    Cached {i+1}/{len(calib_tokens)}...", flush=True)

    print(f"  Cached {len(cached_logits)} samples (top-{TOP_K_LOGITS})", flush=True)

    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    # ==================================================================
    # Step 3: Build student with BitLinear layers
    # ==================================================================
    print("\n" + "=" * 60)
    print("Step 3: Building student model...")
    print("=" * 60, flush=True)

    student = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cpu", token=hf_token,
    )

    gptq_path = "/data/qwen17b_gptq1bit.pt"
    has_gptq = os.path.exists(gptq_path)
    if has_gptq:
        print(f"  Loading GPTQ checkpoint: {gptq_path}", flush=True)
        gptq_ckpt = torch.load(gptq_path, map_location="cpu", weights_only=True)
        gptq_layers = gptq_ckpt.get("optimized_layers", {})
        print(f"  GPTQ has {len(gptq_layers)} layers", flush=True)
    else:
        gptq_layers = {}

    replaced = 0
    bit_modules = {}
    teacher_signs_cache = {}  # Save teacher signs BEFORE replacing
    teacher_scales_cache = {}  # Save teacher group mean-abs BEFORE replacing

    for name, module in list(student.named_modules()):
        if not isinstance(module, nn.Linear) or module.weight.numel() < GROUP_SIZE:
            continue
        cat = classify_layer(name)
        if cat in ("other", "token_embd"):
            continue

        # Cache teacher signs and scales before replacement
        tw = module.weight.data.float()
        pad = (GROUP_SIZE - tw.shape[1] % GROUP_SIZE) % GROUP_SIZE
        tw_p = F.pad(tw, (0, pad)) if pad > 0 else tw
        tw_g = tw_p.reshape(tw.shape[0], -1, GROUP_SIZE)
        t_signs = tw_g.sign()
        t_signs[t_signs == 0] = 1.0
        teacher_signs_cache[name] = t_signs
        teacher_scales_cache[name] = tw_g.abs().mean(dim=2)

        bit = BitLinear(
            module.in_features, module.out_features,
            bias=module.bias is not None, group_size=GROUP_SIZE,
        )
        # Initialize latent weights near sign boundary (not at teacher magnitude)
        teacher_signs = tw.sign()
        teacher_signs[teacher_signs == 0] = 1.0
        bit.weight.data.copy_(teacher_signs * INIT_MAG)

        if name in gptq_layers:
            gptq_scales = gptq_layers[name]["scales"]
            if gptq_scales.shape == bit.scale.shape:
                bit.scale.data.copy_(gptq_scales.float())
            else:
                w = module.weight.data.float()
                pad = (GROUP_SIZE - w.shape[1] % GROUP_SIZE) % GROUP_SIZE
                w_p = F.pad(w, (0, pad)) if pad > 0 else w
                bit.scale.data.copy_(w_p.reshape(w.shape[0], -1, GROUP_SIZE).abs().mean(dim=2))
        else:
            w = module.weight.data.float()
            pad = (GROUP_SIZE - w.shape[1] % GROUP_SIZE) % GROUP_SIZE
            w_p = F.pad(w, (0, pad)) if pad > 0 else w
            bit.scale.data.copy_(w_p.reshape(w.shape[0], -1, GROUP_SIZE).abs().mean(dim=2))

        parts = name.split(".")
        parent = student
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], bit)
        bit_modules[name] = bit
        replaced += 1

    print(f"  Replaced {replaced} layers with BitLinear", flush=True)

    student.to(device)
    student.gradient_checkpointing_enable()

    mem = torch.cuda.memory_allocated() / 1e9
    print(f"  GPU memory: {mem:.1f}GB allocated", flush=True)

    # Use cached teacher signs/scales for forensic comparison
    init_signs = {k: v.cpu() for k, v in teacher_signs_cache.items()}
    init_scales = {k: v.cpu() for k, v in teacher_scales_cache.items()}
    del teacher_signs_cache, teacher_scales_cache

    init_scale_params = {}
    with torch.no_grad():
        for bname, bit in bit_modules.items():
            init_scale_params[bname] = bit.scale.data.abs().clamp(min=1e-6).cpu().clone()

    # ==================================================================
    # Step 4: Two-phase STE distillation
    # ==================================================================
    total_epochs = SIGN_EPOCHS + SCALE_EPOCHS
    print("\n" + "=" * 60)
    print(f"Step 4: Two-phase STE distillation ({total_epochs} epochs)")
    print(f"  Phase 1: {SIGN_EPOCHS} epochs, signs only (scales frozen)")
    print(f"  Phase 2: {SCALE_EPOCHS} epochs, signs + scales (clamped)")
    print("=" * 60, flush=True)

    # Freeze non-BitLinear params
    student.train()
    for name, param in student.named_parameters():
        is_bit = any(name.startswith(bn) for bn in bit_modules.keys())
        if not is_bit:
            param.requires_grad = False

    weight_params = [mod.weight for mod in student.modules() if isinstance(mod, BitLinear)]
    scale_params = [mod.scale for mod in student.modules() if isinstance(mod, BitLinear)]

    def run_epoch(optimizer, phase):
        epoch_loss = 0.0
        n_steps = 0
        indices = torch.randperm(len(cached_logits))
        for idx in indices:
            sample = cached_logits[idx.item()]
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            t_vals = sample["topk_vals"].float().to(device)
            t_ids = sample["topk_ids"].to(device)

            out = student(input_ids)
            s_logits = out.logits[0].float()
            seq_len = min(s_logits.shape[0], t_ids.shape[0])
            s_topk = s_logits[:seq_len].gather(1, t_ids[:seq_len])

            t_probs = F.softmax(t_vals[:seq_len], dim=-1)
            s_log_probs = F.log_softmax(s_topk, dim=-1)
            loss = F.kl_div(s_log_probs, t_probs, reduction="batchmean")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(weight_params + scale_params, GRAD_CLIP)
            optimizer.step()

            with torch.no_grad():
                for mod in student.modules():
                    if isinstance(mod, BitLinear) and not mod._baked:
                        mod.clamp_weights()
                if phase == 2:
                    for bname, bit in bit_modules.items():
                        max_s = init_scale_params[bname].to(device) * MAX_SCALE_RATIO
                        bit.scale.data.clamp_(-max_s, max_s)

            epoch_loss += loss.item()
            n_steps += 1
        return epoch_loss / max(n_steps, 1)

    def compute_forensics():
        cat_m = defaultdict(lambda: {"bit_match": [], "ratio_med": [], "kurtosis": []})
        with torch.no_grad():
            for bname, bit in bit_modules.items():
                cat = classify_layer(bname)
                w = bit.weight
                gs = bit.group_size
                out_dim, in_dim = w.shape
                w_p = F.pad(w, (0, bit.pad_in)) if bit.pad_in > 0 else w
                w_g = w_p.reshape(out_dim, -1, gs)

                cur_signs = w_g.sign()
                cur_signs = torch.where(cur_signs == 0, torch.ones_like(cur_signs), cur_signs)
                t_signs = init_signs[bname].to(device)
                bm = (cur_signs == t_signs).float().mean().item()

                cur_scale = bit.scale.abs().clamp(min=1e-6)
                t_scale = init_scales[bname].to(device).clamp(min=1e-8)
                rm = (cur_scale / t_scale).flatten().median().item()

                baked = (cur_signs * cur_scale.unsqueeze(2)).reshape(out_dim, -1)[:, :in_dim]
                kurt = float(scipy_stats.kurtosis(baked.cpu().reshape(-1).numpy(), fisher=True))

                cat_m[cat]["bit_match"].append(bm)
                cat_m[cat]["ratio_med"].append(rm)
                cat_m[cat]["kurtosis"].append(kurt)
        return cat_m

    def print_metrics(epoch, total, loss, cat_m, phase):
        print(f"\n  Epoch {epoch}/{total} (P{phase}): loss={loss:.4f}", flush=True)
        for cat in ["ffn_up", "ffn_gate", "attn_v", "ffn_down", "attn_o", "attn_k", "attn_q"]:
            if cat in cat_m:
                bm = np.median(cat_m[cat]["bit_match"])
                rm = np.median(cat_m[cat]["ratio_med"])
                ku = np.median(cat_m[cat]["kurtosis"])
                print(f"    {cat:<10}: bit_match={bm:.4f}, ratio_med={rm:.4f}, kurtosis={ku:.4f}", flush=True)

    all_metrics = []

    # -- Phase 1: Signs only (scales frozen) --
    for mod in student.modules():
        if isinstance(mod, BitLinear):
            mod.scale.requires_grad = False

    import bitsandbytes as bnb
    opt1 = bnb.optim.AdamW8bit(
        [{"params": weight_params, "lr": WEIGHT_LR}],
        weight_decay=0.0,
    )

    for epoch in range(1, SIGN_EPOCHS + 1):
        avg_loss = run_epoch(opt1, 1)
        cat_m = compute_forensics()
        print_metrics(epoch, total_epochs, avg_loss, cat_m, 1)
        all_metrics.append({
            "epoch": epoch, "phase": 1, "loss": round(avg_loss, 4),
            "per_category": {cat: {
                "bit_match": round(float(np.median(v["bit_match"])), 4),
                "ratio_med": round(float(np.median(v["ratio_med"])), 4),
                "kurtosis": round(float(np.median(v["kurtosis"])), 4),
            } for cat, v in cat_m.items()},
        })
        if epoch % 4 == 0:
            ckpt = f"/data/qwen17b_ste_v3_epoch{epoch}.pt"
            torch.save({
                "bit_modules": {k: {"weight": v.weight.data.cpu(), "scale": v.scale.data.cpu()}
                                for k, v in bit_modules.items()},
                "epoch": epoch, "phase": 1, "loss": avg_loss,
            }, ckpt)
            print(f"    Saved: {ckpt}", flush=True)
            vol.commit()

    # -- Phase 2: Signs + scales (clamped) --
    for mod in student.modules():
        if isinstance(mod, BitLinear):
            mod.scale.requires_grad = True

    opt2 = bnb.optim.AdamW8bit([
        {"params": weight_params, "lr": WEIGHT_LR * 0.3},
        {"params": scale_params, "lr": SCALE_LR},
    ], weight_decay=0.0)

    for epoch_offset in range(1, SCALE_EPOCHS + 1):
        epoch = SIGN_EPOCHS + epoch_offset
        avg_loss = run_epoch(opt2, 2)
        cat_m = compute_forensics()
        print_metrics(epoch, total_epochs, avg_loss, cat_m, 2)
        all_metrics.append({
            "epoch": epoch, "phase": 2, "loss": round(avg_loss, 4),
            "per_category": {cat: {
                "bit_match": round(float(np.median(v["bit_match"])), 4),
                "ratio_med": round(float(np.median(v["ratio_med"])), 4),
                "kurtosis": round(float(np.median(v["kurtosis"])), 4),
            } for cat, v in cat_m.items()},
        })
        if epoch_offset % 2 == 0 or epoch_offset == SCALE_EPOCHS:
            ckpt = f"/data/qwen17b_ste_v3_epoch{epoch}.pt"
            torch.save({
                "bit_modules": {k: {"weight": v.weight.data.cpu(), "scale": v.scale.data.cpu()}
                                for k, v in bit_modules.items()},
                "epoch": epoch, "phase": 2, "loss": avg_loss,
            }, ckpt)
            print(f"    Saved: {ckpt}", flush=True)
            vol.commit()

    # ==================================================================
    # Step 5: Bake and evaluate
    # ==================================================================
    print("\n" + "=" * 60)
    print("Step 5: Bake to exact binary and evaluate")
    print("=" * 60, flush=True)

    student.eval()
    for mod in student.modules():
        if isinstance(mod, BitLinear):
            mod.bake()

    # Final forensic metrics against teacher
    print("\n  Loading teacher for final bit_match...", flush=True)
    teacher_ref = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, device_map="cpu", token=hf_token,
    )

    final_cat = defaultdict(lambda: {"bit_match": [], "ratio_med": [], "kurtosis": []})
    for bname, bit in bit_modules.items():
        cat = classify_layer(bname)
        teacher_mod = teacher_ref
        for p in bname.split("."):
            teacher_mod = getattr(teacher_mod, p)
        tw = teacher_mod.weight.data.float()
        bw = bit.weight.data.cpu().float()  # baked binary, move to CPU for comparison

        gs = GROUP_SIZE
        pad = (gs - tw.shape[1] % gs) % gs
        tw_p = F.pad(tw, (0, pad)) if pad > 0 else tw
        bw_p = F.pad(bw, (0, pad)) if pad > 0 else bw

        tw_g = tw_p.reshape(tw.shape[0], -1, gs)
        bw_g = bw_p.reshape(bw.shape[0], -1, gs)

        t_signs = tw_g.sign()
        t_signs = torch.where(t_signs == 0, torch.ones_like(t_signs), t_signs)
        b_signs = bw_g.sign()
        b_signs = torch.where(b_signs == 0, torch.ones_like(b_signs), b_signs)

        bm = (t_signs == b_signs).float().mean().item()
        t_sc = tw_g.abs().mean(dim=2)
        b_sc = bw_g.abs().mean(dim=2)
        rm = (b_sc / t_sc.clamp(min=1e-8)).flatten().median().item()
        ku = float(scipy_stats.kurtosis(bw.reshape(-1).numpy(), fisher=True))

        final_cat[cat]["bit_match"].append(bm)
        final_cat[cat]["ratio_med"].append(rm)
        final_cat[cat]["kurtosis"].append(ku)

    del teacher_ref
    gc.collect()

    bonsai_tgt = {
        "ffn_up": (0.70, 2.96, -1.94), "ffn_gate": (0.70, 2.65, -1.89),
        "attn_v": (0.77, 2.71, -1.94), "ffn_down": (0.74, 2.08, -1.93),
        "attn_o": (0.74, 2.32, -1.77), "attn_k": (0.71, 1.83, -1.81),
        "attn_q": (0.71, 1.64, -1.65),
    }

    print("\n  FINAL FORENSIC SUMMARY (baked):")
    print(f"  {'Cat':<10} {'bit_match':>9} {'tgt':>6} {'ratio_med':>10} {'tgt':>6} {'kurtosis':>9} {'tgt':>6}")
    print("  " + "-" * 58)

    final_summary = {}
    for cat in ["ffn_up", "ffn_gate", "attn_v", "ffn_down", "attn_o", "attn_k", "attn_q"]:
        if cat not in final_cat:
            continue
        bm = np.median(final_cat[cat]["bit_match"])
        rm = np.median(final_cat[cat]["ratio_med"])
        ku = np.median(final_cat[cat]["kurtosis"])
        tb, tr, tk = bonsai_tgt.get(cat, (0, 0, 0))
        print(f"  {cat:<10} {bm:>9.4f} {tb:>6.2f} {rm:>10.4f} {tr:>6.2f} {ku:>9.4f} {tk:>6.2f}")
        final_summary[cat] = {
            "bit_match": round(float(bm), 4),
            "ratio_med": round(float(rm), 4),
            "kurtosis": round(float(ku), 4),
        }

    # Text generation
    print("\n  TEXT GENERATION (baked binary):")
    prompts = [
        "What is the capital of France?",
        "Explain gravity in one sentence.",
        "1 + 1 =",
        "The color of the sky is",
        "List three animals:",
        "A truck driver should check their mirrors because",
    ]
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

    # Save results
    results = {
        "final_summary": final_summary,
        "training_metrics": all_metrics,
        "config": {
            "model": model_id, "group_size": GROUP_SIZE,
            "sign_epochs": SIGN_EPOCHS, "scale_epochs": SCALE_EPOCHS,
            "weight_lr": WEIGHT_LR, "scale_lr": SCALE_LR,
            "n_samples": len(calib_texts), "top_k": TOP_K_LOGITS,
            "gptq_init": has_gptq,
        },
    }
    with open("/data/qwen17b_ste_distill_v3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()

    print(f"\n{'=' * 60}")
    print("STE DISTILLATION v3 COMPLETE")
    print(f"{'=' * 60}", flush=True)
    return results


@app.local_entrypoint()
def main():
    print("GPTQ + STE Distillation v3 -- Qwen3-1.7B\n")
    result = run_ste_distillation.remote()
    if result and "final_summary" in result:
        print("\n\nFINAL:")
        for cat, m in result["final_summary"].items():
            print(f"  {cat}: bm={m['bit_match']:.4f}, rm={m['ratio_med']:.4f}, ku={m['kurtosis']:.4f}")
