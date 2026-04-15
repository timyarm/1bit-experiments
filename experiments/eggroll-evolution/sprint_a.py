"""End-to-end QAT with STE — the Bonsai recipe.

Previous block-local approaches proved the mechanics but couldn't match Bonsai's
forensic signature: uniform 25-30% sign flips across ALL depths, negative kurtosis,
depth-correlated scales. Block-local gave 25% shallow / 6% deep — non-uniform.

This script does what Bonsai almost certainly does:
- Replace ALL linear layers with BitLinear simultaneously
- Train end-to-end with KL divergence against teacher logits
- STE gradients flow through the entire network
- All blocks co-adapt — deep blocks learn to compensate for shallow block changes

Key insights from Archie's forensics:
- Depth-indexed scales (correlation 0.877 with depth)
- Category-specific patterns (attn_v treated differently)
- Negative kurtosis (-1.65 to -1.94) = flat weight distribution
- Temperature-scaled KL div for soft target distillation
"""
import modal
import json

app = modal.App("1bit-eggroll-a")

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

# Hyperparameters
GROUP_SIZE = 128
N_STEPS = 1000         # Sprint — 1000 steps to see trajectory
BATCH_SIZE = 2
LR_WEIGHT = 1e-3       # Same as baseline
LR_SCALE = 5e-3
MAX_SEQ_LEN = 256
N_CALIB = 256
KL_TEMP = 2.0
GRAD_ACCUM = 4
EVAL_EVERY = 250       # 4 evals per sprint
GRAD_CKPT = True       # Gradient checkpointing to fit in 24GB


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


def get_block_idx(name: str) -> int:
    """Extract block index from parameter name like 'model.layers.14.self_attn.q_proj'."""
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
    gpu="A100:1",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=65536,
    volumes={"/data": vol},
)
def run_e2e_qat():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import os, gc, time, random
    import numpy as np
    from scipy import stats as scipy_stats
    from collections import defaultdict

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    hf_token = os.environ.get("HF_TOKEN")
    device = "cuda"
    model_id = "Qwen/Qwen3-1.7B"

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset

    class BitLinear(nn.Module):
        def __init__(self, original: nn.Linear, group_size=GROUP_SIZE):
            super().__init__()
            w = original.weight.data.float()
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
            self.has_bias = original.bias is not None
            if self.has_bias:
                self.bias = nn.Parameter(original.bias.data.clone())
            else:
                self.bias = None

        def forward(self, x):
            signs = self.latent_weight.sign()
            signs = signs + (signs == 0).float()
            signs_ste = signs.detach() + self.latent_weight - self.latent_weight.detach()
            scale_abs = self.scale.abs().unsqueeze(2)
            w_q = (signs_ste * scale_abs).reshape(self.out_features, -1)
            if self.pad > 0:
                w_q = w_q[:, :self.in_features]
            out = F.linear(x.float(), w_q, self.bias)
            return out.to(x.dtype)

        def bake(self):
            with torch.no_grad():
                signs = self.latent_weight.sign()
                signs[signs == 0] = 1.0
                scale_abs = self.scale.abs().unsqueeze(2)
                w_q = (signs * scale_abs).reshape(self.out_features, -1)
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
    # Load calibration data
    # ==================================================================
    print("=" * 60)
    print("Loading calibration data...")
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
            if count >= 200:
                break

    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    for row in alpaca.shuffle(seed=42):
        text = (row.get("instruction", "") + " " + row.get("output", "")).strip()
        if len(text) > 100:
            calib_texts.append(text[:2000])
            count += 1
            if count >= N_CALIB:
                break

    print(f"  {len(calib_texts)} calibration samples", flush=True)

    calib_ids = []
    for text in calib_texts:
        toks = tokenizer(text, return_tensors="pt", truncation=True,
                         max_length=MAX_SEQ_LEN, padding=False)
        calib_ids.append(toks["input_ids"].squeeze(0))

    # ==================================================================
    # Pre-cache teacher logits (avoids keeping teacher on GPU during training)
    # ==================================================================
    print("\n" + "=" * 60)
    print("Pre-caching teacher logits...")
    print("=" * 60, flush=True)

    teacher = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, token=hf_token, device_map=device,
    )
    teacher.eval()

    # Cache teacher logits — use top-k to save memory
    # Full logits: 256 samples × 256 tokens × 151k vocab × 4 bytes ≈ 40GB → too much
    # Top-128 logits + indices: manageable (~500MB)
    TOP_K = 128
    teacher_cache = []
    t0 = time.time()
    with torch.no_grad():
        for i, ids in enumerate(calib_ids):
            input_ids = ids.unsqueeze(0).to(device)
            logits = teacher(input_ids).logits[0]  # [seq, vocab]
            # Store top-k values and indices for KL div
            topk_vals, topk_idx = logits.topk(TOP_K, dim=-1)
            teacher_cache.append({
                "topk_vals": topk_vals.cpu().float(),  # [seq, TOP_K]
                "topk_idx": topk_idx.cpu(),             # [seq, TOP_K]
                "input_ids": ids,
            })
            if (i + 1) % 50 == 0:
                print(f"  Cached {i+1}/{len(calib_ids)} samples", flush=True)

    print(f"  Teacher logits cached in {time.time()-t0:.0f}s", flush=True)

    # Free teacher from GPU
    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    # ==================================================================
    # Load student model
    # ==================================================================
    print("\n" + "=" * 60)
    print("Loading student model...")
    print("=" * 60, flush=True)

    student = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, device_map=device, token=hf_token,
    )

    if GRAD_CKPT:
        student.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled", flush=True)

    n_blocks = len(student.model.layers)
    print(f"  {n_blocks} transformer blocks", flush=True)

    # ==================================================================
    # Replace ALL linear layers with BitLinear
    # ==================================================================
    print("\n" + "=" * 60)
    print("Replacing all linear layers with BitLinear...")
    print("=" * 60, flush=True)

    bit_modules = {}  # full_name -> (BitLinear, category, block_idx)
    replace_count = 0

    for name, module in list(student.named_modules()):
        if isinstance(module, nn.Linear):
            cat = classify_layer(name)
            if cat in ("other", "token_embd"):
                continue  # Keep embedding and lm_head as-is

            block_idx = get_block_idx(name)
            parts = name.split(".")
            parent = student
            for p in parts[:-1]:
                parent = getattr(parent, p)

            bit = BitLinear(module)
            setattr(parent, parts[-1], bit)
            bit_modules[name] = (bit, cat, block_idx)
            replace_count += 1

    print(f"  Replaced {replace_count} linear layers with BitLinear", flush=True)

    # Freeze everything except BitLinear params
    for param in student.parameters():
        param.requires_grad = False

    weight_params, scale_params = [], []
    for name, (bit, cat, bidx) in bit_modules.items():
        bit.latent_weight.requires_grad = True
        bit.scale.requires_grad = True
        weight_params.append(bit.latent_weight)
        scale_params.append(bit.scale)

    total_trainable = sum(p.numel() for p in weight_params + scale_params)
    print(f"  {total_trainable:,} trainable parameters", flush=True)

    # ==================================================================
    # Optimizer (constant LR — no scheduler)
    # ==================================================================
    optimizer = torch.optim.AdamW([
        {"params": weight_params, "lr": LR_WEIGHT},
        {"params": scale_params, "lr": LR_SCALE},
    ], weight_decay=0.0)

    # ==================================================================
    # Training loop — end-to-end KL distillation
    # ==================================================================
    print("\n" + "=" * 60)
    print(f"End-to-end QAT: {N_STEPS} steps, batch={BATCH_SIZE}×{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}")
    print(f"  KL temperature: {KL_TEMP}")
    print("=" * 60, flush=True)

    student.train()
    n_samples = len(calib_ids)
    total_t0 = time.time()
    losses_history = []
    best_ppl = float('inf')

    for step in range(N_STEPS):
        optimizer.zero_grad()
        step_loss = 0.0
        valid_accums = 0

        for accum in range(GRAD_ACCUM):
            idx = random.randint(0, n_samples - 1)
            cached = teacher_cache[idx]
            input_ids = cached["input_ids"].unsqueeze(0).to(device)
            teacher_topk_vals = cached["topk_vals"].to(device)  # [seq, TOP_K]
            teacher_topk_idx = cached["topk_idx"].to(device)    # [seq, TOP_K]

            # Student forward pass
            student_logits = student(input_ids).logits[0]  # [seq, vocab]

            # Build KL divergence loss using top-k teacher logits
            # Teacher soft targets (temperature-scaled)
            teacher_log_probs = F.log_softmax(teacher_topk_vals / KL_TEMP, dim=-1)

            # Student logits at teacher's top-k positions
            student_topk_logits = student_logits.gather(1, teacher_topk_idx)
            student_log_probs = F.log_softmax(student_topk_logits / KL_TEMP, dim=-1)

            # KL divergence (teacher || student) — scaled by T²
            kl_loss = F.kl_div(
                student_log_probs,
                teacher_log_probs,
                log_target=True,
                reduction="batchmean",
            ) * (KL_TEMP ** 2)

            # Check for NaN before backward
            if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                continue

            loss = kl_loss / GRAD_ACCUM
            loss.backward()
            step_loss += kl_loss.item()
            valid_accums += 1

        if valid_accums == 0:
            losses_history.append(float('nan'))
            continue

        # Category-weighted gradients: FFN layers get 3x pressure
        CAT_WEIGHTS = {
            "ffn_up": 3.0, "ffn_gate": 3.0, "ffn_down": 2.0,
            "attn_q": 1.0, "attn_k": 1.0, "attn_v": 1.0, "attn_o": 1.0,
        }
        with torch.no_grad():
            for nm, (bit, cat, bidx) in bit_modules.items():
                w = CAT_WEIGHTS.get(cat, 1.0)
                if w != 1.0:
                    if bit.latent_weight.grad is not None:
                        bit.latent_weight.grad *= w
                    if bit.scale.grad is not None:
                        bit.scale.grad *= w

        # Gradient clipping + step
        torch.nn.utils.clip_grad_norm_(weight_params + scale_params, max_norm=1.0)
        optimizer.step()
        # Clamp latent weights
        with torch.no_grad():
            for name, (bit, cat, bidx) in bit_modules.items():
                bit.latent_weight.data.clamp_(-1.0, 1.0)

        avg_loss = step_loss / valid_accums
        losses_history.append(avg_loss)

        # === Logging ===
        if step == 0 or (step + 1) % 50 == 0:
            # Quick forensic snapshot (sample 3 blocks: shallow, mid, deep)
            sample_blocks = [0, n_blocks // 2, n_blocks - 1]
            snap = {}
            for bidx in sample_blocks:
                bm_list, rm_list = [], []
                for nm, (bit, cat, bi) in bit_modules.items():
                    if bi == bidx:
                        bm, rm, _ = bit.forensics()
                        bm_list.append(bm)
                        rm_list.append(rm)
                if bm_list:
                    snap[bidx] = (1 - np.mean(bm_list), np.mean(rm_list))

            snap_str = " | ".join(
                f"B{bi}:{flip*100:.1f}%/{rm:.2f}"
                for bi, (flip, rm) in sorted(snap.items())
            )
            elapsed = time.time() - total_t0
            eta = elapsed / (step + 1) * (N_STEPS - step - 1)
            print(f"  step {step+1:4d}/{N_STEPS}: kl={avg_loss:.4f} "
                  f"[{snap_str}] "
                  f"{elapsed:.0f}s, ~{eta:.0f}s ETA", flush=True)

        # === Full eval checkpoint ===
        if (step + 1) % EVAL_EVERY == 0 or step == N_STEPS - 1:
            print(f"\n  --- Eval at step {step+1} ---", flush=True)

            # Full forensics
            cat_metrics = defaultdict(lambda: {"bit_match": [], "ratio_med": [], "kurtosis": []})
            depth_flips = defaultdict(list)
            for nm, (bit, cat, bidx) in bit_modules.items():
                bm, rm, ku = bit.forensics()
                cat_metrics[cat]["bit_match"].append(bm)
                cat_metrics[cat]["ratio_med"].append(rm)
                cat_metrics[cat]["kurtosis"].append(ku)
                depth_flips[bidx].append(1 - bm)

            bonsai_tgt = {
                "ffn_up": (0.70, 2.96, -1.94), "ffn_gate": (0.70, 2.65, -1.89),
                "attn_v": (0.77, 2.71, -1.94), "ffn_down": (0.74, 2.08, -1.93),
                "attn_o": (0.74, 2.32, -1.77), "attn_k": (0.71, 1.83, -1.81),
                "attn_q": (0.71, 1.64, -1.65),
            }

            print(f"  {'Cat':<10} {'bit_match':>9} {'tgt':>6} {'ratio_med':>10} {'tgt':>6} {'kurtosis':>9} {'tgt':>6}")
            print("  " + "-" * 58)
            for cat in ["attn_q", "attn_k", "attn_v", "attn_o", "ffn_up", "ffn_gate", "ffn_down"]:
                if cat not in cat_metrics:
                    continue
                bm = np.median(cat_metrics[cat]["bit_match"])
                rm = np.median(cat_metrics[cat]["ratio_med"])
                ku = np.median(cat_metrics[cat]["kurtosis"])
                tb, tr, tk = bonsai_tgt.get(cat, (0, 0, 0))
                print(f"  {cat:<10} {bm:>9.4f} {tb:>6.2f} {rm:>10.4f} {tr:>6.2f} {ku:>9.4f} {tk:>6.2f}")

            # Depth uniformity check
            depth_avg = {k: np.mean(v) for k, v in sorted(depth_flips.items())}
            shallow = np.mean([v for k, v in depth_avg.items() if k < 7])
            mid = np.mean([v for k, v in depth_avg.items() if 7 <= k < 21])
            deep = np.mean([v for k, v in depth_avg.items() if k >= 21])
            print(f"  Depth uniformity: shallow={shallow*100:.1f}% mid={mid*100:.1f}% deep={deep*100:.1f}%")

            # Quick text test
            student.eval()
            for prompt in ["What is the capital of France?", "A truck driver should always"]:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                try:
                    with torch.no_grad():
                        out = student.generate(
                            **inputs, max_new_tokens=50, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    gen = out[0][inputs["input_ids"].shape[1]:]
                    text = tokenizer.decode(gen, skip_special_tokens=True)
                except Exception as e:
                    text = f"[ERROR: {e}]"
                print(f"  Q: {prompt}")
                print(f"  A: {text[:200]}", flush=True)

            # Perplexity on 10 samples
            total_nll = 0.0
            total_tokens = 0
            with torch.no_grad():
                for ids in calib_ids[:10]:
                    input_ids = ids.unsqueeze(0).to(device)
                    logits = student(input_ids).logits
                    if torch.isnan(logits).any():
                        continue
                    shift_logits = logits[0, :-1].float()
                    shift_labels = input_ids[0, 1:]
                    nll = F.cross_entropy(shift_logits, shift_labels, reduction='sum')
                    total_nll += nll.item()
                    total_tokens += shift_labels.numel()
            ppl = np.exp(total_nll / total_tokens) if total_tokens > 0 else float('inf')
            print(f"  Perplexity: {ppl:.2f}", flush=True)

            if ppl < best_ppl:
                best_ppl = ppl
                print(f"  *** New best perplexity: {ppl:.2f} — saving checkpoint ***", flush=True)
                # Save BitLinear state for resume/fine-tuning
                ckpt = {}
                for nm, (bit, cat, bidx) in bit_modules.items():
                    ckpt[nm] = {
                        "latent_weight": bit.latent_weight.data.cpu(),
                        "scale": bit.scale.data.cpu(),
                    }
                ckpt["__meta__"] = {"step": step + 1, "ppl": ppl, "best_ppl": best_ppl}
                torch.save(ckpt, "/data/sprint_a_best_checkpoint.pt")
                vol.commit()

            student.train()
            print(flush=True)

    total_time = time.time() - total_t0
    print(f"\nTotal training time: {total_time:.0f}s", flush=True)

    # ==================================================================
    # Final full eval
    # ==================================================================
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60, flush=True)

    student.eval()

    # Final forensics
    cat_metrics = defaultdict(lambda: {"bit_match": [], "ratio_med": [], "kurtosis": []})
    for nm, (bit, cat, bidx) in bit_modules.items():
        bm, rm, ku = bit.forensics()
        cat_metrics[cat]["bit_match"].append(bm)
        cat_metrics[cat]["ratio_med"].append(rm)
        cat_metrics[cat]["kurtosis"].append(ku)

    final_summary = {}
    for cat in ["attn_q", "attn_k", "attn_v", "attn_o", "ffn_up", "ffn_gate", "ffn_down"]:
        if cat not in cat_metrics:
            continue
        bm = np.median(cat_metrics[cat]["bit_match"])
        rm = np.median(cat_metrics[cat]["ratio_med"])
        ku = np.median(cat_metrics[cat]["kurtosis"])
        final_summary[cat] = {"bit_match": round(float(bm), 4),
                              "ratio_med": round(float(rm), 4),
                              "kurtosis": round(float(ku), 4)}

    # Full text generation
    prompts = [
        "What is the capital of France?",
        "Explain gravity in one sentence.",
        "1 + 1 =",
        "The color of the sky is",
        "List three animals:",
        "A truck driver should check their mirrors because",
        "The most important safety rule when driving a semi truck is",
        "To calculate the area of a circle, you need to",
    ]

    gen_results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        try:
            with torch.no_grad():
                out = student.generate(
                    **inputs, max_new_tokens=80, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen = out[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(gen, skip_special_tokens=True)
        except Exception as e:
            text = f"[ERROR: {e}]"
        print(f"  Q: {prompt}")
        print(f"  A: {text[:300]}")
        print(flush=True)
        gen_results.append({"prompt": prompt, "response": text[:300]})

    # Final perplexity
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for ids in calib_ids[:20]:
            input_ids = ids.unsqueeze(0).to(device)
            logits = student(input_ids).logits
            if torch.isnan(logits).any():
                continue
            shift_logits = logits[0, :-1].float()
            shift_labels = input_ids[0, 1:]
            nll = F.cross_entropy(shift_logits, shift_labels, reduction='sum')
            total_nll += nll.item()
            total_tokens += shift_labels.numel()
    ppl = np.exp(total_nll / total_tokens) if total_tokens > 0 else float('inf')
    print(f"\nFinal Perplexity (20 samples): {ppl:.2f}", flush=True)

    # Save results
    results = {
        "forensics": final_summary,
        "generations": gen_results,
        "perplexity": round(float(ppl), 2) if ppl != float('inf') else None,
        "best_perplexity": round(float(best_ppl), 2) if best_ppl != float('inf') else None,
        "config": {
            "approach": "end-to-end QAT with STE + KL distillation",
            "lr_weight": LR_WEIGHT,
            "lr_scale": LR_SCALE,
            "n_steps": N_STEPS,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "kl_temp": KL_TEMP,
            "top_k_teacher": TOP_K,
            "n_blocks": n_blocks,
            "total_time_s": total_time,
        },
        "loss_history": losses_history[-100:],  # Last 100 losses
    }

    with open("/data/sprint_a_catweight_results.json", "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()

    print(f"\n{'=' * 60}")
    print("END-TO-END QAT COMPLETE")
    print(f"{'=' * 60}")
    return results


@app.local_entrypoint()
def main():
    print("End-to-End QAT with STE — Bonsai Recipe\n")
    result = run_e2e_qat.remote()
    if result:
        ppl = result.get('perplexity', 'N/A')
        print(f"\nFinal Perplexity: {ppl}")
        print(f"Best Perplexity: {result.get('best_perplexity', 'N/A')}")
        print("\nFORENSICS:")
        for cat, m in result.get("forensics", {}).items():
            print(f"  {cat}: bm={m['bit_match']:.4f}, rm={m['ratio_med']:.4f}, ku={m['kurtosis']:.4f}")
