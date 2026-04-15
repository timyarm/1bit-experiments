"""Block-local MSE distillation with student-in + teacher-block-target.

Key insight from previous attempts:
- Teacher-in, teacher-out (v1): great per-block metrics (25% flips, Bonsai-matching)
  but accumulated error across 28 blocks → NaN at inference
- Student-in, teacher-out (v2): impossible task — can't match teacher's absolute output
  from corrupted input → MSE clamped at 20, no learning

THIS version: student-in, teacher-BLOCK-out
- For each block i, compute target = teacher_block_i(student_h_in)
- The target is what the teacher block would produce from the SAME corrupted input
- Each block learns the correct transformation, not the absolute output
- Accumulated error is bounded: each block adds ~1 layer of quant error
"""
import modal
import json

app = modal.App("1bit-blocklocal")

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
STEPS_PER_BLOCK = 200
BATCH_SIZE = 8
LR_WEIGHT = 5e-3
LR_SCALE = 1e-2
MAX_SEQ_LEN = 512
N_CALIB = 256


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
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=49152,
    volumes={"/data": vol},
)
def run_blocklocal_distill():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import os, gc, time, random
    import numpy as np

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    from scipy import stats as scipy_stats
    from collections import defaultdict

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
    # Load teacher model (keep on CPU, move blocks to GPU as needed)
    # ==================================================================
    print("\n" + "=" * 60)
    print("Loading teacher model...")
    print("=" * 60, flush=True)

    teacher = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, token=hf_token,
        device_map="cpu",
    )
    teacher.eval()
    n_blocks = len(teacher.model.layers)
    print(f"  {n_blocks} transformer blocks (teacher on CPU)", flush=True)

    # Cache teacher embedding output (= input to block 0)
    # This is the same for student since embedding isn't quantized
    print("  Caching embedding outputs...", flush=True)
    embed_outputs = []
    teacher_embed = teacher.model.embed_tokens.to(device)
    teacher_norm = teacher.model.norm  # final layernorm, keep on CPU for now
    with torch.no_grad():
        for ids in calib_ids:
            h = teacher_embed(ids.unsqueeze(0).to(device))
            embed_outputs.append(h.cpu())
    teacher_embed.cpu()
    print(f"  {len(embed_outputs)} embedding outputs cached", flush=True)

    # ==================================================================
    # Load student model
    # ==================================================================
    print("\n" + "=" * 60)
    print("Loading student model...")
    print("=" * 60, flush=True)

    student = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, device_map=device, token=hf_token,
    )
    rotary_emb = student.model.rotary_emb
    print(f"  Student loaded on GPU", flush=True)

    # ==================================================================
    # Block-local distillation: student-in + teacher-block-target
    # ==================================================================
    print("\n" + "=" * 60)
    print(f"Block-local distillation ({STEPS_PER_BLOCK} steps/block, {n_blocks} blocks)")
    print(f"  Mode: student-in + teacher-block-target")
    print("=" * 60, flush=True)

    cat_metrics = defaultdict(lambda: {"bit_match": [], "ratio_med": [], "kurtosis": []})
    total_t0 = time.time()
    n_samples = len(calib_ids)

    # Running student hidden states (updated after each block bake)
    # Convert to float32 to match student model dtype
    student_h_in = [h.float() for h in embed_outputs]

    for block_idx in range(n_blocks):
        block_t0 = time.time()
        student_block = student.model.layers[block_idx]

        # === Step A: Compute teacher targets ===
        # Move teacher block to GPU, run on student inputs, move back
        teacher_block = teacher.model.layers[block_idx].to(device)
        teacher_block.eval()

        teacher_targets = []
        with torch.no_grad():
            for s_idx in range(n_samples):
                h_in_f32 = student_h_in[s_idx].to(device)
                h_in = h_in_f32.half()  # teacher is fp16
                seq_len = h_in.shape[1]
                pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                pos_emb = rotary_emb(h_in_f32, pos_ids)  # rotary from student (fp32)
                pos_emb = tuple(p.half() for p in pos_emb)  # cast to fp16 for teacher
                t_out = teacher_block(h_in, position_embeddings=pos_emb)
                teacher_targets.append(t_out[0].float().cpu())  # store as fp32
        teacher_block.cpu()
        torch.cuda.empty_cache()

        # === Step B: Replace student block linears with BitLinear ===
        bit_modules = {}
        for name, module in student_block.named_modules():
            if isinstance(module, nn.Linear):
                cat = classify_layer(name)
                if cat in ("other", "token_embd"):
                    continue
                parts = name.split(".")
                parent = student_block
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                bit = BitLinear(module).to(device)
                setattr(parent, parts[-1], bit)
                bit_modules[name] = (bit, cat)

        for param in student.parameters():
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=STEPS_PER_BLOCK, eta_min=LR_WEIGHT * 0.1)

        # === Step C: Training loop ===
        student_block.train()
        losses = []

        for step in range(STEPS_PER_BLOCK):
            batch_indices = random.sample(range(n_samples),
                                          min(BATCH_SIZE, n_samples))
            total_loss = 0.0
            optimizer.zero_grad()

            for idx in batch_indices:
                h_in = student_h_in[idx].to(device)
                h_target = teacher_targets[idx].to(device)

                seq_len = h_in.shape[1]
                pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                pos_emb = rotary_emb(h_in, pos_ids)

                block_out = student_block(h_in, position_embeddings=pos_emb)
                h_out = block_out[0] if isinstance(block_out, tuple) else block_out
                # Ensure matching shapes (Qwen3 blocks sometimes squeeze batch dim)
                if h_out.dim() == 2 and h_target.dim() == 3:
                    h_out = h_out.unsqueeze(0)
                elif h_out.dim() == 3 and h_target.dim() == 2:
                    h_target = h_target.unsqueeze(0)

                # Cosine distance (direction) + normalized MSE (magnitude)
                # Both are scale-invariant — works across all block depths
                h_out_f = h_out.float().reshape(1, -1)
                h_tgt_f = h_target.float().reshape(1, -1)
                cos_loss = 1.0 - F.cosine_similarity(h_out_f, h_tgt_f)
                tgt_scale = h_tgt_f.pow(2).mean().clamp(min=1e-6)
                norm_mse = F.mse_loss(h_out.float(), h_target.float()) / tgt_scale
                loss = cos_loss + 0.1 * norm_mse.clamp(max=10.0)

                # Check for NaN BEFORE backward to prevent poisoning params
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    continue
                loss.backward()
                total_loss += loss.item()

            # NaN check on gradients
            has_nan = any(p.grad is not None and torch.isnan(p.grad).any()
                         for p in weight_params + scale_params)
            if has_nan or total_loss != total_loss:
                optimizer.zero_grad()
                losses.append(float('nan'))
                continue

            torch.nn.utils.clip_grad_norm_(weight_params + scale_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                for name, (bit, cat) in bit_modules.items():
                    bit.latent_weight.data.clamp_(-1.0, 1.0)

            avg_loss = total_loss / len(batch_indices)
            losses.append(avg_loss)

            if step == 0 or (step + 1) % 50 == 0 or step == STEPS_PER_BLOCK - 1:
                bm_list, rm_list = [], []
                for name, (bit, cat) in bit_modules.items():
                    bm, rm, ku = bit.forensics()
                    bm_list.append(bm)
                    rm_list.append(rm)
                avg_bm = np.mean(bm_list)
                avg_rm = np.mean(rm_list)
                print(f"  B{block_idx:2d} s{step+1:3d}/{STEPS_PER_BLOCK}: "
                      f"mse={avg_loss:.6f} flip={(1-avg_bm)*100:.1f}% rm={avg_rm:.4f}",
                      flush=True)

        # === Step D: Record forensics and bake ===
        for name, (bit, cat) in bit_modules.items():
            bm, rm, ku = bit.forensics()
            cat_metrics[cat]["bit_match"].append(bm)
            cat_metrics[cat]["ratio_med"].append(rm)
            cat_metrics[cat]["kurtosis"].append(ku)

            baked_w = bit.bake().float()
            parts = name.split(".")
            parent = student_block
            for p in parts[:-1]:
                parent = getattr(parent, p)
            new_linear = nn.Linear(bit.in_features, bit.out_features,
                                   bias=bit.has_bias, dtype=torch.float32, device=device)
            new_linear.weight.data.copy_(baked_w)
            if bit.has_bias and bit.bias is not None:
                new_linear.bias.data.copy_(bit.bias.data.float())
            new_linear.weight.requires_grad = False
            setattr(parent, parts[-1], new_linear)

        del optimizer, scheduler, bit_modules, weight_params, scale_params
        del teacher_targets
        gc.collect()
        torch.cuda.empty_cache()

        # === Step E: Update student hidden states for next block ===
        if block_idx < n_blocks - 1:
            new_student_h = []
            student_block.eval()
            with torch.no_grad():
                for s_idx in range(n_samples):
                    h_in = student_h_in[s_idx].to(device)
                    seq_len = h_in.shape[1]
                    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                    pos_emb = rotary_emb(h_in, pos_ids)
                    h_out = student_block(h_in, position_embeddings=pos_emb)
                    h_out = h_out[0] if isinstance(h_out, tuple) else h_out
                    new_student_h.append(h_out.cpu())
            student_h_in = new_student_h

        elapsed = time.time() - total_t0
        eta = elapsed / (block_idx + 1) * (n_blocks - block_idx - 1)
        print(f"  B{block_idx:2d} DONE: {time.time()-block_t0:.0f}s, "
              f"mse {losses[0]:.6f}->{losses[-1]:.6f}, "
              f"{elapsed:.0f}s elapsed, ~{eta:.0f}s ETA", flush=True)

    total_time = time.time() - total_t0
    print(f"\nTotal distillation time: {total_time:.0f}s", flush=True)

    # Free teacher
    del teacher
    gc.collect()

    # ==================================================================
    # Forensic summary
    # ==================================================================
    print("\n" + "=" * 60)
    print("FORENSIC SUMMARY")
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
        final_summary[cat] = {"bit_match": round(float(bm), 4),
                              "ratio_med": round(float(rm), 4),
                              "kurtosis": round(float(ku), 4)}

    # ==================================================================
    # Test generation
    # ==================================================================
    print("\n" + "=" * 60)
    print("TEXT GENERATION TEST")
    print("=" * 60, flush=True)

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

    student.eval()

    # Check logit validity first
    print("  Checking logit validity...", flush=True)
    test_ids = tokenizer("Hello", return_tensors="pt").to(device)
    with torch.no_grad():
        test_logits = student(**test_ids).logits
        has_nan = torch.isnan(test_logits).any().item()
        has_inf = torch.isinf(test_logits).any().item()
        logit_range = (test_logits.min().item(), test_logits.max().item())
        print(f"  Logits: nan={has_nan}, inf={has_inf}, "
              f"range=[{logit_range[0]:.1f}, {logit_range[1]:.1f}]", flush=True)

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

    # Sampling test
    print("  --- With sampling (temp=0.7) ---", flush=True)
    for prompt in prompts[:3]:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        try:
            with torch.no_grad():
                out = student.generate(
                    **inputs, max_new_tokens=80, do_sample=True,
                    temperature=0.7, top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen = out[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(gen, skip_special_tokens=True)
        except Exception as e:
            text = f"[ERROR: {e}]"
        print(f"  Q: {prompt}")
        print(f"  A: {text[:300]}")
        print(flush=True)

    # ==================================================================
    # Perplexity
    # ==================================================================
    print("\n" + "=" * 60)
    print("PERPLEXITY CHECK")
    print("=" * 60, flush=True)

    ppl = float('inf')
    try:
        total_nll = 0.0
        total_tokens = 0
        with torch.no_grad():
            for ids in calib_ids[:20]:
                input_ids = ids.unsqueeze(0).to(device)
                logits = student(input_ids).logits
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(f"  WARNING: NaN/Inf in logits", flush=True)
                    continue
                shift_logits = logits[0, :-1].float()
                shift_labels = input_ids[0, 1:]
                nll = F.cross_entropy(shift_logits, shift_labels, reduction='sum')
                total_nll += nll.item()
                total_tokens += shift_labels.numel()
        if total_tokens > 0:
            ppl = np.exp(total_nll / total_tokens)
    except Exception as e:
        print(f"  Perplexity failed: {e}", flush=True)
    print(f"  Perplexity (20 samples): {ppl:.2f}", flush=True)

    # Save
    results = {
        "forensics": final_summary,
        "generations": gen_results,
        "perplexity": round(float(ppl), 2) if ppl != float('inf') else None,
        "config": {
            "approach": "block-local MSE, student-in + teacher-block-target",
            "lr_weight": LR_WEIGHT,
            "lr_scale": LR_SCALE,
            "steps_per_block": STEPS_PER_BLOCK,
            "batch_size": BATCH_SIZE,
            "n_blocks": n_blocks,
            "total_time_s": total_time,
        },
    }
    with open("/data/blocklocal_v3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()

    print(f"\n{'=' * 60}")
    print("BLOCK-LOCAL DISTILLATION v3 COMPLETE")
    print(f"{'=' * 60}")
    return results


@app.local_entrypoint()
def main():
    print("Block-Local MSE v3: student-in + teacher-block-target\n")
    result = run_blocklocal_distill.remote()
    if result:
        ppl = result.get('perplexity', 'N/A')
        print(f"\nPerplexity: {ppl}")
        print("\nFORENSICS:")
        for cat, m in result.get("forensics", {}).items():
            print(f"  {cat}: bm={m['bit_match']:.4f}, rm={m['ratio_med']:.4f}, ku={m['kurtosis']:.4f}")
