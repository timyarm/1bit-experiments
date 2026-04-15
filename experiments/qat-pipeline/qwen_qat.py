"""Fast 1-bit QAT recipe proving ground — Qwen3-0.6B on T4.

Same architecture family Archie reverse-engineered Bonsai from.
Text-only, ~5-10 min per run, ~$0.10/run on T4.

Proves: STE gradients flow, signs flip to 0.70-0.79, scales reach
Archie's forensic targets, coherent output from 1-bit model.

Then we port the proven recipe to Gemma 4 E4B.

Usage:
  modal run scripts/modal_qwen_1bit_qat.py
"""
import modal
import json

app = modal.App("1bit-qwen-qat")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.51.0",
        "accelerate>=1.5.0",
        "huggingface_hub>=0.28.0",
        "bitsandbytes>=0.45.0",
        "datasets>=3.0.0",
    )
)

vol = modal.Volume.from_name("1bit-checkpoints", create_if_missing=True)

# Archie's category hierarchy (normalized loss weights)
CATEGORY_WEIGHTS = {
    "ffn_up":   1.0,
    "ffn_gate": 0.85,
    "attn_v":   0.90,
    "ffn_down": 0.60,
    "attn_o":   0.50,
    "attn_k":   0.30,
    "attn_q":   0.25,
    "embed":    0.0,
    "lm_head":  0.0,
    "other":    0.10,
}


def classify_layer(name: str) -> str:
    if "up_proj" in name: return "ffn_up"
    if "down_proj" in name: return "ffn_down"
    if "gate_proj" in name: return "ffn_gate"
    if "q_proj" in name: return "attn_q"
    if "k_proj" in name: return "attn_k"
    if "v_proj" in name: return "attn_v"
    if "o_proj" in name: return "attn_o"
    if "embed" in name: return "embed"
    if "lm_head" in name: return "lm_head"
    return "other"


@app.function(
    image=image,
    gpu="T4:1",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=32768,
    volumes={"/data": vol},
)
def run_qat():
    """Qwen3-0.6B 1-bit QAT: prove Bonsai recipe on small transformer."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import time
    import os
    import gc
    import numpy as np

    import bitsandbytes as bnb

    hf_token = os.environ.get("HF_TOKEN")
    model_id = "Qwen/Qwen3-0.6B"
    device = "cuda"
    TOP_K = 500  # smaller vocab (151K vs 256K), less top-K needed

    print("=" * 60)
    print("1-BIT BONSAI QAT — Qwen3-0.6B (recipe proving ground)")
    print("  Archie's recipe on the exact architecture he analyzed")
    print("  Goal: bit_match 0.70-0.79, ratio_med ~2-3x, coherent output")
    print("=" * 60)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

    # ══════════════════════════════════════════════════════════
    # BitLinear: 1-bit with bf16 STE
    # ══════════════════════════════════════════════════════════
    class BitLinear(nn.Module):
        def __init__(self, in_features, out_features, bias=True, group_size=128):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.group_size = group_size
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter("bias", None)
            n_groups = max(1, (out_features * in_features + group_size - 1) // group_size)
            self.scale = nn.Parameter(torch.ones(n_groups))
            self._baked = False

        def forward(self, x):
            if self._baked:
                return F.linear(x, self.weight, self.bias)
            w_bin = self._quantize_ste(self.weight)
            return F.linear(x, w_bin, self.bias)

        def _quantize_ste(self, w):
            """Identity STE: forward=sign*scale, backward=identity.

            No tanh — eliminates gradient vanishing and train/eval gap.
            Gradient flows directly through sign() as if it were identity.
            Weight clamping (external) prevents explosion.
            """
            flat = w.reshape(-1)
            gs = self.group_size
            remainder = flat.numel() % gs
            if remainder != 0:
                flat = F.pad(flat, (0, gs - remainder))
            groups = flat.reshape(-1, gs)
            n_groups = groups.shape[0]
            scale = self.scale[:n_groups].abs().clamp(min=1e-6).unsqueeze(1)
            # Identity STE: hard sign in forward, gradient=1 in backward
            signs = groups.sign()
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            # STE trick: forward uses signs, backward uses groups (identity)
            binary_w = signs.detach() + groups - groups.detach()
            quantized = binary_w * scale
            result = quantized.reshape(-1)[:w.numel()].reshape(w.shape)
            return result

        def clamp_weights(self, max_ratio=1.5):
            """Clamp latent weights to ±max_ratio*scale per group.
            Prevents weight explosion while allowing sign flips near zero.
            """
            with torch.no_grad():
                flat = self.weight.reshape(-1)
                gs = self.group_size
                rem = flat.numel() % gs
                if rem:
                    # Can't do per-group clamping easily with remainder
                    s_max = self.scale.abs().max().item() * max_ratio
                    self.weight.data.clamp_(-s_max, s_max)
                else:
                    groups = flat.reshape(-1, gs)
                    n = groups.shape[0]
                    s = self.scale[:n].abs().clamp(min=1e-6).unsqueeze(1) * max_ratio
                    groups.clamp_(-s, s)
                    self.weight.data = groups.reshape(-1)[:self.weight.numel()].reshape(self.weight.shape)

        def bake_for_inference(self):
            with torch.no_grad():
                w = self.weight.reshape(-1)
                gs = self.group_size
                rem = w.numel() % gs
                if rem:
                    w = F.pad(w, (0, gs - rem))
                groups = w.reshape(-1, gs)
                n_groups = groups.shape[0]
                scale = self.scale[:n_groups].abs().clamp(min=1e-6).unsqueeze(1)
                signs = groups.sign()
                signs = torch.where(signs == 0, torch.ones_like(signs), signs)
                baked = (signs * scale).reshape(-1)[:self.out_features * self.in_features]
                baked = baked.reshape(self.out_features, self.in_features).to(self.weight.dtype)
                self.weight = nn.Parameter(baked, requires_grad=False)
                del self.scale
                self._baked = True

        @classmethod
        def from_linear(cls, linear, group_size=128):
            has_bias = linear.bias is not None
            bit = cls(linear.in_features, linear.out_features, bias=has_bias,
                      group_size=group_size)
            with torch.no_grad():
                bit.weight.copy_(linear.weight)  # Keep continuous teacher weights!
                if has_bias:
                    bit.bias.copy_(linear.bias)
                # Initialize scales from teacher's actual weight magnitudes
                flat = linear.weight.reshape(-1)
                gs = group_size
                remainder = flat.numel() % gs
                if remainder != 0:
                    flat = F.pad(flat, (0, gs - remainder))
                groups = flat.reshape(-1, gs)
                bit.scale = nn.Parameter(groups.abs().mean(dim=1).clone())
            return bit.to(linear.weight.device)

    # ══════════════════════════════════════════════════════════
    # Phase 1: Cache teacher logits
    # ══════════════════════════════════════════════════════════
    logits_path = "/data/qwen06b_teacher_logits.pt"

    if os.path.exists(logits_path):
        print(f"\n  Found cached teacher logits — loading", flush=True)
        cached = torch.load(logits_path, map_location="cpu", weights_only=True)
        cached_topk_vals = cached["topk_vals"]
        cached_topk_ids = cached["topk_ids"]
        cached_input_ids = cached["input_ids"]
        print(f"  Loaded {len(cached_topk_vals)} cached samples", flush=True)
    else:
        print(f"\n[Phase 1] Caching teacher logits...", flush=True)

        teacher = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cpu", token=hf_token,
        )
        teacher.to(device)
        teacher.eval()

        vram = torch.cuda.memory_allocated() / 1e9
        print(f"  Teacher VRAM: {vram:.1f} GB", flush=True)

        # Load diverse text data: SlimOrca (150) + Alpaca (100)
        from datasets import load_dataset

        samples = []

        print("  Loading SlimOrca...", flush=True)
        orca = load_dataset("Open-Orca/SlimOrca", split="train", streaming=True)
        count = 0
        for item in orca:
            if count >= 150:
                break
            convos = item.get("conversations", [])
            human_text = ""
            for turn in convos:
                if turn.get("from") == "human":
                    human_text = turn.get("value", "")
                    break
            if not human_text or len(human_text) < 20:
                continue
            if len(human_text) > 400:
                human_text = human_text[:400]
            samples.append(human_text)
            count += 1
        print(f"  SlimOrca: {count}", flush=True)

        print("  Loading Alpaca...", flush=True)
        alpaca = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
        count = 0
        for item in alpaca:
            if count >= 100:
                break
            instruction = item.get("instruction", "")
            inp = item.get("input", "")
            if not instruction or len(instruction) < 10:
                continue
            prompt = instruction
            if inp:
                prompt = f"{instruction}\n\n{inp}"
            if len(prompt) > 400:
                prompt = prompt[:400]
            samples.append(prompt)
            count += 1
        print(f"  Alpaca: {count}", flush=True)
        print(f"  Total: {len(samples)} samples", flush=True)

        cached_topk_vals = []
        cached_topk_ids = []
        cached_input_ids = []

        t_cache = time.time()
        for i, text in enumerate(samples):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            input_ids = inputs["input_ids"].to(device)
            with torch.no_grad():
                out = teacher(input_ids=input_ids)
                logits = out.logits[0].cpu().float()  # [seq_len, vocab]
                topk_vals, topk_ids = logits.topk(TOP_K, dim=-1)
                cached_topk_vals.append(topk_vals)
                cached_topk_ids.append(topk_ids)
                cached_input_ids.append(input_ids[0].cpu())
            del out, logits
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t_cache
                print(f"    [{i+1}/{len(samples)}] {elapsed:.0f}s", flush=True)

        torch.save({
            "topk_vals": cached_topk_vals,
            "topk_ids": cached_topk_ids,
            "input_ids": cached_input_ids,
        }, logits_path)
        vol.commit()
        print(f"  Cached {len(samples)} samples in {time.time()-t_cache:.0f}s", flush=True)

        del teacher
        gc.collect()
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════
    # Phase 2: Load student, swap to BitLinear, train
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("[Phase 2] Loading Qwen3-0.6B for 1-bit QAT")
    print(f"{'=' * 60}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cpu", token=hf_token,
    )

    # Swap to BitLinear, save base signs for forensic tracking
    layer_categories = {}
    base_signs = {}
    base_scales = {}  # store teacher scale magnitudes for ratio_med
    swapped = 0

    print("  Swapping to BitLinear...", flush=True)
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or module.weight.numel() < 128:
            continue
        parts = name.split('.')
        parent = model
        for p in parts[:-1]:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        attr = parts[-1]

        # Save base model signs + scale magnitudes for forensics
        base_signs[name] = module.weight.detach().sign().cpu()
        flat = module.weight.detach().reshape(-1)
        gs = 128
        rem = flat.numel() % gs
        if rem:
            flat = F.pad(flat, (0, gs - rem))
        base_scales[name] = flat.reshape(-1, gs).abs().mean(dim=1).cpu()

        bit = BitLinear.from_linear(module, group_size=128)
        swapped += 1

        if attr.isdigit():
            parent[int(attr)] = bit
        else:
            setattr(parent, attr, bit)
        layer_categories[name] = classify_layer(name)

    gc.collect()
    print(f"  {swapped} layers swapped to BitLinear", flush=True)

    # Category counts
    cat_counts = {}
    for cat in layer_categories.values():
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    print(f"  Categories: {cat_counts}", flush=True)

    # Set up training: weights + scales trainable
    weight_params = []
    scale_params = []
    trainable = 0
    frozen = 0

    for name, param in model.named_parameters():
        param.requires_grad = False
        frozen += param.numel()

    for name, mod in model.named_modules():
        if isinstance(mod, BitLinear):
            mod.weight.requires_grad = True
            weight_params.append(mod.weight)
            trainable += mod.weight.numel()
            frozen -= mod.weight.numel()
            if hasattr(mod, 'scale'):
                mod.scale.requires_grad = True
                scale_params.append(mod.scale)
                trainable += mod.scale.numel()
                frozen -= mod.scale.numel()

    print(f"  Trainable: {trainable:,} | Frozen: {frozen:,}", flush=True)

    model = model.bfloat16()
    model.to(device)
    try:
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing: ON", flush=True)
    except Exception:
        print("  Gradient checkpointing: not available", flush=True)

    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM after loading: {vram:.1f} GB", flush=True)

    # Optimizer: 8-bit Adam, dual LR
    WEIGHT_LR = 5e-4
    SCALE_LR = 1e-2
    optimizer = bnb.optim.Adam8bit([
        {"params": weight_params, "lr": WEIGHT_LR},
        {"params": scale_params, "lr": SCALE_LR},
    ])

    n_samples = len(cached_topk_vals)
    n_epochs = 10
    temperature = 2.0
    log_interval = 10

    print(f"\n  Training config:")
    print(f"    Samples: {n_samples}")
    print(f"    Epochs: {n_epochs}")
    print(f"    Temperature: {temperature}")
    print(f"    Weight LR: {WEIGHT_LR}, Scale LR: {SCALE_LR}")
    print(f"    STE: bf16 | Top-K: {TOP_K}", flush=True)

    # ── Forensic metrics ──
    def compute_forensic_metrics():
        metrics = {}
        for name, mod in model.named_modules():
            if not isinstance(mod, BitLinear) or mod._baked:
                continue
            cat = layer_categories.get(name, "other")
            if cat not in metrics:
                metrics[cat] = {"bit_match": [], "ratio_med": [], "kurtosis": []}

            with torch.no_grad():
                w = mod.weight.detach().cpu().float()
                current_signs = w.sign()
                original_signs = base_signs.get(name)
                if original_signs is not None:
                    match = (current_signs == original_signs).float().mean().item()
                    metrics[cat]["bit_match"].append(match)

                # ratio_med: current scale / base scale (Archie's definition)
                current_scale = mod.scale.detach().cpu().float().abs()
                orig_scale = base_scales.get(name)
                if orig_scale is not None:
                    ratios = current_scale[:orig_scale.shape[0]] / (orig_scale + 1e-8)
                    metrics[cat]["ratio_med"].append(ratios.median().item())

                # kurtosis of weight distribution
                flat = w.reshape(-1)
                if flat.numel() > 1000:
                    mean = flat.mean()
                    std = flat.std()
                    if std > 1e-8:
                        kurt = ((flat - mean) / std).pow(4).mean().item() - 3.0
                        metrics[cat]["kurtosis"].append(kurt)

        result = {}
        for cat, vals in metrics.items():
            result[cat] = {}
            for metric, arr in vals.items():
                if arr:
                    result[cat][metric] = round(np.mean(arr), 4)
        return result

    # ── Pre-training forensic snapshot ──
    print(f"\n  Forensic metrics (pre-training):", flush=True)
    initial_metrics = compute_forensic_metrics()
    for cat in ["ffn_up", "ffn_gate", "attn_v", "ffn_down", "attn_k", "attn_q", "embed"]:
        if cat in initial_metrics:
            m = initial_metrics[cat]
            print(f"    {cat:10s}: bit_match={m.get('bit_match','?'):>6} "
                  f"ratio_med={m.get('ratio_med','?'):>8} "
                  f"kurtosis={m.get('kurtosis','?'):>8}", flush=True)

    # ── Training loop ──
    print(f"\n  Starting QAT training...", flush=True)
    t_train = time.time()
    step = 0
    losses = []
    ooms = 0

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        indices = torch.randperm(n_samples).tolist()

        for idx in indices:
            topk_vals = cached_topk_vals[idx]
            topk_ids = cached_topk_ids[idx]
            t_input_ids = cached_input_ids[idx]
            seq_len = t_input_ids.shape[0]

            if seq_len > 256:
                continue

            try:
                input_ids = t_input_ids.unsqueeze(0).to(device)
                outputs = model(input_ids=input_ids)
                student_logits = outputs.logits[0]

                s_at_topk = student_logits.gather(1, topk_ids.to(device).long())
                t_at_topk = topk_vals.to(device).float()

                t_probs = F.softmax(t_at_topk / temperature, dim=-1)
                s_log_probs = F.log_softmax(s_at_topk.float() / temperature, dim=-1)
                loss = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (temperature ** 2)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # Clamp weights to ±1.5*scale — prevents explosion,
                # keeps train/eval gap small, allows sign flips near zero
                for mod in model.modules():
                    if isinstance(mod, BitLinear) and not mod._baked:
                        mod.clamp_weights(max_ratio=1.5)

                loss_val = loss.item()
                epoch_loss += loss_val
                epoch_steps += 1
                step += 1

                if step % log_interval == 0:
                    avg = epoch_loss / epoch_steps
                    vram_now = torch.cuda.memory_allocated() / 1e9
                    vram_peak = torch.cuda.max_memory_allocated() / 1e9
                    elapsed = time.time() - t_train
                    print(f"    Step {step} | Loss: {loss_val:.4f} | Avg: {avg:.4f} | "
                          f"VRAM: {vram_now:.1f}/{vram_peak:.1f}GB | {elapsed:.0f}s", flush=True)

                losses.append({"step": step, "epoch": epoch, "loss": round(loss_val, 6)})

                del outputs, student_logits, s_at_topk, t_at_topk, t_probs, s_log_probs, loss
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                ooms += 1
                print(f"    OOM at step {step}, seq_len={seq_len} (total: {ooms})", flush=True)
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue
            except Exception as e:
                print(f"    Error at step {step}: {e}", flush=True)
                continue

        avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else float('nan')
        print(f"\n  Epoch {epoch+1}/{n_epochs}: avg loss = {avg_epoch_loss:.4f}, "
              f"steps = {epoch_steps}, OOMs = {ooms}", flush=True)

        # Forensic metrics each epoch
        print(f"\n  Forensic metrics (epoch {epoch+1}):", flush=True)
        epoch_metrics = compute_forensic_metrics()
        for cat in ["ffn_up", "ffn_gate", "attn_v", "ffn_down", "attn_k", "attn_q", "embed"]:
            if cat in epoch_metrics:
                m = epoch_metrics[cat]
                print(f"    {cat:10s}: bit_match={m.get('bit_match','?'):>6} "
                      f"ratio_med={m.get('ratio_med','?'):>8} "
                      f"kurtosis={m.get('kurtosis','?'):>8}", flush=True)

        # Save checkpoint
        ckpt_path = f"/data/qwen06b_1bit_epoch{epoch+1}.pt"
        ckpt = {}
        for name, mod in model.named_modules():
            if isinstance(mod, BitLinear) and hasattr(mod, 'scale'):
                ckpt[name] = {
                    "weight": mod.weight.detach().cpu(),
                    "scales": mod.scale.detach().cpu(),
                }
        torch.save(ckpt, ckpt_path)
        vol.commit()
        print(f"  Checkpoint saved: {ckpt_path}", flush=True)

    train_time = time.time() - t_train
    print(f"\n  Training done: {step} steps, {train_time:.0f}s, {ooms} OOMs", flush=True)

    # ══════════════════════════════════════════════════════════
    # Phase 3: Inference test
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("[Phase 3] Post-QAT inference test")
    print(f"{'=' * 60}", flush=True)

    model.eval()
    for name, mod in model.named_modules():
        if isinstance(mod, BitLinear) and not mod._baked:
            mod.bake_for_inference()

    gc.collect()
    torch.cuda.empty_cache()
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM after baking: {vram:.1f} GB", flush=True)

    test_prompts = [
        "What is the capital of France?",
        "Explain photosynthesis in simple terms.",
        "Write a short poem about the ocean.",
        "What are three important safety considerations when driving a truck?",
        "If a car is traveling at 60 mph, how far does it travel in 2.5 hours?",
        "Describe what you would see looking out the front window of a truck on a highway.",
        "What is machine learning?",
        "List the planets in our solar system in order.",
    ]

    results = []
    for i, prompt in enumerate(test_prompts):
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            t_start = time.time()
            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=100, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen_time = time.time() - t_start
            gen_tokens = output[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(gen_tokens, skip_special_tokens=True)

            print(f"\n  Q: {prompt}")
            print(f"  A: {response[:200]}")
            print(f"  ({gen_time:.1f}s)", flush=True)

            results.append({
                "question": prompt,
                "response": response,
                "gen_time_sec": round(gen_time, 2),
            })
        except Exception as e:
            print(f"  ERROR on '{prompt[:50]}': {e}", flush=True)
            results.append({"question": prompt, "error": str(e)})

    # ── Final summary ──
    final_metrics = compute_forensic_metrics()

    output = {
        "model_id": model_id,
        "format": "Q1_0_g128 + Bonsai-recipe QAT",
        "training": {
            "n_samples": n_samples,
            "n_epochs": n_epochs,
            "total_steps": step,
            "total_ooms": ooms,
            "train_time_sec": round(train_time, 1),
            "final_avg_loss": round(losses[-1]["loss"], 6) if losses else None,
            "weight_lr": WEIGHT_LR,
            "scale_lr": SCALE_LR,
        },
        "forensic_metrics": final_metrics,
        "initial_metrics": initial_metrics,
        "loss_curve": losses[::max(1, len(losses)//50)],
        "inference_results": results,
    }
    with open("/data/qwen06b_1bit_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    vol.commit()

    print(f"\n{'=' * 60}")
    print("QWEN3-0.6B 1-BIT QAT COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Steps: {step}, Time: {train_time:.0f}s, OOMs: {ooms}")

    # Print forensic comparison
    print(f"\n  Forensic metrics (initial → final):")
    for cat in ["ffn_up", "ffn_gate", "attn_v", "ffn_down", "attn_k", "attn_q"]:
        if cat in final_metrics and cat in initial_metrics:
            i_m = initial_metrics[cat]
            f_m = final_metrics[cat]
            print(f"    {cat:10s}: bit_match {i_m.get('bit_match','?'):>6} → {f_m.get('bit_match','?'):>6} "
                  f"| ratio_med {i_m.get('ratio_med','?'):>6} → {f_m.get('ratio_med','?'):>6} "
                  f"| kurtosis {i_m.get('kurtosis','?'):>7} → {f_m.get('kurtosis','?'):>7}", flush=True)

    coherent = len([r for r in results if "response" in r])
    print(f"\n  Inference: {coherent}/{len(results)} successful")
    for r in results[:4]:
        if "response" in r:
            print(f"\n  Q: {r['question']}")
            print(f"  A: {r['response'][:150]}")

    return output


@app.local_entrypoint()
def main():
    print("1-bit Bonsai QAT — Qwen3-0.6B (fast iteration)\n")
    result = run_qat.remote()
    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("loss_curve", "inference_results")}, indent=2))
    if "inference_results" in result:
        for r in result["inference_results"][:4]:
            if "response" in r:
                print(f"\n  Q: {r['question']}")
                print(f"  A: {r['response'][:150]}")
