"""Bonsai 8B Scale Personality Test — Native 1-bit Loading

NO fp16 conversion. Load weights directly as signs + scales.
Model VRAM: ~2GB (not 16GB). Leaves room for training ALL layers.

LoRA-style scale deltas: original_scale + alpha * learned_delta
"""
import modal
import os

app = modal.App("bonsai8b-personality")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "accelerate", "numpy",
                 "huggingface_hub", "safetensors", "datasets")
)


@app.function(image=image, gpu="T4", timeout=3600,
              secrets=[modal.Secret.from_name("huggingface-secret")])
def test():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    import time
    import math
    import gc
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    from datasets import load_dataset

    DEVICE = "cuda"
    GROUP_SIZE = 128
    ALPHA = 0.05  # LoRA-style: original + alpha * delta
    random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("BONSAI 8B: Scale Personality (Native 1-bit, LoRA deltas)")
    print("=" * 60, flush=True)

    hf_token = os.environ.get("HF_TOKEN")

    # ─── Load model skeleton in fp16 but SMALL memory footprint ───
    # We load the model structure, then replace weights with 1-bit
    print("\nLoading Qwen3-8B skeleton...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with low_cpu_mem_usage to minimize peak VRAM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", dtype=torch.float16, device_map="cpu",
        low_cpu_mem_usage=True, token=hf_token
    )
    print(f"  Skeleton loaded in {time.time()-t0:.1f}s (CPU)", flush=True)

    n_layers = len(model.model.layers)
    print(f"  {n_layers} layers, hidden=4096, FFN=12288", flush=True)

    # ─── Load Bonsai 8B weights as 1-bit ───
    print("\n  Loading Bonsai 8B weights...", flush=True)
    bonsai_path = snapshot_download("prism-ml/Bonsai-8B-unpacked", token=hf_token)

    # Extract signs and scales from Bonsai weights
    signs_map = {}      # name -> packed_signs (int32, CPU)
    scales_map = {}     # name -> group_scales (fp16, CPU)
    linear_names = []

    # Process each shard
    for shard_idx in range(1, 5):
        shard_file = f"{bonsai_path}/model-0000{shard_idx}-of-00004.safetensors"
        print(f"    Processing shard {shard_idx}/4...", flush=True)
        with safe_open(shard_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                if not key.endswith(".weight"):
                    continue
                if "layers." not in key:
                    continue

                tensor = f.get_tensor(key)
                if tensor.dim() != 2:
                    continue
                name = key.replace(".weight", "")
                out_f, in_f = tensor.shape

                # Extract signs
                signs = tensor.sign()
                signs[signs == 0] = 1

                # Pack signs into int32
                bits = (signs > 0).to(torch.int32)
                pad = (32 - in_f % 32) % 32
                if pad > 0:
                    bits = F.pad(bits, (0, pad))
                n_ints = bits.shape[1] // 32
                bits_r = bits.view(out_f, n_ints, 32)
                packed = torch.zeros(out_f, n_ints, dtype=torch.int32)
                for bit in range(32):
                    packed |= bits_r[:, :, bit] << bit

                # Extract scales (per-group absmean)
                pad_s = (GROUP_SIZE - in_f % GROUP_SIZE) % GROUP_SIZE
                w_abs = F.pad(tensor.abs(), (0, pad_s)) if pad_s > 0 else tensor.abs()
                n_groups = w_abs.shape[1] // GROUP_SIZE
                scales = w_abs.view(out_f, n_groups, GROUP_SIZE).mean(dim=2).half()

                signs_map[name] = packed
                scales_map[name] = scales
                linear_names.append(name)

                del tensor, signs, bits, packed, scales
                gc.collect()

    print(f"  Extracted {len(signs_map)} linear layers as 1-bit", flush=True)

    # ─── Apply 1-bit weights to model and move to GPU ───
    def unpack_and_apply(name, module, scales_override=None):
        packed = signs_map[name]
        scales = scales_override if scales_override is not None else scales_map[name]
        out_f = packed.shape[0]
        in_f = module.weight.shape[1]

        signs = torch.zeros(out_f, in_f, dtype=torch.float16)
        for bit in range(32):
            col_indices = torch.arange(bit, in_f, 32)
            if len(col_indices) == 0: continue
            n_valid = min(len(col_indices), packed.shape[1])
            extracted = ((packed[:, :n_valid] >> bit) & 1).to(torch.float16)
            signs[:, col_indices[:n_valid]] = extracted * 2 - 1

        n_groups = scales.shape[1]
        se = scales.unsqueeze(2).expand(-1, -1, GROUP_SIZE)
        sf = se.reshape(out_f, n_groups * GROUP_SIZE)[:, :in_f]
        module.weight.data = (signs * sf).to(DEVICE)

    print("  Applying 1-bit weights on CPU, then moving to GPU...", flush=True)

    # Apply 1-bit weights while model is on CPU (avoids 16GB GPU spike)
    for name in linear_names:
        parts = name.split('.')
        module = model
        for p in parts:
            module = getattr(module, p)
        # Apply on CPU
        packed = signs_map[name]
        scales = scales_map[name]
        out_f = packed.shape[0]
        in_f = module.weight.shape[1]
        signs = torch.zeros(out_f, in_f, dtype=torch.float16)
        for bit in range(32):
            col_indices = torch.arange(bit, in_f, 32)
            if len(col_indices) == 0: continue
            n_valid = min(len(col_indices), packed.shape[1])
            extracted = ((packed[:, :n_valid] >> bit) & 1).to(torch.float16)
            signs[:, col_indices[:n_valid]] = extracted * 2 - 1
        n_groups = scales.shape[1]
        se = scales.unsqueeze(2).expand(-1, -1, GROUP_SIZE)
        sf = se.reshape(out_f, n_groups * GROUP_SIZE)[:, :in_f]
        module.weight.data = (signs * sf)  # Stay on CPU

    # Move to GPU — weights are still fp16 tensors (~16GB)
    # Use device_map auto to shard across GPU + CPU if needed
    gc.collect()
    model = model.to(DEVICE)

    torch.cuda.empty_cache()
    gc.collect()
    model.eval()

    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"  VRAM used: {vram:.2f}GB", flush=True)

    # ─── PPL + eval ───
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    wiki_texts = [t for t in wiki["text"] if len(t.strip()) > 100]
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    math_texts = [f"Q: {gsm[i]['question']}\nA: {gsm[i]['answer']}" for i in range(200)]

    def compute_ppl(texts, n=20):
        model.eval()
        tl, tt = 0, 0
        for text in texts[:n]:
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).input_ids.to(DEVICE)
            if ids.shape[1] < 10: continue
            with torch.no_grad():
                out = model(ids, labels=ids)
                if out.loss is not None and not torch.isnan(out.loss):
                    tl += out.loss.item() * ids.shape[1]
                    tt += ids.shape[1]
        return math.exp(tl / max(tt, 1))

    # ─── Baseline ───
    print(f"\n{'='*60}")
    print("BASELINE")
    print(f"{'='*60}", flush=True)
    base_math = compute_ppl(math_texts)
    base_lang = compute_ppl(wiki_texts)
    print(f"  Math PPL: {base_math:.2f}  Lang PPL: {base_lang:.2f}", flush=True)

    # ─── Train scale personality with LoRA-style deltas ───
    def train_personality(name, texts, epochs=10, lr=1e-3):
        print(f"\n  Training {name} personality ({epochs} epochs, alpha={ALPHA})...", flush=True)

        # Create delta parameters for ALL layers
        deltas = {}
        param_list = []
        for lin_name in linear_names:
            d = nn.Parameter(torch.zeros_like(scales_map[lin_name].float()).to(DEVICE))
            deltas[lin_name] = d
            param_list.append(d)

        optimizer = torch.optim.Adam(param_list, lr=lr)

        for epoch in range(epochs):
            total_loss, n = 0, 0
            random.shuffle(texts)
            for text in texts[:20]:
                ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=48).input_ids.to(DEVICE)
                if ids.shape[1] < 5: continue
                torch.cuda.empty_cache()

                # Apply current scales: original + alpha * delta
                for lin_name in linear_names:
                    parts = lin_name.split('.')
                    module = model
                    for p in parts:
                        module = getattr(module, p)
                    new_scales = (scales_map[lin_name].to(DEVICE) + ALPHA * deltas[lin_name]).half().clamp(min=1e-6)
                    unpack_and_apply(lin_name, module, new_scales)

                model.train()
                out = model(ids, labels=ids)
                if out.loss is None or torch.isnan(out.loss): continue

                optimizer.zero_grad()
                out.loss.backward()

                # Manual delta gradients from weight gradients
                for lin_name in linear_names:
                    parts = lin_name.split('.')
                    module = model
                    for p in parts:
                        module = getattr(module, p)
                    if module.weight.grad is None: continue

                    packed = signs_map[lin_name]
                    out_f = packed.shape[0]
                    in_f = module.weight.shape[1]
                    signs = torch.zeros(out_f, in_f, dtype=torch.float16, device=DEVICE)
                    for bit in range(32):
                        col_indices = torch.arange(bit, in_f, 32, device=DEVICE)
                        if len(col_indices) == 0: continue
                        n_valid = min(len(col_indices), packed.shape[1])
                        extracted = ((packed[:, :n_valid].to(DEVICE) >> bit) & 1).to(torch.float16)
                        signs[:, col_indices[:n_valid]] = extracted * 2 - 1

                    gts = module.weight.grad.float() * signs.float()
                    n_groups = deltas[lin_name].shape[1]
                    pad = (GROUP_SIZE - in_f % GROUP_SIZE) % GROUP_SIZE
                    if pad > 0: gts = F.pad(gts, (0, pad))
                    gpg = gts.view(out_f, n_groups, GROUP_SIZE).mean(dim=2)
                    deltas[lin_name].grad = (gpg * ALPHA).to(deltas[lin_name].dtype)

                torch.nn.utils.clip_grad_norm_(param_list, 1.0)
                optimizer.step()
                total_loss += out.loss.item()
                n += 1
                del ids, out
                torch.cuda.empty_cache()

            print(f"    Epoch {epoch+1}: CE={total_loss/max(n,1):.4f}", flush=True)

        # Return final scales
        result = {}
        for lin_name in linear_names:
            result[lin_name] = (scales_map[lin_name].to(DEVICE) + ALPHA * deltas[lin_name].detach()).half().clamp(min=1e-6).cpu()
        del deltas, param_list, optimizer
        torch.cuda.empty_cache()
        return result

    # ─── Train personalities ───
    print(f"\n{'='*60}")
    print("MATH PERSONALITY")
    print(f"{'='*60}", flush=True)
    math_scales = train_personality("MATH", math_texts)

    print(f"\n{'='*60}")
    print("LANG PERSONALITY")
    print(f"{'='*60}", flush=True)
    lang_scales = train_personality("LANG", wiki_texts)

    # ─── SWAP TEST ───
    print(f"\n{'='*60}")
    print("SCALE SWAP TEST")
    print(f"{'='*60}", flush=True)

    def apply_all_scales(scale_table):
        for lin_name in linear_names:
            parts = lin_name.split('.')
            module = model
            for p in parts:
                module = getattr(module, p)
            unpack_and_apply(lin_name, module, scale_table[lin_name].to(DEVICE))

    # Original
    apply_all_scales(scales_map)
    orig_m = compute_ppl(math_texts)
    orig_l = compute_ppl(wiki_texts)
    print(f"  Original: Math={orig_m:.2f}  Lang={orig_l:.2f}", flush=True)

    # Math personality
    apply_all_scales(math_scales)
    math_m = compute_ppl(math_texts)
    math_l = compute_ppl(wiki_texts)
    print(f"  Math:     Math={math_m:.2f}  Lang={math_l:.2f}", flush=True)

    # Lang personality
    apply_all_scales(lang_scales)
    lang_m = compute_ppl(math_texts)
    lang_l = compute_ppl(wiki_texts)
    print(f"  Lang:     Math={lang_m:.2f}  Lang={lang_l:.2f}", flush=True)

    # ─── Results ───
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  {'Table':<12} {'Math PPL':>10} {'Lang PPL':>10}")
    print(f"  {'-'*34}")
    print(f"  {'Original':<12} {orig_m:>10.2f} {orig_l:>10.2f}")
    print(f"  {'Math':<12} {math_m:>10.2f} {math_l:>10.2f}")
    print(f"  {'Lang':<12} {lang_m:>10.2f} {lang_l:>10.2f}")

    math_wins = math_m < lang_m
    lang_wins = lang_l < math_l
    math_improves = math_m < orig_m
    lang_improves = lang_l < orig_l

    print(f"\n  Math personality better at math than lang: {'YES' if math_wins else 'NO'}")
    print(f"  Lang personality better at lang than math: {'YES' if lang_wins else 'NO'}")
    print(f"  Math personality improves over original:   {'YES' if math_improves else 'NO'}")
    print(f"  Lang personality improves over original:   {'YES' if lang_improves else 'NO'}")

    if math_wins and lang_wins:
        print(f"\n  SCALE PERSONALITIES WORK ON 8B!")
    elif math_wins or lang_wins:
        print(f"\n  PARTIAL — one direction differentiates")
    else:
        print(f"\n  NO DIFFERENTIATION at 8B")


@app.local_entrypoint()
def main():
    test.remote()
