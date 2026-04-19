"""T4 Burst — three concurrent experiments on separate T4 GPUs.

  1. Scale math v2 + GSM8K n=500  — validate 28% headline at larger n
  2. LoRA rank=16 + GSM8K n=500   — matched-size baseline at larger n
  3. STE sign QAT K=15% + GSM8K n=100  — clean sign capacity test (Exp 24)

All three spawn concurrently. Wall clock ~2hr, total cost ~$1.50-2.

Run:   modal run experiments/scale-personalities/modal_t4_burst.py
Logs:  modal app logs <app-id>
"""
import modal

app = modal.App("1bit-t4-burst")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.51.0",
        "accelerate>=1.5.0",
        "huggingface_hub>=0.28.0",
        "datasets>=3.0.0",
        "numpy>=1.26.0",
    )
)

vol = modal.Volume.from_name("1bit-t4-burst-results", create_if_missing=True)

BONSAI = "prism-ml/Bonsai-1.7B-unpacked"
TRAIN_EXAMPLES = 150
DOMAIN_MIX = 0.7
MAX_LEN = 128
EPOCHS = 3
LR = 1e-4
REG_LAMBDA = 0.1
TOKEN_SELECT_RATIO = 0.6
GROUP_SIZE = 128
N_EVAL_MAIN = 200
N_EVAL_SIGN = 100
K_SIGN_PCT = 15.0
LORA_RANK = 16
LORA_ALPHA = 16


# ── Shared code injected into each remote function ────────────────────────────

PACKED_BITLINEAR_SRC = '''
import torch, torch.nn as nn, torch.nn.functional as F

class _BitLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, signs, scales, out_features, in_features, group_size):
        scales_exp = scales.unsqueeze(1).expand(-1, group_size).reshape(out_features, in_features)
        weight = signs * scales_exp.half()
        output = F.linear(x, weight)
        ctx.save_for_backward(x, signs, scales)
        ctx.out_features = out_features
        ctx.in_features = in_features
        ctx.group_size = group_size
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, signs, scales = ctx.saved_tensors
        out_f, in_f, gs = ctx.out_features, ctx.in_features, ctx.group_size
        scales_exp = scales.unsqueeze(1).expand(-1, gs).reshape(out_f, in_f)
        weight = signs * scales_exp.half()
        grad_x = grad_output.matmul(weight.to(grad_output.dtype))
        grad_w = grad_output.reshape(-1, out_f).t().matmul(x.reshape(-1, in_f))
        grad_scales = (grad_w * signs).reshape(-1, gs).sum(dim=1).float()
        return grad_x, None, grad_scales, None, None, None

class PackedBitLinear(nn.Module):
    def __init__(self, weight, group_size=128):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.group_size = group_size
        signs = weight.sign().half()
        signs[signs == 0] = 1
        self.register_buffer("signs", signs)
        w_flat = weight.reshape(-1, group_size)
        scales = w_flat.abs().mean(dim=1)
        self.scales = nn.Parameter(scales.float())
        self.register_buffer("_original_scales", scales.clone())

    def forward(self, x):
        return _BitLinearFn.apply(x, self.signs, self.scales,
                                   self.out_features, self.in_features, self.group_size)

def convert_model(model, group_size=128, skip_names=("lm_head",)):
    converted = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear): continue
        if any(s in name for s in skip_names): continue
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = dict(model.named_modules())[parent_name] if parent_name else model
        setattr(parent, child_name, PackedBitLinear(module.weight.data, group_size))
        converted += 1
    return converted
'''


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _get_data():
    from datasets import load_dataset
    import numpy as np
    math_texts, diverse_texts = [], []
    ds = load_dataset("gsm8k", "main", split="train", streaming=True)
    for ex in ds:
        math_texts.append(f"Question: {ex['question']}\nAnswer: {ex['answer']}")
        if len(math_texts) >= TRAIN_EXAMPLES * 2: break
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
    for ex in ds:
        if len(ex['text'].strip()) > 200:
            diverse_texts.append(ex['text'])
        if len(diverse_texts) >= TRAIN_EXAMPLES: break
    rng = np.random.default_rng(42)
    n_math = int(TRAIN_EXAMPLES * DOMAIN_MIX)
    mixed = math_texts[:n_math] + diverse_texts[:TRAIN_EXAMPLES - n_math]
    return [mixed[i] for i in rng.permutation(len(mixed))]


def _eval_gsm8k(model, tokenizer, n, device, log):
    import re, torch
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test", streaming=True)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for ex in ds:
            if total >= n: break
            prompt = f"Question: {ex['question']}\nAnswer:"
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model.generate(
                    **toks, max_new_tokens=200,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id)
            resp = tokenizer.decode(out[0][toks.input_ids.shape[1]:],
                                    skip_special_tokens=True)
            pred = re.findall(r"-?\d+\.?\d*", resp.replace(",", ""))
            gold = re.findall(r"-?\d+\.?\d*",
                              ex["answer"].split("####")[-1].replace(",", ""))
            try:
                if pred and gold and abs(float(pred[-1]) - float(gold[-1])) < 0.01:
                    correct += 1
            except ValueError:
                pass
            total += 1
            if total % 25 == 0:
                log(f"  [{total}/{n}] acc={correct/total:.1%}")
    return correct / max(total, 1)


# ── Function 1: Scale math v2 + GSM8K n=500 ───────────────────────────────────

@app.function(
    image=image, gpu="T4", timeout=14400,
    volumes={"/results": vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_scale_eval():
    import sys, time, json, types
    import torch, torch.nn as nn, torch.nn.functional as F

    t0 = time.time()
    DEVICE = "cuda"

    def log(msg): print(f"{time.strftime('%H:%M:%S')} | [SCALE] {msg}", flush=True)

    log("Scale math v2 + GSM8K n=500")

    pbl = types.ModuleType("packed_bitlinear")
    exec(PACKED_BITLINEAR_SRC, pbl.__dict__)
    sys.modules["packed_bitlinear"] = pbl
    PackedBitLinear = pbl.PackedBitLinear
    convert_model   = pbl.convert_model

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BONSAI, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BONSAI, dtype=torch.float16, trust_remote_code=True,
        device_map="cpu", attn_implementation="eager")
    convert_model(model)
    model = model.to(DEVICE)
    log(f"Model loaded. VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Freeze everything except scales
    for p in model.parameters():
        p.requires_grad_(False)
    scale_params, orig_snapshots = [], {}
    for name, module in model.named_modules():
        if isinstance(module, PackedBitLinear):
            module.scales.requires_grad_(True)
            scale_params.append(module.scales)
            orig_snapshots[name] = module.scales.data.clone()

    log(f"Scale params: {sum(p.numel() for p in scale_params):,}")

    texts = _get_data()
    log(f"Data: {len(texts)} examples")

    optimizer = torch.optim.AdamW(scale_params, lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, EPOCHS * len(texts)))

    model.train()
    for epoch in range(EPOCHS):
        total_loss, n = 0.0, 0
        for text in texts:
            toks = tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=MAX_LEN).to(DEVICE)
            if toks.input_ids.shape[1] < 10: continue
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(**toks)
                logits = out.logits[:, :-1]
                labels = toks.input_ids[:, 1:]
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    labels.reshape(-1), reduction="none")
                threshold = torch.quantile(ce, 1.0 - TOKEN_SELECT_RATIO)
                mask = (ce >= threshold).float()
                task_loss = (ce * mask).sum() / (mask.sum() + 1e-8)
                reg = sum(F.mse_loss(m.scales, orig_snapshots[nm])
                          for nm, m in model.named_modules()
                          if isinstance(m, PackedBitLinear))
                loss = task_loss + REG_LAMBDA * reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scale_params, 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += task_loss.item(); n += 1
        log(f"  Epoch {epoch+1}/{EPOCHS}: loss={total_loss/max(n,1):.4f}")

    log(f"Eval n={N_EVAL_MAIN}...")
    acc = _eval_gsm8k(model, tokenizer, N_EVAL_MAIN, DEVICE, log)

    log(f"RESULT: {acc:.1%} GSM8K (n={N_EVAL_MAIN})  runtime={( time.time()-t0)/60:.1f}min")
    result = {"experiment": "scale_math_v2", "gsm8k": float(acc),
              "n_eval": N_EVAL_MAIN, "runtime_min": (time.time()-t0)/60}
    with open("/results/scale_eval_n500.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    return result


# ── Function 2: LoRA rank=16 + GSM8K n=500 ────────────────────────────────────

@app.function(
    image=image, gpu="T4", timeout=14400,
    volumes={"/results": vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_lora_baseline():
    import sys, time, json, types
    import torch, torch.nn as nn, torch.nn.functional as F

    t0 = time.time()
    DEVICE = "cuda"

    def log(msg): print(f"{time.strftime('%H:%M:%S')} | [LORA] {msg}", flush=True)

    log(f"LoRA rank={LORA_RANK} + GSM8K n={N_EVAL_MAIN}")

    pbl = types.ModuleType("packed_bitlinear")
    exec(PACKED_BITLINEAR_SRC, pbl.__dict__)
    sys.modules["packed_bitlinear"] = pbl
    PackedBitLinear = pbl.PackedBitLinear
    convert_model   = pbl.convert_model

    class LoRALinear(nn.Module):
        def __init__(self, base, rank, alpha):
            super().__init__()
            self.base = base
            dev = base.signs.device
            self.lora_A = nn.Parameter(
                torch.randn(rank, base.in_features, device=dev, dtype=torch.float32) * 0.02)
            self.lora_B = nn.Parameter(
                torch.zeros(base.out_features, rank, device=dev, dtype=torch.float32))
            self.scaling = alpha / rank

        def forward(self, x):
            return self.base(x) + F.linear(x, (self.lora_B @ self.lora_A).to(x.dtype)) * self.scaling

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BONSAI, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BONSAI, dtype=torch.float16, trust_remote_code=True,
        device_map="cpu", attn_implementation="eager")
    convert_model(model)
    model = model.to(DEVICE)

    # Wrap PackedBitLinear layers with LoRA
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, PackedBitLinear):
                setattr(parent, child_name, LoRALinear(child, LORA_RANK, LORA_ALPHA))

    for name, p in model.named_parameters():
        p.requires_grad_('lora_A' in name or 'lora_B' in name)

    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
    scale_params = sum(m.scales.numel() for m in model.modules() if isinstance(m, PackedBitLinear))
    lora_count = sum(p.numel() for p in lora_params)
    log(f"LoRA params: {lora_count:,} ({lora_count*4/1e6:.1f}MB)  "
        f"vs scales: {scale_params:,} ({scale_params*2/1e6:.1f}MB fp16)")
    log(f"VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB")

    texts = _get_data()
    optimizer = torch.optim.AdamW(lora_params, lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, EPOCHS * len(texts)))

    model.train()
    for epoch in range(EPOCHS):
        total_loss, n = 0.0, 0
        for text in texts:
            toks = tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=MAX_LEN).to(DEVICE)
            if toks.input_ids.shape[1] < 10: continue
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(**toks)
                logits = out.logits[:, :-1]
                labels = toks.input_ids[:, 1:]
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    labels.reshape(-1), reduction="none")
                threshold = torch.quantile(ce, 1.0 - TOKEN_SELECT_RATIO)
                mask = (ce >= threshold).float()
                reg = sum(F.mse_loss(p, torch.zeros_like(p)) for p in lora_params)
                loss = (ce * mask).sum() / (mask.sum() + 1e-8) + REG_LAMBDA * reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += loss.item(); n += 1
        log(f"  Epoch {epoch+1}/{EPOCHS}: loss={total_loss/max(n,1):.4f}")

    log(f"Eval n={N_EVAL_MAIN}...")
    acc = _eval_gsm8k(model, tokenizer, N_EVAL_MAIN, DEVICE, log)

    log(f"RESULT: {acc:.1%} GSM8K (n={N_EVAL_MAIN})  runtime={(time.time()-t0)/60:.1f}min")
    result = {"experiment": "lora_rank16", "gsm8k": float(acc),
              "n_eval": N_EVAL_MAIN, "lora_params": lora_count,
              "lora_mb": lora_count*4/1e6, "runtime_min": (time.time()-t0)/60}
    with open("/results/lora_n500.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    return result


# ── Function 3: STE sign QAT K=15% + GSM8K n=100 ─────────────────────────────
# Fix vs first attempt: base stays fp16 (3.2GB), only K=15% selected elements
# stored as fp32 patches (~960MB). Total ~4.5GB vs 6.4GB+ that OOM'd on T4.

@app.function(
    image=image, gpu="T4", timeout=14400,
    volumes={"/results": vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_sign_qat():
    import sys, time, json, types
    import numpy as np
    import torch, torch.nn as nn, torch.nn.functional as F

    t0 = time.time()
    DEVICE = "cuda"

    def log(msg): print(f"{time.strftime('%H:%M:%S')} | [SIGN] {msg}", flush=True)

    log(f"STE sign QAT K={K_SIGN_PCT}% + GSM8K n={N_EVAL_SIGN}")
    log("Exp 24 — base fp16, fp32 patches only for selected K% elements")

    # ── STE autograd ──────────────────────────────────────────────────────────
    class _STESign(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x): return x.sign()
        @staticmethod
        def backward(ctx, grad): return grad

    def ste_sign(x): return _STESign.apply(x)

    # ── SignPatchWrapper: fp16 base + fp32 patches for selected groups only ────
    class SignPatchWrapper(nn.Module):
        """Wraps PackedBitLinear. Keeps fp16 signs as frozen base.
        Adds fp32 patch_signs only for selected group elements (~K% of total).
        Forward: base weight with STE override at selected positions via index_put.
        Memory: fp16 base (unchanged) + K%*total*4 bytes for patches.
        """
        def __init__(self, pbl, selected_flat_idx_cpu):
            super().__init__()
            self.out_features = pbl.out_features
            self.in_features = pbl.in_features
            self.group_size = pbl.group_size
            self.register_buffer("signs", pbl.signs)           # fp16, frozen
            self.register_buffer("scales", pbl.scales.data)    # fp32, frozen
            idx = torch.tensor(selected_flat_idx_cpu, dtype=torch.long)
            self.register_buffer("sel_idx", idx)
            # fp32 patch values initialized to ±0.1 — close to boundary so
            # small gradients cross zero. Sign direction preserved, STE
            # forward still gives ±1.0 since sign(±0.1) = ±1.
            orig = pbl.signs.float().reshape(-1)[idx]          # ±1.0 fp32
            init = orig * 0.1                                  # ±0.1 fp32
            self.patch_signs = nn.Parameter(init)
            self.register_buffer("orig_patch", orig.clone())   # keep ±1.0 for flip counting

        def commit_patches(self):
            """Write binarized patches into signs buffer. After this, forward
            uses the fast path (no index_put) — call before eval."""
            with torch.no_grad():
                binary = self.patch_signs.data.sign()
                binary[binary == 0] = 1.0
                signs_flat = self.signs.reshape(-1).float()
                signs_flat[self.sel_idx] = binary
                self.signs.copy_(signs_flat.reshape(self.signs.shape).half())
            self.patch_signs = None  # disable slow path

        def forward(self, x):
            gs, out_f, in_f = self.group_size, self.out_features, self.in_features
            scales_exp = self.scales.unsqueeze(1).expand(-1, gs).reshape(out_f, in_f)
            weight = self.signs * scales_exp.half()

            if self.patch_signs is not None:
                # Training path: STE override at selected positions via index_put
                binary = ste_sign(self.patch_signs)
                sel_scales = scales_exp.reshape(-1)[self.sel_idx]
                patch_vals = (binary * sel_scales).half()
                weight = weight.reshape(-1).index_put((self.sel_idx,), patch_vals).reshape(out_f, in_f)

            return F.linear(x, weight)

    # ── Load model with standard PackedBitLinear (fp16 signs) ─────────────────
    pbl = types.ModuleType("packed_bitlinear")
    exec(PACKED_BITLINEAR_SRC, pbl.__dict__)
    sys.modules["packed_bitlinear"] = pbl
    PackedBitLinear = pbl.PackedBitLinear
    convert_model   = pbl.convert_model

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BONSAI, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BONSAI, dtype=torch.float16, trust_remote_code=True,
        device_map="cpu", attn_implementation="eager")
    convert_model(model)
    model = model.to(DEVICE)
    log(f"Model loaded. VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB")

    total_signs = sum(m.signs.numel() for m in model.modules()
                      if isinstance(m, PackedBitLinear))
    total_groups = total_signs // GROUP_SIZE
    log(f"Total signs: {total_signs:,}  groups: {total_groups:,}")

    # ── Ranking: use scale gradients as group-importance proxy ────────────────
    log("Ranking pass (20 examples)...")
    for p in model.parameters(): p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, PackedBitLinear): m.scales.requires_grad_(True)

    texts = _get_data()
    opt_rank = torch.optim.SGD([m.scales for m in model.modules()
                                 if isinstance(m, PackedBitLinear)], lr=0.0)
    opt_rank.zero_grad()
    model.train()
    for text in texts[:20]:
        toks = tokenizer(text, return_tensors="pt", truncation=True,
                         max_length=MAX_LEN).to(DEVICE)
        if toks.input_ids.shape[1] < 10: continue
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(**toks)
            loss = F.cross_entropy(
                out.logits[:, :-1].reshape(-1, out.logits.size(-1)).float(),
                toks.input_ids[:, 1:].reshape(-1))
        loss.backward()

    layer_names, layer_imps, layer_sizes = [], [], []
    for name, module in model.named_modules():
        if not isinstance(module, PackedBitLinear): continue
        if module.scales.grad is None: continue
        imp = module.scales.grad.abs().cpu().float().numpy()
        layer_names.append(name)
        layer_imps.append(imp)
        layer_sizes.append(len(imp))

    for m in model.modules():
        if isinstance(m, PackedBitLinear): m.scales.requires_grad_(False)
    opt_rank.zero_grad()
    torch.cuda.empty_cache()

    # ── Select top K% groups ──────────────────────────────────────────────────
    all_imp = np.concatenate(layer_imps)
    k_groups = max(1, int(len(all_imp) * K_SIGN_PCT / 100.0))
    top_k_global = np.argpartition(all_imp, -k_groups)[-k_groups:]
    offsets = np.concatenate([[0], np.cumsum(layer_sizes[:-1])])

    selected = {}  # layer_name → sorted list of flat element indices within that layer
    for g_idx in top_k_global:
        layer_i = int(np.searchsorted(offsets, g_idx, side='right') - 1)
        name = layer_names[layer_i]
        g_local = int(g_idx - offsets[layer_i])
        start = g_local * GROUP_SIZE
        end = start + GROUP_SIZE
        selected.setdefault(name, []).extend(range(start, end))

    total_patch_signs = sum(len(v) for v in selected.values())
    log(f"K={K_SIGN_PCT}%: {k_groups:,} groups → {total_patch_signs:,} patch elements "
        f"({100*total_patch_signs/total_signs:.1f}% of all signs, "
        f"{total_patch_signs*4/1e6:.0f}MB fp32)")

    # ── Replace selected modules with SignPatchWrapper ─────────────────────────
    patch_params = []
    module_map = dict(model.named_modules())
    for name, flat_indices in selected.items():
        module = module_map[name]
        if not isinstance(module, PackedBitLinear): continue
        flat_indices = sorted(set(i for i in flat_indices
                                  if i < module.out_features * module.in_features))
        wrapper = SignPatchWrapper(module, flat_indices)
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = module_map[parent_name] if parent_name else model
        setattr(parent, child_name, wrapper)
        patch_params.append(wrapper.patch_signs)

    torch.cuda.empty_cache()
    log(f"Wrapped {len(patch_params)} modules. VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB")

    # ── Train ─────────────────────────────────────────────────────────────────
    # Adam: per-parameter gradient normalization handles the scale issue.
    # No grad clipping — clipping 211M params to norm=1.0 made each grad ~7e-7,
    # neutering the optimizer (0 flips in previous run).
    LR_SIGNS = 1e-3
    optimizer = torch.optim.Adam(patch_params, lr=LR_SIGNS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, EPOCHS * len(texts)))

    model.train()
    for epoch in range(EPOCHS):
        total_loss, n = 0.0, 0
        for text in texts:
            toks = tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=MAX_LEN).to(DEVICE)
            if toks.input_ids.shape[1] < 10: continue
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(**toks)
                logits = out.logits[:, :-1]
                labels = toks.input_ids[:, 1:]
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    labels.reshape(-1), reduction="none")
                threshold = torch.quantile(ce, 1.0 - TOKEN_SELECT_RATIO)
                mask_t = (ce >= threshold).float()
                loss = (ce * mask_t).sum() / (mask_t.sum() + 1e-8)
            loss.backward()
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += loss.item(); n += 1
            if n % 30 == 0: torch.cuda.empty_cache()
        log(f"  Epoch {epoch+1}/{EPOCHS}: loss={total_loss/max(n,1):.4f}  "
            f"VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB")

    # ── Count flips + rebinarize ──────────────────────────────────────────────
    total_flips = 0
    for module in model.modules():
        if not isinstance(module, SignPatchWrapper): continue
        with torch.no_grad():
            current = module.patch_signs.data.sign()
            current[current == 0] = 1.0
            flipped = (current != module.orig_patch).sum().item()
            total_flips += flipped
            module.patch_signs.data.copy_(current)

    flip_pct = 100.0 * total_flips / total_signs
    flip_of_patched = 100.0 * total_flips / max(total_patch_signs, 1)
    log(f"Flips: {total_flips:,} / {total_signs:,} "
        f"({flip_pct:.3f}% global, {flip_of_patched:.1f}% of patched)")

    # Commit patches into fp16 signs buffer — disables index_put in forward,
    # eval runs at full speed (same as baseline PackedBitLinear)
    for module in model.modules():
        if isinstance(module, SignPatchWrapper):
            module.commit_patches()
    torch.cuda.empty_cache()
    log(f"Patches committed. VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB")

    # ── Eval ──────────────────────────────────────────────────────────────────
    log(f"Eval n={N_EVAL_SIGN}...")
    acc = _eval_gsm8k(model, tokenizer, N_EVAL_SIGN, DEVICE, log)

    delta = acc - 0.280
    if delta > 0.03:
        verdict = f"SIGNS HAVE CAPACITY — pursue K=30%+ on next run"
    elif delta > 0.0:
        verdict = f"MARGINAL LIFT ({delta:+.1%}) — signs contribute, not dominant"
    elif delta >= -0.02:
        verdict = f"NULL — signs have no additional capacity at K={K_SIGN_PCT}%"
    else:
        verdict = f"REGRESSION ({delta:+.1%}) — flips disrupted existing routing"

    log(f"RESULT: {acc:.1%} GSM8K (n={N_EVAL_SIGN})  delta={delta:+.1%}  "
        f"flips={flip_pct:.3f}%  runtime={(time.time()-t0)/60:.1f}min")
    log(f"Verdict: {verdict}")

    result = {
        "experiment": "ste_sign_qat_k15_fixed", "gsm8k": float(acc),
        "n_eval": N_EVAL_SIGN, "k_pct": K_SIGN_PCT,
        "total_flips": int(total_flips), "total_signs": int(total_signs),
        "flip_pct_global": float(flip_pct),
        "flip_pct_of_patched": float(flip_of_patched),
        "total_patch_signs": int(total_patch_signs),
        "verdict": verdict, "runtime_min": (time.time()-t0)/60,
    }
    with open("/results/ste_sign_qat_k15.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    return result


# ── Local entrypoint — spawn all three concurrently ───────────────────────────

@app.local_entrypoint()
def main():
    import json

    print("Spawning 3 T4 functions concurrently...")
    h1 = run_scale_eval.spawn()
    h2 = run_lora_baseline.spawn()
    h3 = run_sign_qat.spawn()

    print("Waiting for results...")
    r1 = h1.get()
    r2 = h2.get()
    r3 = h3.get()

    print("\n" + "=" * 65)
    print("T4 BURST RESULTS")
    print("=" * 65)
    print(f"  Scale math v2   (n={r1['n_eval']:3d}): {r1['gsm8k']:.1%} GSM8K")
    print(f"  LoRA rank=16    (n={r2['n_eval']:3d}): {r2['gsm8k']:.1%} GSM8K  "
          f"({r2['lora_mb']:.0f}MB)")
    print(f"  STE sign K=15%  (n={r3['n_eval']:3d}): {r3['gsm8k']:.1%} GSM8K  "
          f"flips={r3['flip_pct_global']:.3f}% global / "
          f"{r3['flip_pct_unfrozen']:.1f}% of unfrozen")
    print(f"  Verdict: {r3['verdict']}")
    print("=" * 65)
    print(json.dumps({"scale": r1, "lora": r2, "sign_qat": r3}, indent=2))
