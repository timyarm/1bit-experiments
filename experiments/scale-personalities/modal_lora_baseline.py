"""LoRA baseline — Modal A100 version.

Same experiment as lora_baseline.py but runs on Modal A100 (no WSL2 TDR).
Logs stream live via: modal logs <app-id>  or  modal app logs trucksim-lora-baseline

Run: modal run experiments/scale-personalities/modal_lora_baseline.py
"""
import modal

app = modal.App("trucksim-lora-baseline")

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

vol = modal.Volume.from_name("trucksim-vlm-eval", create_if_missing=True)

LORA_RANK  = 16
LORA_ALPHA = 16
MAX_LEN    = 256
TRAIN_EXAMPLES = 150
LR         = 1e-4
EPOCHS     = 3
REG_LAMBDA = 0.1
TOKEN_SELECT_RATIO = 0.6
DOMAIN_MIX = 0.7
N_GSM8K_EVAL = 100

SCALE_BASELINE = 0.053
SCALE_MATH     = 0.280
SCALE_BLEND    = 0.400


PACKED_BITLINEAR_SRC = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        grad_w_signed = (grad_w * signs).reshape(-1, gs)
        grad_scales = grad_w_signed.sum(dim=1).float()
        return grad_x, None, grad_scales, None, None, None

class PackedBitLinear(nn.Module):
    def __init__(self, weight, group_size=128):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.group_size = group_size
        signs_fp16 = weight.sign().half()
        signs_fp16[signs_fp16 == 0] = 1
        self.register_buffer("signs", signs_fp16)
        n_groups = (self.out_features * self.in_features) // group_size
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
    print(f"Converted {converted} layers to PackedBitLinear")
    return converted
'''


@app.function(
    image=image,
    gpu="A100",
    timeout=7200,
    volumes={"/checkpoints": vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_lora_baseline():
    import sys, os, time, json, re, types
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # Inject packed_bitlinear as a module
    pbl = types.ModuleType("packed_bitlinear")
    exec(PACKED_BITLINEAR_SRC, pbl.__dict__)
    sys.modules["packed_bitlinear"] = pbl
    PackedBitLinear = pbl.PackedBitLinear
    convert_model   = pbl.convert_model

    DEVICE = "cuda"
    t0 = time.time()

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')} | {msg}", flush=True)

    log("=" * 60)
    log("LORA BASELINE — MATCHED PARAMETER COUNT")
    log(f"Rank={LORA_RANK}, Alpha={LORA_ALPHA}, LR={LR}, Epochs={EPOCHS}")
    log("Hypothesis: do LoRA adapters match scale personality GSM8K gains?")
    log("=" * 60)

    # ── LoRA wrapper ──────────────────────────────────────────────────────────

    class LoRALinear(nn.Module):
        def __init__(self, base, rank, alpha):
            super().__init__()
            self.base = base
            out_f = base.out_features
            in_f  = base.in_features
            dev   = base.signs.device
            self.lora_A = nn.Parameter(torch.randn(rank, in_f,  device=dev, dtype=torch.float32) * 0.02)
            self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=dev, dtype=torch.float32))
            self.scaling = alpha / rank

        def forward(self, x):
            w = (self.lora_B @ self.lora_A).to(x.dtype)
            return self.base(x) + F.linear(x, w) * self.scaling

    def apply_lora(model, rank, alpha):
        replaced = 0
        for parent_name, parent in list(model.named_modules()):
            for child_name, child in list(parent.named_children()):
                if isinstance(child, PackedBitLinear):
                    setattr(parent, child_name, LoRALinear(child, rank, alpha))
                    replaced += 1
        return replaced

    def freeze_non_lora(model):
        for name, p in model.named_parameters():
            p.requires_grad = ('lora_A' in name or 'lora_B' in name)

    def count_lora_params(model):
        return sum(p.numel() for name, p in model.named_parameters()
                   if 'lora_A' in name or 'lora_B' in name)

    # ── model ─────────────────────────────────────────────────────────────────

    log("\n--- MODEL ---")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "prism-ml/Bonsai-1.7B-unpacked", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "prism-ml/Bonsai-1.7B-unpacked", dtype=torch.float16,
        trust_remote_code=True, device_map="cpu")
    convert_model(model)
    model = model.to(DEVICE)

    scale_param_count = sum(m.scales.numel() for m in model.modules()
                            if isinstance(m, PackedBitLinear))
    scale_bytes = scale_param_count * 2
    log(f"  Scale table: {scale_param_count:,} params ({scale_bytes/1e6:.1f}MB fp16)")

    n_replaced = apply_lora(model, LORA_RANK, LORA_ALPHA)
    freeze_non_lora(model)

    lora_param_count = count_lora_params(model)
    lora_bytes = lora_param_count * 4  # fp32
    log(f"  LoRA layers:  {n_replaced}")
    log(f"  LoRA params:  {lora_param_count:,} ({lora_bytes/1e6:.1f}MB fp32)")
    log(f"  Ratio vs scales: {lora_param_count/scale_param_count:.2f}x")
    log(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # ── data ──────────────────────────────────────────────────────────────────

    log("\n--- DATA ---")
    from datasets import load_dataset

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
    texts = [mixed[i] for i in rng.permutation(len(mixed))]
    log(f"  {len(texts)} examples ({int(DOMAIN_MIX*100)}% math)")

    # ── train ─────────────────────────────────────────────────────────────────

    log("\n--- TRAIN ---")
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, EPOCHS * len(texts)))
    model.train()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        n = 0
        for text in texts:
            toks = tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=MAX_LEN).to(DEVICE)
            if toks.input_ids.shape[1] < 10: continue
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(**toks)
                logits = outputs.logits[:, :-1]
                labels = toks.input_ids[:, 1:]
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    labels.reshape(-1), reduction="none")
                threshold = torch.quantile(ce, 1.0 - TOKEN_SELECT_RATIO)
                mask = (ce >= threshold).float()
                task_loss = (ce * mask).sum() / (mask.sum() + 1e-8)
                reg = sum(F.mse_loss(p, torch.zeros_like(p))
                          for nm, p in model.named_parameters()
                          if "lora_A" in nm or "lora_B" in nm)
                loss = task_loss + REG_LAMBDA * reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            epoch_loss += task_loss.item()
            n += 1
        log(f"  epoch {epoch+1}/{EPOCHS}  loss={epoch_loss/max(n,1):.4f}")

    log(f"  Training done in {(time.time()-t0)/60:.1f} min")

    # ── eval ──────────────────────────────────────────────────────────────────

    log(f"\n--- EVAL (GSM8K n={N_GSM8K_EVAL}) ---")
    ds = load_dataset("gsm8k", "main", split="test", streaming=True)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for ex in ds:
            if total >= N_GSM8K_EVAL: break
            prompt = f"Question: {ex['question']}\nAnswer:"
            toks = tokenizer(prompt, return_tensors="pt").to(DEVICE)
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
            if total % 10 == 0:
                log(f"  [{total}/{N_GSM8K_EVAL}] running acc={correct/total:.1%}")

    gsm = correct / max(total, 1)

    # ── results ───────────────────────────────────────────────────────────────

    delta = gsm - SCALE_MATH
    if abs(delta) <= 0.03:
        verdict = f"MATCHES scale training (Δ={delta:+.1%}) — scales win on zero inference overhead"
    elif delta > 0.03:
        verdict = f"EXCEEDS scale training (Δ={delta:+.1%}) — LoRA is stronger"
    else:
        verdict = f"BELOW scale training (Δ={delta:+.1%}) — scales have genuine inductive advantage"

    log(f"\n{'='*60}")
    log("RESULTS — LORA vs SCALE PERSONALITIES")
    log(f"{'='*60}")
    log(f"  LoRA (rank={LORA_RANK}, {lora_bytes/1e6:.0f}MB fp32):  {gsm:.1%} GSM8K")
    log(f"  Scale baselines (same data, same recipe):")
    log(f"    No training:            {SCALE_BASELINE:.1%}")
    log(f"    Scale math only:        {SCALE_MATH:.1%}  ({scale_bytes/1e6:.0f}MB fp16)")
    log(f"    Scale blend (flat 0.7): {SCALE_BLEND:.1%}  (no extra training)")
    log(f"  Verdict: {verdict}")
    log(f"  Total runtime: {(time.time()-t0)/60:.1f} min")

    results = {
        "lora_gsm8k": gsm, "lora_rank": LORA_RANK,
        "lora_params": lora_param_count, "lora_mb": lora_bytes / 1e6,
        "scale_params": scale_param_count, "scale_mb": scale_bytes / 1e6,
        "scale_math_gsm8k": SCALE_MATH, "delta_vs_scale_math": delta,
        "verdict": verdict,
    }
    out = "/checkpoints/lora_baseline_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()
    log(f"Saved: {out}")
    return results


@app.local_entrypoint()
def main():
    result = run_lora_baseline.remote()
    print("\nFINAL RESULT:")
    import json
    print(json.dumps(result, indent=2))
