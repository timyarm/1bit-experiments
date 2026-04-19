"""Blend Validation — re-validate the 40% headline at n=200.

Trains math + knowledge scale tables sequentially on T4, then blends at α=0.7
and evals GSM8K n=200. Checks whether 40% (measured at n=50-100) holds at
larger sample size.

Previous numbers (n=50-100):
  Math only:             28%
  Knowledge only:        ~35% (from blend paper)
  Math+knowledge α=0.7:  40%

Expected at n=200 if proportional to math correction (28%→23%):
  Blend: ~33-35%

Run: modal run experiments/scale-personalities/modal_blend_validation.py
Cost: ~$0.55, ~110 min on single T4
"""
import modal

app = modal.App("1bit-blend-validation")

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

vol = modal.Volume.from_name("1bit-blend-results", create_if_missing=True)

BONSAI = "prism-ml/Bonsai-1.7B-unpacked"
TRAIN_EXAMPLES = 150
DOMAIN_MIX = 0.7
MAX_LEN = 128
EPOCHS = 3
LR = 1e-4
REG_LAMBDA = 0.1
TOKEN_SELECT_RATIO = 0.6
GROUP_SIZE = 128
N_EVAL = 200
BLEND_ALPHA = 0.7   # math weight; knowledge = 1 - alpha

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


def _load_model(log):
    import sys, types, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    pbl = types.ModuleType("packed_bitlinear")
    exec(PACKED_BITLINEAR_SRC, pbl.__dict__)
    sys.modules["packed_bitlinear"] = pbl
    tokenizer = AutoTokenizer.from_pretrained(BONSAI, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BONSAI, dtype=torch.float16, trust_remote_code=True,
        device_map="cpu", attn_implementation="eager")
    pbl.convert_model(model)
    model = model.to("cuda")
    log(f"  Model loaded. VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB")
    return model, tokenizer, pbl.PackedBitLinear


def _get_data(domain):
    """Return 150 mixed training examples for a domain."""
    from datasets import load_dataset
    import numpy as np
    domain_texts, diverse_texts = [], []

    if domain == "math":
        ds = load_dataset("gsm8k", "main", split="train", streaming=True)
        for ex in ds:
            domain_texts.append(f"Question: {ex['question']}\nAnswer: {ex['answer']}")
            if len(domain_texts) >= TRAIN_EXAMPLES * 2: break
    elif domain == "knowledge":
        ds = load_dataset("trivia_qa", "rc", split="train", streaming=True)
        for ex in ds:
            if ex.get("answer", {}).get("value"):
                domain_texts.append(
                    f"Question: {ex['question']}\nAnswer: {ex['answer']['value']}")
            if len(domain_texts) >= TRAIN_EXAMPLES * 2: break

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
    for ex in ds:
        if len(ex['text'].strip()) > 200:
            diverse_texts.append(ex['text'])
        if len(diverse_texts) >= TRAIN_EXAMPLES: break

    rng = np.random.default_rng(42)
    n_domain = int(TRAIN_EXAMPLES * DOMAIN_MIX)
    mixed = domain_texts[:n_domain] + diverse_texts[:TRAIN_EXAMPLES - n_domain]
    return [mixed[i] for i in rng.permutation(len(mixed))]


def _train_scales(model, tokenizer, texts, PackedBitLinear, log):
    import torch, torch.nn.functional as F
    DEVICE = "cuda"

    for p in model.parameters(): p.requires_grad_(False)
    scale_params, orig_snapshots = [], {}
    for name, module in model.named_modules():
        if isinstance(module, PackedBitLinear):
            module.scales.requires_grad_(True)
            scale_params.append(module.scales)
            orig_snapshots[name] = module.scales.data.clone()

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
        log(f"    epoch {epoch+1}/{EPOCHS}: loss={total_loss/max(n,1):.4f}")

    # Return trained scale snapshot
    return {name: module.scales.data.clone()
            for name, module in model.named_modules()
            if isinstance(module, PackedBitLinear)}


def _apply_scales(model, scales_dict, PackedBitLinear):
    """Load a scale snapshot into the model."""
    import torch
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, PackedBitLinear) and name in scales_dict:
                module.scales.data.copy_(scales_dict[name])


def _reset_scales(model, PackedBitLinear):
    """Restore original scales."""
    import torch
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PackedBitLinear):
                module.scales.data.copy_(module._original_scales)
    for p in model.parameters(): p.requires_grad_(False)


def _eval_gsm8k(model, tokenizer, n, log):
    import re, torch
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test", streaming=True)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for ex in ds:
            if total >= n: break
            prompt = f"Question: {ex['question']}\nAnswer:"
            toks = tokenizer(prompt, return_tensors="pt").to("cuda")
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


@app.function(
    image=image, gpu="T4", timeout=14400,
    volumes={"/results": vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_blend_validation():
    import time, json, torch

    t0 = time.time()
    def log(msg): print(f"{time.strftime('%H:%M:%S')} | {msg}", flush=True)

    log("BLEND VALIDATION — re-validate 40% headline at n=200")
    log(f"Blend α={BLEND_ALPHA} math + {1-BLEND_ALPHA:.1f} knowledge")

    model, tokenizer, PackedBitLinear = _load_model(log)

    # Skip baseline + individual evals — already confirmed from prior runs:
    #   Baseline (HF unpacked): ~17%
    #   Math only: 23.0% (n=200, confirmed twice)
    #   Knowledge only: 5.5% (n=200, GSM8K — trained on TriviaQA, expected)
    # Just need: train both → blend → eval n=200

    # ── Math scales ───────────────────────────────────────────────────────────
    log("[1/2] Training math scales...")
    texts_math = _get_data("math")
    log(f"  {len(texts_math)} examples")
    math_scales = _train_scales(model, tokenizer, texts_math, PackedBitLinear, log)
    log("  Math scales trained.")

    # ── Knowledge scales ──────────────────────────────────────────────────────
    log("[2/2] Training knowledge scales...")
    _reset_scales(model, PackedBitLinear)
    texts_knowledge = _get_data("knowledge")
    log(f"  {len(texts_knowledge)} examples")
    knowledge_scales = _train_scales(model, tokenizer, texts_knowledge, PackedBitLinear, log)
    log("  Knowledge scales trained.")

    # ── Blend α=0.7 + eval n=200 ─────────────────────────────────────────────
    log(f"Blending math×{BLEND_ALPHA} + knowledge×{1-BLEND_ALPHA:.1f}...")
    blend_scales = {
        name: BLEND_ALPHA * math_scales[name] + (1 - BLEND_ALPHA) * knowledge_scales[name]
        for name in math_scales
    }
    _apply_scales(model, blend_scales, PackedBitLinear)

    log(f"Eval blend n={N_EVAL}...")
    acc_blend = _eval_gsm8k(model, tokenizer, N_EVAL, log)
    log(f"  Blend α=0.7: {acc_blend:.1%}")

    # ── Results ───────────────────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("BLEND VALIDATION RESULTS")
    log("=" * 60)
    log(f"  Baseline (HF unpacked):    ~17%  (confirmed prior run)")
    log(f"  Math only:                 23.0% (confirmed prior run, n=200)")
    log(f"  Knowledge only (GSM8K):    5.5%  (confirmed prior run, n=200)")
    log(f"  Blend α=0.7:               {acc_blend:.1%}  (prev headline: 40%) ← THE NUMBER")
    log(f"  Runtime: {(time.time()-t0)/60:.1f} min")
    log("=" * 60)

    result = {
        "acc_baseline_hf": 0.17,
        "acc_math_n200": 0.23,
        "acc_knowledge_gsm8k_n200": 0.055,
        "acc_blend_07": float(acc_blend),
        "n_eval": N_EVAL,
        "blend_alpha": BLEND_ALPHA,
        "prev_headline_n100": 0.40,
        "runtime_min": (time.time()-t0)/60,
    }
    with open("/results/blend_validation_v2.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    log("Saved: /results/blend_validation_v2.json")
    return result


@app.local_entrypoint()
def main():
    import json
    print("Running blend validation on T4...")
    result = run_blend_validation.remote()
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"  Baseline:      {result['acc_baseline']:.1%}")
    print(f"  Math only:     {result['acc_math']:.1%}  (prev headline: 28%)")
    print(f"  Blend α=0.7:   {result['acc_blend_07']:.1%}  (prev headline: 40%)")
    print(f"  Alpha sweep:   {result['alpha_sweep']}")
    print("=" * 60)
    print(json.dumps(result, indent=2))
