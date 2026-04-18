"""Targeted Late-Layer Scale Training — Exp 23.

Hypothesis: ffn_up and gate_proj scales in late transformer layers (20-27)
are the highest-gradient, highest-impact parameters for domain adaptation
(per Bonsai forensics: category hierarchy ffn_up > gate > ... > q, and
depth-scale correlation 0.877 means late layers carry more per-scale impact).

If targeted training matches full-model training (28% GSM8K) with 3-10x
fewer parameters, this validates the forensics and shows efficient adaptation.

Baseline: 28.0% GSM8K (all scales trained, Exp 7)
Hypothesis: late ffn_up+gate (layers 20-27) → 25-30% with ~10% of params

Monitor: ssh -p 2222 timyarm@100.110.173.110 "tail -f /tmp/targeted_scales.log"
"""
import sys, os, time, json, re, types, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_PATH = "/tmp/targeted_scales.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_PATH, mode='w'),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("targeted")

DEVICE = "cuda"
MAX_LEN = 128
TRAIN_EXAMPLES = 150
LR = 1e-4
EPOCHS = 3
REG_LAMBDA = 0.1
TOKEN_SELECT_RATIO = 0.6
DOMAIN_MIX = 0.7
N_EVAL = 100
CKPT = os.path.expanduser("~/freigent/apps/trucksim/checkpoints")

# Which layer types and depths to train
TARGET_TYPES = ("mlp.up_proj", "mlp.gate_proj")  # ffn_up and gate per forensics hierarchy
TARGET_LAYERS = list(range(20, 28))  # late 8 of 28 layers (indices 20-27)

SCALE_BASELINE = 0.053
SCALE_MATH_FULL = 0.280

PACKED_BITLINEAR_SRC = '''
import torch, torch.nn as nn, torch.nn.functional as F

class _BitLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, signs, scales, out_features, in_features, group_size):
        weight = signs * scales.flatten().repeat_interleave(group_size).view(out_features, in_features).to(signs.dtype)
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
        weight = signs * scales.flatten().repeat_interleave(gs).view(out_f, in_f).to(signs.dtype)
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


def inject_pbl():
    pbl = types.ModuleType("packed_bitlinear")
    exec(PACKED_BITLINEAR_SRC, pbl.__dict__)
    sys.modules["packed_bitlinear"] = pbl
    return pbl.PackedBitLinear, pbl.convert_model


def is_targeted(name):
    """Return True if this layer name matches TARGET_TYPES in TARGET_LAYERS."""
    for layer_idx in TARGET_LAYERS:
        for typ in TARGET_TYPES:
            if f"layers.{layer_idx}.{typ}" in name:
                return True
    return False


def get_data():
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
    return [mixed[i] for i in rng.permutation(len(mixed))]


def train(model, tokenizer, texts, scale_params, orig_snapshots):
    optimizer = torch.optim.AdamW(scale_params, lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, EPOCHS * len(texts)))

    model.train()
    for epoch in range(EPOCHS):
        total_loss, total_reg, n = 0.0, 0.0, 0
        for text in texts:
            tokens = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=MAX_LEN).to(DEVICE)
            if tokens.input_ids.shape[1] < 10: continue

            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(**tokens)
                logits = outputs.logits[:, :-1]
                labels = tokens.input_ids[:, 1:]
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    labels.reshape(-1), reduction='none')
                threshold = torch.quantile(ce, 1.0 - TOKEN_SELECT_RATIO)
                mask = (ce >= threshold).float()
                task_loss = (ce * mask).sum() / (mask.sum() + 1e-8)

                reg_loss = torch.tensor(0.0, device=DEVICE)
                for name, module in model.named_modules():
                    if hasattr(module, 'scales') and module.scales.requires_grad and name in orig_snapshots:
                        reg_loss = reg_loss + F.mse_loss(module.scales, orig_snapshots[name])

                loss = task_loss + REG_LAMBDA * reg_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(scale_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += task_loss.item()
            total_reg += reg_loss.item()
            n += 1

            if n % 30 == 0:
                torch.cuda.empty_cache()

        log.info(f"  Epoch {epoch+1}/{EPOCHS}: loss={total_loss/max(n,1):.4f}  "
                 f"reg={total_reg/max(n,1):.6f}  VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB")


def eval_gsm8k(model, tokenizer, n=N_EVAL):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test", streaming=True)
    questions = []
    for ex in ds:
        questions.append((ex['question'], ex['answer']))
        if len(questions) >= n: break

    model.eval()
    correct = 0
    for i, (q, a) in enumerate(questions):
        prompt = f"Question: {q}\nAnswer:"
        tokens = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            out = model.generate(
                **tokens, max_new_tokens=200, do_sample=False,
                temperature=1.0, pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][tokens.input_ids.shape[1]:], skip_special_tokens=True)
        # Extract final number
        nums = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', gen.replace(',', ''))
        pred = nums[-1] if nums else ""
        ref_nums = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', a.replace(',', ''))
        ref = ref_nums[-1] if ref_nums else ""
        if pred == ref:
            correct += 1
        if (i + 1) % 10 == 0:
            log.info(f"  [{i+1}/{n}] acc={correct/(i+1):.1%}")
    acc = correct / n
    return acc


def main():
    t0 = time.time()
    log.info("TARGETED LATE-LAYER SCALE TRAINING — Exp 23")
    log.info(f"Target: {TARGET_TYPES} in layers {TARGET_LAYERS[0]}-{TARGET_LAYERS[-1]}")
    log.info(f"Baseline to compare: {SCALE_MATH_FULL:.1%} GSM8K (all scales)")

    log.info("GPU warmup...")
    _d = torch.ones(256, 256, device=DEVICE)
    for _ in range(10): _d = _d @ _d.T
    torch.cuda.synchronize()
    del _d
    import time as t; t.sleep(3)

    PackedBitLinear, convert_model = inject_pbl()

    log.info("\n--- MODEL ---")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "prism-ml/Bonsai-1.7B-unpacked", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "prism-ml/Bonsai-1.7B-unpacked", dtype=torch.float16,
        trust_remote_code=True, device_map="cpu", attn_implementation="eager")
    n_converted = convert_model(model)
    model = model.to(DEVICE)
    torch.cuda.synchronize()
    log.info(f"  Converted {n_converted} layers, VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Freeze all params, then selectively unfreeze targeted scales
    for p in model.parameters():
        p.requires_grad_(False)

    scale_params = []
    orig_snapshots = {}
    targeted_count = 0
    total_count = 0

    for name, module in model.named_modules():
        if not isinstance(module, PackedBitLinear): continue
        total_count += 1
        if is_targeted(name):
            module.scales.requires_grad_(True)
            scale_params.append(module.scales)
            orig_snapshots[name] = module.scales.data.clone().detach()
            targeted_count += 1
        else:
            module.scales.requires_grad_(False)

    targeted_params = sum(p.numel() for p in scale_params)
    total_scale_params = sum(m.scales.numel() for m in model.modules()
                             if isinstance(m, PackedBitLinear))

    log.info(f"  Targeted layers: {targeted_count}/{total_count} "
             f"({100*targeted_count/total_count:.0f}%)")
    log.info(f"  Targeted params: {targeted_params:,} / {total_scale_params:,} "
             f"({100*targeted_params/total_scale_params:.1f}%) = {targeted_params*4/1e6:.1f}MB")

    log.info("\n--- DATA ---")
    texts = get_data()
    log.info(f"  {len(texts)} examples ({int(DOMAIN_MIX*100)}% math)")

    log.info("\n--- TRAIN ---")
    train(model, tokenizer, texts, scale_params, orig_snapshots)

    log.info(f"\n--- EVAL (GSM8K n={N_EVAL}) ---")
    acc = eval_gsm8k(model, tokenizer, N_EVAL)

    log.info("")
    log.info("=" * 60)
    log.info("RESULTS — TARGETED LATE-LAYER SCALES")
    log.info("=" * 60)
    log.info(f"  Targeted (ffn_up+gate layers 20-27): {acc:.1%} GSM8K")
    log.info(f"  Full scale training (Exp 7):          {SCALE_MATH_FULL:.1%} GSM8K")
    log.info(f"  No training:                          {SCALE_BASELINE:.1%} GSM8K")
    delta = acc - SCALE_MATH_FULL
    if delta >= 0.0:
        verdict = f"MATCHES OR EXCEEDS full training (Δ=+{delta:.1%}) with {100*targeted_params/total_scale_params:.0f}% params"
    elif delta >= -0.03:
        verdict = f"NEAR-MATCH (Δ={delta:.1%}) with {100*targeted_params/total_scale_params:.0f}% params — efficient"
    else:
        verdict = f"BELOW full training (Δ={delta:.1%}) — late ffn_up+gate insufficient alone"
    log.info(f"  Verdict: {verdict}")
    log.info("=" * 60)

    results = {
        "targeted_acc": float(acc),
        "full_baseline": SCALE_MATH_FULL,
        "no_training": SCALE_BASELINE,
        "delta": float(delta),
        "targeted_layers": TARGET_LAYERS,
        "targeted_types": list(TARGET_TYPES),
        "targeted_params": targeted_params,
        "total_scale_params": total_scale_params,
        "targeted_pct": float(targeted_params / total_scale_params),
        "verdict": verdict,
    }
    os.makedirs(CKPT, exist_ok=True)
    with open(f"{CKPT}/targeted_late_scales_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"\n  Saved: {CKPT}/targeted_late_scales_results.json")
    log.info(f"  Total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"CRASHED: {e}", exc_info=True)
        sys.exit(1)
    finally:
        log.info("DONE")
