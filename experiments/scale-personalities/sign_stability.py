"""Sign stability probe — Bonsai 1.7B on local GPU.

Measures how much each layer's signs "want to change" during math training.
Signs stay FROZEN (buffers, not parameters) — zero extra VRAM.
We intercept the backward pass and compute the would-be sign gradient
as a side-channel, without materializing full weight matrices.

Method:
  - grad_sign_i = grad_w_i × scale_g  (already computed in backward)
  - We capture |grad_sign| per layer via the custom autograd function
  - Accumulate across 50 steps, report per transformer-layer mean

If early layers have low |grad_sign| and late layers have high |grad_sign|
→ signs crystallize early→late → progressive freezing is the right
curriculum for native 1-bit training at scale.

Monitor: ssh -p 2222 timyarm@100.110.173.110 "tail -f /tmp/sign_stability.log"
"""
import sys
import os
import time
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_PATH = "/tmp/sign_stability.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_PATH, mode='w'),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("sign_stability")

DEVICE = "cuda"
MAX_LEN = 128
TRAIN_STEPS = 50
LR = 1e-4
TOKEN_SELECT_RATIO = 0.6
CKPT = os.path.expanduser("~/freigent/apps/trucksim/checkpoints")

# Global accumulator: layer_name → list of mean |grad_sign| per step
SIGN_GRAD_ACCUM = {}


def make_instrumented_bitlinear_src():
    return '''
import torch
import torch.nn as nn
import torch.nn.functional as F

# Module-level accumulator injected from outside
_ACCUM = {}
_LAYER_REGISTRY = {}  # signs_data_ptr → layer_name

class _BitLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, signs, scales, out_features, in_features, group_size, layer_name):
        weight = signs * scales.flatten().repeat_interleave(group_size).view(out_features, in_features).to(signs.dtype)
        output = F.linear(x, weight)
        ctx.save_for_backward(x, signs, scales)
        ctx.out_features = out_features
        ctx.in_features = in_features
        ctx.group_size = group_size
        ctx.layer_name = layer_name
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, signs, scales = ctx.saved_tensors
        out_f, in_f, gs = ctx.out_features, ctx.in_features, ctx.group_size
        layer_name = ctx.layer_name

        weight = signs * scales.flatten().repeat_interleave(gs).view(out_f, in_f).to(signs.dtype)
        grad_x = grad_output.matmul(weight.to(grad_output.dtype))

        # Scale gradient (existing)
        grad_w = grad_output.reshape(-1, out_f).t().matmul(x.reshape(-1, in_f))
        grad_w_signed = (grad_w * signs).reshape(-1, gs)
        grad_scales = grad_w_signed.sum(dim=1).float()

        # Sign gradient (side-channel — not returned, just measured)
        # grad_sign_i = grad_w_i * scale_g  (how much sign_i wants to flip)
        scales_exp = scales.flatten().repeat_interleave(gs).view(out_f, in_f).float()
        sign_grad_mag = (grad_w.float().abs() * scales_exp.abs()).mean().item()
        if layer_name not in _ACCUM:
            _ACCUM[layer_name] = []
        _ACCUM[layer_name].append(sign_grad_mag)

        return grad_x, None, grad_scales, None, None, None, None


class PackedBitLinear(nn.Module):
    def __init__(self, weight, group_size=128, layer_name="unknown"):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.group_size = group_size
        self.layer_name = layer_name
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
                                   self.out_features, self.in_features,
                                   self.group_size, self.layer_name)

def convert_model(model, group_size=128, skip_names=("lm_head",)):
    converted = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear): continue
        if any(s in name for s in skip_names): continue
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = dict(model.named_modules())[parent_name] if parent_name else model
        setattr(parent, child_name, PackedBitLinear(module.weight.data, group_size, layer_name=name))
        converted += 1
    print(f"Converted {converted} layers to PackedBitLinear")
    return converted
'''


def inject_pbl():
    import types
    src = make_instrumented_bitlinear_src()
    pbl = types.ModuleType("packed_bitlinear")
    exec(src, pbl.__dict__)
    # Wire the accumulator into the module's namespace
    pbl._ACCUM = SIGN_GRAD_ACCUM
    sys.modules["packed_bitlinear"] = pbl
    return pbl.PackedBitLinear, pbl.convert_model, pbl._ACCUM


def get_data():
    from datasets import load_dataset
    texts = []
    ds = load_dataset("gsm8k", "main", split="train", streaming=True)
    for ex in ds:
        texts.append(f"Question: {ex['question']}\nAnswer: {ex['answer']}")
        if len(texts) >= TRAIN_STEPS * 2:
            break
    return texts[:TRAIN_STEPS]


def layer_depth(name):
    parts = name.split('.')
    for i, p in enumerate(parts):
        if p == 'layers' and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return -1


def main():
    t0 = time.time()
    log.info("SIGN STABILITY PROBE — Bonsai 1.7B")
    log.info(f"Steps={TRAIN_STEPS}, MaxLen={MAX_LEN}")
    log.info("Signs stay frozen. Measuring would-be gradient via backward side-channel.")

    log.info("GPU warmup...")
    _d = torch.ones(256, 256, device=DEVICE)
    for _ in range(10):
        _d = _d @ _d.T
    torch.cuda.synchronize()
    del _d
    time.sleep(3)

    PackedBitLinear, convert_model, accum = inject_pbl()

    log.info("\n--- MODEL ---")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "prism-ml/Bonsai-1.7B-unpacked", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "prism-ml/Bonsai-1.7B-unpacked", dtype=torch.float16,
        trust_remote_code=True, device_map="cpu", attn_implementation="eager")
    convert_model(model)
    model = model.to(DEVICE)
    torch.cuda.synchronize()
    log.info(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Only scales train — signs stay frozen buffers
    for name, p in model.named_parameters():
        p.requires_grad = ('scales' in name)

    scale_params = [p for n, p in model.named_parameters() if p.requires_grad]
    log.info(f"  Scale params training: {sum(p.numel() for p in scale_params):,}")

    log.info("\n--- DATA ---")
    texts = get_data()
    log.info(f"  {len(texts)} math examples loaded")

    log.info("\n--- PROBE ---")
    optimizer = torch.optim.AdamW(scale_params, lr=LR, weight_decay=0.01)
    model.train()

    for step, text in enumerate(texts):
        tokens = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=MAX_LEN).to(DEVICE)
        if tokens.input_ids.shape[1] < 10:
            continue

        with torch.amp.autocast('cuda', dtype=torch.float16):
            outputs = model(**tokens)
            logits = outputs.logits[:, :-1]
            labels = tokens.input_ids[:, 1:]
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                labels.reshape(-1), reduction='none')
            threshold = torch.quantile(ce, 1.0 - TOKEN_SELECT_RATIO)
            mask = (ce >= threshold).float()
            loss = (ce * mask).sum() / (mask.sum() + 1e-8)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(scale_params, 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            torch.cuda.empty_cache()
            n_captured = sum(len(v) for v in accum.values())
            log.info(f"  step {step+1}/{TRAIN_STEPS}  loss={loss.item():.4f}  captured={n_captured} layer-steps")

    log.info("\n--- RESULTS ---")

    # Mean |grad_sign| per layer name
    mean_grads = {name: np.mean(vals) for name, vals in accum.items() if vals}

    # Group by transformer depth
    by_depth = {}
    for name, g in mean_grads.items():
        d = layer_depth(name)
        if d >= 0:
            by_depth.setdefault(d, []).append(g)

    depth_mean = {d: np.mean(v) for d, v in by_depth.items()}
    sorted_depths = sorted(depth_mean)

    if not sorted_depths:
        log.error("No layer data captured — check accumulator wiring.")
        return

    all_g = [depth_mean[d] for d in sorted_depths]
    min_g, max_g = min(all_g), max(all_g)
    n = len(sorted_depths)
    early = np.mean(all_g[:n//3])
    late  = np.mean(all_g[2*n//3:])

    log.info("")
    log.info("Per-transformer-layer mean |sign gradient| (higher = wants to change more):")
    log.info(f"  {'Layer':>5}  {'|grad|':>10}  bar")
    log.info("  " + "-" * 40)
    for d in sorted_depths:
        g = depth_mean[d]
        bar = "█" * max(1, int(25 * (g - min_g) / max(max_g - min_g, 1e-10)))
        log.info(f"  {d:>5}  {g:>10.4e}  {bar}")

    ratio = late / max(early, 1e-12)
    log.info("")
    log.info(f"  Early layers (0-{n//3-1}):  mean = {early:.4e}")
    log.info(f"  Late  layers ({2*n//3}-{n-1}): mean = {late:.4e}")
    log.info(f"  Late/Early ratio: {ratio:.2f}x")
    log.info("")

    if ratio > 1.5:
        verdict = f"CONFIRMED ({ratio:.1f}x) — sign pressure INCREASES with depth. Signs crystallize early→late. Progressive freeze validated."
    elif ratio < 0.67:
        verdict = f"INVERTED ({ratio:.1f}x) — sign pressure DECREASES with depth. Late layers stabilize first."
    else:
        verdict = f"UNIFORM ({ratio:.1f}x) — no strong depth trend."
    log.info(f"  Verdict: {verdict}")

    results = {
        "depth_mean_sign_grad": {str(d): depth_mean[d] for d in sorted_depths},
        "early_mean": early,
        "late_mean": late,
        "late_early_ratio": ratio,
        "verdict": verdict,
        "n_steps": TRAIN_STEPS,
    }
    out = f"{CKPT}/sign_stability_results.json"
    os.makedirs(CKPT, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"\n  Saved: {out}")
    log.info(f"  Total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"CRASHED: {e}", exc_info=True)
        sys.exit(1)
    finally:
        log.info("DONE")
