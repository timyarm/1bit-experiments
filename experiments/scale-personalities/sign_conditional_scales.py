"""Sign-conditional scales — Exp 20.

Hypothesis: within each 128-weight group, positive-sign and negative-sign
weights don't have to share the same magnitude. Bonsai's QAT likely produced
sign-asymmetric groups (positive weights ≠ negative weights in magnitude).
Two trainable fp16 scales per group — one for +1 signs, one for -1 signs —
captures this asymmetry for free.

Current:  w_i = sign_i × scale_g               (1 fp16 per group)
Proposed: w_i = sign_i × scale_{sign_i, g}     (2 fp16 per group, 44MB total)

If scale_pos ≠ scale_neg after training → asymmetry is real.
If GSM8K > 28% (math personality baseline) → sign-conditional scales improve.

Setup: identical to scale math training (v2 recipe):
  - Bonsai 1.7B, 150 examples, 70% math / 30% wiki
  - AdamW lr=1e-4, Rho-1 top-60%, elastic band λ=0.1, 3 epochs, seq_len=128
  - Eval: GSM8K test n=100, 0-shot greedy

Baseline: 28.0% GSM8K (scale math only, Exp 7, same recipe)

Monitor: ssh -p 2222 timyarm@100.110.173.110 "tail -f /tmp/sign_cond.log"
"""
import sys, os, time, json, re, types, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_PATH = "/tmp/sign_cond.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_PATH, mode='w'),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("sign_cond")

DEVICE = "cuda"
MAX_LEN = 128
TRAIN_EXAMPLES = 150
LR = 1e-4
EPOCHS = 3
REG_LAMBDA = 0.1
TOKEN_SELECT_RATIO = 0.6
DOMAIN_MIX = 0.7
N_GSM8K_EVAL = 100
CKPT = os.path.expanduser("~/freigent/apps/trucksim/checkpoints")

SCALE_BASELINE = 0.053
SCALE_MATH     = 0.280


# ── Sign-conditional PackedBitLinear ─────────────────────────────────────────

class _SignCondBitLinearFn(torch.autograd.Function):
    """Forward: w_i = sign_i × scale_pos_g  if sign_i > 0
                      sign_i × scale_neg_g  if sign_i < 0
    Backward: grad flows to scale_pos for positive-sign groups,
              grad flows to scale_neg for negative-sign groups.
    Signs stay frozen.
    """
    @staticmethod
    def forward(ctx, x, signs, scale_pos, scale_neg, out_f, in_f, gs):
        # pos_mask: 1 where sign=+1, 0 where sign=-1
        pos_mask = (signs > 0).to(signs.dtype)          # fp16 {0,1}
        neg_mask = 1.0 - pos_mask                        # fp16 {0,1}

        sp = scale_pos.flatten().repeat_interleave(gs).view(out_f, in_f)
        sn = scale_neg.flatten().repeat_interleave(gs).view(out_f, in_f)

        # Effective scale per weight: pos weights use scale_pos, neg use scale_neg
        eff_scale = pos_mask * sp.to(signs.dtype) + neg_mask * sn.to(signs.dtype)
        weight = signs * eff_scale
        output = F.linear(x, weight)

        # Don't save pos_mask/neg_mask — recompute from signs in backward (saves ~1.6GB)
        ctx.save_for_backward(x, signs, scale_pos, scale_neg)
        ctx.out_f, ctx.in_f, ctx.gs = out_f, in_f, gs
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, signs, scale_pos, scale_neg = ctx.saved_tensors
        out_f, in_f, gs = ctx.out_f, ctx.in_f, ctx.gs

        pos_mask = (signs > 0).to(signs.dtype)
        neg_mask = 1.0 - pos_mask
        sp = scale_pos.flatten().repeat_interleave(gs).view(out_f, in_f)
        sn = scale_neg.flatten().repeat_interleave(gs).view(out_f, in_f)
        eff_scale = pos_mask * sp.to(signs.dtype) + neg_mask * sn.to(signs.dtype)
        weight = signs * eff_scale

        grad_x = grad_output.matmul(weight.to(grad_output.dtype))
        grad_w = grad_output.reshape(-1, out_f).t().matmul(x.reshape(-1, in_f))

        # grad_scale_pos: only positive-sign weights contribute
        grad_w_pos = (grad_w * signs * pos_mask).reshape(-1, gs)
        grad_scale_pos = grad_w_pos.sum(dim=1).float()

        # grad_scale_neg: only negative-sign weights contribute
        grad_w_neg = (grad_w * signs * neg_mask).reshape(-1, gs)
        grad_scale_neg = grad_w_neg.sum(dim=1).float()

        return grad_x, None, grad_scale_pos, grad_scale_neg, None, None, None


class SignCondPackedBitLinear(nn.Module):
    """1-bit linear with separate fp32 scales for +1 and -1 sign groups."""

    def __init__(self, weight, group_size=128):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.group_size = group_size

        signs_fp16 = weight.sign().half()
        signs_fp16[signs_fp16 == 0] = 1
        self.register_buffer('signs', signs_fp16)

        n_groups = (self.out_features * self.in_features) // group_size
        w_flat = weight.reshape(-1, group_size)

        # Initialize from actual pos/neg absmeans
        pos_mask = (signs_fp16.reshape(-1, group_size) > 0).float()
        neg_mask = 1.0 - pos_mask
        w_abs = weight.reshape(-1, group_size).abs()

        pos_counts = pos_mask.sum(dim=1).clamp(min=1)
        neg_counts = neg_mask.sum(dim=1).clamp(min=1)
        scale_pos_init = (w_abs * pos_mask).sum(dim=1) / pos_counts
        scale_neg_init = (w_abs * neg_mask).sum(dim=1) / neg_counts

        # Fallback: groups where all signs are same polarity get absmean
        absmean = w_flat.abs().mean(dim=1)
        scale_pos_init = torch.where(pos_counts > 0.5, scale_pos_init, absmean)
        scale_neg_init = torch.where(neg_counts > 0.5, scale_neg_init, absmean)

        self.scale_pos = nn.Parameter(scale_pos_init.float())
        self.scale_neg = nn.Parameter(scale_neg_init.float())

        # For elastic band regularization
        self.register_buffer('_orig_scale_pos', scale_pos_init.clone())
        self.register_buffer('_orig_scale_neg', scale_neg_init.clone())

    def forward(self, x):
        return _SignCondBitLinearFn.apply(
            x, self.signs, self.scale_pos, self.scale_neg,
            self.out_features, self.in_features, self.group_size)


def convert_model_sign_cond(model, group_size=128, skip_names=('lm_head',)):
    converted = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear): continue
        if any(s in name for s in skip_names): continue
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        parent = dict(model.named_modules())[parent_name] if parent_name else model
        setattr(parent, child_name, SignCondPackedBitLinear(module.weight.data, group_size))
        converted += 1
    log.info(f"  Converted {converted} layers to SignCondPackedBitLinear")
    return converted


# ── data ─────────────────────────────────────────────────────────────────────

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


# ── train ─────────────────────────────────────────────────────────────────────

def train(model, tokenizer, texts):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, EPOCHS * len(texts)))
    model.train()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        n = 0
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

                # Elastic band: pull both scales toward their init
                reg = sum(
                    F.mse_loss(m.scale_pos, m._orig_scale_pos) +
                    F.mse_loss(m.scale_neg, m._orig_scale_neg)
                    for m in model.modules()
                    if isinstance(m, SignCondPackedBitLinear)
                )
                loss = task_loss + REG_LAMBDA * reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += task_loss.item()
            n += 1
            if n % 10 == 0:
                torch.cuda.empty_cache()

        log.info(f"  epoch {epoch+1}/{EPOCHS}  loss={epoch_loss/max(n,1):.4f}")


# ── eval ──────────────────────────────────────────────────────────────────────

def eval_gsm8k(model, tokenizer):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test", streaming=True)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for ex in ds:
            if total >= N_GSM8K_EVAL: break
            prompt = f"Question: {ex['question']}\nAnswer:"
            toks = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                out = model.generate(
                    **toks, max_new_tokens=200,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id)
            resp = tokenizer.decode(out[0][toks.input_ids.shape[1]:],
                                    skip_special_tokens=True)
            pred = re.findall(r'-?\d+\.?\d*', resp.replace(',', ''))
            gold = re.findall(r'-?\d+\.?\d*',
                              ex['answer'].split('####')[-1].replace(',', ''))
            try:
                if pred and gold and abs(float(pred[-1]) - float(gold[-1])) < 0.01:
                    correct += 1
            except ValueError:
                pass
            total += 1
            if total % 10 == 0:
                log.info(f"  [{total}/{N_GSM8K_EVAL}] acc={correct/total:.1%}")
    return correct / max(total, 1)


# ── asymmetry analysis ────────────────────────────────────────────────────────

def analyze_asymmetry(model):
    """Report how much scale_pos vs scale_neg diverged after training."""
    ratios = []
    for m in model.modules():
        if isinstance(m, SignCondPackedBitLinear):
            sp = m.scale_pos.detach().float()
            sn = m.scale_neg.detach().float()
            ratio = (sp / sn.clamp(min=1e-8)).mean().item()
            ratios.append(ratio)
    if ratios:
        log.info(f"  scale_pos/scale_neg ratio: mean={np.mean(ratios):.4f}  "
                 f"std={np.std(ratios):.4f}  min={np.min(ratios):.4f}  max={np.max(ratios):.4f}")
        log.info(f"  (1.0 = symmetric, >1.0 = positive signs trained larger)")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log.info("SIGN-CONDITIONAL SCALES — Exp 20")
    log.info("Two fp16 scales per group: scale_pos (for +1 signs) + scale_neg (for -1 signs)")
    log.info(f"Baseline to beat: {SCALE_MATH:.1%} GSM8K (scale math only, same recipe)")

    log.info("GPU warmup...")
    _d = torch.ones(256, 256, device=DEVICE)
    for _ in range(10): _d = _d @ _d.T
    torch.cuda.synchronize()
    del _d
    time.sleep(3)

    log.info("\n--- MODEL ---")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "prism-ml/Bonsai-1.7B-unpacked", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "prism-ml/Bonsai-1.7B-unpacked", dtype=torch.float16,
        trust_remote_code=True, device_map="cpu", attn_implementation="eager")

    n_layers = convert_model_sign_cond(model)
    model = model.to(DEVICE)
    torch.cuda.synchronize()

    n_params = sum(p.numel() for m in model.modules()
                   if isinstance(m, SignCondPackedBitLinear)
                   for p in [m.scale_pos, m.scale_neg])
    log.info(f"  Sign-cond scale params: {n_params:,} ({n_params*4/1e6:.1f}MB fp32)")
    log.info(f"  vs standard scales:     {n_params//2:,} ({n_params*2/1e6:.1f}MB fp16)")
    log.info(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Only scale_pos and scale_neg train
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, SignCondPackedBitLinear):
            m.scale_pos.requires_grad_(True)
            m.scale_neg.requires_grad_(True)

    # Smoke test
    log.info("  Smoke test...")
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        _t = tokenizer("test", return_tensors="pt").to(DEVICE)
        _o = model(**_t)
        torch.cuda.synchronize()
    log.info(f"  OK: logits={_o.logits.shape}")
    del _t, _o

    log.info("\n--- DATA ---")
    texts = get_data()
    log.info(f"  {len(texts)} examples ({int(DOMAIN_MIX*100)}% math)")

    log.info("\n--- TRAIN ---")
    train(model, tokenizer, texts)
    log.info(f"  Trained in {(time.time()-t0)/60:.1f} min")

    log.info("\n--- ASYMMETRY ANALYSIS ---")
    analyze_asymmetry(model)

    log.info(f"\n--- EVAL (GSM8K n={N_GSM8K_EVAL}) ---")
    gsm = eval_gsm8k(model, tokenizer)

    delta = gsm - SCALE_MATH
    if delta > 0.02:
        verdict = f"BEATS standard scales (Δ={delta:+.1%}) — sign asymmetry is real and exploitable"
    elif delta > -0.02:
        verdict = f"MATCHES standard scales (Δ={delta:+.1%}) — groups are roughly symmetric"
    else:
        verdict = f"BELOW standard scales (Δ={delta:+.1%}) — extra params don't help"

    log.info(f"\n{'='*60}")
    log.info("RESULTS — SIGN-CONDITIONAL SCALES")
    log.info(f"{'='*60}")
    log.info(f"  Sign-conditional scales:  {gsm:.1%} GSM8K")
    log.info(f"  Standard scales (math):   {SCALE_MATH:.1%} GSM8K  (baseline)")
    log.info(f"  No training:              {SCALE_BASELINE:.1%} GSM8K")
    log.info(f"  Verdict: {verdict}")
    log.info(f"  Total: {(time.time()-t0)/60:.1f} min")

    results = {
        "gsm8k": gsm, "baseline_math": SCALE_MATH,
        "delta": delta, "verdict": verdict,
        "n_scale_params": n_params, "mb_fp32": n_params * 4 / 1e6,
    }
    out = f"{CKPT}/sign_conditional_results.json"
    os.makedirs(CKPT, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"  Saved: {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"CRASHED: {e}", exc_info=True)
        sys.exit(1)
    finally:
        log.info("DONE")
