"""EFI Sign Unfreeze — Exp 22.

Extreme Feature Importance: unfreeze the top K% of signs by gradient
magnitude, train them alongside scales on math data. If sign CAPACITY
is the bottleneck (not just scale amplitude), this should push past
the 28% GSM8K scale-only ceiling.

Theory:
  - Scale training ceiling: ~28% (scales fully exploited)
  - Sign flips change the routing structure itself
  - Top K% signs = highest gradient pressure = most "wrong" for math
  - Flipping them correctly should unlock new reasoning circuits

Setup:
  - Bonsai 1.7B, same v2 recipe (AdamW lr=1e-4, Rho-1, elastic band)
  - Ranking pass: 1 forward+backward, rank signs by |grad_w * scale|
  - Unfreeze top K=1% signs (~16M of 1.6B)
  - SGD for sign params (no Adam state = saves ~128MB VRAM)
  - AdamW for scale params (as normal)
  - After training: binarize updated signs back to {-1,+1} fp16
  - Eval: GSM8K n=100, same 0-shot protocol

Baseline: 28.0% GSM8K (scale math only)
Hypothesis: signs at K=1% → 32-35%+ (sign capacity was bottleneck)

Monitor: ssh -p 2222 timyarm@100.110.173.110 "tail -f /tmp/efi.log"
"""
import sys, os, time, json, re, types, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_PATH = "/tmp/efi.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_PATH, mode='w'),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("efi")

DEVICE = "cuda"
MAX_LEN = 128
TRAIN_EXAMPLES = 150
LR_SCALES = 1e-4
LR_SIGNS  = 1e-3   # higher LR for signs — they need to flip decisively
EPOCHS = 3
REG_LAMBDA = 0.1
TOKEN_SELECT_RATIO = 0.6
DOMAIN_MIX = 0.7
N_GSM8K_EVAL = 100
K_PERCENT = 1.0    # % of signs to unfreeze
CKPT = os.path.expanduser("~/freigent/apps/trucksim/checkpoints")

SCALE_BASELINE = 0.053
SCALE_MATH     = 0.280

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
    return converted
'''


def inject_pbl():
    pbl = types.ModuleType("packed_bitlinear")
    exec(PACKED_BITLINEAR_SRC, pbl.__dict__)
    sys.modules["packed_bitlinear"] = pbl
    return pbl.PackedBitLinear, pbl.convert_model


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


def rank_signs_by_gradient(model, tokenizer, texts, PackedBitLinear):
    """Rank sign GROUPS by |grad_scale| — kept at group level to avoid OOM.
    All signs in a group share the same importance, so per-sign expansion
    wastes 128× RAM (820M floats = 3.2GB) with no new information."""
    log.info("  Ranking signs by gradient magnitude (1 pass, group level)...")
    model.train()

    # Group-level importance only — [n_groups] per layer, not [out_f, in_f]
    sign_importance = {}  # name → [n_groups] cpu tensor

    for i, text in enumerate(texts[:20]):
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
            loss = ce.mean()

        loss.backward()

        for name, module in model.named_modules():
            if isinstance(module, PackedBitLinear):
                if module.scales.grad is not None:
                    imp = module.scales.grad.abs().flatten()  # [n_groups] only
                    if name not in sign_importance:
                        sign_importance[name] = imp.detach().cpu()
                    else:
                        sign_importance[name] += imp.detach().cpu()

        model.zero_grad()
        torch.cuda.empty_cache()

        if i % 5 == 0:
            log.info(f"    ranking step {i+1}/20")

    # Vectorized top-K GROUP selection (~6.4M groups = 25MB, not 820M signs = 3.2GB)
    total_signs = sum(m.signs.numel() for m in model.modules()
                      if isinstance(m, PackedBitLinear))
    total_groups = sum(m.scales.numel() for m in model.modules()
                       if isinstance(m, PackedBitLinear))
    k_groups = max(1, int(total_groups * K_PERCENT / 100))

    layer_names = list(sign_importance.keys())
    layer_imps = [sign_importance[n].numpy() for n in layer_names]
    layer_sizes = [len(f) for f in layer_imps]
    all_imp = np.concatenate(layer_imps)  # [total_groups]

    top_k_global = np.argpartition(all_imp, -k_groups)[-k_groups:]

    offsets = np.cumsum([0] + layer_sizes[:-1])
    top_k = []
    for g_idx in top_k_global:
        layer_i = int(np.searchsorted(offsets, g_idx, side='right') - 1)
        group_local = int(g_idx - offsets[layer_i])
        top_k.append((float(all_imp[g_idx]), layer_names[layer_i], group_local))

    log.info(f"  Total signs: {total_signs:,} in {total_groups:,} groups")
    log.info(f"  Unfreezing top {K_PERCENT}% = {k_groups:,} groups ≈ {k_groups*128:,} signs")
    return top_k, sign_importance


def unfreeze_top_signs(model, top_k_info, PackedBitLinear):
    """Convert top-K group positions to trainable float sign parameters.
    top_k_info contains (importance, name, group_local_idx) triples."""
    module_group_indices = {}
    for importance, name, group_local in top_k_info:
        module_group_indices.setdefault(name, []).append(group_local)

    sign_params = []
    module_sign_params = {}

    for name, module in model.named_modules():
        if not isinstance(module, PackedBitLinear): continue
        if name not in module_group_indices: continue

        gs = module.group_size
        n_signs = module.signs.numel()
        group_arr = np.array(module_group_indices[name], dtype=np.int64)

        # Vectorized group→sign expansion
        sign_idx = (group_arr[:, None] * gs + np.arange(gs, dtype=np.int64)[None, :]).flatten()
        sign_idx = sign_idx[sign_idx < n_signs]  # clip to valid range
        indices = torch.from_numpy(sign_idx)

        flat_signs = module.signs.flatten().float()
        selected_vals = flat_signs[indices].detach()
        sign_param = nn.Parameter(selected_vals.to(DEVICE))
        sign_params.append(sign_param)
        module_sign_params[name] = (module, indices, sign_param)

    total_unfrozen = sum(p.numel() for p in sign_params)
    log.info(f"  Created {len(sign_params)} sign parameter tensors ({total_unfrozen:,} signs, {total_unfrozen*4/1e6:.1f}MB fp32)")
    return sign_params, module_sign_params


class EFIModel(nn.Module):
    """Wraps model with EFI sign overrides applied during forward."""

    def __init__(self, base_model, module_sign_params, PackedBitLinear):
        super().__init__()
        self.base = base_model
        self.sign_params = nn.ParameterList()
        self._overrides = {}  # name → (module, flat_indices, param_idx)

        for name, (module, indices, param) in module_sign_params.items():
            param_idx = len(self.sign_params)
            self.sign_params.append(param)
            self._overrides[name] = (module, indices, param_idx)

        self._PackedBitLinear = PackedBitLinear

    def apply_sign_overrides(self):
        """Write current sign param values back into module sign buffers."""
        for name, (module, indices, param_idx) in self._overrides.items():
            param = self.sign_params[param_idx]
            flat = module.signs.flatten()
            flat[indices] = param.detach().half()
            module.signs.copy_(flat.view_as(module.signs))

    def forward(self, **kwargs):
        self.apply_sign_overrides()
        return self.base(**kwargs)

    def generate(self, **kwargs):
        self.apply_sign_overrides()
        return self.base.generate(**kwargs)


def train(model, tokenizer, texts, sign_params, scale_params):
    optimizer_scales = torch.optim.AdamW(scale_params, lr=LR_SCALES, weight_decay=0.01)
    optimizer_signs  = torch.optim.SGD(sign_params, lr=LR_SIGNS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_scales, T_max=max(1, EPOCHS * len(texts)))

    model.base.train()
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

                reg = sum(
                    F.mse_loss(m.scales, m._original_scales.to(m.scales.device))
                    for m in model.base.modules()
                    if isinstance(m, model._PackedBitLinear)
                )
                loss = task_loss + REG_LAMBDA * reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(scale_params + sign_params, 1.0)
            optimizer_scales.step()
            optimizer_signs.step()
            scheduler.step()
            optimizer_scales.zero_grad()
            optimizer_signs.zero_grad()
            epoch_loss += task_loss.item()
            n += 1
            if n % 10 == 0:
                torch.cuda.empty_cache()

        log.info(f"  epoch {epoch+1}/{EPOCHS}  loss={epoch_loss/max(n,1):.4f}")


def eval_gsm8k(model, tokenizer):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test", streaming=True)
    model.base.eval()
    correct = total = 0
    with torch.no_grad():
        for ex in ds:
            if total >= N_GSM8K_EVAL: break
            prompt = f"Question: {ex['question']}\nAnswer:"
            toks = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                out = model.generate(
                    input_ids=toks.input_ids,
                    attention_mask=toks.attention_mask,
                    max_new_tokens=200, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id)
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


def analyze_sign_flips(model, module_sign_params):
    """How many of the unfrozen signs actually flipped polarity?"""
    flipped = 0
    total = 0
    for name, (module, indices, param) in module_sign_params.items():
        original = module.signs.flatten()[indices].float()
        current = param.detach()
        flips = ((original * current) < 0).sum().item()
        flipped += flips
        total += len(indices)
    pct = 100 * flipped / max(total, 1)
    log.info(f"  Sign flips: {flipped:,} / {total:,} ({pct:.1f}%) changed polarity")
    return flipped, total


def main():
    t0 = time.time()
    log.info("EFI SIGN UNFREEZE — Exp 22")
    log.info(f"K={K_PERCENT}% signs unfrozen, SGD lr={LR_SIGNS}, scales AdamW lr={LR_SCALES}")
    log.info(f"Baseline to beat: {SCALE_MATH:.1%} GSM8K (scale math only)")

    log.info("GPU warmup...")
    _d = torch.ones(256, 256, device=DEVICE)
    for _ in range(10): _d = _d @ _d.T
    torch.cuda.synchronize()
    del _d
    time.sleep(3)

    PackedBitLinear, convert_model = inject_pbl()

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

    # Freeze everything except scales before ranking — lm_head (not converted) would
    # otherwise compute a 32000×2048 gradient matrix during backward, causing OOM
    for name, p in model.named_parameters():
        p.requires_grad_('scales' in name)
    log.info(f"  Scale params: {sum(p.numel() for n,p in model.named_parameters() if p.requires_grad):,}")

    log.info("\n--- DATA ---")
    texts = get_data()
    log.info(f"  {len(texts)} examples ({int(DOMAIN_MIX*100)}% math)")

    log.info("\n--- RANKING SIGNS ---")
    top_k_info, _ = rank_signs_by_gradient(model, tokenizer, texts, PackedBitLinear)

    log.info("\n--- UNFREEZING TOP SIGNS ---")
    sign_params_list, module_sign_params = unfreeze_top_signs(model, top_k_info, PackedBitLinear)
    efi_model = EFIModel(model, module_sign_params, PackedBitLinear)

    # Freeze everything except scales + selected sign params
    for p in model.parameters():
        p.requires_grad_(False)
    scale_params = [m.scales for m in model.modules()
                    if isinstance(m, PackedBitLinear)]
    for p in scale_params:
        p.requires_grad_(True)

    sign_vram = sum(p.numel() for p in sign_params_list) * 4 / 1e6
    scale_vram = sum(p.numel() for p in scale_params) * 4 / 1e6
    log.info(f"  Sign params (fp32): {sum(p.numel() for p in sign_params_list):,} ({sign_vram:.1f}MB)")
    log.info(f"  Scale params (fp32): {sum(p.numel() for p in scale_params):,} ({scale_vram:.1f}MB)")
    log.info(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Smoke test
    log.info("  Smoke test...")
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        _t = tokenizer("test", return_tensors="pt").to(DEVICE)
        _o = efi_model(**_t)
        torch.cuda.synchronize()
    log.info(f"  OK: logits={_o.logits.shape}")
    del _t, _o

    log.info("\n--- TRAIN ---")
    train(efi_model, tokenizer, texts, sign_params_list, scale_params)
    log.info(f"  Trained in {(time.time()-t0)/60:.1f} min")

    log.info("\n--- SIGN FLIP ANALYSIS ---")
    analyze_sign_flips(efi_model, module_sign_params)

    log.info(f"\n--- EVAL (GSM8K n={N_GSM8K_EVAL}) ---")
    gsm = eval_gsm8k(efi_model, tokenizer)

    delta = gsm - SCALE_MATH
    if delta > 0.02:
        verdict = f"BEATS scale-only (Δ={delta:+.1%}) — sign capacity was bottleneck"
    elif delta > -0.02:
        verdict = f"MATCHES scale-only (Δ={delta:+.1%}) — sign flips don't add signal"
    else:
        verdict = f"BELOW scale-only (Δ={delta:+.1%}) — sign flips hurt"

    log.info(f"\n{'='*60}")
    log.info("RESULTS — EFI SIGN UNFREEZE")
    log.info(f"{'='*60}")
    log.info(f"  EFI K={K_PERCENT}% signs:   {gsm:.1%} GSM8K")
    log.info(f"  Scale-only baseline:  {SCALE_MATH:.1%} GSM8K")
    log.info(f"  No training:          {SCALE_BASELINE:.1%} GSM8K")
    log.info(f"  Verdict: {verdict}")
    log.info(f"  Total: {(time.time()-t0)/60:.1f} min")

    results = {
        "gsm8k": gsm, "baseline_math": SCALE_MATH, "delta": delta,
        "k_percent": K_PERCENT, "verdict": verdict,
    }
    out = f"{CKPT}/efi_results.json"
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
