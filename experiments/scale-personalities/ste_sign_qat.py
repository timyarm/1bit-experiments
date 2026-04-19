"""Proper STE Sign QAT — Exp 24.

Tests whether signs have genuine additional capacity beyond scales, using a
clean method. Unlike Exp 22 (EFI), which did greedy sign swapping without STE
(making the result inconclusive), this uses a proper straight-through estimator:

  - Forward pass: uses sign(w_fp32) — binary computation, same as inference
  - Backward pass: STE — gradient flows through sign() as if identity
  - Optimizer knows signs will be binarized; optimizes toward flipping

Scales are FROZEN throughout — this isolates whether signs themselves can
carry additional math capability on top of the 28% scale-only ceiling.

After training: count how many sign_weights crossed zero (actual flips),
rebinarize, eval GSM8K.

Exp 22 null result (27%) was inconclusive — method was too broken to answer
the question. This gives a clean answer.

K=1% selected for local 6GB GPU feasibility.
Baseline: 28.0% scale-only (Exp 7), 27.0% EFI broken (Exp 22)
Hypothesis: proper STE → 29-35% if signs have capacity; ~28% if they don't

Monitor: ssh -p 2222 timyarm@100.110.173.110 "tail -f /tmp/ste_sign.log"
"""
import sys, os, time, json, re, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_PATH = "/tmp/ste_sign.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_PATH, mode='w'),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("ste")

DEVICE = "cuda"
MAX_LEN = 128
TRAIN_EXAMPLES = 150
LR_SIGNS = 1e-2        # needs to be high enough to push across the ±1 → flip boundary
EPOCHS = 3
TOKEN_SELECT_RATIO = 0.6
DOMAIN_MIX = 0.7
N_GSM8K_EVAL = 100
K_PERCENT = 1.0        # % of weight groups to unfreeze
GROUP_SIZE = 128

SCALE_BASELINE = 0.053
SCALE_MATH = 0.280
EFI_RESULT = 0.270

CKPT = os.path.expanduser("~/freigent/apps/trucksim/checkpoints")


# ── STE autograd function ─────────────────────────────────────────────────────

class _STESign(torch.autograd.Function):
    """Forward: binary sign. Backward: identity (straight-through)."""
    @staticmethod
    def forward(ctx, x):
        return x.sign()

    @staticmethod
    def backward(ctx, grad):
        return grad  # pass gradient through unchanged


def ste_sign(x):
    return _STESign.apply(x)


# ── STE BitLinear ─────────────────────────────────────────────────────────────

class STEBitLinear(nn.Module):
    """1-bit linear layer with trainable fp32 sign weights and frozen scales.

    sign_weights initialized to ±1.0. STE makes the optimizer aware of the
    binarization boundary — gradients push sign_weights toward crossing zero,
    which is an actual sign flip.

    Scales are frozen buffers (we're isolating sign capacity, not scale).
    """
    def __init__(self, weight_fp16, group_size=GROUP_SIZE):
        super().__init__()
        self.out_features, self.in_features = weight_fp16.shape
        self.group_size = group_size

        # Extract signs as ±1.0 fp32 — these become the trainable params
        signs = weight_fp16.sign().float()
        signs[signs == 0] = 1.0
        self.sign_weights = nn.Parameter(signs)  # will selectively freeze groups

        # Scales as frozen buffer
        w_flat = weight_fp16.reshape(-1, group_size)
        scales = w_flat.abs().mean(dim=1).float()
        self.register_buffer("scales", scales)
        self.register_buffer("orig_signs", signs.clone())  # for flip counting

    def forward(self, x):
        gs = self.group_size
        out_f, in_f = self.out_features, self.in_features
        # STE: binary forward, gradient flows through
        binary = ste_sign(self.sign_weights)  # [out_f, in_f]
        weight = binary * self.scales.flatten().repeat_interleave(gs).view(out_f, in_f)
        return F.linear(x, weight.to(x.dtype))


def convert_to_ste(model, skip_names=("lm_head",)):
    converted = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if any(s in name for s in skip_names):
            continue
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = dict(model.named_modules())[parent_name] if parent_name else model
        setattr(parent, child_name, STEBitLinear(module.weight.data))
        converted += 1
    return converted


# ── Importance ranking ────────────────────────────────────────────────────────

def rank_groups_by_importance(model, tokenizer, texts, n_rank=20):
    """One backward pass to rank groups by |grad_sign_weight|.

    Since sign_weights are fp32 ±1.0, their gradient magnitude tells us which
    groups are under the most optimization pressure — highest expected flip value.
    """
    log.info("  Ranking pass...")

    # Freeze everything except sign_weights for ranking
    for p in model.parameters():
        p.requires_grad_(False)
    for name, module in model.named_modules():
        if isinstance(module, STEBitLinear):
            module.sign_weights.requires_grad_(True)

    model.train()
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=0.0)

    # Accumulate gradients over n_rank examples
    optimizer.zero_grad()
    for text in texts[:n_rank]:
        tokens = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=MAX_LEN).to(DEVICE)
        if tokens.input_ids.shape[1] < 10:
            continue
        with torch.amp.autocast('cuda', dtype=torch.float16):
            out = model(**tokens)
            logits = out.logits[:, :-1]
            labels = tokens.input_ids[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                labels.reshape(-1))
        loss.backward()

    # Collect group-level importance: mean |grad| per group
    layer_names, layer_imps, layer_sizes = [], [], []
    for name, module in model.named_modules():
        if not isinstance(module, STEBitLinear):
            continue
        if module.sign_weights.grad is None:
            continue
        gs = module.group_size
        grad_flat = module.sign_weights.grad.abs().reshape(-1, gs).mean(dim=1)
        imp = grad_flat.cpu().float().numpy()
        layer_names.append(name)
        layer_imps.append(imp)
        layer_sizes.append(len(imp))

    # Freeze sign_weights again — ranking pass done
    for name, module in model.named_modules():
        if isinstance(module, STEBitLinear):
            module.sign_weights.requires_grad_(False)
    optimizer.zero_grad()

    return layer_names, layer_imps, layer_sizes


def select_top_k_groups(layer_names, layer_imps, layer_sizes, k_pct):
    all_imp = np.concatenate(layer_imps)
    total_groups = len(all_imp)
    k_groups = max(1, int(total_groups * k_pct / 100.0))

    top_k_global = np.argpartition(all_imp, -k_groups)[-k_groups:]
    offsets = np.concatenate([[0], np.cumsum(layer_sizes[:-1])])

    # Map global indices → (layer_name, group_local_idx)
    selected = {}  # layer_name → set of group indices
    for g_idx in top_k_global:
        layer_i = int(np.searchsorted(offsets, g_idx, side='right') - 1)
        group_local = int(g_idx - offsets[layer_i])
        name = layer_names[layer_i]
        selected.setdefault(name, set()).add(group_local)

    log.info(f"  Top K={k_pct}%: {k_groups:,} groups across {len(selected)} layers")
    return selected, k_groups


def unfreeze_top_groups(model, selected_groups):
    """Unfreeze sign_weights for selected groups only.

    For each selected group g, sign_weights rows [g*gs : (g+1)*gs] become
    trainable. All other rows stay frozen via a custom gradient hook.
    """
    sign_params = []

    for name, module in model.named_modules():
        if not isinstance(module, STEBitLinear):
            continue
        if name not in selected_groups:
            continue

        groups = sorted(selected_groups[name])
        gs = module.group_size
        out_f = module.out_features

        # Build row mask: True for rows belonging to selected groups
        mask = torch.zeros(out_f, dtype=torch.bool)
        for g in groups:
            start = g * gs
            end = min(start + gs, out_f)
            mask[start:end] = True

        module.sign_weights.requires_grad_(True)

        # Hook: zero out gradients for non-selected rows after each backward
        mask_cuda = mask.to(DEVICE)
        def make_hook(m):
            def hook(grad):
                return grad * m.unsqueeze(1).float()
            return hook
        module.sign_weights.register_hook(make_hook(mask_cuda))

        sign_params.append(module.sign_weights)

    return sign_params


# ── Data ──────────────────────────────────────────────────────────────────────

def get_data():
    from datasets import load_dataset
    math_texts, diverse_texts = [], []
    ds = load_dataset("gsm8k", "main", split="train", streaming=True)
    for ex in ds:
        math_texts.append(f"Question: {ex['question']}\nAnswer: {ex['answer']}")
        if len(math_texts) >= TRAIN_EXAMPLES * 2:
            break
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
    for ex in ds:
        if len(ex['text'].strip()) > 200:
            diverse_texts.append(ex['text'])
        if len(diverse_texts) >= TRAIN_EXAMPLES:
            break
    rng = np.random.default_rng(42)
    n_math = int(TRAIN_EXAMPLES * DOMAIN_MIX)
    mixed = math_texts[:n_math] + diverse_texts[:TRAIN_EXAMPLES - n_math]
    return [mixed[i] for i in rng.permutation(len(mixed))]


# ── Training ──────────────────────────────────────────────────────────────────

def train(model, tokenizer, texts, sign_params):
    optimizer = torch.optim.SGD(sign_params, lr=LR_SIGNS, momentum=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, EPOCHS * len(texts)))

    model.train()
    for epoch in range(EPOCHS):
        total_loss, n = 0.0, 0
        for text in texts:
            tokens = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=MAX_LEN).to(DEVICE)
            if tokens.input_ids.shape[1] < 10:
                continue
            with torch.amp.autocast('cuda', dtype=torch.float16):
                out = model(**tokens)
                logits = out.logits[:, :-1]
                labels = tokens.input_ids[:, 1:]
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    labels.reshape(-1), reduction='none')
                threshold = torch.quantile(ce, 1.0 - TOKEN_SELECT_RATIO)
                mask = (ce >= threshold).float()
                loss = (ce * mask).sum() / (mask.sum() + 1e-8)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(sign_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n += 1
            if n % 30 == 0:
                torch.cuda.empty_cache()

        log.info(f"  Epoch {epoch+1}/{EPOCHS}: loss={total_loss/max(n,1):.4f}  "
                 f"VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB")


def count_flips(model):
    total_signs, total_flips = 0, 0
    for module in model.modules():
        if not isinstance(module, STEBitLinear):
            continue
        current = module.sign_weights.data.sign()
        current[current == 0] = 1.0
        orig = module.orig_signs
        flipped = (current != orig).sum().item()
        total_flips += flipped
        total_signs += orig.numel()
    return total_flips, total_signs


def rebinarize(model):
    """Snap all sign_weights back to {-1,+1}."""
    for module in model.modules():
        if isinstance(module, STEBitLinear):
            with torch.no_grad():
                s = module.sign_weights.data.sign()
                s[s == 0] = 1.0
                module.sign_weights.data.copy_(s)


# ── Eval ──────────────────────────────────────────────────────────────────────

def eval_gsm8k(model, tokenizer, n=N_GSM8K_EVAL):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test", streaming=True)
    questions = []
    for ex in ds:
        questions.append((ex['question'], ex['answer']))
        if len(questions) >= n:
            break

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
        nums = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', gen.replace(',', ''))
        pred = nums[-1] if nums else ""
        ref_nums = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', a.replace(',', ''))
        ref = ref_nums[-1] if ref_nums else ""
        if pred == ref:
            correct += 1
        if (i + 1) % 10 == 0:
            log.info(f"  [{i+1}/{n}] acc={correct/(i+1):.1%}")
    return correct / n


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log.info("STE SIGN QAT — Exp 24")
    log.info(f"K={K_PERCENT}%, LR_signs={LR_SIGNS}, epochs={EPOCHS}")
    log.info(f"Baseline: scale-only {SCALE_MATH:.1%}, EFI broken {EFI_RESULT:.1%}")

    log.info("\n--- MODEL ---")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "prism-ml/Bonsai-1.7B-unpacked", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "prism-ml/Bonsai-1.7B-unpacked", dtype=torch.float16,
        trust_remote_code=True, device_map="cpu", attn_implementation="eager")
    n_conv = convert_to_ste(model)
    model = model.to(DEVICE)
    torch.cuda.synchronize()
    log.info(f"  Converted {n_conv} layers to STE  "
             f"VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Freeze all sign_weights by default
    for module in model.modules():
        if isinstance(module, STEBitLinear):
            module.sign_weights.requires_grad_(False)

    log.info("\n--- DATA ---")
    texts = get_data()
    log.info(f"  {len(texts)} examples ({int(DOMAIN_MIX*100)}% math)")

    log.info("\n--- RANKING ---")
    layer_names, layer_imps, layer_sizes = rank_groups_by_importance(
        model, tokenizer, texts, n_rank=20)
    total_groups = sum(layer_sizes)
    total_signs = sum(
        m.sign_weights.numel() for m in model.modules()
        if isinstance(m, STEBitLinear))
    log.info(f"  Total groups: {total_groups:,}  Total signs: {total_signs:,}")

    selected_groups, k_groups = select_top_k_groups(
        layer_names, layer_imps, layer_sizes, K_PERCENT)
    log.info(f"  Selected {k_groups:,} groups = {k_groups*GROUP_SIZE:,} signs "
             f"({100*k_groups*GROUP_SIZE/total_signs:.2f}% of all signs)")

    sign_params = unfreeze_top_groups(model, selected_groups)
    log.info(f"  Unfrozen {len(sign_params)} parameter tensors with gradient hooks")
    torch.cuda.empty_cache()
    log.info(f"  VRAM after unfreeze: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    log.info("\n--- TRAIN (STE) ---")
    train(model, tokenizer, texts, sign_params)

    log.info("\n--- SIGN FLIP COUNT ---")
    flips, total = count_flips(model)
    flip_pct = 100.0 * flips / total
    log.info(f"  Actual flips: {flips:,} / {total:,} ({flip_pct:.3f}%)")
    log.info(f"  Of unfrozen signs: {100.0*flips/(k_groups*GROUP_SIZE):.1f}% crossed zero")

    log.info("\n--- REBINARIZE ---")
    rebinarize(model)
    log.info("  Done.")

    log.info(f"\n--- EVAL (GSM8K n={N_GSM8K_EVAL}) ---")
    acc = eval_gsm8k(model, tokenizer, N_GSM8K_EVAL)

    log.info("")
    log.info("=" * 60)
    log.info("RESULTS — STE SIGN QAT")
    log.info("=" * 60)
    log.info(f"  No training:           {SCALE_BASELINE:.1%}")
    log.info(f"  Scale-only (Exp 7):    {SCALE_MATH:.1%}")
    log.info(f"  EFI broken (Exp 22):   {EFI_RESULT:.1%}  ← inconclusive method")
    log.info(f"  STE signs K={K_PERCENT}%:     {acc:.1%}  ← clean result")
    delta = acc - SCALE_MATH
    log.info(f"  Delta vs scale-only:   {delta:+.1%}")
    log.info(f"  Sign flips:            {flips:,} ({flip_pct:.3f}% of all signs)")
    if delta > 0.03:
        verdict = "SIGNS HAVE CAPACITY — worth pursuing at higher K% on A100"
    elif delta > 0.0:
        verdict = "MARGINAL LIFT — signs contribute but are not the main bottleneck"
    elif delta >= -0.02:
        verdict = "NULL — signs have no additional capacity beyond scales at K=1%"
    else:
        verdict = "REGRESSION — sign flips disrupted existing routing structure"
    log.info(f"  Verdict: {verdict}")
    log.info("=" * 60)

    results = {
        "acc": float(acc),
        "delta_vs_scale_only": float(delta),
        "scale_baseline": SCALE_MATH,
        "efi_broken": EFI_RESULT,
        "k_percent": K_PERCENT,
        "k_groups": int(k_groups),
        "total_groups": int(total_groups),
        "total_signs": int(total_signs),
        "actual_flips": int(flips),
        "flip_pct": float(flip_pct),
        "verdict": verdict,
    }
    os.makedirs(CKPT, exist_ok=True)
    out_path = f"{CKPT}/ste_sign_qat_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"\n  Saved: {out_path}")
    log.info(f"  Total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"CRASHED: {e}", exc_info=True)
        sys.exit(1)
    finally:
        log.info("DONE")
