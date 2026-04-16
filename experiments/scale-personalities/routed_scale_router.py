"""Per-token scale router on Bonsai 1.7B — Experiment 1.

Loads 4 profile scale tables (baseline + math-v2 + knowledge-v2 + code-v2),
replaces each Linear with a RoutedBitLinear that holds all 4 profiles, and
trains a small router head to pick (soft blend) which profile to use per token.

Architecture:
  x → embeds → router → routing[B,S,4] → for each RoutedBitLinear:
    out_p = F.linear(x, signs * scales_p)  for p in 0..3
    out = sum_p routing[b,s,p] * out_p

Profile scales are FROZEN. Only the router head trains.

This tests the compounding hypothesis directly: can per-token routing
recover math scales' ARC-Easy collapse while preserving the HellaSwag gain?

Monitor: tail -f /tmp/router_train.log
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import sys
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("/tmp/router_train.log", mode='w'),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("router")

DEVICE = "cuda"
GROUP_SIZE = 128
CKPT_DIR = os.path.expanduser("~/freigent/apps/trucksim/checkpoints")
PROFILES = ["baseline", "math", "knowledge", "code"]
PROFILE_FILES = {
    "baseline": "original_scales.pt",
    "math": "math_scales_v2.pt",
    "knowledge": "knowledge_scales_v2.pt",
    "code": "code_scales_v2.pt",
}

# Training config
MAX_LEN = 64
EPOCHS = 3
TRAIN_EXAMPLES_PER_DOMAIN = 120  # 4 domains * 120 = 480 examples
LR = 5e-4
ENTROPY_BONUS = 0.001
DOMAIN_CE_WEIGHT = 1.0  # V2: teach router to classify domain from input

# Map training domain → profile index (PROFILES = ["baseline", "math", "knowledge", "code"])
DOMAIN_TO_PROFILE = {"general": 0, "math": 1, "knowledge": 2, "code": 3}


# ─── RoutedBitLinear ────────────────────────────────────────────────────

class _RoutedBitFn(torch.autograd.Function):
    """Custom autograd: weight is NOT stored in graph (recomputed in backward).

    Saves VRAM by not keeping the [out, in] weight tensor per-layer.
    Mirrors the PackedBitLinear technique but with routing through profile scales.
    """

    @staticmethod
    def forward(ctx, x, signs, profile_scales, routing, out_features, in_features, group_size):
        blended = (routing.view(-1, 1) * profile_scales).sum(dim=0)  # [n_groups]
        scales_exp = blended.unsqueeze(1).expand(-1, group_size).reshape(out_features, in_features)
        weight = signs * scales_exp.half()
        output = F.linear(x, weight)
        ctx.save_for_backward(x, signs, profile_scales, routing)
        ctx.out_features = out_features
        ctx.in_features = in_features
        ctx.group_size = group_size
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, signs, profile_scales, routing = ctx.saved_tensors
        out_f, in_f, gs = ctx.out_features, ctx.in_features, ctx.group_size

        # Recompute weight for grad_x
        blended = (routing.view(-1, 1) * profile_scales).sum(dim=0)
        scales_exp = blended.unsqueeze(1).expand(-1, gs).reshape(out_f, in_f)
        weight = signs * scales_exp.half()
        grad_x = grad_output.matmul(weight.to(grad_output.dtype))

        # grad w.r.t. blended scales per group
        grad_w = grad_output.reshape(-1, out_f).t().matmul(x.reshape(-1, in_f))  # [out, in]
        grad_w_signed = (grad_w * signs.to(grad_w.dtype)).reshape(-1, gs)
        grad_blended = grad_w_signed.sum(dim=1)  # [n_groups]

        # grad w.r.t. routing[p] = sum_g grad_blended[g] * profile_scales[p, g]
        grad_routing = profile_scales.to(grad_blended.dtype) @ grad_blended  # [P]

        return grad_x, None, None, grad_routing, None, None, None


class RoutedBitLinear(nn.Module):
    """1-bit linear with P profile scales; sequence-level routing blends scales.

    _current_routing is set externally before forward. If None, uses profile 0.
    Uses custom autograd to avoid storing [out, in] weight per-layer (saves VRAM).
    """

    def __init__(self, weight: torch.Tensor, profile_scales_list, group_size: int = 128):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.group_size = group_size
        self.n_profiles = len(profile_scales_list)

        signs = weight.sign().half()
        signs[signs == 0] = 1
        self.register_buffer('signs', signs)

        # Stack profile scales: [P, n_groups] fp16
        stacked = torch.stack([s.half().flatten() for s in profile_scales_list])
        self.register_buffer('profile_scales', stacked)

        self._current_routing = None

    def forward(self, x):
        if self._current_routing is None:
            # Inference default: baseline profile (no graph)
            blended = self.profile_scales[0]
            scales_exp = blended.unsqueeze(1).expand(-1, self.group_size).reshape(
                self.out_features, self.in_features
            )
            weight = self.signs * scales_exp.half()
            return F.linear(x, weight)

        return _RoutedBitFn.apply(
            x, self.signs, self.profile_scales, self._current_routing,
            self.out_features, self.in_features, self.group_size,
        )


# ─── Router ──────────────────────────────────────────────────────────────

class SequenceScaleRouter(nn.Module):
    """Small MLP on mean-pooled embeddings → softmax over P profiles.

    V1 routes at sequence level (one profile blend per input). Per-token
    routing is future work once we have more VRAM.

    Bias init [5, 0, 0, 0] → softmax ≈ [0.98, 0.007, 0.007, 0.007] so
    training starts with blended scales ≈ baseline (numerically stable).
    """

    def __init__(self, hidden_dim, n_profiles, mid=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.GELU(),
            nn.Linear(mid, n_profiles),
        )
        nn.init.zeros_(self.net[-1].weight)
        bias = torch.zeros(n_profiles)
        bias[0] = 5.0  # prefer baseline profile at init for stability
        with torch.no_grad():
            self.net[-1].bias.copy_(bias)

    def forward(self, embeds, return_logits=False):
        # embeds: [B, S, hidden] → mean-pool over S → [B, hidden]
        pooled = embeds.mean(dim=1)
        logits = self.net(pooled)  # [B, P]
        probs = F.softmax(logits, dim=-1)
        if probs.shape[0] == 1:
            probs = probs.squeeze(0)
            logits = logits.squeeze(0)
        if return_logits:
            return probs, logits
        return probs


# ─── Loading ─────────────────────────────────────────────────────────────

def load_profile_scales():
    """Load 4 profile scale dicts. Returns dict: profile_name → {layer_name: scales}."""
    out = {}
    for name, fname in PROFILE_FILES.items():
        path = os.path.join(CKPT_DIR, fname)
        scales = torch.load(path, weights_only=False)
        log.info(f"  loaded {name}: {len(scales)} layers")
        out[name] = scales
    return out


def build_routed_model(model_id="prism-ml/Bonsai-1.7B-unpacked"):
    """Load Bonsai 1.7B and replace Linear with RoutedBitLinear containing all 4 profiles."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading Bonsai 1.7B base model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, trust_remote_code=True
    )

    log.info("Loading 4 profile scales...")
    all_profiles = load_profile_scales()

    log.info("Converting Linear → RoutedBitLinear...")
    converted = 0
    skipped = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if 'lm_head' in name:
            skipped += 1
            continue

        # Gather scales for this layer from all 4 profiles
        profile_scales_list = []
        missing = False
        for p in PROFILES:
            if name not in all_profiles[p]:
                log.warning(f"    {name} missing from {p}, using profile 0 scales")
                missing = True
                break
            profile_scales_list.append(all_profiles[p][name])

        if missing or len(profile_scales_list) != 4:
            skipped += 1
            continue

        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        parent = dict(model.named_modules())[parent_name] if parent_name else model

        routed = RoutedBitLinear(module.weight.data, profile_scales_list, GROUP_SIZE)
        setattr(parent, child_name, routed)
        converted += 1

    log.info(f"  converted: {converted}, skipped: {skipped}")
    return model, tokenizer


def set_routing(model, routing):
    """Attach routing tensor to all RoutedBitLinear modules in model."""
    for m in model.modules():
        if isinstance(m, RoutedBitLinear):
            m._current_routing = routing


def clear_routing(model):
    for m in model.modules():
        if isinstance(m, RoutedBitLinear):
            m._current_routing = None


# ─── Data ────────────────────────────────────────────────────────────────

def get_train_data(n_per_domain=TRAIN_EXAMPLES_PER_DOMAIN):
    """Mix of 4 domains: math (gsm8k), knowledge (trivia_qa), code (code_search_net), general (wikitext)."""
    from datasets import load_dataset
    texts = []
    domains = {
        "math": ("gsm8k", "main", "train"),
        "knowledge": ("trivia_qa", "unfiltered", "train"),
        "code": ("code_search_net", "python", "train"),
        "general": ("wikitext", "wikitext-2-raw-v1", "train"),
    }
    for dom, (ds_name, cfg, split) in domains.items():
        log.info(f"  loading {dom}...")
        try:
            if cfg:
                ds = load_dataset(ds_name, cfg, split=split, streaming=True)
            else:
                ds = load_dataset(ds_name, split=split, streaming=True)
        except Exception as e:
            log.warning(f"    {dom} load failed: {e}")
            continue

        taken = 0
        for ex in ds:
            if taken >= n_per_domain:
                break
            text = None
            if dom == "math":
                text = f"Question: {ex['question']}\nAnswer: {ex['answer']}"
            elif dom == "knowledge":
                q = ex.get('question', '')
                a = ex.get('answer', {}).get('value', '')
                if q and a:
                    text = f"Question: {q}\nAnswer: {a}"
            elif dom == "code":
                code = ex.get('func_code_string', '')
                if code and len(code) > 100:
                    text = code
            elif dom == "general":
                if len(ex.get('text', '').strip()) > 200:
                    text = ex['text']

            if text:
                texts.append((dom, text))
                taken += 1

        log.info(f"    {dom}: {taken}")

    np.random.shuffle(texts)
    return texts


# ─── Training ────────────────────────────────────────────────────────────

def train_router(model, router, tokenizer, train_data, epochs=EPOCHS, lr=LR):
    """Train ONLY the router. Profile scales frozen."""
    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False
    # Train router only
    for p in router.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in router.parameters())
    log.info(f"  router trainable: {trainable:,} params")

    optimizer = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_data)
    )

    model.eval()  # No dropout
    router.train()

    embed_layer = model.get_input_embeddings()

    for epoch in range(epochs):
        total_loss = 0
        total_entropy = 0
        total_domain_loss = 0
        domain_dist = {d: np.zeros(len(PROFILES)) for d in ["math", "knowledge", "code", "general"]}
        n = 0

        for dom, text in train_data:
            tokens = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=MAX_LEN
            ).to(DEVICE)
            if tokens.input_ids.shape[1] < 10:
                continue

            with torch.amp.autocast('cuda', dtype=torch.float16):
                embeds = embed_layer(tokens.input_ids)  # [B, S, hidden]

            # Router and blending in fp32 for numeric stability
            routing, route_logits = router(embeds.float(), return_logits=True)  # [P], [P] fp32

            set_routing(model, routing)
            try:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    out = model(
                        inputs_embeds=embeds,
                        labels=tokens.input_ids,
                        use_cache=False,
                    )
            finally:
                clear_routing(model)

            lm_loss = out.loss
            if not torch.isfinite(lm_loss):
                log.warning(f"    non-finite lm_loss on {dom} — skipping")
                optimizer.zero_grad()
                n += 1
                continue

            # Domain classification loss — teaches router to condition on input
            domain_idx = DOMAIN_TO_PROFILE[dom]
            target = torch.tensor([domain_idx], device=DEVICE)
            domain_loss = F.cross_entropy(route_logits.unsqueeze(0), target)

            entropy = -(routing * torch.log(routing + 1e-9)).sum()
            total_loss_term = lm_loss + DOMAIN_CE_WEIGHT * domain_loss - ENTROPY_BONUS * entropy

            total_loss_term.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += lm_loss.item()
            total_entropy += entropy.item()
            total_domain_loss = total_domain_loss + domain_loss.item() if n > 0 else domain_loss.item()
            # Track mean routing per domain
            mean_route = routing.detach().cpu().float().numpy()
            if mean_route.ndim > 1:
                mean_route = mean_route.mean(axis=tuple(range(mean_route.ndim - 1)))
            domain_dist[dom] += mean_route
            n += 1

        avg_loss = total_loss / max(n, 1)
        avg_ent = total_entropy / max(n, 1)
        avg_dom = total_domain_loss / max(n, 1)
        vram = torch.cuda.memory_allocated() / 1e9
        log.info(f"  Epoch {epoch+1}/{epochs}: lm={avg_loss:.4f} dom_ce={avg_dom:.4f} entropy={avg_ent:.3f} VRAM={vram:.1f}GB")
        for d, dist in domain_dist.items():
            dist = dist / max(n, 1) * 4  # normalize (rough)
            # normalize to sum to 1
            if dist.sum() > 0:
                dist = dist / dist.sum()
            log.info(f"    {d:>10}: " + " ".join(f"{PROFILES[i][:4]}={dist[i]:.2f}" for i in range(len(PROFILES))))


# ─── Evaluation ──────────────────────────────────────────────────────────

@torch.no_grad()
def generate_with_routing(model, router, tokenizer, prompt, max_tokens=30):
    """Greedy generation with per-token routing."""
    model.eval()
    router.eval()

    tokens = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = tokens.input_ids

    embed_layer = model.get_input_embeddings()
    generated = []

    for _ in range(max_tokens):
        with torch.amp.autocast('cuda', dtype=torch.float16):
            embeds = embed_layer(input_ids)
            routing = router(embeds.float()).half()

            set_routing(model, routing)
            try:
                out = model(inputs_embeds=embeds)
            finally:
                clear_routing(model)

        next_logits = out.logits[0, -1, :]
        next_id = next_logits.argmax().item()

        if next_id == tokenizer.eos_token_id:
            break
        generated.append(next_id)
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=DEVICE)], dim=1)

    return tokenizer.decode(generated, skip_special_tokens=True)


def eval_router_benchmarks(model, router, tokenizer, n_per_bench=100):
    """Run TriviaQA, ARC-Easy, HellaSwag with router active. Compare to baseline."""
    from datasets import load_dataset
    import re

    results = {}

    # TriviaQA
    log.info(f"  TriviaQA ({n_per_bench}q)...")
    ds = load_dataset("trivia_qa", "unfiltered", split="validation", streaming=True)
    correct = total = 0
    for ex in ds:
        if total >= n_per_bench:
            break
        q = ex['question']
        answers = ex.get('answer', {}).get('aliases', [])
        if not answers:
            continue
        resp = generate_with_routing(model, router, tokenizer, f"Question: {q}\nAnswer:", 30)
        if any(a.lower() in resp.lower() for a in answers):
            correct += 1
        total += 1
    results['trivia'] = {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}
    log.info(f"    trivia: {correct}/{total} = {correct/max(total,1):.1%}")

    # ARC-Easy
    log.info(f"  ARC-Easy ({n_per_bench}q)...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test", streaming=True)
    correct = total = 0
    for ex in ds:
        if total >= n_per_bench:
            break
        q = ex['question']
        choices = ex['choices']
        labels = choices['label']
        texts = choices['text']
        answer_key = ex['answerKey']
        options = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
        prompt = f"Question: {q}\n{options}\nAnswer:"
        resp = generate_with_routing(model, router, tokenizer, prompt, 3)
        resp_clean = resp.strip().upper()
        if resp_clean and resp_clean[0] == answer_key:
            correct += 1
        total += 1
    results['arc_easy'] = {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}
    log.info(f"    arc_easy: {correct}/{total} = {correct/max(total,1):.1%}")

    # HellaSwag
    log.info(f"  HellaSwag ({n_per_bench}q)...")
    ds = load_dataset("Rowan/hellaswag", split="validation", streaming=True)
    correct = total = 0
    for ex in ds:
        if total >= n_per_bench:
            break
        ctx = ex['ctx']
        endings = ex['endings']
        label = int(ex['label'])
        options = "\n".join(f"{i+1}. {e}" for i, e in enumerate(endings))
        prompt = f"Context: {ctx}\n\nWhich continuation is most likely?\n{options}\nAnswer:"
        resp = generate_with_routing(model, router, tokenizer, prompt, 3)
        nums = re.findall(r'[1-4]', resp[:10])
        if nums and int(nums[0]) - 1 == label:
            correct += 1
        total += 1
    results['hellaswag'] = {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}
    log.info(f"    hellaswag: {correct}/{total} = {correct/max(total,1):.1%}")

    return results


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("PER-TOKEN SCALE ROUTER — BONSAI 1.7B")
    log.info("=" * 60)

    t_start = time.time()

    model, tokenizer = build_routed_model()
    model = model.to(DEVICE)
    # Enable gradient checkpointing to fit transient weights in 6GB
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        log.info("  gradient checkpointing enabled (non-reentrant)")
    except TypeError:
        model.gradient_checkpointing_enable()
        log.info("  gradient checkpointing enabled (default)")
    except Exception as e:
        log.warning(f"  checkpointing not available: {e}")
    torch.cuda.empty_cache()
    log.info(f"  model on GPU, VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB")

    # Build router
    hidden_dim = model.config.hidden_size
    # Router stays fp32 for numeric stability; cheap (~260K params)
    router = SequenceScaleRouter(hidden_dim, len(PROFILES)).to(DEVICE)
    log.info(f"  router: {hidden_dim} → {len(PROFILES)} profiles")

    # Data
    log.info("Loading training data...")
    train_data = get_train_data()
    log.info(f"  {len(train_data)} examples total")

    # Train
    log.info("Training router...")
    train_router(model, router, tokenizer, train_data)

    # Save router
    router_path = os.path.join(CKPT_DIR, "per_token_router_v1.pt")
    torch.save({
        "router_state_dict": router.state_dict(),
        "profiles": PROFILES,
        "hidden_dim": hidden_dim,
        "config": {
            "epochs": EPOCHS,
            "lr": LR,
            "max_len": MAX_LEN,
            "entropy_bonus": ENTROPY_BONUS,
        }
    }, router_path)
    log.info(f"  saved router to {router_path}")

    # Evaluate
    log.info("Evaluating router on benchmarks (100q each)...")
    results = eval_router_benchmarks(model, router, tokenizer, n_per_bench=100)

    # Save results
    out_path = os.path.join(CKPT_DIR, "router_benchmark_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"  saved results to {out_path}")

    log.info(f"\n{'='*60}")
    log.info("ROUTER vs baseline (150q prior run):")
    log.info(f"{'='*60}")
    log.info(f"  TriviaQA:  router {results['trivia']['accuracy']:.1%} vs baseline 9.3%")
    log.info(f"  ARC-Easy:  router {results['arc_easy']['accuracy']:.1%} vs baseline 64.7%")
    log.info(f"  HellaSwag: router {results['hellaswag']['accuracy']:.1%} vs baseline 35.3%")
    log.info(f"\nTotal: {(time.time() - t_start)/60:.0f} min")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"CRASHED: {e}", exc_info=True)
    finally:
        log.info("DONE")
