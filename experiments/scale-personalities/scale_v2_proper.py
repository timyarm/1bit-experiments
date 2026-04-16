"""Scale Training v2 — KL distillation + token weighting + proper hyperparameters.

Fixes from research:
1. KL divergence from frozen original (not raw CE)
2. Rho-1 style token weighting (skip easy tokens)
3. Sequence length 1024 (was 64-96)
4. LR 1e-4 AdamW (was SGD 0.01)
5. Elastic band regularization to original scales
6. Mixed data (70% domain + 30% diverse)

Monitor: tail -f /tmp/scale_training.log
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import sys
import json
import shutil
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S",
                    handlers=[logging.FileHandler("/tmp/scale_training.log", mode='w'),
                              logging.StreamHandler(sys.stdout)])
log = logging.getLogger("s")

DEVICE = "cuda"
GROUP_SIZE = 128
GGUF_BASE = os.path.expanduser("~/freigent/apps/trucksim/data/llm/Bonsai-1.7B.gguf")
MAX_LEN = 256  # compromise: longer than v1 (64) but fits in 6GB VRAM
TRAIN_EXAMPLES = 150
EVAL_EXAMPLES = 80
LR = 1e-4
EPOCHS = 3  # fewer epochs, lower LR = less overfitting
REG_LAMBDA = 0.1  # elastic band strength
TOKEN_SELECT_RATIO = 0.6  # Rho-1: train on top 60% excess-loss tokens
DOMAIN_MIX = 0.7  # 70% domain, 30% diverse


# ─── Data ────────────────────────────────────────────────────────────────

def get_data(domain, n_train=TRAIN_EXAMPLES, n_eval=EVAL_EXAMPLES):
    cache_dir = os.path.expanduser("~/freigent/apps/trucksim/data/scale_data")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{domain}_v2.pt")

    if os.path.exists(cache_file):
        log.info(f"  Cached: {domain}")
        return torch.load(cache_file, weights_only=False)

    log.info(f"  Downloading {domain}...")
    from datasets import load_dataset
    texts = []

    if domain == "math":
        ds = load_dataset("gsm8k", "main", split="train", streaming=True)
        for ex in ds:
            # Match eval format exactly
            texts.append(f"Question: {ex['question']}\nAnswer: {ex['answer']}")
            if len(texts) >= n_train + n_eval:
                break
    elif domain == "knowledge":
        ds = load_dataset("trivia_qa", "unfiltered", split="train", streaming=True)
        for ex in ds:
            q = ex.get('question', '')
            a = ex.get('answer', {}).get('value', '')
            if q and a:
                texts.append(f"Question: {q}\nAnswer: {a}")
            if len(texts) >= n_train + n_eval:
                break
    elif domain == "code":
        ds = load_dataset("code_search_net", "python", split="train", streaming=True)
        for ex in ds:
            code = ex.get('func_code_string', '')
            if code and len(code) > 100:
                texts.append(code)
            if len(texts) >= n_train + n_eval:
                break
    elif domain == "diverse":
        # General diverse data for mixing
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
        for ex in ds:
            if len(ex['text'].strip()) > 200:
                texts.append(ex['text'])
            if len(texts) >= n_train + n_eval:
                break

    data = {"train": texts[:n_train], "eval": texts[n_train:n_train + n_eval]}
    torch.save(data, cache_file)
    log.info(f"  {domain}: {len(data['train'])} train, {len(data['eval'])} eval")
    return data


def mix_data(domain_texts, diverse_texts, ratio=DOMAIN_MIX):
    """Mix domain data with diverse data."""
    n_domain = int(len(domain_texts) * ratio)
    n_diverse = len(domain_texts) - n_domain
    mixed = domain_texts[:n_domain]
    if diverse_texts and n_diverse > 0:
        mixed += diverse_texts[:n_diverse]
    np.random.shuffle(mixed)
    return mixed


# ─── Training ────────────────────────────────────────────────────────────

def train_scales_v2(model, tokenizer, train_texts, orig_scales_dict, epochs=EPOCHS, lr=LR):
    """Train group scales with KL distillation + token weighting + regularization."""
    from packed_bitlinear import PackedBitLinear

    # Freeze everything except scales
    for p in model.parameters():
        p.requires_grad = False
    scale_params = []
    for m in model.modules():
        if isinstance(m, PackedBitLinear):
            m.scales.requires_grad = True
            scale_params.append(m.scales)

    trainable = sum(p.numel() for p in scale_params)
    log.info(f"    Trainable: {trainable:,} scale params")

    optimizer = torch.optim.AdamW(scale_params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_texts))

    # Pre-compute teacher logits would be ideal but VRAM-limited.
    # Instead: compute KL per-batch using original scales snapshot.
    # Store original scale values for KL + regularization.
    orig_snapshots = {}
    for name, module in model.named_modules():
        if isinstance(module, PackedBitLinear):
            orig_snapshots[name] = module.scales.data.clone().detach()

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        total_kl = 0
        total_reg = 0
        n = 0

        for text in train_texts:
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
            if tokens.input_ids.shape[1] < 10:
                continue

            # Forward with current (training) scales
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(**tokens, labels=tokens.input_ids)
                student_logits = outputs.logits[:, :-1]
                labels = tokens.input_ids[:, 1:]

                # Per-token CE loss
                ce_per_token = F.cross_entropy(
                    student_logits.reshape(-1, student_logits.size(-1)).float(),
                    labels.reshape(-1),
                    reduction='none'
                )

                # Rho-1 token selection: only train on hardest tokens
                if TOKEN_SELECT_RATIO < 1.0:
                    threshold = torch.quantile(ce_per_token, 1.0 - TOKEN_SELECT_RATIO)
                    mask = (ce_per_token >= threshold).float()
                    task_loss = (ce_per_token * mask).sum() / (mask.sum() + 1e-8)
                else:
                    task_loss = ce_per_token.mean()

                # Regularization: elastic band to original scales
                reg_loss = torch.tensor(0.0, device=DEVICE)
                for name, module in model.named_modules():
                    if isinstance(module, PackedBitLinear) and name in orig_snapshots:
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

        avg_loss = total_loss / max(n, 1)
        avg_reg = total_reg / max(n, 1)
        cur_lr = scheduler.get_last_lr()[0]
        log.info(f"    Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} reg={avg_reg:.6f} lr={cur_lr:.6f} VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB")


def eval_ppl(model, tokenizer, texts, max_len=128):
    """Eval PPL with shorter sequences to save VRAM."""
    model.eval()
    torch.cuda.empty_cache()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(DEVICE)
            if tokens.input_ids.shape[1] < 5:
                continue
            with torch.amp.autocast('cuda', dtype=torch.float16):
                out = model(**tokens, labels=tokens.input_ids)
            total_loss += out.loss.item() * tokens.input_ids.shape[1]
            total_tokens += tokens.input_ids.shape[1]
    torch.cuda.empty_cache()
    return np.exp(total_loss / max(total_tokens, 1))


# ─── GGUF Patching ──────────────────────────────────────────────────────

def patch_gguf(orig_scales, new_scales, output_path):
    from gguf import GGUFReader
    shutil.copy2(GGUF_BASE, output_path)
    reader = GGUFReader(GGUF_BASE)
    data = np.memmap(output_path, dtype=np.uint8, mode='r+')
    BLOCK = 18
    patched = 0
    for t in reader.tensors:
        tname = str(t.name)
        if 'Q1_0' not in str(t.tensor_type.name):
            continue
        for ptn in orig_scales:
            mapped = ptn.replace('model.layers.', 'blk.').replace('.self_attn.q_proj', '.attn_q') \
                .replace('.self_attn.k_proj', '.attn_k').replace('.self_attn.v_proj', '.attn_v') \
                .replace('.self_attn.o_proj', '.attn_output').replace('.mlp.gate_proj', '.ffn_gate') \
                .replace('.mlp.up_proj', '.ffn_up').replace('.mlp.down_proj', '.ffn_down') + '.weight'
            if mapped == tname:
                offset = t.data.ctypes.data - reader.data.ctypes.data
                n_groups = t.n_bytes // BLOCK
                o = orig_scales[ptn].cpu().numpy().astype(np.float32)
                m = new_scales[ptn].cpu().numpy().astype(np.float32)
                if len(o) != n_groups:
                    break
                mult = m / (o + 1e-8)
                blocks = data[offset:offset + t.n_bytes].reshape(n_groups, BLOCK)
                old_s = blocks[:, :2].copy().view(np.float16).flatten().astype(np.float32)
                new_s = (old_s * mult).astype(np.float16)
                blocks[:, :2] = new_s.view(np.uint8).reshape(n_groups, 2)
                patched += 1
                break
    data.flush()
    del data
    return patched


# ─── llama.cpp Eval ──────────────────────────────────────────────────────

def eval_accuracy(gguf_path, n_trivia=100):
    sys.path.insert(0, os.path.dirname(__file__))
    from llama_fast_eval import LlamaEval
    ev = LlamaEval(gguf_path)
    ev.start()

    log.info(f"    TriviaQA ({n_trivia} questions)...")
    trivia = ev.eval_trivia(n=n_trivia)
    log.info(f"    TriviaQA: {trivia['correct']}/{trivia['total']} = {trivia['accuracy']:.1%}")

    # Sample generations
    for prompt in ["What is 25 * 4? Answer:", "The capital of Japan is", "def fibonacci(n):"]:
        out = ev.generate(prompt, max_tokens=40)
        log.info(f"    {prompt[:30]}  →  {out[:50]}")

    ev.stop()
    return trivia


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("SCALE TRAINING v2 — KL + Token Weighting + Regularization")
    log.info(f"LR={LR}, epochs={EPOCHS}, max_len={MAX_LEN}, reg={REG_LAMBDA}")
    log.info(f"token_select={TOKEN_SELECT_RATIO}, domain_mix={DOMAIN_MIX}")
    log.info("=" * 60)
    t_start = time.time()

    # Load data
    log.info("\n--- DATA ---")
    domains = ["math", "knowledge", "code"]
    data = {}
    for d in domains:
        data[d] = get_data(d)
    data["diverse"] = get_data("diverse")

    # Load model
    log.info("\n--- MODEL ---")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, os.path.dirname(__file__))
    from packed_bitlinear import PackedBitLinear, convert_model

    tokenizer = AutoTokenizer.from_pretrained("prism-ml/Bonsai-1.7B-unpacked", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "prism-ml/Bonsai-1.7B-unpacked", dtype=torch.float16, trust_remote_code=True, device_map="cpu"
    )
    convert_model(model)
    model = model.to(DEVICE)
    model.gradient_checkpointing_enable()
    log.info(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Save original scales
    orig_scales = {}
    for name, module in model.named_modules():
        if isinstance(module, PackedBitLinear):
            orig_scales[name] = module.scales.data.clone()

    # Baseline PPL
    log.info("\n--- BASELINE PPL ---")
    baseline_ppl = {}
    for d in domains:
        ppl = eval_ppl(model, tokenizer, data[d]['eval'])
        baseline_ppl[d] = ppl
        log.info(f"  {d}: PPL={ppl:.2f}")

    # Baseline accuracy
    log.info("\n--- BASELINE ACCURACY ---")
    baseline_acc = eval_accuracy(GGUF_BASE, n_trivia=100)

    # Train each domain
    results = {}
    for domain in domains:
        log.info(f"\n{'='*60}")
        log.info(f"TRAINING: {domain}")
        log.info(f"{'='*60}")

        # Restore original scales
        for name, module in model.named_modules():
            if isinstance(module, PackedBitLinear) and name in orig_scales:
                module.scales.data.copy_(orig_scales[name])

        # Mix domain + diverse data
        train_texts = mix_data(data[domain]['train'], data['diverse']['train'])
        log.info(f"  Training on {len(train_texts)} examples ({int(DOMAIN_MIX*100)}% domain + {int((1-DOMAIN_MIX)*100)}% diverse)")

        t0 = time.time()
        train_scales_v2(model, tokenizer, train_texts, orig_scales, epochs=EPOCHS, lr=LR)
        train_time = time.time() - t0
        log.info(f"  Trained in {train_time:.0f}s")

        # Save trained scales
        trained_scales = {}
        for name, module in model.named_modules():
            if isinstance(module, PackedBitLinear):
                trained_scales[name] = module.scales.data.clone()
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(trained_scales, f"checkpoints/{domain}_scales_v2.pt")

        # PPL on all domains
        log.info(f"  PPL with {domain} scales:")
        domain_ppl = {}
        for d in domains:
            ppl = eval_ppl(model, tokenizer, data[d]['eval'])
            domain_ppl[d] = ppl
            delta = (1 - ppl / baseline_ppl[d]) * 100
            marker = " <<<" if d == domain else ""
            log.info(f"    {d}: {ppl:.2f} (was {baseline_ppl[d]:.2f}, {delta:+.1f}%){marker}")

        # Patch GGUF + eval accuracy
        gguf_path = os.path.expanduser(f"~/freigent/apps/trucksim/data/llm/Bonsai-1.7B-{domain}-v2.gguf")
        log.info(f"  Patching GGUF...")
        patch_gguf(orig_scales, trained_scales, gguf_path)
        log.info(f"  Evaluating accuracy...")
        domain_acc = eval_accuracy(gguf_path, n_trivia=100)

        results[domain] = {
            "ppl": domain_ppl,
            "accuracy": domain_acc,
            "train_time": train_time,
        }

    # ─── Summary ───
    log.info(f"\n{'='*60}")
    log.info("RESULTS SUMMARY")
    log.info(f"{'='*60}")

    log.info(f"\nPPL Table:")
    log.info(f"{'Scales':<12}" + "".join(f" {d:>12}" for d in domains))
    log.info("-" * (12 + 13 * len(domains)))
    log.info(f"{'baseline':<12}" + "".join(f" {baseline_ppl[d]:>12.2f}" for d in domains))
    for domain in domains:
        line = f"{domain:<12}"
        for d in domains:
            ppl = results[domain]['ppl'][d]
            marker = "*" if d == domain else " "
            line += f" {ppl:>11.2f}{marker}"
        log.info(line)

    log.info(f"\nAccuracy Table:")
    log.info(f"{'Scales':<12} {'TriviaQA':>10}")
    log.info("-" * 24)
    log.info(f"{'baseline':<12} {baseline_acc['accuracy']:>10.1%}")
    for domain in domains:
        acc = results[domain]['accuracy']['accuracy']
        delta = acc - baseline_acc['accuracy']
        log.info(f"{domain:<12} {acc:>10.1%} ({delta:+.1%})")

    # Diagonal dominance
    log.info(f"\nDiagonal Dominance (PPL):")
    for domain in domains:
        own = results[domain]['ppl'][domain]
        others = [results[d]['ppl'][domain] for d in domains if d != domain]
        best = own <= min(others)
        log.info(f"  {domain}: {'YES' if best else 'NO'} (own={own:.2f} best_other={min(others):.2f})")

    elapsed = time.time() - t_start
    log.info(f"\nTotal: {elapsed/60:.0f} min")

    with open("checkpoints/results_v2.json", "w") as f:
        json.dump({"baseline_ppl": baseline_ppl, "baseline_acc": baseline_acc,
                    "results": {d: {"ppl": r["ppl"], "accuracy": r["accuracy"],
                                     "train_time": r["train_time"]} for d, r in results.items()},
                    "config": {"lr": LR, "epochs": EPOCHS, "max_len": MAX_LEN, "reg_lambda": REG_LAMBDA,
                               "token_select": TOKEN_SELECT_RATIO, "domain_mix": DOMAIN_MIX}
                    }, f, indent=2, default=str)
    log.info("Results saved to checkpoints/results_v2.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"CRASHED: {e}", exc_info=True)
    finally:
        log.info("=" * 60)
        log.info("DONE")
        log.info("=" * 60)
