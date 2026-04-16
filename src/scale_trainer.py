"""Scale personality training with fast eval loop.

Train: PyTorch (modify scales via gradient) — ~160 tok/s forward pass
Eval:  llama.cpp server (120 tok/s generation)

Flow:
  1. Load model in PyTorch (fp16, 3.4GB)
  2. Train scales on domain data (fast forward pass, no generation needed)
  3. Export modified scales → patch GGUF
  4. Eval via llama.cpp at 120 tok/s
  5. Repeat with different domain / approach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import os
import sys
import struct
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S",
                    handlers=[logging.FileHandler("/tmp/scale_training.log", mode='w'),
                              logging.StreamHandler(sys.stdout)])
log = logging.getLogger("scales")


# ─── Config ──────────────────────────────────────────────────────────────

MODEL_ID = "prism-ml/Bonsai-1.7B-unpacked"
GGUF_PATH = os.path.expanduser("~/freigent/apps/trucksim/data/llm/Bonsai-1.7B.gguf")
DEVICE = "cuda"
GROUP_SIZE = 128
LR = 0.01
EPOCHS = 10


# ─── PyTorch Scale Training (forward pass only, no generation) ───────────

def load_model():
    """Load Bonsai 1.7B fp16 for scale training."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading model for training...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, trust_remote_code=True, device_map=DEVICE
    )
    log.info(f"Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    return model, tokenizer


def get_scale_params(model):
    """Extract group scale values from all linear layers.

    Bonsai weights are sign * scale stored as fp16.
    Group scale = absmean of each group of GROUP_SIZE weights.
    """
    scales = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'lm_head' not in name:
            w = module.weight.data
            w_flat = w.reshape(-1, GROUP_SIZE)
            group_scales = w_flat.abs().mean(dim=1)
            scales[name] = {
                'signs': w.sign(),
                'group_scales': group_scales,
                'shape': w.shape,
            }
    return scales


def apply_scale_multipliers(model, multipliers):
    """Apply scale multipliers to model weights.

    multipliers: dict of {layer_name: tensor of per-group multipliers}
    Each weight becomes: sign * (original_group_scale * multiplier)
    """
    for name, module in model.named_modules():
        if name in multipliers:
            w = module.weight.data
            signs = w.sign()
            m = multipliers[name].to(w.device)
            # Expand multiplier per group → per weight
            m_expanded = m.unsqueeze(1).expand(-1, GROUP_SIZE).reshape(w.shape)
            # Apply: new_weight = sign * |old_weight| * multiplier
            module.weight.data = signs * w.abs() * m_expanded


def train_scales(model, tokenizer, texts, epochs=EPOCHS, lr=LR, answer_only=False):
    """Train scale multipliers using CE loss on domain text.

    Instead of training scales directly, train a multiplier per group.
    Multiplier starts at 1.0 (no change). Gradient adjusts it.
    """
    # Create trainable multipliers for each linear layer
    multipliers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'lm_head' not in name:
            n_groups = module.weight.numel() // GROUP_SIZE
            m = torch.ones(n_groups, device=DEVICE, requires_grad=True)
            multipliers[name] = m

    optimizer = torch.optim.SGD(list(multipliers.values()), lr=lr)

    log.info(f"Training {len(multipliers)} layers, {sum(m.numel() for m in multipliers.values()):,} scale groups")

    for epoch in range(epochs):
        epoch_loss = 0
        n = 0

        for text in texts[:20]:
            tokens = tokenizer(text, return_tensors="pt", truncation=True,
                              max_length=256).to(DEVICE)
            if tokens.input_ids.shape[1] < 5:
                continue

            # Apply current multipliers
            apply_scale_multipliers(model, multipliers)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(**tokens, labels=tokens.input_ids)

                if answer_only:
                    logits = outputs.logits[:, :-1]
                    labels = tokens.input_ids[:, 1:]
                    loss_per_token = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1), reduction='none'
                    ).reshape(labels.shape)
                    # Last 20% = answer region
                    answer_start = labels.shape[1] * 4 // 5
                    mask = torch.zeros_like(loss_per_token)
                    mask[:, answer_start:] = 1.0
                    loss = (loss_per_token * mask).sum() / (mask.sum() + 1e-8)
                else:
                    loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            n += 1

        avg = epoch_loss / max(n, 1)
        log.info(f"  Epoch {epoch+1}/{epochs}: loss={avg:.4f} | VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB")

    return multipliers


def save_multipliers(multipliers, path):
    """Save multiplier tensors."""
    data = {name: m.detach().cpu() for name, m in multipliers.items()}
    torch.save(data, path)
    log.info(f"Saved multipliers to {path}")


def load_domain_data(domain, n_samples=200):
    """Load domain-specific training data."""
    from datasets import load_dataset

    log.info(f"Loading {domain} data ({n_samples} samples)...")

    if domain == "math":
        ds = load_dataset("gsm8k", "main", split="train", streaming=True)
        return [f"Question: {ex['question']}\nAnswer: {ex['answer']}"
                for i, ex in enumerate(ds) if i < n_samples]
    elif domain == "language":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
        return [ex['text'] for ex in ds if len(ex['text'].strip()) > 100][:n_samples]
    elif domain == "code":
        ds = load_dataset("code_search_net", "python", split="train", streaming=True)
        return [ex['func_code_string'] for ex in ds if ex.get('func_code_string')][:n_samples]
    else:
        raise ValueError(f"Unknown domain: {domain}")


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    from llama_fast_eval import LlamaEval

    log.info("=" * 60)
    log.info("Scale Personality Training — Fast Loop")
    log.info("=" * 60)

    save_dir = Path("checkpoints/bonsai17b_scales")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Start llama.cpp for fast eval
    log.info("Starting llama.cpp server for eval...")
    evaluator = LlamaEval(GGUF_PATH)
    evaluator.start()

    # Baseline eval at 120 tok/s
    log.info("\n--- BASELINE EVAL (llama.cpp, 120 tok/s) ---")
    bench = evaluator.bench_tok_s(100)
    log.info(f"Speed: {bench['tok_s']:.1f} tok/s")

    gsm_base = evaluator.eval_gsm8k(n=30)
    log.info(f"GSM8K baseline: {gsm_base['correct']}/{gsm_base['total']} = {gsm_base['accuracy']:.1%}")

    trivia_base = evaluator.eval_trivia(n=30)
    log.info(f"TriviaQA baseline: {trivia_base['correct']}/{trivia_base['total']} = {trivia_base['accuracy']:.1%}")

    # Load PyTorch model for training
    model, tokenizer = load_model()

    domains = ["math", "language", "code"]
    results = {"baseline": {"gsm8k": gsm_base, "trivia": trivia_base}}

    for domain in domains:
        log.info(f"\n{'='*60}")
        log.info(f"Training: {domain} scales")
        log.info(f"{'='*60}")

        texts = load_domain_data(domain)
        t0 = time.time()
        multipliers = train_scales(model, tokenizer, texts, epochs=EPOCHS, lr=LR)
        elapsed = time.time() - t0
        log.info(f"Trained in {elapsed:.1f}s")

        save_multipliers(multipliers, save_dir / f"multipliers_{domain}.pt")

        # TODO: Patch GGUF with new scales and eval via llama.cpp
        # For now, eval PPL in PyTorch (fast forward pass, no generation)
        log.info(f"Evaluating PPL...")
        ppl_results = {}
        model.eval()
        for eval_domain in domains:
            eval_texts = load_domain_data(eval_domain, n_samples=50)
            total_loss = 0
            total_tokens = 0
            with torch.no_grad():
                for text in eval_texts[:30]:
                    tokens = tokenizer(text, return_tensors="pt", truncation=True,
                                      max_length=256).to(DEVICE)
                    if tokens.input_ids.shape[1] < 5:
                        continue
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        out = model(**tokens, labels=tokens.input_ids)
                    total_loss += out.loss.item() * tokens.input_ids.shape[1]
                    total_tokens += tokens.input_ids.shape[1]
            ppl = np.exp(total_loss / max(total_tokens, 1))
            ppl_results[eval_domain] = ppl
            log.info(f"  {domain}_scales on {eval_domain}: PPL={ppl:.2f}")

        results[domain] = {"ppls": ppl_results, "train_time": elapsed}

        # Reset model weights for next domain
        del model
        torch.cuda.empty_cache()
        model, tokenizer = load_model()

    # Summary
    log.info(f"\n{'='*60}")
    log.info("SUMMARY — Diagonal Dominance")
    log.info(f"{'='*60}")
    log.info(f"{'Scales':<15} {'Math PPL':>10} {'Lang PPL':>10} {'Code PPL':>10}")
    log.info("-" * 47)
    for domain in domains:
        ppls = results[domain]["ppls"]
        line = f"{domain:<15}"
        for d in domains:
            marker = " *" if d == domain else ""
            line += f" {ppls[d]:>9.2f}{marker}"
        log.info(line)

    # Save results
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nResults saved to {save_dir / 'results.json'}")

    evaluator.stop()


if __name__ == "__main__":
    main()
