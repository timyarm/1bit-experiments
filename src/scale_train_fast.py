"""Fast scale training: PyTorch gradients + llama.cpp eval.

Train: PyTorch backprop on scales (~5 min)
Export: Patch GGUF scales (instant)
Eval: llama.cpp server (340 tok/s)

Usage:
    python scale_train_fast.py
    # Monitor: tail -f /tmp/scale_training.log
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import os
import sys
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S",
                    handlers=[logging.FileHandler("/tmp/scale_training.log", mode='w'),
                              logging.StreamHandler(sys.stdout)])
log = logging.getLogger("scales")

# ─── Config ──────────────────────────────────────────────────────────────

MODEL_ID = "prism-ml/Bonsai-1.7B-unpacked"
GGUF_BASE = os.path.expanduser("~/freigent/apps/trucksim/data/llm/Bonsai-1.7B.gguf")
GGUF_PATCHED = os.path.expanduser("~/freigent/apps/trucksim/data/llm/Bonsai-1.7B-patched.gguf")
SAVE_DIR = Path("checkpoints/bonsai17b_scales")
DEVICE = "cuda"
GROUP_SIZE = 128
LR = 0.01
EPOCHS = 10
TRAIN_EXAMPLES = 20
EVAL_QUESTIONS = 30


# ─── Data ────────────────────────────────────────────────────────────────

def load_data(domain, n=200):
    from datasets import load_dataset
    log.info(f"Loading {domain} data...")
    if domain == "math":
        ds = load_dataset("gsm8k", "main", split="train", streaming=True)
        return [f"Question: {ex['question']}\nAnswer: {ex['answer']}" for i, ex in enumerate(ds) if i < n]
    elif domain == "language":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
        return [ex['text'] for ex in ds if len(ex['text'].strip()) > 100][:n]
    elif domain == "code":
        ds = load_dataset("code_search_net", "python", split="train", streaming=True)
        return [ex['func_code_string'] for ex in ds if ex.get('func_code_string')][:n]


# ─── Train (PyTorch) ─────────────────────────────────────────────────────

def train_domain(model, tokenizer, domain, epochs=EPOCHS, lr=LR):
    """Train scales on one domain. Returns per-group multipliers."""
    from packed_bitlinear import PackedBitLinear
    texts = load_data(domain)

    # Snapshot original scales
    originals = {}
    for name, module in model.named_modules():
        if isinstance(module, PackedBitLinear):
            originals[name] = module.scales.data.clone()

    # Freeze everything, unfreeze only PackedBitLinear scales
    for p in model.parameters():
        p.requires_grad = False
    for name, module in model.named_modules():
        if isinstance(module, PackedBitLinear):
            module.scales.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Trainable scale params: {trainable:,}")

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n = 0
        for text in texts[:TRAIN_EXAMPLES]:
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
            if tokens.input_ids.shape[1] < 5:
                continue
            with torch.amp.autocast('cuda', dtype=torch.float16):
                loss = model(**tokens, labels=tokens.input_ids).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            n += 1

        log.info(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/max(n,1):.4f} VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB")

    # Compute per-group multipliers: new_scale / old_scale
    multipliers = {}
    for name, module in model.named_modules():
        if isinstance(module, PackedBitLinear) and name in originals:
            old_s = originals[name]
            new_s = module.scales.data
            mult = (new_s / (old_s + 1e-8)).cpu()
            multipliers[name] = mult
            # Restore original scales for next domain
            module.scales.data.copy_(old_s)

    torch.cuda.empty_cache()
    return multipliers


# ─── Eval PPL (PyTorch, fast forward pass) ───────────────────────────────

def eval_ppl(model, tokenizer, domain, n=50):
    """Quick PPL eval via forward pass. No generation needed."""
    texts = load_data(domain, n)
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    model.eval()
    torch.cuda.empty_cache()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for text in texts[:n]:
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
            if tokens.input_ids.shape[1] < 5:
                continue
            with torch.amp.autocast('cuda', dtype=torch.float16):
                out = model(**tokens, labels=tokens.input_ids)
            total_loss += out.loss.item() * tokens.input_ids.shape[1]
            total_tokens += tokens.input_ids.shape[1]
    torch.cuda.empty_cache()
    return np.exp(total_loss / max(total_tokens, 1))


# ─── GGUF Patching ──────────────────────────────────────────────────────

def patch_gguf(multipliers):
    """Apply multipliers to GGUF and save patched copy."""
    sys.path.insert(0, os.path.dirname(__file__))
    from gguf_scale_patcher import apply_multipliers_to_gguf

    log.info("Patching GGUF with trained scales...")
    # Convert multiplier dict to use torch tensors
    torch_mults = {}
    for name, mult in multipliers.items():
        if isinstance(mult, np.ndarray):
            torch_mults[name] = torch.from_numpy(mult)
        else:
            torch_mults[name] = mult

    apply_multipliers_to_gguf(GGUF_BASE, GGUF_PATCHED, torch_mults)
    log.info(f"Patched GGUF saved to {GGUF_PATCHED}")


# ─── Eval via llama.cpp ─────────────────────────────────────────────────

def eval_llama(gguf_path, n_questions=EVAL_QUESTIONS):
    """Eval accuracy via llama.cpp server at 340 tok/s."""
    sys.path.insert(0, os.path.dirname(__file__))
    from llama_fast_eval import LlamaEval

    ev = LlamaEval(gguf_path)
    ev.start()

    results = {}

    bench = ev.bench_tok_s(50)
    results['tok_s'] = bench['tok_s']
    log.info(f"  Speed: {bench['tok_s']:.0f} tok/s")

    trivia = ev.eval_trivia(n=n_questions)
    results['trivia'] = trivia
    log.info(f"  TriviaQA: {trivia['correct']}/{trivia['total']} = {trivia['accuracy']:.1%}")

    ev.stop()
    return results


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Scale Training — Fast Loop (PyTorch train + llama.cpp eval)")
    log.info("=" * 60)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # ─── Baseline eval via llama.cpp ───
    log.info("\n--- BASELINE EVAL (llama.cpp) ---")
    baseline_llama = eval_llama(GGUF_BASE)
    # Kill server and free GPU memory before PyTorch training
    import subprocess
    subprocess.run("kill $(pgrep llama-server) 2>/dev/null", shell=True)
    time.sleep(2)
    torch.cuda.empty_cache()

    # ─── Load PyTorch model with packed 1-bit (0.84GB instead of 3.44GB) ───
    log.info("\nLoading PyTorch model (packed 1-bit)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, os.path.dirname(__file__))
    from packed_bitlinear import convert_model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, trust_remote_code=True, device_map="cpu"
    )
    convert_model(model)
    model = model.to(DEVICE)
    log.info(f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # ─── Baseline PPL ───
    log.info("\n--- BASELINE PPL ---")
    domains = ["math", "language", "code"]
    baseline_ppl = {}
    for d in domains:
        ppl = eval_ppl(model, tokenizer, d)
        baseline_ppl[d] = ppl
        log.info(f"  {d}: PPL={ppl:.2f}")

    # ─── Train each domain ───
    results = {"baseline": {"ppl": baseline_ppl, "llama": baseline_llama}}

    for domain in domains:
        log.info(f"\n{'='*60}")
        log.info(f"TRAINING: {domain}")
        log.info(f"{'='*60}")

        t0 = time.time()
        multipliers = train_domain(model, tokenizer, domain, epochs=EPOCHS, lr=LR)
        train_time = time.time() - t0
        log.info(f"Trained in {train_time:.1f}s")

        # Save multipliers
        torch.save(multipliers, SAVE_DIR / f"mult_{domain}.pt")

        # Apply multipliers and measure PPL
        log.info(f"\nPPL with {domain} scales:")
        from packed_bitlinear import PackedBitLinear
        originals_eval = {}
        for name, module in model.named_modules():
            if isinstance(module, PackedBitLinear) and name in multipliers:
                originals_eval[name] = module.scales.data.clone()
                mult = multipliers[name].to(DEVICE)
                module.scales.data *= mult

        ppl_results = {}
        for d in domains:
            ppl = eval_ppl(model, tokenizer, d)
            ppl_results[d] = ppl
            marker = " <<<" if d == domain else ""
            log.info(f"  {d}: PPL={ppl:.2f}{marker}")

        # Restore scales
        for name, orig in originals_eval.items():
            dict(model.named_modules())[name].scales.data.copy_(orig)

        # Patch GGUF and eval via llama.cpp
        log.info(f"\nPatching GGUF and eval via llama.cpp...")
        patch_gguf(multipliers)
        llama_results = eval_llama(GGUF_PATCHED)

        results[domain] = {
            "ppl": ppl_results,
            "llama": llama_results,
            "train_time": train_time,
        }

    # ─── Summary ───
    log.info(f"\n{'='*60}")
    log.info("RESULTS SUMMARY")
    log.info(f"{'='*60}")

    log.info(f"\n{'Scales':<12} {'Math PPL':>10} {'Lang PPL':>10} {'Code PPL':>10} {'TriviaQA':>10}")
    log.info("-" * 55)
    bp = baseline_ppl
    log.info(f"{'baseline':<12} {bp['math']:>10.2f} {bp['language']:>10.2f} {bp['code']:>10.2f} {baseline_llama['trivia']['accuracy']:>10.1%}")
    for domain in domains:
        r = results[domain]
        p = r['ppl']
        ta = r['llama']['trivia']['accuracy']
        log.info(f"{domain:<12} {p['math']:>10.2f} {p['language']:>10.2f} {p['code']:>10.2f} {ta:>10.1%}")

    # Diagonal dominance check
    log.info("\nDiagonal dominance:")
    for domain in domains:
        own_ppl = results[domain]['ppl'][domain]
        others = [results[d]['ppl'][domain] for d in domains if d != domain]
        best = own_ppl <= min(others)
        log.info(f"  {domain}: {'YES' if best else 'NO'} (own={own_ppl:.2f}, others={[f'{x:.2f}' for x in others]})")

    # Save
    with open(SAVE_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nResults saved to {SAVE_DIR / 'results.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"CRASHED: {e}", exc_info=True)
    finally:
        log.info("=" * 60)
        log.info("RUN COMPLETE (or crashed — check above for errors)")
        log.info(f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB" if torch.cuda.is_available() else "no CUDA")
        log.info("=" * 60)
