"""Proper scale training — real datasets, proper eval, publishable results.

Train: Group scales via PyTorch + gradient checkpointing (3.6GB VRAM)
Patch: GGUF in 22s via numpy memmap
Eval: llama.cpp at 400 tok/s, 100+ questions

Monitor: tail -f /tmp/scale_training.log
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
import sys
import logging
import shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S",
                    handlers=[logging.FileHandler("/tmp/scale_training.log", mode='w'),
                              logging.StreamHandler(sys.stdout)])
log = logging.getLogger("s")

DEVICE = "cuda"
GROUP_SIZE = 128
GGUF_BASE = os.path.expanduser("~/freigent/apps/trucksim/data/llm/Bonsai-1.7B.gguf")


# ─── Data: stream from HuggingFace, cache locally ────────────────────────

def get_data(domain, n_train=200, n_eval=100):
    """Load train + eval splits. Cache to disk after first download."""
    cache_dir = os.path.expanduser(f"~/freigent/apps/trucksim/data/scale_data")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{domain}.pt")

    if os.path.exists(cache_file):
        log.info(f"  Loading cached {domain} data")
        return torch.load(cache_file, weights_only=False)

    log.info(f"  Downloading {domain} data from HuggingFace...")
    from datasets import load_dataset

    texts = []
    if domain == "math":
        ds = load_dataset("gsm8k", "main", split="train", streaming=True)
        for ex in ds:
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
            if code and len(code) > 50:
                texts.append(code)
            if len(texts) >= n_train + n_eval:
                break
    elif domain == "language":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
        for ex in ds:
            if len(ex['text'].strip()) > 100:
                texts.append(ex['text'])
            if len(texts) >= n_train + n_eval:
                break

    train = texts[:n_train]
    eval_set = texts[n_train:n_train + n_eval]
    data = {"train": train, "eval": eval_set}
    torch.save(data, cache_file)
    log.info(f"  Cached {len(train)} train + {len(eval_set)} eval examples")
    return data


# ─── Training ────────────────────────────────────────────────────────────

def train_scales(model, tokenizer, texts, epochs=5, lr=0.005, max_len=96):
    """Train group scales on domain data."""
    from packed_bitlinear import PackedBitLinear

    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, PackedBitLinear):
            m.scales.requires_grad = True

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        n = 0
        for text in texts:
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(DEVICE)
            if tokens.input_ids.shape[1] < 5:
                continue
            with torch.amp.autocast('cuda', dtype=torch.float16):
                loss = model(**tokens, labels=tokens.input_ids).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            n += 1
        log.info(f"    Epoch {epoch+1}/{epochs}: loss={total_loss/max(n,1):.4f} VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB")


def eval_ppl(model, tokenizer, texts, max_len=96):
    """Evaluate PPL on text samples."""
    model.eval()
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
    return np.exp(total_loss / max(total_tokens, 1))


# ─── GGUF Patching ──────────────────────────────────────────────────────

def patch_gguf(orig_scales, new_scales, output_path):
    """Patch GGUF with new group scales using numpy memmap. ~22s."""
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

def eval_accuracy(gguf_path, n_trivia=100, n_gsm=50):
    """Eval via llama.cpp server."""
    sys.path.insert(0, os.path.dirname(__file__))
    from llama_fast_eval import LlamaEval

    ev = LlamaEval(gguf_path)
    ev.start()

    results = {}

    # TriviaQA
    log.info(f"    Evaluating TriviaQA ({n_trivia} questions)...")
    trivia = ev.eval_trivia(n=n_trivia)
    results['trivia'] = trivia
    log.info(f"    TriviaQA: {trivia['correct']}/{trivia['total']} = {trivia['accuracy']:.1%}")

    # Quick generation samples
    samples = [
        ("What is 25 * 4?", 30),
        ("The largest planet in our solar system is", 20),
        ("Write a Python function to reverse a string:", 50),
    ]
    for prompt, max_tok in samples:
        out = ev.generate(prompt, max_tokens=max_tok)
        log.info(f"    Q: {prompt[:40]}  A: {out[:60]}")

    ev.stop()
    return results


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("SCALE PERSONALITIES — PROPER EXPERIMENT")
    log.info("=" * 60)
    t_start = time.time()

    # Load data
    log.info("\n--- LOADING DATA ---")
    domains = ["math", "knowledge", "code", "language"]
    data = {}
    for d in domains:
        data[d] = get_data(d, n_train=200, n_eval=100)
        log.info(f"  {d}: {len(data[d]['train'])} train, {len(data[d]['eval'])} eval")

    # Load model
    log.info("\n--- LOADING MODEL ---")
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

    # Baseline accuracy via llama.cpp
    log.info("\n--- BASELINE ACCURACY (llama.cpp) ---")
    baseline_acc = eval_accuracy(GGUF_BASE, n_trivia=100)

    # Train each domain
    results = {}
    for domain in domains:
        log.info(f"\n{'='*60}")
        log.info(f"TRAINING: {domain} ({len(data[domain]['train'])} examples)")
        log.info(f"{'='*60}")

        # Restore original scales
        for name, module in model.named_modules():
            if isinstance(module, PackedBitLinear) and name in orig_scales:
                module.scales.data.copy_(orig_scales[name])

        # Train
        t0 = time.time()
        train_scales(model, tokenizer, data[domain]['train'], epochs=5, lr=0.005)
        train_time = time.time() - t0
        log.info(f"  Trained in {train_time:.0f}s")

        # Save trained scales
        trained_scales = {}
        for name, module in model.named_modules():
            if isinstance(module, PackedBitLinear):
                trained_scales[name] = module.scales.data.clone()
        torch.save(trained_scales, f"checkpoints/{domain}_scales.pt")

        # Eval PPL on all domains
        log.info(f"  PPL with {domain} scales:")
        domain_ppl = {}
        for d in domains:
            ppl = eval_ppl(model, tokenizer, data[d]['eval'])
            domain_ppl[d] = ppl
            marker = " <<<" if d == domain else ""
            log.info(f"    {d}: {ppl:.2f} (baseline: {baseline_ppl[d]:.2f}){marker}")

        # Patch GGUF and eval accuracy
        log.info(f"  Patching GGUF...")
        gguf_path = os.path.expanduser(f"~/freigent/apps/trucksim/data/llm/Bonsai-1.7B-{domain}.gguf")
        n_patched = patch_gguf(orig_scales, trained_scales, gguf_path)
        log.info(f"  Patched {n_patched} tensors")

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

    # PPL table
    log.info(f"\n{'Scales':<12}", end="")
    for d in domains:
        log.info(f" {d:>10}", end="")
    log.info("")
    log.info("-" * (12 + 11 * len(domains)))
    log.info(f"{'baseline':<12}", end="")
    for d in domains:
        log.info(f" {baseline_ppl[d]:>10.2f}", end="")
    log.info("")
    for domain in domains:
        log.info(f"{domain:<12}", end="")
        for d in domains:
            ppl = results[domain]['ppl'][d]
            marker = "*" if d == domain else " "
            log.info(f" {ppl:>9.2f}{marker}", end="")
        log.info("")

    # Accuracy table
    log.info(f"\n{'Scales':<12} {'TriviaQA':>10}")
    log.info("-" * 24)
    log.info(f"{'baseline':<12} {baseline_acc['trivia']['accuracy']:>10.1%}")
    for domain in domains:
        acc = results[domain]['accuracy']['trivia']['accuracy']
        log.info(f"{domain:<12} {acc:>10.1%}")

    # Diagonal dominance
    log.info("\nDiagonal dominance (PPL):")
    for domain in domains:
        own = results[domain]['ppl'][domain]
        others_on_domain = [results[d]['ppl'][domain] for d in domains if d != domain]
        is_best = own <= min(others_on_domain)
        log.info(f"  {domain}: {'YES' if is_best else 'NO'} (own={own:.2f} vs best_other={min(others_on_domain):.2f})")

    elapsed = time.time() - t_start
    log.info(f"\nTotal time: {elapsed/60:.1f} min")

    # Save full results
    import json
    with open("checkpoints/results_proper.json", "w") as f:
        json.dump({
            "baseline_ppl": baseline_ppl,
            "baseline_acc": baseline_acc,
            "domain_results": {d: {"ppl": r["ppl"], "train_time": r["train_time"],
                                    "accuracy": r["accuracy"]} for d, r in results.items()},
        }, f, indent=2, default=str)
    log.info("Results saved to checkpoints/results_proper.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"CRASHED: {e}", exc_info=True)
    finally:
        log.info("DONE")
