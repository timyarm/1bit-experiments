"""Scale Personality Training — Bonsai 1.7B on local GPU (6GB VRAM).

Loads Bonsai 1.7B native 1-bit, trains domain-specific scale tables,
evaluates accuracy before/after. Designed for GTX 1660 Super.

Monitor mid-run:
    tail -f /tmp/scale_training.log

Usage:
    python local_bonsai17b_scales.py
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
from pathlib import Path
from collections import defaultdict

# ─── Logging — file + stdout, unbuffered ─────────────────────────────────────

LOG_PATH = "/tmp/scale_training.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_PATH, mode='w'),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("scales")
# Flush after every write
for h in log.handlers:
    if hasattr(h, 'stream'):
        h.stream = os.fdopen(os.dup(h.stream.fileno()), 'w', buffering=1)

log.info(f"Logging to {LOG_PATH} — tail -f {LOG_PATH} to monitor")

def vram():
    """Return VRAM usage string."""
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_mem / 1e9
        return f"{used:.1f}/{total:.1f}GB"
    return "cpu"


# ─── Config ──────────────────────────────────────────────────────────────────

MODEL_ID = "prism-ml/Bonsai-1.7B-unpacked"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GROUP_SIZE = 128
DTYPE = torch.float16

# Training
LR = 0.01
EPOCHS = 10
EVAL_SAMPLES = 50  # quick iteration


# ─── NativeBitLinear ─────────────────────────────────────────────────────────

class NativeBitLinear(nn.Module):
    """1-bit linear layer: frozen signs + trainable scales."""

    def __init__(self, weight_tensor, group_size=128):
        super().__init__()
        self.out_features, self.in_features = weight_tensor.shape
        self.group_size = group_size

        # Extract signs and scales from the weight tensor
        signs = torch.sign(weight_tensor)
        signs[signs == 0] = 1  # no zeros in binary

        # Pack signs as int8 for storage (1 bit per weight conceptually, stored as +1/-1)
        self.register_buffer('signs', signs.to(torch.int8))

        # Compute group scales (absmean per group)
        w_flat = weight_tensor.reshape(-1, group_size)
        scales = w_flat.abs().mean(dim=1)
        self.scales = nn.Parameter(scales.float())  # trainable!

        self.original_scales = scales.clone()
        self._cached_weight = None

    def _rebuild_weight(self):
        """Reconstruct full weight matrix from signs + scales. Cache it."""
        scales_expanded = self.scales.unsqueeze(1).expand(-1, self.group_size)
        scales_expanded = scales_expanded.reshape(self.out_features, self.in_features)
        self._cached_weight = (self.signs.float() * scales_expanded).half()

    def invalidate_cache(self):
        self._cached_weight = None

    def forward(self, x):
        if self._cached_weight is None or self.scales.requires_grad:
            # Training: rebuild every pass (scales changing via grad)
            scales_expanded = self.scales.unsqueeze(1).expand(-1, self.group_size)
            scales_expanded = scales_expanded.reshape(self.out_features, self.in_features)
            weight = self.signs.float() * scales_expanded
            return F.linear(x, weight.to(x.dtype))
        else:
            # Inference: use cached weight (fast)
            return F.linear(x, self._cached_weight.to(x.dtype))


# ─── Model Loading ───────────────────────────────────────────────────────────

def load_bonsai_1bit(model_id=MODEL_ID):
    """Load Bonsai 1.7B and convert all Linear layers to NativeBitLinear."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=DTYPE, trust_remote_code=True, device_map="cpu"
    )

    # Convert Linear → NativeBitLinear
    converted = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and 'lm_head' not in name:
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = dict(model.named_modules())[parent_name] if parent_name else model

            bit_linear = NativeBitLinear(module.weight.data, GROUP_SIZE)
            setattr(parent, child_name, bit_linear)
            converted += 1

    log.info(f"  Converted {converted} layers to NativeBitLinear")

    # Count params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Total params: {total:,}, Trainable (scales): {trainable:,}")

    model = model.to(DEVICE)
    vram = torch.cuda.memory_allocated() / 1e9
    log.info(f"  VRAM: {vram:.2f} GB")

    return model, tokenizer


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_domain_data(domain, n_samples=200):
    """Load domain-specific training data from HuggingFace."""
    from datasets import load_dataset

    if domain == "math":
        ds = load_dataset("gsm8k", "main", split="train", streaming=True)
        texts = []
        for ex in ds:
            texts.append(f"Question: {ex['question']}\nAnswer: {ex['answer']}")
            if len(texts) >= n_samples:
                break
        return texts

    elif domain == "language":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
        texts = []
        for ex in ds:
            if len(ex['text'].strip()) > 100:
                texts.append(ex['text'])
                if len(texts) >= n_samples:
                    break
        return texts

    elif domain == "code":
        ds = load_dataset("code_search_net", "python", split="train", streaming=True)
        texts = []
        for ex in ds:
            if ex.get('func_code_string'):
                texts.append(ex['func_code_string'])
                if len(texts) >= n_samples:
                    break
        return texts

    elif domain == "factual":
        ds = load_dataset("natural_questions", "default", split="train", streaming=True,
                          trust_remote_code=True)
        texts = []
        for ex in ds:
            q = ex.get('question', {})
            if isinstance(q, dict):
                q = q.get('text', '')
            texts.append(str(q))
            if len(texts) >= n_samples:
                break
        return texts

    else:
        raise ValueError(f"Unknown domain: {domain}")


# ─── Training ────────────────────────────────────────────────────────────────

def train_scales(model, tokenizer, texts, epochs=EPOCHS, lr=LR, answer_only=False):
    """Train scale parameters on domain text.

    If answer_only=True, only compute loss on answer tokens (after "Answer:" or "####").
    """
    # Freeze everything except scales
    for name, param in model.named_parameters():
        param.requires_grad = 'scales' in name

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr
    )

    model.train()
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0

        for text in texts[:20]:  # 20 examples per epoch for speed
            tokens = tokenizer(text, return_tensors="pt", truncation=True,
                              max_length=256).to(DEVICE)

            if tokens.input_ids.shape[1] < 5:
                continue

            with torch.cuda.amp.autocast(dtype=DTYPE):
                outputs = model(**tokens, labels=tokens.input_ids)

                if answer_only:
                    # Find answer boundary and mask loss
                    logits = outputs.logits[:, :-1]
                    labels = tokens.input_ids[:, 1:]
                    loss_per_token = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                        reduction='none'
                    ).reshape(labels.shape)

                    # Find "####" or last 20% of tokens as answer region
                    text_decoded = tokenizer.decode(tokens.input_ids[0])
                    answer_start = len(labels[0]) * 4 // 5  # last 20% as fallback
                    for marker in ["####", "Answer:", "answer:"]:
                        idx = text_decoded.find(marker)
                        if idx >= 0:
                            # Convert char position to token position (approximate)
                            answer_start = max(1, int(idx / len(text_decoded) * len(labels[0])))
                            break

                    mask = torch.zeros_like(loss_per_token)
                    mask[:, answer_start:] = 1.0
                    loss = (loss_per_token * mask).sum() / (mask.sum() + 1e-8)
                else:
                    loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        log.info(f"    Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} | VRAM={vram()}")

    return losses


def save_scales(model, path):
    """Save only the scale parameters."""
    scales = {}
    for name, param in model.named_parameters():
        if 'scales' in name:
            scales[name] = param.data.cpu()
    torch.save(scales, path)
    log.info(f"  Saved scales to {path} ({os.path.getsize(path) / 1024:.0f} KB)")


def load_scales(model, path):
    """Load scale parameters into model."""
    scales = torch.load(path, map_location=DEVICE, weights_only=True)
    state = model.state_dict()
    for name, val in scales.items():
        if name in state:
            state[name] = val.to(DEVICE)
    model.load_state_dict(state)
    log.info(f"  Loaded scales from {path}")


def reset_scales(model):
    """Reset scales to original values."""
    for module in model.modules():
        if isinstance(module, NativeBitLinear):
            module.scales.data.copy_(module.original_scales.to(module.scales.device))


# ─── Evaluation ──────────────────────────────────────────────────────────────

def eval_ppl(model, tokenizer, texts, label=""):
    """Evaluate perplexity on a set of texts."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for text in texts[:EVAL_SAMPLES]:
            tokens = tokenizer(text, return_tensors="pt", truncation=True,
                              max_length=256).to(DEVICE)
            if tokens.input_ids.shape[1] < 5:
                continue
            with torch.cuda.amp.autocast(dtype=DTYPE):
                outputs = model(**tokens, labels=tokens.input_ids)
            total_loss += outputs.loss.item() * tokens.input_ids.shape[1]
            total_tokens += tokens.input_ids.shape[1]

    ppl = np.exp(total_loss / max(total_tokens, 1))
    if label:
        log.info(f"  {label}: PPL={ppl:.2f}")
    return ppl


def eval_gsm8k(model, tokenizer, n=EVAL_SAMPLES):
    """Evaluate GSM8K accuracy — extract numeric answer, check correctness."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test", streaming=True)

    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for ex in ds:
            if total >= n:
                break

            prompt = f"Question: {ex['question']}\nAnswer:"
            tokens = tokenizer(prompt, return_tensors="pt", truncation=True,
                              max_length=256).to(DEVICE)

            with torch.cuda.amp.autocast(dtype=DTYPE):
                output = model.generate(
                    tokens.input_ids,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(output[0][tokens.input_ids.shape[1]:],
                                       skip_special_tokens=True)

            # Extract numeric answer
            pred_num = extract_number(response)
            true_answer = ex['answer'].split("####")[-1].strip()
            true_num = extract_number(true_answer)

            if pred_num is not None and true_num is not None and abs(pred_num - true_num) < 0.01:
                correct += 1
            total += 1

    acc = correct / max(total, 1)
    log.info(f"  GSM8K: {correct}/{total} = {acc:.1%}")
    return acc


def eval_trivia(model, tokenizer, n=EVAL_SAMPLES):
    """Evaluate TriviaQA accuracy."""
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "unfiltered", split="validation", streaming=True)

    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for ex in ds:
            if total >= n:
                break

            question = ex['question']
            answers = ex.get('answer', {}).get('aliases', [])
            if not answers:
                continue

            prompt = f"Question: {question}\nAnswer:"
            tokens = tokenizer(prompt, return_tensors="pt", truncation=True,
                              max_length=128).to(DEVICE)

            with torch.cuda.amp.autocast(dtype=DTYPE):
                output = model.generate(
                    tokens.input_ids,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(output[0][tokens.input_ids.shape[1]:],
                                       skip_special_tokens=True).strip().lower()

            if any(a.lower() in response for a in answers):
                correct += 1
            total += 1

    acc = correct / max(total, 1)
    log.info(f"  TriviaQA: {correct}/{total} = {acc:.1%}")
    return acc


def extract_number(text):
    """Extract the last number from a string."""
    import re
    numbers = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            return None
    return None


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Scale Personality Training — Bonsai 1.7B (Local GPU)")
    log.info("=" * 60)

    model, tokenizer = load_bonsai_1bit()

    save_dir = Path("checkpoints/bonsai17b_scales")
    save_dir.mkdir(parents=True, exist_ok=True)

    domains = ["math", "language", "code"]

    # ─── Baseline eval ───
    log.info("\n--- BASELINE (original scales) ---")
    baseline = {}
    for domain in domains:
        texts = load_domain_data(domain, n_samples=EVAL_SAMPLES)
        ppl = eval_ppl(model, tokenizer, texts, label=f"baseline/{domain}")
        baseline[domain] = ppl

    log.info("\n--- BASELINE ACCURACY ---")
    baseline_gsm = eval_gsm8k(model, tokenizer, n=EVAL_SAMPLES)
    baseline_trivia = eval_trivia(model, tokenizer, n=EVAL_SAMPLES)

    # ─── Train each domain ───
    results = {}
    for domain in domains:
        log.info(f"\n{'='*60}")
        log.info(f"Training: {domain} scales")
        log.info(f"{'='*60}")

        reset_scales(model)  # start from original scales each time

        texts = load_domain_data(domain, n_samples=200)
        log.info(f"  Loaded {len(texts)} {domain} examples")

        t0 = time.time()
        losses = train_scales(model, tokenizer, texts, epochs=EPOCHS, lr=LR)
        elapsed = time.time() - t0
        log.info(f"  Trained in {elapsed:.1f}s")

        # Save scales
        save_scales(model, save_dir / f"scales_{domain}.pt")

        # Eval PPL on all domains (diagonal dominance check)
        log.info(f"\n  PPL with {domain} scales:")
        domain_ppls = {}
        for eval_domain in domains:
            eval_texts = load_domain_data(eval_domain, n_samples=EVAL_SAMPLES)
            ppl = eval_ppl(model, tokenizer, eval_texts, label=f"  {domain}_scales/{eval_domain}")
            domain_ppls[eval_domain] = ppl

        results[domain] = {
            "ppls": domain_ppls,
            "losses": losses,
            "train_time": elapsed,
        }

    # ─── Train with answer-only loss ───
    log.info(f"\n{'='*60}")
    log.info(f"Training: math scales (ANSWER-ONLY loss)")
    log.info(f"{'='*60}")

    reset_scales(model)
    math_texts = load_domain_data("math", n_samples=200)
    t0 = time.time()
    losses = train_scales(model, tokenizer, math_texts, epochs=EPOCHS, lr=LR, answer_only=True)
    elapsed = time.time() - t0
    log.info(f"  Trained in {elapsed:.1f}s")
    save_scales(model, save_dir / "scales_math_answer_only.pt")

    # Accuracy comparison
    log.info(f"\n--- ACCURACY: math_answer_only scales ---")
    ao_gsm = eval_gsm8k(model, tokenizer, n=EVAL_SAMPLES)
    ao_trivia = eval_trivia(model, tokenizer, n=EVAL_SAMPLES)

    # Also check standard math scales accuracy
    log.info(f"\n--- ACCURACY: math_standard scales ---")
    load_scales(model, save_dir / "scales_math.pt")
    std_gsm = eval_gsm8k(model, tokenizer, n=EVAL_SAMPLES)
    std_trivia = eval_trivia(model, tokenizer, n=EVAL_SAMPLES)

    # ─── Summary ───
    log.info(f"\n{'='*60}")
    log.info("SUMMARY — Diagonal Dominance Check")
    log.info(f"{'='*60}\n")

    log.info(f"{'Scales':<20} {'Math PPL':>10} {'Lang PPL':>10} {'Code PPL':>10}")
    log.info("-" * 52)
    log.info(f"{'baseline':<20} {baseline['math']:>10.2f} {baseline['language']:>10.2f} {baseline['code']:>10.2f}")
    for domain in domains:
        ppls = results[domain]["ppls"]
        marker = lambda d: " *" if d == domain else ""
        log.info(f"{domain + ' scales':<20} {ppls['math']:>10.2f}{marker('math')} {ppls['language']:>10.2f}{marker('language')} {ppls['code']:>10.2f}{marker('code')}")

    log.info(f"\n{'='*60}")
    log.info("ACCURACY COMPARISON")
    log.info(f"{'='*60}\n")
    log.info(f"{'Config':<25} {'GSM8K':>10} {'TriviaQA':>10}")
    log.info("-" * 47)
    log.info(f"{'baseline':<25} {baseline_gsm:>10.1%} {baseline_trivia:>10.1%}")
    log.info(f"{'math_standard':<25} {std_gsm:>10.1%} {std_trivia:>10.1%}")
    log.info(f"{'math_answer_only':<25} {ao_gsm:>10.1%} {ao_trivia:>10.1%}")

    # Save results
    all_results = {
        "baseline": {"ppls": baseline, "gsm8k": baseline_gsm, "trivia": baseline_trivia},
        "domain_results": {d: {"ppls": r["ppls"], "train_time": r["train_time"]} for d, r in results.items()},
        "answer_only": {"gsm8k": ao_gsm, "trivia": ao_trivia},
        "standard_math": {"gsm8k": std_gsm, "trivia": std_trivia},
    }
    with open(save_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"\nResults saved to {save_dir / 'results.json'}")


if __name__ == "__main__":
    main()
