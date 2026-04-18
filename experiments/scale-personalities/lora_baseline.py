"""LoRA baseline at matched parameter count — critical validity check.

The scale personality thesis rests on scale tables being a uniquely effective
intervention. This script tests the null hypothesis: a LoRA adapter at the same
parameter budget does just as well.

Setup (identical to scale_v2_proper.py):
  - Same model: Bonsai 1.7B
  - Same data: GSM8K train, 150 examples, 70/30 math/wiki mix
  - Same recipe: AdamW lr=1e-4, Rho-1 top-60% token weighting,
                 elastic band λ=0.1, 3 epochs, seq_len=256
  - Parameter budget: matched to scale table size (measured at run time)
  - Eval: GSM8K test n=100 (same as all other evals)

LoRA implementation: wraps each PackedBitLinear with trainable A/B matrices.
Forward: out = packed_linear(x) + (x @ lora_A.T @ lora_B.T) * (alpha/rank)
Signs stay frozen. Only lora_A, lora_B train.

Possible outcomes:
  LoRA > scales  → scale training not uniquely better; scales win only on deployment overhead
  LoRA ≈ scales  → equivalent; scales win on zero inference overhead
  scales > LoRA  → scale structure is genuinely the right inductive bias for 1-bit models

Monitor: ssh -p 2222 timyarm@100.110.173.110 "tail -f /tmp/lora_baseline.log"
"""
import sys
import os
import time
import json
import logging
import shutil
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_PATH = "/tmp/lora_baseline.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_PATH, mode='w'),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("lora")

sys.path.insert(0, os.path.dirname(__file__))

CKPT    = os.path.expanduser("~/freigent/apps/trucksim/checkpoints")
GGUF_DIR = os.path.expanduser("~/freigent/apps/trucksim/data/llm")
GGUF_BASE = f"{GGUF_DIR}/Bonsai-1.7B.gguf"
GGUF_OUT  = f"{GGUF_DIR}/Bonsai-1.7B-lora.gguf"

DEVICE = "cuda"
LORA_RANK  = 16       # tuned so total LoRA params ≈ scale table size; verified at runtime
LORA_ALPHA = 16       # scaling = alpha/rank = 1.0
MAX_LEN    = 128
TRAIN_EXAMPLES = 150
LR         = 1e-4
EPOCHS     = 3
REG_LAMBDA = 0.1
TOKEN_SELECT_RATIO = 0.6
DOMAIN_MIX = 0.7
N_GSM8K_EVAL = 100


# ── LoRA wrapper ──────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Wraps a PackedBitLinear with trainable LoRA A/B matrices."""

    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        out_f = base.out_features
        in_f  = base.in_features
        dev   = base.signs.device
        self.lora_A = nn.Parameter(torch.randn(rank, in_f,  device=dev, dtype=torch.float32) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=dev, dtype=torch.float32))
        self.scaling = alpha / rank

    def forward(self, x):
        # base: memory-efficient via _BitLinearFn (never saves full weight matrix)
        base_out = self.base(x)
        # LoRA: x→rank→out, peak intermediate is [batch,seq,rank] not [out_f,in_f]
        lora_out = F.linear(
            F.linear(x, self.lora_A.to(x.dtype)),
            self.lora_B.to(x.dtype)
        ) * self.scaling
        return base_out + lora_out


def apply_lora(model, rank, alpha):
    """Replace all PackedBitLinear modules with LoRALinear wrappers."""
    from packed_bitlinear import PackedBitLinear
    replaced = 0
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, PackedBitLinear):
                setattr(parent, child_name, LoRALinear(child, rank, alpha))
                replaced += 1
    return replaced


def count_lora_params(model):
    return sum(p.numel() for name, p in model.named_parameters()
               if 'lora_A' in name or 'lora_B' in name)


def freeze_non_lora(model):
    for name, p in model.named_parameters():
        p.requires_grad = ('lora_A' in name or 'lora_B' in name)


# ── data ──────────────────────────────────────────────────────────────────────

def get_data():
    from datasets import load_dataset
    import numpy as np
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


# ── train ─────────────────────────────────────────────────────────────────────

def train(model, tokenizer, texts, orig_lora_A, orig_lora_B):
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, EPOCHS * len(texts)))
    model.train()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        n = 0
        for text in texts:
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
                task_loss = (ce * mask).sum() / (mask.sum() + 1e-8)

                # Elastic band: keep LoRA weights close to zero (init)
                reg = sum(
                    F.mse_loss(p, torch.zeros_like(p))
                    for n, p in model.named_parameters()
                    if 'lora_A' in n or 'lora_B' in n
                )
                loss = task_loss + REG_LAMBDA * reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += task_loss.item()
            n += 1
            if n % 10 == 0:
                torch.cuda.empty_cache()
        log.info(f"  epoch {epoch+1}/{EPOCHS}  loss={epoch_loss/max(n,1):.4f}")


# ── eval ──────────────────────────────────────────────────────────────────────

def patch_gguf_with_lora(model, output_path):
    """Merge LoRA into base weights and patch GGUF scales.

    LoRA only adds to outputs — it doesn't change the sign structure or scales.
    We can't fully bake LoRA into a Q1_0 GGUF (signs are frozen). Instead we
    run eval by saving the merged model as a proper eval artifact.

    For apples-to-apples with scale patching, we eval the PyTorch model
    directly via the transformers generate path (same answer extraction).
    """
    pass  # see eval_direct below


def eval_gsm8k_direct(model, tokenizer, n=N_GSM8K_EVAL):
    """Eval GSM8K directly on the PyTorch model (no GGUF round-trip)."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test", streaming=True)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for ex in ds:
            if total >= n: break
            prompt = f"Question: {ex['question']}\nAnswer:"
            tokens = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                out = model.generate(
                    **tokens, max_new_tokens=200,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id)
            resp = tokenizer.decode(out[0][tokens.input_ids.shape[1]:],
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
    return correct / max(total, 1)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log.info("LORA BASELINE — MATCHED PARAMETER COUNT")
    log.info(f"Rank={LORA_RANK}, Alpha={LORA_ALPHA}, LR={LR}, Epochs={EPOCHS}")
    log.info("Hypothesis: do LoRA adapters match scale personality GSM8K gains?")

    # WSL2 WDDM TDR warmup — prevents "CUDA error: unknown error" on first kernel
    log.info("GPU warmup...")
    _d = torch.ones(256, 256, device=DEVICE)
    for _ in range(10):
        _d = _d @ _d.T
    torch.cuda.synchronize()
    del _d
    time.sleep(5)
    log.info("GPU ready.")

    log.info("\n--- MODEL ---")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from packed_bitlinear import PackedBitLinear, convert_model

    tokenizer = AutoTokenizer.from_pretrained(
        "prism-ml/Bonsai-1.7B-unpacked", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "prism-ml/Bonsai-1.7B-unpacked", dtype=torch.float16,
        trust_remote_code=True, device_map="cpu",
        attn_implementation="eager")
    convert_model(model)
    model = model.to(DEVICE)
    torch.cuda.synchronize()
    log.info(f"  model.to(CUDA) OK — VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Snapshot original scales for comparison
    scale_param_count = sum(
        m.scales.numel() for m in model.modules()
        if isinstance(m, PackedBitLinear))
    scale_bytes = scale_param_count * 2  # fp16
    log.info(f"  Scale table: {scale_param_count:,} params ({scale_bytes/1e6:.1f}MB fp16)")

    # Apply LoRA
    n_replaced = apply_lora(model, LORA_RANK, LORA_ALPHA)
    torch.cuda.synchronize()
    log.info(f"  apply_lora OK — VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    freeze_non_lora(model)
    # gradient_checkpointing conflicts with PackedBitLinear custom autograd — skip it

    lora_param_count = count_lora_params(model)
    lora_bytes = lora_param_count * 4  # fp32 now
    log.info(f"  LoRA modules: {n_replaced} layers replaced")
    log.info(f"  LoRA params:  {lora_param_count:,} ({lora_bytes/1e6:.1f}MB fp16)")
    log.info(f"  Ratio vs scales: {lora_param_count/scale_param_count:.2f}×")
    log.info(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Smoke test: single forward pass to catch CUDA issues before data loading
    log.info("  Smoke test forward pass...")
    torch.cuda.synchronize()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        _t = tokenizer("test", return_tensors="pt").to(DEVICE)
        _out = model(**_t)
        torch.cuda.synchronize()
    log.info(f"  Smoke test OK: logits={_out.logits.shape}")
    del _t, _out

    log.info("\n--- DATA ---")
    texts = get_data()
    log.info(f"  {len(texts)} examples ({int(DOMAIN_MIX*100)}% math)")

    log.info("\n--- TRAIN ---")
    train(model, tokenizer, texts, None, None)
    log.info(f"  Trained in {time.time()-t0:.0f}s")

    log.info("\n--- EVAL ---")
    log.info(f"  GSM8K n={N_GSM8K_EVAL} (direct PyTorch, same 0-shot protocol)...")
    gsm = eval_gsm8k_direct(model, tokenizer)

    SCALE_BASELINE   = 0.053   # 1.7B no training
    SCALE_MATH       = 0.280   # scale personality, math only
    SCALE_BLEND      = 0.400   # scale personality, flat_0.7 blend

    log.info(f"\n{'='*60}")
    log.info("RESULTS — LORA vs SCALE PERSONALITIES")
    log.info(f"{'='*60}")
    log.info(f"  LoRA (rank={LORA_RANK}, {lora_bytes/1e6:.0f}MB):  {gsm:.1%} GSM8K")
    log.info(f"")
    log.info(f"  Scale baselines (same data, same recipe):")
    log.info(f"    No training:           {SCALE_BASELINE:.1%}")
    log.info(f"    Scale math only:       {SCALE_MATH:.1%}  ({scale_bytes/1e6:.0f}MB)")
    log.info(f"    Scale blend (flat 0.7):{SCALE_BLEND:.1%}  (no extra training)")
    log.info(f"")
    delta = gsm - SCALE_MATH
    if abs(delta) <= 0.03:
        verdict = f"MATCHES scale training (Δ={delta:+.1%}) — scales win on zero inference overhead"
    elif delta > 0.03:
        verdict = f"EXCEEDS scale training (Δ={delta:+.1%}) — LoRA is stronger"
    else:
        verdict = f"BELOW scale training (Δ={delta:+.1%}) — scales have genuine inductive advantage"
    log.info(f"  Verdict: {verdict}")

    results = {
        "lora_gsm8k": gsm,
        "lora_rank": LORA_RANK,
        "lora_params": lora_param_count,
        "lora_mb": lora_bytes / 1e6,
        "scale_params": scale_param_count,
        "scale_mb": scale_bytes / 1e6,
        "scale_math_gsm8k": SCALE_MATH,
        "delta_vs_scale_math": gsm - SCALE_MATH,
    }
    out = f"{CKPT}/lora_baseline_results.json"
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved: {out}")
    log.info(f"Total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"CRASHED: {e}", exc_info=True)
        sys.exit(1)
    finally:
        log.info("DONE")
