"""Data efficiency curve: train math scales on {10, 30, 100, 300} examples,
eval GSM8K at each point.

Question being tested: are we data-starved at 150? If the curve is still rising
at 300, yes — the 5.3x result undersells what this recipe can do. If it plateaus
before 150, then 150 is already saturated and more data wouldn't help.

Uses the same v2 recipe as scale_v2_proper.py (AdamW 1e-4, Rho-1 token selection,
elastic band reg, 3 epochs, seq_len 256). Only the number of training examples
varies.
"""
import sys
import os
import time
import json
import logging
import shutil
import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler("/tmp/data_efficiency.log", mode='w'),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("deff")

sys.path.insert(0, os.path.dirname(__file__))

CKPT = os.path.expanduser("~/freigent/apps/trucksim/checkpoints")
GGUF_DIR = os.path.expanduser("~/freigent/apps/trucksim/data/llm")
GGUF_BASE = f"{GGUF_DIR}/Bonsai-1.7B.gguf"
GGUF_OUT = f"{GGUF_DIR}/Bonsai-1.7B-deff.gguf"

DEVICE = "cuda"
MAX_LEN = 256
LR = 1e-4
EPOCHS = 3
REG_LAMBDA = 0.1
TOKEN_SELECT_RATIO = 0.6
DOMAIN_MIX = 0.7

N_TRAIN_POINTS = [10, 30, 100, 300]
N_GSM8K_EVAL = 100


def get_math_data(n_needed):
    """Load GSM8K train split + wikitext diverse pool, mix 70/30."""
    from datasets import load_dataset

    math_texts = []
    ds = load_dataset("gsm8k", "main", split="train", streaming=True)
    for ex in ds:
        math_texts.append(f"Question: {ex['question']}\nAnswer: {ex['answer']}")
        if len(math_texts) >= n_needed * 2:
            break

    diverse_texts = []
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
    for ex in ds:
        if len(ex['text'].strip()) > 200:
            diverse_texts.append(ex['text'])
        if len(diverse_texts) >= n_needed:
            break
    return math_texts, diverse_texts


def mix_data(domain_texts, diverse_texts, n_total, ratio=DOMAIN_MIX, seed=42):
    rng = np.random.default_rng(seed)
    n_domain = int(n_total * ratio)
    n_diverse = n_total - n_domain
    mixed = domain_texts[:n_domain] + diverse_texts[:n_diverse]
    idx = rng.permutation(len(mixed))
    return [mixed[i] for i in idx]


def train_math_scales(model, tokenizer, train_texts, orig_snapshots, scale_params):
    optimizer = torch.optim.AdamW(scale_params, lr=LR, weight_decay=0.01)
    total_steps = max(1, EPOCHS * len(train_texts))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    model.train()

    from packed_bitlinear import PackedBitLinear
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        n = 0
        for text in train_texts:
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
            if tokens.input_ids.shape[1] < 10:
                continue

            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(**tokens, labels=tokens.input_ids)
                student_logits = outputs.logits[:, :-1]
                labels = tokens.input_ids[:, 1:]

                ce_per_token = F.cross_entropy(
                    student_logits.reshape(-1, student_logits.size(-1)).float(),
                    labels.reshape(-1),
                    reduction='none',
                )
                threshold = torch.quantile(ce_per_token, 1.0 - TOKEN_SELECT_RATIO)
                mask = (ce_per_token >= threshold).float()
                task_loss = (ce_per_token * mask).sum() / (mask.sum() + 1e-8)

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

            epoch_loss += task_loss.item()
            n += 1

        log.info(f"    epoch {epoch+1}/{EPOCHS}  loss={epoch_loss/max(n,1):.4f}")


def patch_gguf_with_scales(trained_scales, orig_scales, output_path):
    from gguf import GGUFReader
    shutil.copy2(GGUF_BASE, output_path)
    reader = GGUFReader(GGUF_BASE)
    data = np.memmap(output_path, dtype=np.uint8, mode='r+')
    BLOCK = 18

    def map_name(pt_name):
        n = pt_name.replace('model.layers.', 'blk.')
        n = n.replace('.self_attn.q_proj', '.attn_q').replace('.self_attn.k_proj', '.attn_k')
        n = n.replace('.self_attn.v_proj', '.attn_v').replace('.self_attn.o_proj', '.attn_output')
        n = n.replace('.mlp.gate_proj', '.ffn_gate').replace('.mlp.up_proj', '.ffn_up')
        n = n.replace('.mlp.down_proj', '.ffn_down')
        return n + '.weight'

    name_to_trained = {map_name(k): v for k, v in trained_scales.items()}
    patched = 0
    for t in reader.tensors:
        tname = str(t.name)
        if tname not in name_to_trained:
            continue
        scales = name_to_trained[tname].cpu().numpy().astype(np.float16).flatten()
        offset = t.data_offset
        n_groups = t.n_bytes // BLOCK
        if len(scales) != n_groups:
            continue
        blocks = data[offset:offset + t.n_bytes].reshape(n_groups, BLOCK)
        blocks[:, :2] = scales.view(np.uint8).reshape(n_groups, 2)
        patched += 1
    data.flush()
    return patched


def eval_gsm8k(gguf_path, n=N_GSM8K_EVAL):
    from llama_fast_eval import LlamaEval
    from datasets import load_dataset
    import re

    ev = LlamaEval(gguf_path, n_ctx=1024)
    ev.start()
    try:
        ds = load_dataset("gsm8k", "main", split="test", streaming=True)
        correct = total = 0
        for ex in ds:
            if total >= n:
                break
            resp = ev.generate(f"Question: {ex['question']}\nAnswer:", max_tokens=200)
            resp = resp if isinstance(resp, str) else ""
            pred_nums = re.findall(r'-?\d+\.?\d*', resp.replace(',', ''))
            gold_nums = re.findall(r'-?\d+\.?\d*', ex['answer'].split('####')[-1].replace(',', ''))
            try:
                if pred_nums and gold_nums and abs(float(pred_nums[-1]) - float(gold_nums[-1])) < 0.01:
                    correct += 1
            except ValueError:
                pass
            total += 1
        return correct / max(total, 1)
    finally:
        ev.stop()


def main():
    t0 = time.time()
    log.info(f"DATA EFFICIENCY CURVE — math scales @ n = {N_TRAIN_POINTS}")
    log.info(f"Eval: GSM8K test split n={N_GSM8K_EVAL}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from packed_bitlinear import PackedBitLinear, convert_model

    log.info("\n--- MODEL ---")
    tokenizer = AutoTokenizer.from_pretrained("prism-ml/Bonsai-1.7B-unpacked", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "prism-ml/Bonsai-1.7B-unpacked", dtype=torch.float16,
        trust_remote_code=True, device_map="cpu",
    )
    convert_model(model)
    model = model.to(DEVICE)
    model.gradient_checkpointing_enable()
    log.info(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Freeze everything except scales
    for p in model.parameters():
        p.requires_grad = False
    scale_params = []
    orig_scales = {}
    for name, module in model.named_modules():
        if isinstance(module, PackedBitLinear):
            module.scales.requires_grad = True
            scale_params.append(module.scales)
            orig_scales[name] = module.scales.data.clone()

    log.info(f"  Trainable scale params: {sum(p.numel() for p in scale_params):,}")

    # Cache data for largest size
    log.info("\n--- DATA ---")
    math_texts, diverse_texts = get_math_data(max(N_TRAIN_POINTS))
    log.info(f"  Loaded {len(math_texts)} math + {len(diverse_texts)} diverse pool")

    results = {}

    # Baseline eval (no training)
    log.info("\n--- BASELINE (n=0) ---")
    shutil.copy2(GGUF_BASE, GGUF_OUT)
    base_acc = eval_gsm8k(GGUF_OUT)
    log.info(f"  baseline GSM8K: {base_acc:.1%}")
    results["0"] = {"n_train": 0, "gsm8k": base_acc, "train_time": 0.0}

    for n_train in N_TRAIN_POINTS:
        log.info(f"\n{'='*50}")
        log.info(f"n_train = {n_train}")
        log.info(f"{'='*50}")

        # Restore original scales
        for name, module in model.named_modules():
            if isinstance(module, PackedBitLinear) and name in orig_scales:
                module.scales.data.copy_(orig_scales[name])

        train_texts = mix_data(math_texts, diverse_texts, n_train, seed=42)
        log.info(f"  Training on {len(train_texts)} examples ({int(DOMAIN_MIX*100)}% math)")

        t_train = time.time()
        train_math_scales(model, tokenizer, train_texts, orig_scales, scale_params)
        train_time = time.time() - t_train

        # Snapshot trained scales
        trained_scales = {}
        for name, module in model.named_modules():
            if isinstance(module, PackedBitLinear):
                trained_scales[name] = module.scales.data.clone()

        # Patch + eval
        log.info(f"  Patching GGUF and evaluating...")
        patch_gguf_with_scales(trained_scales, orig_scales, GGUF_OUT)
        acc = eval_gsm8k(GGUF_OUT)
        log.info(f"  n={n_train}  GSM8K={acc:.1%}  train_time={train_time:.0f}s")
        results[str(n_train)] = {"n_train": n_train, "gsm8k": acc, "train_time": train_time}

    # Summary
    log.info(f"\n{'='*50}")
    log.info("DATA EFFICIENCY CURVE")
    log.info(f"{'='*50}")
    log.info(f"{'n_train':<10}{'GSM8K':<10}{'delta_vs_base':<15}")
    for k, r in results.items():
        delta = r['gsm8k'] - base_acc
        log.info(f"{r['n_train']:<10}{r['gsm8k']:<10.1%}{delta:+.1%}")

    out_path = f"{CKPT}/data_efficiency_results.json"
    os.makedirs(CKPT, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved: {out_path}")
    log.info(f"Total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"CRASHED: {e}", exc_info=True)
    finally:
        log.info("DONE")
