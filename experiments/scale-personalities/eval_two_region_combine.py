"""Eval for Exp 17: Combine Exp 16 early_ffn scales + Exp 17 late_knowledge scales.

Since the GPU is in a bad state for PyTorch (WDDM TDR on WSL2), we reuse the
already-trained Exp 16 early_ffn scales as the "early_math" component.

Exp 16 early_ffn: trained mlp.gate_proj/up_proj layers 0-11 on GSM8K v2 recipe
                  → 22.0% GSM8K (21.4% of params)
Exp 17 late_knowledge: trained mlp.gate_proj/up_proj layers 12-23 on TriviaQA v2 recipe
                       → saved to coop_late_knowledge_scales.pt

Combine:
  - early-ffn scales (layers 0-11): from early_math_early_ffn_scales.pt
  - late-ffn scales  (layers 12-23): from coop_late_knowledge_scales.pt
  - everything else: baseline (unchanged)

Hypothesis: each region gets scales tuned for what it actually does → exceed 28% GSM8K
while recovering MMLU (flat_0.7 got 40.0% GSM8K / 41.7% MMLU, no training).

GGUF patching uses numpy only (no PyTorch CUDA) — llama.cpp eval has its own CUDA context.
"""
import sys
import os
import json
import time
import logging
import shutil
import re
import numpy as np
import torch

LOG_PATH = "/tmp/two_region_eval.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_PATH, mode='w'),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("coop_eval")

sys.path.insert(0, os.path.dirname(__file__))

CKPT = os.path.expanduser("~/freigent/apps/trucksim/checkpoints")
GGUF_DIR = os.path.expanduser("~/freigent/apps/trucksim/data/llm")
GGUF_BASE = f"{GGUF_DIR}/Bonsai-1.7B.gguf"
GGUF_OUT  = f"{GGUF_DIR}/Bonsai-1.7B-coop.gguf"

EARLY_CUTOFF = 12
FFN_TYPES = {"mlp.gate_proj", "mlp.up_proj"}

FULL_MATH_BASELINE  = 0.280
FLAT_BLEND_BASELINE = 0.400
FLAT_MMLU_BASELINE  = 0.417

N_GSM8K_EVAL = 100
N_MMLU_EVAL  = 50

MMLU_SUBJECTS = ["high_school_world_history", "world_religions", "nutrition",
                 "global_facts", "miscellaneous", "sociology"]


def get_layer_idx(name):
    if "model.layers." not in name:
        return None
    try:
        return int(name.split("model.layers.")[1].split(".")[0])
    except (IndexError, ValueError):
        return None


def is_early_ffn(name):
    idx = get_layer_idx(name)
    return idx is not None and idx < EARLY_CUTOFF and any(t in name for t in FFN_TYPES)


def is_late_ffn(name):
    idx = get_layer_idx(name)
    return idx is not None and idx >= EARLY_CUTOFF and any(t in name for t in FFN_TYPES)


def patch_gguf(combined_scales, output_path):
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

    name_to_scales = {map_name(k): v for k, v in combined_scales.items()}
    patched = 0
    for t in reader.tensors:
        tname = str(t.name)
        if tname not in name_to_scales:
            continue
        scales = name_to_scales[tname].cpu().numpy().astype(np.float16).flatten()
        offset = t.data_offset
        n_groups = t.n_bytes // BLOCK
        if len(scales) != n_groups:
            continue
        blocks = data[offset:offset + t.n_bytes].reshape(n_groups, BLOCK)
        blocks[:, :2] = scales.view(np.uint8).reshape(n_groups, 2)
        patched += 1
    data.flush()
    return patched


def eval_gsm8k(ev, n=N_GSM8K_EVAL):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test", streaming=True)
    correct = total = 0
    for ex in ds:
        if total >= n:
            break
        resp = ev.generate(f"Question: {ex['question']}\nAnswer:", max_tokens=200)
        resp = resp if isinstance(resp, str) else ""
        pred = re.findall(r'-?\d+\.?\d*', resp.replace(',', ''))
        gold = re.findall(r'-?\d+\.?\d*', ex['answer'].split('####')[-1].replace(',', ''))
        try:
            if pred and gold and abs(float(pred[-1]) - float(gold[-1])) < 0.01:
                correct += 1
        except ValueError:
            pass
        total += 1
    return correct / max(total, 1)


def eval_mmlu(ev, n=N_MMLU_EVAL):
    from datasets import load_dataset
    per_subj = max(1, n // len(MMLU_SUBJECTS))
    correct = total = 0
    for subj in MMLU_SUBJECTS:
        if total >= n:
            break
        try:
            ds = load_dataset("cais/mmlu", subj, split="test", streaming=True)
        except Exception:
            continue
        taken = 0
        for ex in ds:
            if taken >= per_subj or total >= n:
                break
            options = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(ex['choices']))
            resp = ev.generate(f"Question: {ex['question']}\n{options}\nAnswer:", max_tokens=3)
            resp = resp if isinstance(resp, str) else ""
            if resp.strip().upper()[:1] == chr(65 + ex['answer']):
                correct += 1
            total += 1
            taken += 1
    return correct / max(total, 1)


def main():
    t0 = time.time()
    log.info("TWO-REGION COMBINE + EVAL (Exp 17)")
    log.info("Using Exp 16 early_ffn scales + Exp 17 late_knowledge scales")

    early_path = f"{CKPT}/early_math_early_ffn_scales.pt"
    late_path  = f"{CKPT}/coop_late_knowledge_scales.pt"

    log.info(f"Loading: {early_path}")
    early_scales = torch.load(early_path, weights_only=False)
    log.info(f"Loading: {late_path}")
    late_scales  = torch.load(late_path, weights_only=False)

    log.info(f"Early scales tensors: {len(early_scales)}")
    log.info(f"Late scales tensors:  {len(late_scales)}")

    # Build combined: early-ffn from early_math, late-ffn from late_knowledge
    combined = {k: v.clone() for k, v in early_scales.items()}
    early_swapped = late_swapped = 0
    for name in combined:
        if is_early_ffn(name):
            early_swapped += 1   # already correct
        elif is_late_ffn(name) and name in late_scales:
            combined[name] = late_scales[name].clone()
            late_swapped += 1

    log.info(f"Combined: {early_swapped} early-ffn tensors (math) + {late_swapped} late-ffn tensors (knowledge)")

    n_patched = patch_gguf(combined, GGUF_OUT)
    log.info(f"Patched {n_patched} tensors into GGUF → {GGUF_OUT}")

    from llama_fast_eval import LlamaEval
    ev = LlamaEval(GGUF_OUT, n_ctx=1024)
    ev.start()
    log.info(f"Evaluating GSM8K n={N_GSM8K_EVAL}...")
    gsm = eval_gsm8k(ev)
    log.info(f"Evaluating MMLU n={N_MMLU_EVAL}...")
    mmlu = eval_mmlu(ev)
    ev.stop()

    log.info(f"\n{'='*55}")
    log.info("RESULTS — TWO-REGION COOPERATIVE TRAINING (Exp 17)")
    log.info(f"{'='*55}")
    log.info(f"  GSM8K n={N_GSM8K_EVAL}:  {gsm:.1%}")
    log.info(f"  MMLU  n={N_MMLU_EVAL}:   {mmlu:.1%}")
    log.info(f"")
    log.info(f"  Comparison:")
    log.info(f"    early_ffn only (math training):    22.0% GSM8K  (Exp 16)")
    log.info(f"    full model (math training):        {FULL_MATH_BASELINE:.1%} GSM8K  (Exp 13/v2)")
    log.info(f"    flat_0.7 blend (no training):      {FLAT_BLEND_BASELINE:.1%} GSM8K / {FLAT_MMLU_BASELINE:.1%} MMLU")
    log.info(f"    two-region coop (this run):        {gsm:.1%} GSM8K / {mmlu:.1%} MMLU")

    gsm_vs_full = gsm - FULL_MATH_BASELINE
    gsm_verdict = "EXCEEDS full-model" if gsm_vs_full > 0.02 else \
                  "MATCHES full-model" if abs(gsm_vs_full) <= 0.02 else "below full-model"
    log.info(f"\n  GSM8K verdict: {gsm_verdict} ({gsm_vs_full:+.1%})")

    results = {
        "gsm8k": gsm, "mmlu": mmlu,
        "gsm_vs_full_model": gsm_vs_full,
        "components": {
            "early_ffn": "early_math_early_ffn_scales.pt (Exp 16)",
            "late_ffn": "coop_late_knowledge_scales.pt (Exp 17)",
        }
    }
    out_path = f"{CKPT}/coop_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"  Saved: {out_path}")
    log.info(f"\nTotal: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"CRASHED: {e}", exc_info=True)
        sys.exit(1)
    finally:
        log.info("DONE")
