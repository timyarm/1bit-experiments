"""Exp 17b: Late-knowledge overlay on full math scales.

Proper cooperative test:
  - Base: math_scales_v2.pt (ALL layers trained on math → 28.0% GSM8K)
  - Overlay: replace late-ffn (layers 12-23, gate/up) with knowledge_scales_v2.pt

This tests: "full math everywhere, knowledge training overlaid on late layers only"
vs the Exp 17a result where we trained early-ffn on math + late-ffn on knowledge from scratch.

Exp 17a result for reference: 25.0% GSM8K / 43.8% MMLU
flat_0.7 for reference:        40.0% GSM8K / 41.7% MMLU
full math baseline:             28.0% GSM8K

No training needed — just scale arithmetic + GGUF patch + eval.
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

LOG_PATH = "/tmp/late_overlay_eval.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_PATH, mode='w'),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("overlay")

sys.path.insert(0, os.path.dirname(__file__))

CKPT    = os.path.expanduser("~/freigent/apps/trucksim/checkpoints")
GGUF_DIR = os.path.expanduser("~/freigent/apps/trucksim/data/llm")
GGUF_BASE = f"{GGUF_DIR}/Bonsai-1.7B.gguf"
GGUF_OUT  = f"{GGUF_DIR}/Bonsai-1.7B-overlay.gguf"

EARLY_CUTOFF = 12
FFN_TYPES = {"mlp.gate_proj", "mlp.up_proj"}
N_GSM8K = 100
N_MMLU  = 50
MMLU_SUBJECTS = ["high_school_world_history", "world_religions", "nutrition",
                 "global_facts", "miscellaneous", "sociology"]


def get_layer_idx(name):
    if "model.layers." not in name:
        return None
    try:
        return int(name.split("model.layers.")[1].split(".")[0])
    except (IndexError, ValueError):
        return None


def is_late_ffn(name):
    idx = get_layer_idx(name)
    return idx is not None and idx >= EARLY_CUTOFF and any(t in name for t in FFN_TYPES)


def patch_gguf(scales, output_path):
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

    name_to_scales = {map_name(k): v for k, v in scales.items()}
    patched = 0
    for t in reader.tensors:
        tname = str(t.name)
        if tname not in name_to_scales:
            continue
        s = name_to_scales[tname].cpu().numpy().astype(np.float16).flatten()
        offset = t.data_offset
        n_groups = t.n_bytes // BLOCK
        if len(s) != n_groups:
            continue
        blocks = data[offset:offset + t.n_bytes].reshape(n_groups, BLOCK)
        blocks[:, :2] = s.view(np.uint8).reshape(n_groups, 2)
        patched += 1
    data.flush()
    return patched


def eval_gsm8k(ev, n=N_GSM8K):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test", streaming=True)
    correct = total = 0
    for ex in ds:
        if total >= n: break
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


def eval_mmlu(ev, n=N_MMLU):
    from datasets import load_dataset
    per_subj = max(1, n // len(MMLU_SUBJECTS))
    correct = total = 0
    for subj in MMLU_SUBJECTS:
        if total >= n: break
        try:
            ds = load_dataset("cais/mmlu", subj, split="test", streaming=True)
        except Exception:
            continue
        taken = 0
        for ex in ds:
            if taken >= per_subj or total >= n: break
            opts = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(ex['choices']))
            resp = ev.generate(f"Question: {ex['question']}\n{opts}\nAnswer:", max_tokens=3)
            resp = resp if isinstance(resp, str) else ""
            if resp.strip().upper()[:1] == chr(65 + ex['answer']):
                correct += 1
            total += 1; taken += 1
    return correct / max(total, 1)


def main():
    t0 = time.time()
    log.info("LATE-KNOWLEDGE OVERLAY ON FULL MATH SCALES (Exp 17b)")
    log.info("Base: math_scales_v2 (28% GSM8K) | Overlay: knowledge_scales_v2 on late-ffn only")

    math_scales = torch.load(f"{CKPT}/math_scales_v2.pt", weights_only=False)
    know_scales  = torch.load(f"{CKPT}/knowledge_scales_v2.pt", weights_only=False)
    log.info(f"Loaded {len(math_scales)} math tensors, {len(know_scales)} knowledge tensors")

    # Start from full math scales, overlay late-ffn with knowledge
    combined = {k: v.clone() for k, v in math_scales.items()}
    overlaid = 0
    for name in combined:
        if is_late_ffn(name) and name in know_scales:
            combined[name] = know_scales[name].clone()
            overlaid += 1
    log.info(f"Overlaid {overlaid} late-ffn tensors with knowledge scales")

    n_patched = patch_gguf(combined, GGUF_OUT)
    log.info(f"Patched {n_patched} tensors → {GGUF_OUT}")

    from llama_fast_eval import LlamaEval
    ev = LlamaEval(GGUF_OUT, n_ctx=1024)
    ev.start()
    log.info(f"GSM8K n={N_GSM8K}...")
    gsm = eval_gsm8k(ev)
    log.info(f"MMLU n={N_MMLU}...")
    mmlu = eval_mmlu(ev)
    ev.stop()

    log.info(f"\n{'='*60}")
    log.info("RESULTS — LATE-KNOWLEDGE OVERLAY (Exp 17b)")
    log.info(f"{'='*60}")
    log.info(f"  This run:                  GSM8K={gsm:.1%}  MMLU={mmlu:.1%}")
    log.info(f"  full math (baseline):      GSM8K=28.0%  MMLU=?")
    log.info(f"  Exp 17a coop:              GSM8K=25.0%  MMLU=43.8%")
    log.info(f"  flat_0.7 blend:            GSM8K=40.0%  MMLU=41.7%")
    log.info(f"")
    log.info(f"  GSM8K cost of knowledge overlay: {gsm - 0.28:+.1%}")
    log.info(f"  MMLU gain over flat_0.7:         {mmlu - 0.417:+.1%}")

    results = {"gsm8k": gsm, "mmlu": mmlu,
               "gsm_delta_vs_full_math": gsm - 0.28,
               "mmlu_delta_vs_flat07": mmlu - 0.417}
    with open(f"{CKPT}/overlay_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"  Saved: {CKPT}/overlay_results.json")
    log.info(f"\nTotal: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"CRASHED: {e}", exc_info=True)
        sys.exit(1)
    finally:
        log.info("DONE")
