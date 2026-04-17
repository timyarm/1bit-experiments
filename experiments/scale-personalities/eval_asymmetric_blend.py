"""Asymmetric per-depth blend sweep.

Finding from eval_layerwise_blend.py:
  - Math scales live in EARLY layers (not late as probe predicted)
  - Late layers carry knowledge/general capability
  - Flat α=0.7 still wins because it doesn't force hard layer boundaries

This script tests graduated asymmetric blends: early layers get higher α (more math),
late layers get lower α (more knowledge). Goal: beat flat_0.7 (40% GSM8K, 41.7% MMLU)
on both benchmarks simultaneously.

Two sweeps:
  1. Alpha grid (split fixed at layer 12):
       early_alpha ∈ {0.8, 0.9, 1.0}  ×  late_alpha ∈ {0.3, 0.4, 0.5}
       = 9 combos + flat_0.7 control = 10 evals

  2. Crossover sweep (best combo from above applied, split varies):
       split ∈ {6, 8, 10, 14, 16, 18}  (where early ends, late begins)
       = 6 more evals

Total: ~16 evals × 3 min = ~50 min. No new training.
"""
import sys
import os
import time
import json
import logging
import shutil
import itertools
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler("/tmp/asymblend_eval.log", mode='w'),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("asym")

sys.path.insert(0, os.path.dirname(__file__))
from llama_fast_eval import LlamaEval

CKPT = os.path.expanduser("~/freigent/apps/trucksim/checkpoints")
GGUF_DIR = os.path.expanduser("~/freigent/apps/trucksim/data/llm")
GGUF_BASE = f"{GGUF_DIR}/Bonsai-1.7B.gguf"
GGUF_OUT = f"{GGUF_DIR}/Bonsai-1.7B-asym.gguf"

N_LAYERS = 24
N_PER_BENCH = 50

EARLY_ALPHAS = [0.8, 0.9, 1.0]
LATE_ALPHAS = [0.3, 0.4, 0.5]
DEFAULT_SPLIT = 12   # layers < split get early_alpha, >= split get late_alpha

CROSSOVER_SPLITS = [6, 8, 10, 14, 16, 18]
CROSSOVER_EARLY = float(os.environ.get("CROSSOVER_EARLY", "0.9"))
CROSSOVER_LATE = float(os.environ.get("CROSSOVER_LATE", "0.4"))


def get_layer_idx(pt_name):
    if "model.layers." not in pt_name:
        return None
    try:
        return int(pt_name.split("model.layers.")[1].split(".")[0])
    except (IndexError, ValueError):
        return None


def blend_asymmetric(math_scales, knowledge_scales, early_alpha, late_alpha, split):
    """Blend with different α for layers < split vs >= split."""
    blended = {}
    for name in math_scales:
        if name not in knowledge_scales:
            continue
        idx = get_layer_idx(name)
        if idx is None:
            a = (early_alpha + late_alpha) / 2   # embeddings: neutral
        elif idx < split:
            a = early_alpha
        else:
            a = late_alpha
        blended[name] = a * math_scales[name] + (1 - a) * knowledge_scales[name]
    return blended


def patch_gguf(blended_scales, output_path):
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

    name_to_scales = {map_name(k): v for k, v in blended_scales.items()}
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


def eval_gsm8k(ev, n=N_PER_BENCH):
    from datasets import load_dataset
    import re
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


MMLU_SUBJECTS = ["high_school_world_history", "world_religions", "nutrition",
                 "global_facts", "miscellaneous", "sociology"]


def eval_mmlu(ev, n=N_PER_BENCH):
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
            resp_clean = resp.strip().upper()
            if resp_clean and resp_clean[0] == chr(65 + ex['answer']):
                correct += 1
            total += 1
            taken += 1
    return correct / max(total, 1)


def run_eval(label, blended):
    patch_gguf(blended, GGUF_OUT)
    ev = LlamaEval(GGUF_OUT, n_ctx=1024)
    ev.start()
    gsm = eval_gsm8k(ev)
    mmlu = eval_mmlu(ev)
    ev.stop()
    time.sleep(1)
    log.info(f"  {label:<30} GSM8K={gsm:.1%}  MMLU={mmlu:.1%}  avg={(gsm+mmlu)/2:.1%}")
    return {"gsm8k": gsm, "mmlu": mmlu, "avg": (gsm + mmlu) / 2}


def main():
    t0 = time.time()
    log.info("ASYMMETRIC BLEND SWEEP")
    log.info(f"Early alphas: {EARLY_ALPHAS}  Late alphas: {LATE_ALPHAS}  Split: {DEFAULT_SPLIT}")
    log.info(f"Crossover splits: {CROSSOVER_SPLITS} (early={CROSSOVER_EARLY}, late={CROSSOVER_LATE})")
    log.info(f"Baseline to beat: GSM8K=40.0%  MMLU=41.7%  (flat_0.7)")

    math_scales = torch.load(f"{CKPT}/math_scales_v2.pt", weights_only=False)
    knowledge_scales = torch.load(f"{CKPT}/knowledge_scales_v2.pt", weights_only=False)
    log.info(f"Loaded scales: {len(math_scales)} tensors")

    results = {}

    # ── Sweep 1: alpha grid (fixed split=12) ──────────────────────────────────
    log.info("\n" + "="*55)
    log.info("SWEEP 1: Alpha grid (split=12)")
    log.info("="*55)

    # flat_0.7 control first
    label = "flat_0.7 (control)"
    blended = blend_asymmetric(math_scales, knowledge_scales, 0.7, 0.7, DEFAULT_SPLIT)
    results[label] = run_eval(label, blended)

    best_gsm = results[label]["gsm8k"]
    best_label = label

    for early_a, late_a in itertools.product(EARLY_ALPHAS, LATE_ALPHAS):
        label = f"e{early_a:.1f}_l{late_a:.1f}_s{DEFAULT_SPLIT}"
        blended = blend_asymmetric(math_scales, knowledge_scales, early_a, late_a, DEFAULT_SPLIT)
        results[label] = run_eval(label, blended)
        if results[label]["gsm8k"] > best_gsm:
            best_gsm = results[label]["gsm8k"]
            best_label = label
            best_early, best_late = early_a, late_a

    log.info(f"\nBest on GSM8K after sweep 1: {best_label} → {best_gsm:.1%}")

    # ── Sweep 2: crossover layer (best alpha or CROSSOVER_EARLY/LATE) ─────────
    log.info("\n" + "="*55)
    log.info(f"SWEEP 2: Crossover layer (early={CROSSOVER_EARLY}, late={CROSSOVER_LATE})")
    log.info("="*55)

    for split in CROSSOVER_SPLITS:
        label = f"e{CROSSOVER_EARLY:.1f}_l{CROSSOVER_LATE:.1f}_s{split}"
        blended = blend_asymmetric(math_scales, knowledge_scales,
                                   CROSSOVER_EARLY, CROSSOVER_LATE, split)
        results[label] = run_eval(label, blended)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info(f"\n{'='*65}")
    log.info("ASYMMETRIC BLEND SUMMARY")
    log.info(f"{'='*65}")
    log.info(f"{'Profile':<30}{'GSM8K':<10}{'MMLU':<10}{'Avg':<10}")

    baseline_gsm = results["flat_0.7 (control)"]["gsm8k"]
    baseline_mmlu = results["flat_0.7 (control)"]["mmlu"]

    # Sort by GSM8K descending
    for label, r in sorted(results.items(), key=lambda x: -x[1]["gsm8k"]):
        beat_gsm = "✓" if r["gsm8k"] > baseline_gsm else " "
        beat_mmlu = "✓" if r["mmlu"] > baseline_mmlu else " "
        log.info(f"{label:<30}{r['gsm8k']:<10.1%}{r['mmlu']:<10.1%}{r['avg']:<10.1%}"
                 f"  GSM:{beat_gsm} MMLU:{beat_mmlu}")

    winners = [(k, v) for k, v in results.items()
               if v["gsm8k"] > baseline_gsm and v["mmlu"] > baseline_mmlu]
    if winners:
        log.info(f"\n{'='*65}")
        log.info("BEATS flat_0.7 ON BOTH BENCHMARKS:")
        for k, v in sorted(winners, key=lambda x: -x[1]["avg"]):
            log.info(f"  {k}: GSM8K={v['gsm8k']:.1%} MMLU={v['mmlu']:.1%}")
    else:
        log.info("\nNo single profile beats flat_0.7 on both — flat blend remains optimal")

    out_path = f"{CKPT}/asymblend_results.json"
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
