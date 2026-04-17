"""Scale interpolation curve: math ↔ knowledge scales blended at 11 ratios.

For each alpha in [0.0, 0.1, ..., 1.0]:
  blended_scales = alpha * math_scales + (1 - alpha) * knowledge_scales
Patch GGUF. Eval on GSM8K (n=50) and MMLU-Knowledge (n=50).

Produces a tradeoff curve: can we find a sweet spot that's better than
either profile alone on BOTH benchmarks?
"""
import sys
import os
import time
import json
import logging
import shutil
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler("/tmp/interp_eval.log", mode='w'),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("interp")

sys.path.insert(0, os.path.dirname(__file__))
from llama_fast_eval import LlamaEval

CKPT = os.path.expanduser("~/freigent/apps/trucksim/checkpoints")
GGUF_DIR = os.path.expanduser("~/freigent/apps/trucksim/data/llm")
GGUF_BASE = f"{GGUF_DIR}/Bonsai-1.7B.gguf"
GGUF_OUT = f"{GGUF_DIR}/Bonsai-1.7B-interp.gguf"

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_PER_BENCH = 50


def patch_gguf(blended_scales, output_path, verbose=False):
    """Patch GGUF with given blended scales dict. Uses numpy memmap."""
    from gguf import GGUFReader
    shutil.copy2(GGUF_BASE, output_path)
    reader = GGUFReader(GGUF_BASE)
    data = np.memmap(output_path, dtype=np.uint8, mode='r+')
    BLOCK = 18  # 2 bytes fp16 scale + 16 bytes packed signs
    patched = 0

    def map_name(pt_name):
        n = pt_name.replace('model.layers.', 'blk.')
        n = n.replace('.self_attn.q_proj', '.attn_q').replace('.self_attn.k_proj', '.attn_k')
        n = n.replace('.self_attn.v_proj', '.attn_v').replace('.self_attn.o_proj', '.attn_output')
        n = n.replace('.mlp.gate_proj', '.ffn_gate').replace('.mlp.up_proj', '.ffn_up')
        n = n.replace('.mlp.down_proj', '.ffn_down')
        return n + '.weight'

    name_to_scales = {map_name(k): v for k, v in blended_scales.items()}

    for t in reader.tensors:
        tname = str(t.name)
        if tname not in name_to_scales:
            continue
        scales = name_to_scales[tname].cpu().numpy().astype(np.float16).flatten()
        offset = t.data_offset
        for g in range(len(scales)):
            block_start = offset + g * BLOCK
            data[block_start:block_start + 2] = scales[g].view(np.uint8).tobytes()
        patched += 1
    data.flush()
    if verbose:
        log.info(f"  patched {patched} tensors")


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


def main():
    t0 = time.time()
    log.info(f"SCALE INTERPOLATION CURVE — {len(ALPHAS)} points × GSM8K({N_PER_BENCH}) + MMLU({N_PER_BENCH})")
    log.info(f"alpha = 0.0 means pure knowledge, 1.0 means pure math")

    math_scales = torch.load(f"{CKPT}/math_scales_v2.pt", weights_only=False)
    knowledge_scales = torch.load(f"{CKPT}/knowledge_scales_v2.pt", weights_only=False)
    log.info(f"  math: {len(math_scales)} layers, knowledge: {len(knowledge_scales)} layers")

    results = {}
    for alpha in ALPHAS:
        log.info(f"\n--- alpha = {alpha:.1f} ---")
        blended = {}
        for name in math_scales:
            if name in knowledge_scales:
                blended[name] = alpha * math_scales[name] + (1 - alpha) * knowledge_scales[name]

        patch_gguf(blended, GGUF_OUT)

        ev = LlamaEval(GGUF_OUT, n_ctx=1024)
        ev.start()
        gsm = eval_gsm8k(ev)
        mmlu = eval_mmlu(ev)
        ev.stop()
        time.sleep(1)

        log.info(f"  alpha={alpha:.1f}  GSM8K={gsm:.1%}  MMLU={mmlu:.1%}")
        results[f"{alpha:.1f}"] = {"gsm8k": gsm, "mmlu": mmlu}

    log.info(f"\n{'='*50}")
    log.info("INTERPOLATION CURVE")
    log.info(f"{'='*50}")
    log.info(f"{'alpha':<8}{'GSM8K':<10}{'MMLU':<10}")
    for a in ALPHAS:
        r = results[f"{a:.1f}"]
        log.info(f"{a:<8.1f}{r['gsm8k']:<10.1%}{r['mmlu']:<10.1%}")

    # Look for sweet spots
    best_avg = max(results.items(), key=lambda kv: (kv[1]['gsm8k'] + kv[1]['mmlu']) / 2)
    log.info(f"\nBest average: alpha={best_avg[0]}  GSM8K={best_avg[1]['gsm8k']:.1%}  MMLU={best_avg[1]['mmlu']:.1%}")

    out_path = f"{CKPT}/interpolation_results.json"
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
