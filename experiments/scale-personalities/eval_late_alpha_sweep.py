"""Exp 18: Late-layer α sweep — find Pareto optimum on late-ffn blend.

Finding from Exp 17b: full math base + pure knowledge in late-ffn (α_late=0)
gave GSM8K=31% (+3pp over full math 28%) but MMLU collapsed to 20.8%.

This sweeps α_late ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7} where:
  α_late=1.0 → full math everywhere (28% GSM8K baseline)
  α_late=0.0 → pure knowledge in late-ffn (31% GSM8K / 20.8% MMLU, Exp 17b)

Early layers always locked at full math (α_early=1.0).

No training — pure scale arithmetic + GGUF patch + eval.
~6 evals × 3 min = ~20 min total.

Monitor: ssh -p 2222 timyarm@100.110.173.110 "tail -f /tmp/late_alpha_sweep.log"
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

LOG_PATH = "/tmp/late_alpha_sweep.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_PATH, mode='w'),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("sweep")

sys.path.insert(0, os.path.dirname(__file__))

CKPT     = os.path.expanduser("~/freigent/apps/trucksim/checkpoints")
GGUF_DIR = os.path.expanduser("~/freigent/apps/trucksim/data/llm")
GGUF_BASE = f"{GGUF_DIR}/Bonsai-1.7B.gguf"
GGUF_OUT  = f"{GGUF_DIR}/Bonsai-1.7B-sweep.gguf"

EARLY_CUTOFF = 12
FFN_TYPES = {"mlp.gate_proj", "mlp.up_proj"}

ALPHA_LATES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]   # α=1.0 is just full math (known: 28% GSM8K)

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


def build_scales(math_scales, know_scales, alpha_late):
    """Full math base; late-ffn blended at alpha_late*math + (1-alpha_late)*knowledge."""
    combined = {k: v.clone() for k, v in math_scales.items()}
    for name in combined:
        if is_late_ffn(name) and name in know_scales:
            combined[name] = alpha_late * math_scales[name] + (1 - alpha_late) * know_scales[name]
    return combined


def patch_gguf(scales, output_path):
    from gguf import GGUFReader
    shutil.copy2(GGUF_BASE, output_path)
    reader = GGUFReader(GGUF_BASE)
    data = np.memmap(output_path, dtype=np.uint8, mode='r+')
    BLOCK = 18

    def map_name(n):
        n = n.replace('model.layers.', 'blk.')
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
    log.info("LATE-LAYER α SWEEP (Exp 18)")
    log.info("Early layers: full math (α=1.0) | Late-ffn: sweep α_late")
    log.info(f"Points: {ALPHA_LATES}  (+ known endpoints: α=1.0→28%GSM8K, α=0.0→31%GSM8K/20.8%MMLU)")

    math_scales = torch.load(f"{CKPT}/math_scales_v2.pt", weights_only=False)
    know_scales  = torch.load(f"{CKPT}/knowledge_scales_v2.pt", weights_only=False)
    log.info(f"Loaded {len(math_scales)} math + {len(know_scales)} knowledge tensors")

    from llama_fast_eval import LlamaEval

    results = {
        "alpha_1.0": {"gsm8k": 0.280, "mmlu": None, "avg": None},  # known
        "alpha_0.0": {"gsm8k": 0.310, "mmlu": 0.208, "avg": None},  # Exp 17b
    }

    log.info(f"\n{'Alpha':<12}{'GSM8K':<10}{'MMLU':<10}{'Avg':<10}")
    log.info(f"{'1.0 (math)':<12}{'28.0%':<10}{'—':<10}{'—':<10}  [known baseline]")
    log.info(f"{'0.0 (know)':<12}{'31.0%':<10}{'20.8%':<10}{'—':<10}  [Exp 17b]")

    for alpha_late in ALPHA_LATES:
        label = f"alpha_{alpha_late:.1f}"
        scales = build_scales(math_scales, know_scales, alpha_late)
        patch_gguf(scales, GGUF_OUT)

        ev = LlamaEval(GGUF_OUT, n_ctx=1024)
        ev.start()
        gsm  = eval_gsm8k(ev)
        mmlu = eval_mmlu(ev)
        ev.stop()
        time.sleep(1)

        avg = (gsm + mmlu) / 2
        results[label] = {"gsm8k": gsm, "mmlu": mmlu, "avg": avg, "alpha_late": alpha_late}
        log.info(f"{alpha_late:<12.1f}{gsm:<10.1%}{mmlu:<10.1%}{avg:<10.1%}")

    # Summary
    log.info(f"\n{'='*60}")
    log.info("SUMMARY — LATE-LAYER α SWEEP")
    log.info(f"{'='*60}")
    log.info(f"{'Alpha':<12}{'GSM8K':<10}{'MMLU':<10}{'Avg':<10}  Notes")

    best_gsm = best_gsm_label = None
    best_avg = best_avg_label = None

    for label, r in sorted(results.items()):
        if r["mmlu"] is None:
            continue
        gsm, mmlu, avg = r["gsm8k"], r["mmlu"], r.get("avg") or (r["gsm8k"]+r["mmlu"])/2
        beat_flat = "✓ beats flat_0.7 avg" if avg > 0.409 else ""
        a = r.get("alpha_late", label)
        log.info(f"{str(a):<12}{gsm:<10.1%}{mmlu:<10.1%}{avg:<10.1%}  {beat_flat}")
        if best_gsm is None or gsm > best_gsm:
            best_gsm, best_gsm_label = gsm, label
        if best_avg is None or avg > best_avg:
            best_avg, best_avg_label = avg, label

    log.info(f"\n  Best GSM8K: {best_gsm_label} → {best_gsm:.1%}")
    log.info(f"  Best avg:   {best_avg_label} → {best_avg:.1%}")
    log.info(f"  flat_0.7 reference: GSM8K=40.0%  MMLU=41.7%  avg=40.9%")

    out = f"{CKPT}/late_alpha_sweep_results.json"
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
