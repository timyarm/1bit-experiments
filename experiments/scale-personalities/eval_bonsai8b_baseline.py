"""Bonsai 8B 0-shot baseline eval — establish clean measured number.

The README currently shows ~20% GSM8K for Bonsai 8B, sourced from a
byproduct measurement during v1 recipe experiments. This script runs
our exact eval harness on the unmodified Bonsai 8B GGUF to get a
properly measured baseline using the same protocol as all other numbers.

Monitor: ssh -p 2222 timyarm@100.110.173.110 "tail -f /tmp/bonsai8b_baseline.log"
"""
import sys
import os
import json
import time
import logging
import re

LOG_PATH = "/tmp/bonsai8b_baseline.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_PATH, mode='w'),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("b8b")

sys.path.insert(0, os.path.dirname(__file__))

GGUF_DIR = os.path.expanduser("~/freigent/apps/trucksim/data/llm")
GGUF_8B  = f"{GGUF_DIR}/Bonsai-8B.gguf"
CKPT     = os.path.expanduser("~/freigent/apps/trucksim/checkpoints")

N_GSM8K = 100
N_MMLU  = 50
MMLU_SUBJECTS = ["high_school_world_history", "world_religions", "nutrition",
                 "global_facts", "miscellaneous", "sociology"]


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
    log.info("BONSAI 8B BASELINE EVAL")
    log.info(f"GGUF: {GGUF_8B}")
    log.info(f"Protocol: 0-shot, same harness as all 1.7B results")
    log.info(f"GSM8K n={N_GSM8K}, MMLU n={N_MMLU}")

    from llama_fast_eval import LlamaEval
    ev = LlamaEval(GGUF_8B, n_ctx=1024)
    ev.start()

    log.info(f"\nGSM8K n={N_GSM8K}...")
    gsm = eval_gsm8k(ev)
    log.info(f"GSM8K: {gsm:.1%}")

    log.info(f"\nMMlu n={N_MMLU}...")
    mmlu = eval_mmlu(ev)
    log.info(f"MMLU: {mmlu:.1%}")

    ev.stop()

    log.info(f"\n{'='*55}")
    log.info("BONSAI 8B BASELINE RESULTS")
    log.info(f"{'='*55}")
    log.info(f"  GSM8K (0-shot, n={N_GSM8K}): {gsm:.1%}")
    log.info(f"  MMLU  (0-shot, n={N_MMLU}):  {mmlu:.1%}")
    log.info(f"")
    log.info(f"  Context:")
    log.info(f"    Bonsai 1.7B baseline:           5.3% GSM8K")
    log.info(f"    Bonsai 1.7B + scale personalities: 40.0% GSM8K")
    log.info(f"    Bonsai 8B (this run):           {gsm:.1%} GSM8K")

    results = {"gsm8k": gsm, "mmlu": mmlu, "n_gsm8k": N_GSM8K, "n_mmlu": N_MMLU,
               "protocol": "0-shot, same harness as 1.7B evals"}
    with open(f"{CKPT}/bonsai8b_baseline.json", 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved: {CKPT}/bonsai8b_baseline.json")
    log.info(f"Total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"CRASHED: {e}", exc_info=True)
        sys.exit(1)
    finally:
        log.info("DONE")
