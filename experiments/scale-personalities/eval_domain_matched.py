"""Domain-matched benchmarks — test each profile on its INTENDED domain.

Previous eval tested all profiles on TriviaQA/ARC-Easy/HellaSwag (general benchmarks).
This runs:
  - math profile → GSM8K (multi-step arithmetic)
  - knowledge profile → MMLU (factual recall across 57 subjects)
  - code profile → MBPP (Python function synthesis with test cases)

All 4 models are evaluated on all 3 benchmarks so we can see the full cross-domain
matrix and whether the matched-domain number is actually a lift.

Runtime: ~45-60 min on GTX 1660 Super via llama.cpp.
Monitor: tail -f /tmp/domain_eval.log
"""

import sys
import os
import time
import json
import logging
import re
import subprocess
import tempfile
import signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("/tmp/domain_eval.log", mode='w'),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("domain_eval")

sys.path.insert(0, os.path.dirname(__file__))
from llama_fast_eval import LlamaEval


def safe_generate(ev, prompt, max_tokens=100):
    """Wrap generate with error handling — llama.cpp can return error dicts."""
    try:
        r = ev.generate(prompt, max_tokens=max_tokens)
        return r if isinstance(r, str) else ""
    except (KeyError, Exception) as e:
        return ""

GGUF_DIR = os.path.expanduser("~/freigent/apps/trucksim/data/llm")
MODELS = {
    "baseline": f"{GGUF_DIR}/Bonsai-1.7B.gguf",
    "math": f"{GGUF_DIR}/Bonsai-1.7B-math-v2.gguf",
    "knowledge": f"{GGUF_DIR}/Bonsai-1.7B-knowledge-v2.gguf",
    "code": f"{GGUF_DIR}/Bonsai-1.7B-code-v2.gguf",
}

N_GSM = 150
N_MMLU = 150
N_MBPP = 100  # MBPP takes longer — code exec + longer generations


def eval_gsm8k(ev, n=N_GSM):
    """GSM8K — multi-step arithmetic word problems."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test", streaming=True)
    correct = 0
    total = 0
    for ex in ds:
        if total >= n:
            break
        prompt = f"Question: {ex['question']}\nAnswer:"
        resp = safe_generate(ev, prompt, max_tokens=200)

        pred = _last_number(resp)
        gold = _last_number(ex['answer'].split("####")[-1])
        if pred is not None and gold is not None and abs(pred - gold) < 0.01:
            correct += 1
        total += 1
    return {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}


MMLU_KNOWLEDGE_SUBJECTS = [
    # Knowledge-heavy subjects (avoiding pure reasoning)
    "high_school_world_history", "high_school_us_history", "prehistory",
    "world_religions", "nutrition", "global_facts",
    "human_aging", "miscellaneous", "international_law",
    "marketing", "public_relations", "sociology",
]


def eval_mmlu(ev, n=N_MMLU):
    """MMLU knowledge subjects — multiple choice factual knowledge."""
    from datasets import load_dataset
    per_subj = max(1, n // len(MMLU_KNOWLEDGE_SUBJECTS))
    correct = 0
    total = 0
    for subj in MMLU_KNOWLEDGE_SUBJECTS:
        if total >= n:
            break
        try:
            ds = load_dataset("cais/mmlu", subj, split="test", streaming=True)
        except Exception as e:
            log.warning(f"  skip {subj}: {e}")
            continue
        taken = 0
        for ex in ds:
            if taken >= per_subj or total >= n:
                break
            q = ex['question']
            choices = ex['choices']
            answer = ex['answer']  # 0-3 int

            options = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
            prompt = f"Question: {q}\n{options}\nAnswer:"
            resp = safe_generate(ev, prompt, max_tokens=3)
            resp_clean = resp.strip().upper()

            if resp_clean and resp_clean[0] == chr(65 + answer):
                correct += 1
            total += 1
            taken += 1
    return {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}


def eval_mbpp(ev, n=N_MBPP):
    """MBPP — generate Python function, run against test cases."""
    from datasets import load_dataset
    ds = load_dataset("mbpp", split="test", streaming=True)
    correct = 0
    total = 0
    for ex in ds:
        if total >= n:
            break
        text = ex['text']
        tests = ex['test_list']

        prompt = (
            f"Write a Python function that solves this task.\n"
            f"Task: {text}\n"
            f"Example test:\n{tests[0] if tests else ''}\n"
            f"```python\n"
        )
        resp = safe_generate(ev, prompt, max_tokens=300)

        code = _extract_code(resp)
        if code and _run_tests(code, tests):
            correct += 1
        total += 1
    return {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}


def _last_number(text):
    nums = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def _extract_code(text):
    """Pull the first code block or the text itself if no fences."""
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            block = parts[1]
            if block.startswith("python\n"):
                block = block[len("python\n"):]
            elif block.startswith("python"):
                block = block[len("python"):]
            return block.strip()
    # Take up to the next blank line
    lines = text.split("\n\n")[0]
    return lines.strip()


def _run_tests(code, tests, timeout=3):
    """Run generated code + test assertions in a subprocess. Return True if all pass."""
    if not code or not tests:
        return False
    full = code + "\n\n" + "\n".join(tests)
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full)
            tmp = f.name
        try:
            result = subprocess.run(
                ["python3", tmp],
                capture_output=True,
                timeout=timeout,
            )
            return result.returncode == 0
        finally:
            os.unlink(tmp)
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def main():
    log.info("=" * 60)
    log.info(f"DOMAIN-MATCHED EVAL — GSM8K({N_GSM}) / MMLU({N_MMLU}) / MBPP({N_MBPP})")
    log.info("=" * 60)
    t_start = time.time()

    all_results = {}

    for model_name, gguf_path in MODELS.items():
        log.info(f"\n{'='*60}")
        log.info(f"MODEL: {model_name}")
        log.info(f"{'='*60}")

        ev = LlamaEval(gguf_path, n_ctx=1024)
        ev.start()
        results = {}

        log.info(f"  GSM8K ({N_GSM}q)...")
        t0 = time.time()
        gsm = eval_gsm8k(ev, N_GSM)
        log.info(f"  GSM8K: {gsm['correct']}/{gsm['total']} = {gsm['accuracy']:.1%} ({time.time()-t0:.0f}s)")
        results['gsm8k'] = gsm

        log.info(f"  MMLU ({N_MMLU}q)...")
        t0 = time.time()
        mmlu = eval_mmlu(ev, N_MMLU)
        log.info(f"  MMLU: {mmlu['correct']}/{mmlu['total']} = {mmlu['accuracy']:.1%} ({time.time()-t0:.0f}s)")
        results['mmlu_knowledge'] = mmlu

        log.info(f"  MBPP ({N_MBPP}q)...")
        t0 = time.time()
        mbpp = eval_mbpp(ev, N_MBPP)
        log.info(f"  MBPP: {mbpp['correct']}/{mbpp['total']} = {mbpp['accuracy']:.1%} ({time.time()-t0:.0f}s)")
        results['mbpp'] = mbpp

        all_results[model_name] = results
        ev.stop()
        time.sleep(2)

    log.info(f"\n{'='*60}")
    log.info("DOMAIN-MATCHED RESULTS")
    log.info(f"{'='*60}")
    benchmarks = ['gsm8k', 'mmlu_knowledge', 'mbpp']
    headers = ['GSM8K', 'MMLU-K', 'MBPP']
    log.info(f"\n{'Scales':<12}" + "".join(f" {h:>10}" for h in headers))
    log.info("-" * 44)
    for m in MODELS:
        line = f"{m:<12}"
        for b in benchmarks:
            line += f" {all_results[m][b]['accuracy']:>10.1%}"
        log.info(line)

    base = all_results['baseline']
    log.info(f"\nDelta from baseline:")
    for m in MODELS:
        if m == 'baseline':
            continue
        line = f"  {m:<10}"
        for b in benchmarks:
            d = all_results[m][b]['accuracy'] - base[b]['accuracy']
            line += f" {d:>+10.1%}"
        log.info(line)

    log.info(f"\nTotal: {(time.time() - t_start)/60:.0f} min")

    out_path = os.path.expanduser("~/freigent/apps/trucksim/checkpoints/domain_matched_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"CRASHED: {e}", exc_info=True)
    finally:
        log.info("DONE")
