"""MBPP-only eval with fixed code extraction. All 4 profiles, n=100."""
import sys
import os
import time
import json
import logging
import subprocess
import tempfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler("/tmp/mbpp_eval.log", mode='w'),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("mbpp")

sys.path.insert(0, os.path.dirname(__file__))
from llama_fast_eval import LlamaEval

GGUF_DIR = os.path.expanduser("~/freigent/apps/trucksim/data/llm")
MODELS = {
    "baseline":  f"{GGUF_DIR}/Bonsai-1.7B.gguf",
    "math":      f"{GGUF_DIR}/Bonsai-1.7B-math-v2.gguf",
    "knowledge": f"{GGUF_DIR}/Bonsai-1.7B-knowledge-v2.gguf",
    "code":      f"{GGUF_DIR}/Bonsai-1.7B-code-v2.gguf",
}
N = 100


def extract_code(text):
    """Prompt opens ```python fence; code is everything BEFORE first closing ```."""
    if not isinstance(text, str):
        return ""
    if "```" in text:
        return text.split("```")[0].strip()
    return text.split("\n\n")[0].strip()


def run_tests(code, tests, timeout=3):
    if not code or not tests:
        return False
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code + "\n\n" + "\n".join(tests))
            tmp = f.name
        try:
            r = subprocess.run(["python3", tmp], capture_output=True, timeout=timeout)
            return r.returncode == 0
        finally:
            os.unlink(tmp)
    except Exception:
        return False


def safe_generate(ev, prompt, max_tokens=300):
    try:
        r = ev.generate(prompt, max_tokens=max_tokens)
        return r if isinstance(r, str) else ""
    except Exception:
        return ""


def eval_mbpp(ev, n=N):
    from datasets import load_dataset
    ds = load_dataset("mbpp", split="test", streaming=True)
    correct = total = 0
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
        code = extract_code(resp)
        if run_tests(code, tests):
            correct += 1
        total += 1
    return {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}


def main():
    t0 = time.time()
    log.info(f"MBPP EVAL — fixed extractor, n={N} per model")
    all_results = {}
    for name, path in MODELS.items():
        log.info(f"\n=== {name} ===")
        ev = LlamaEval(path, n_ctx=1024)
        ev.start()
        tm = time.time()
        r = eval_mbpp(ev, N)
        log.info(f"  MBPP: {r['correct']}/{r['total']} = {r['accuracy']:.1%} ({time.time()-tm:.0f}s)")
        all_results[name] = r
        ev.stop()
        time.sleep(2)

    log.info("\n=== SUMMARY ===")
    for name, r in all_results.items():
        log.info(f"  {name:<10} MBPP: {r['accuracy']:.1%} ({r['correct']}/{r['total']})")

    # Deltas vs baseline
    base = all_results['baseline']['accuracy']
    log.info("\nDeltas from baseline:")
    for name, r in all_results.items():
        if name == 'baseline':
            continue
        log.info(f"  {name:<10} {r['accuracy'] - base:+.1%}")

    out_path = os.path.expanduser("~/freigent/apps/trucksim/checkpoints/mbpp_fixed_results.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"\nSaved: {out_path}")
    log.info(f"Total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"CRASHED: {e}", exc_info=True)
    finally:
        log.info("DONE")
