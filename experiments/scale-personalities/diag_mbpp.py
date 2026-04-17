"""Diagnostic: print raw MBPP outputs so we can tell if it's extraction or generation."""
import sys
import os
import subprocess
import tempfile
import re

sys.path.insert(0, os.path.dirname(__file__))
from llama_fast_eval import LlamaEval

MODEL_PATH = os.path.expanduser("~/freigent/apps/trucksim/data/llm/Bonsai-1.7B.gguf")
N = 10


def extract_fixed(text):
    """V2: prompt opens a ```python fence, so response starts with code and
    closes with ```. The CODE is everything BEFORE the first ``` in response."""
    if not isinstance(text, str):
        return ""
    # Everything up to the first code-fence terminator
    if "```" in text:
        code = text.split("```")[0]
    else:
        # No fence at all — take up to first blank line
        code = text.split("\n\n")[0]
    return code.strip()


def run_tests(code, tests, timeout=3):
    if not code or not tests:
        return False, "empty"
    full = code + "\n\n" + "\n".join(tests)
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full)
            tmp = f.name
        try:
            r = subprocess.run(["python3", tmp], capture_output=True, timeout=timeout)
            if r.returncode == 0:
                return True, "pass"
            return False, (r.stderr.decode() or "")[:200]
        finally:
            os.unlink(tmp)
    except Exception as e:
        return False, str(e)[:200]


def main():
    from datasets import load_dataset
    ds = load_dataset("mbpp", split="test", streaming=True)

    ev = LlamaEval(MODEL_PATH, n_ctx=1024)
    ev.start()

    total = 0
    passes = 0
    for ex in ds:
        if total >= N:
            break
        text = ex['text']
        tests = ex['test_list']
        prompt = (
            f"Write a Python function that solves this task.\n"
            f"Task: {text}\n"
            f"Example test:\n{tests[0] if tests else ''}\n"
            f"```python\n"
        )
        resp = ev.generate(prompt, max_tokens=300)
        code = extract_fixed(resp)
        ok, err = run_tests(code, tests)

        print(f"\n# Example {total + 1}: {'PASS' if ok else 'FAIL'}")
        print(f"  task: {text[:80]}")
        print(f"  code: {code[:200].replace(chr(10), ' | ')}")
        if not ok:
            print(f"  err:  {err[:120]}")

        if ok:
            passes += 1
        total += 1

    ev.stop()
    print(f"\n{'='*60}")
    print(f"FIXED EXTRACTOR: {passes}/{total} = {passes/max(total,1):.1%}")


if __name__ == "__main__":
    main()
