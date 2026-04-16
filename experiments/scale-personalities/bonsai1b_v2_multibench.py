"""Multi-benchmark eval on patched GGUFs via llama.cpp.

Evaluates: TriviaQA, ARC-Easy, HellaSwag
On: baseline, math-v2, knowledge-v2, code-v2

Monitor: tail -f /tmp/scale_eval.log
"""

import sys
import os
import time
import json
import logging
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S",
                    handlers=[logging.FileHandler("/tmp/scale_eval.log", mode='w'),
                              logging.StreamHandler(sys.stdout)])
log = logging.getLogger("eval")

sys.path.insert(0, os.path.dirname(__file__))
from llama_fast_eval import LlamaEval

GGUF_DIR = os.path.expanduser("~/freigent/apps/trucksim/data/llm")
MODELS = {
    "baseline": f"{GGUF_DIR}/Bonsai-1.7B.gguf",
    "math": f"{GGUF_DIR}/Bonsai-1.7B-math-v2.gguf",
    "knowledge": f"{GGUF_DIR}/Bonsai-1.7B-knowledge-v2.gguf",
    "code": f"{GGUF_DIR}/Bonsai-1.7B-code-v2.gguf",
}

N_QUESTIONS = 150  # per benchmark


def eval_trivia(ev, n=N_QUESTIONS):
    """TriviaQA — factual recall."""
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "unfiltered", split="validation", streaming=True)
    correct = 0
    total = 0
    for ex in ds:
        if total >= n:
            break
        q = ex['question']
        answers = ex.get('answer', {}).get('aliases', [])
        if not answers:
            continue
        resp = ev.generate(f"Question: {q}\nAnswer:", max_tokens=30)
        if any(a.lower() in resp.lower() for a in answers):
            correct += 1
        total += 1
    return {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}


def eval_arc_easy(ev, n=N_QUESTIONS):
    """ARC-Easy — science multiple choice."""
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test", streaming=True)
    correct = 0
    total = 0
    for ex in ds:
        if total >= n:
            break
        question = ex['question']
        choices = ex['choices']
        labels = choices['label']
        texts = choices['text']
        answer_key = ex['answerKey']

        # Format as multiple choice
        options = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
        prompt = f"Question: {question}\n{options}\nAnswer:"
        resp = ev.generate(prompt, max_tokens=5)
        resp_clean = resp.strip().upper()

        # Check if the model picked the right letter
        if resp_clean and resp_clean[0] == answer_key:
            correct += 1
        total += 1
    return {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}


def eval_hellaswag(ev, n=N_QUESTIONS):
    """HellaSwag — sentence completion."""
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation", streaming=True)
    correct = 0
    total = 0
    for ex in ds:
        if total >= n:
            break
        ctx = ex['ctx']
        endings = ex['endings']
        label = int(ex['label'])

        # Ask model to pick best continuation
        options = "\n".join(f"{i+1}. {e}" for i, e in enumerate(endings))
        prompt = f"Context: {ctx}\n\nWhich continuation is most likely?\n{options}\nAnswer:"
        resp = ev.generate(prompt, max_tokens=5)

        # Parse response for number
        nums = re.findall(r'[1-4]', resp[:10])
        if nums and int(nums[0]) - 1 == label:
            correct += 1
        total += 1
    return {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}


def main():
    log.info("=" * 60)
    log.info(f"MULTI-BENCHMARK EVAL — {N_QUESTIONS} questions per benchmark")
    log.info("=" * 60)
    t_start = time.time()

    all_results = {}

    for model_name, gguf_path in MODELS.items():
        log.info(f"\n{'='*60}")
        log.info(f"MODEL: {model_name}")
        log.info(f"{'='*60}")

        ev = LlamaEval(gguf_path)
        ev.start()

        results = {}

        # TriviaQA
        log.info(f"  TriviaQA ({N_QUESTIONS}q)...")
        t0 = time.time()
        trivia = eval_trivia(ev, N_QUESTIONS)
        log.info(f"  TriviaQA: {trivia['correct']}/{trivia['total']} = {trivia['accuracy']:.1%} ({time.time()-t0:.0f}s)")
        results['trivia'] = trivia

        # ARC-Easy
        log.info(f"  ARC-Easy ({N_QUESTIONS}q)...")
        t0 = time.time()
        arc = eval_arc_easy(ev, N_QUESTIONS)
        log.info(f"  ARC-Easy: {arc['correct']}/{arc['total']} = {arc['accuracy']:.1%} ({time.time()-t0:.0f}s)")
        results['arc_easy'] = arc

        # HellaSwag
        log.info(f"  HellaSwag ({N_QUESTIONS}q)...")
        t0 = time.time()
        hella = eval_hellaswag(ev, N_QUESTIONS)
        log.info(f"  HellaSwag: {hella['correct']}/{hella['total']} = {hella['accuracy']:.1%} ({time.time()-t0:.0f}s)")
        results['hellaswag'] = hella

        all_results[model_name] = results
        ev.stop()
        time.sleep(2)

    # ─── Summary ───
    log.info(f"\n{'='*60}")
    log.info("RESULTS SUMMARY")
    log.info(f"{'='*60}")

    benchmarks = ['trivia', 'arc_easy', 'hellaswag']
    log.info(f"\n{'Scales':<12}" + "".join(f" {'TriviaQA':>10} {'ARC-Easy':>10} {'HellaSwag':>10}"))
    log.info("-" * 44)
    for model_name in MODELS:
        line = f"{model_name:<12}"
        for b in benchmarks:
            acc = all_results[model_name][b]['accuracy']
            line += f" {acc:>10.1%}"
        log.info(line)

    # Delta from baseline
    log.info(f"\nDelta from baseline:")
    base = all_results['baseline']
    for model_name in MODELS:
        if model_name == 'baseline':
            continue
        line = f"  {model_name:<10}"
        for b in benchmarks:
            delta = all_results[model_name][b]['accuracy'] - base[b]['accuracy']
            line += f" {delta:>+10.1%}"
        log.info(line)

    elapsed = time.time() - t_start
    log.info(f"\nTotal: {elapsed/60:.0f} min")

    # Save
    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("Saved to checkpoints/benchmark_results.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"CRASHED: {e}", exc_info=True)
    finally:
        log.info("DONE")
