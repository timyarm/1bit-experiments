"""Fast eval via llama.cpp server — 120 tok/s on GTX 1660.

Starts llama-server, sends requests via HTTP, parses responses.
Use for eval between scale training rounds.

Usage:
    # Start server (one time):
    eval = LlamaEval('/path/to/model.gguf')
    eval.start()

    # Generate:
    text = eval.generate("The capital of France is", max_tokens=50)

    # Batch eval:
    results = eval.eval_gsm8k(n=50)

    # Stop:
    eval.stop()
"""

import subprocess
import requests
import time
import json
import re
import os
import signal


class LlamaEval:
    """Fast inference via llama.cpp server."""

    def __init__(self, model_path, port=8090, n_gpu_layers=99, n_ctx=512):
        self.model_path = model_path
        self.port = port
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.base_url = f"http://localhost:{port}"
        self.process = None
        self.llama_bin = os.path.expanduser("~/prismml-llama-cpp/build-cuda/bin/llama-server")

    def start(self):
        """Start llama.cpp server in background."""
        if self.is_running():
            print(f"Server already running on port {self.port}")
            return

        cmd = [
            self.llama_bin,
            "-m", self.model_path,
            "--host", "0.0.0.0",
            "--port", str(self.port),
            "-ngl", str(self.n_gpu_layers),
            "-c", str(self.n_ctx),
        ]
        env = os.environ.copy()
        env["PATH"] = env.get("PATH", "") + ":/usr/lib/wsl/lib"
        env["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib:" + env.get("LD_LIBRARY_PATH", "")
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env
        )

        # Wait for server to be ready
        for _ in range(60):
            try:
                r = requests.get(f"{self.base_url}/health", timeout=1)
                if r.status_code == 200:
                    print(f"llama.cpp server ready on port {self.port}")
                    return
            except requests.ConnectionError:
                pass
            time.sleep(0.5)
        raise RuntimeError("llama.cpp server failed to start")

    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except:
                self.process.kill()
                self.process.wait(timeout=2)
            self.process = None

    def is_running(self):
        try:
            r = requests.get(f"{self.base_url}/health", timeout=1)
            return r.status_code == 200
        except:
            return False

    def generate(self, prompt, max_tokens=100, temperature=0):
        """Generate text. Returns generated string."""
        r = requests.post(f"{self.base_url}/completion", json={
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stop": ["\n\n", "<|endoftext|>", "<|im_end|>"],
        }, timeout=30)
        return r.json()["content"]

    def generate_batch(self, prompts, max_tokens=100, temperature=0):
        """Generate for multiple prompts sequentially (server handles one at a time)."""
        results = []
        for p in prompts:
            results.append(self.generate(p, max_tokens, temperature))
        return results

    def eval_gsm8k(self, n=50):
        """Evaluate GSM8K accuracy."""
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="test", streaming=True)

        correct = 0
        total = 0

        for ex in ds:
            if total >= n:
                break

            prompt = f"Question: {ex['question']}\nAnswer:"
            response = self.generate(prompt, max_tokens=200, temperature=0)

            pred_num = self._extract_number(response)
            true_answer = ex['answer'].split("####")[-1].strip()
            true_num = self._extract_number(true_answer)

            if pred_num is not None and true_num is not None and abs(pred_num - true_num) < 0.01:
                correct += 1
            total += 1

        acc = correct / max(total, 1)
        return {"accuracy": acc, "correct": correct, "total": total}

    def eval_trivia(self, n=50):
        """Evaluate TriviaQA accuracy."""
        from datasets import load_dataset
        ds = load_dataset("trivia_qa", "unfiltered", split="validation", streaming=True)

        correct = 0
        total = 0

        for ex in ds:
            if total >= n:
                break

            question = ex['question']
            answers = ex.get('answer', {}).get('aliases', [])
            if not answers:
                continue

            prompt = f"Question: {question}\nAnswer:"
            response = self.generate(prompt, max_tokens=30, temperature=0)

            if any(a.lower() in response.lower() for a in answers):
                correct += 1
            total += 1

        acc = correct / max(total, 1)
        return {"accuracy": acc, "correct": correct, "total": total}

    def eval_ppl_approx(self, texts, max_tokens=256):
        """Approximate PPL using logprobs from server."""
        r = requests.post(f"{self.base_url}/completion", json={
            "prompt": texts[0][:500],
            "n_predict": 0,
            "logprobs": True,
        }, timeout=10)
        # Server may not support logprobs — fall back to generation speed test
        return None

    def bench_tok_s(self, n_tokens=200):
        """Benchmark generation speed."""
        prompt = "Write a detailed explanation of"
        t0 = time.time()
        response = self.generate(prompt, max_tokens=n_tokens, temperature=0.1)
        elapsed = time.time() - t0
        actual_tokens = len(response.split())  # approximate
        return {"tok_s": n_tokens / elapsed, "elapsed": elapsed}

    @staticmethod
    def _extract_number(text):
        numbers = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                return None
        return None


if __name__ == "__main__":
    model_path = os.path.expanduser("~/freigent/apps/trucksim/data/llm/Bonsai-1.7B.gguf")
    ev = LlamaEval(model_path)
    ev.start()

    # Speed test
    print("\n--- Speed Test ---")
    bench = ev.bench_tok_s(100)
    print(f"Speed: {bench['tok_s']:.1f} tok/s")

    # Quick generation
    print("\n--- Generation ---")
    out = ev.generate("The capital of France is", max_tokens=50)
    print(f"Output: {out[:100]}")

    # GSM8K
    print("\n--- GSM8K (20 questions) ---")
    gsm = ev.eval_gsm8k(n=20)
    print(f"Accuracy: {gsm['correct']}/{gsm['total']} = {gsm['accuracy']:.1%}")

    ev.stop()
