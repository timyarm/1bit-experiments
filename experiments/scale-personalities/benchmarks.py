"""Bonsai 8B Scale Personalities — ACCURACY BENCHMARKS

Loads saved scale tables from volume, runs proper accuracy benchmarks:
  - GSM8K: solve math problems, check final answer
  - MMLU: multiple choice accuracy
  - HumanEval-style: code generation correctness
  - Function calling: structured output accuracy
  - IFEval-style: instruction following compliance

Compares: original scales vs each profile's scales
"""
import modal
import os

app = modal.App("bonsai8b-benchmarks")

vol = modal.Volume.from_name("scale-personalities-8b")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "accelerate", "numpy",
                 "huggingface_hub", "safetensors", "datasets")
)


@app.function(image=image, gpu="T4", timeout=7200,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/checkpoints": vol})
def benchmark(table_name: str):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import random, time, math, gc, json, re
    from math import ceil
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset

    DEVICE = "cuda"
    GROUP_SIZE = 128
    random.seed(42)
    torch.manual_seed(42)

    print("=" * 70)
    print("BONSAI 8B: ACCURACY BENCHMARKS ON SCALE PERSONALITIES")
    print("=" * 70, flush=True)

    hf_token = (os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                or None)
    os.environ["HF_TOKEN"] = hf_token

    # ─── NativeBitLinear (same as training script) ───
    class NativeBitLinear(nn.Module):
        def __init__(self, in_features, out_features, bias=False, group_size=128):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.group_size = group_size
            n_ints = ceil(in_features / 32)
            n_groups = ceil(in_features / group_size)
            self.register_buffer('packed_signs', torch.zeros(out_features, n_ints, dtype=torch.int32))
            self.group_scales = nn.Parameter(torch.zeros(out_features, n_groups, dtype=torch.float16))
            self.bias = None

        def forward(self, x):
            if not hasattr(self, '_bit_shifts'):
                self._bit_shifts = torch.arange(32, device=x.device, dtype=torch.int32)
            shifts = self._bit_shifts.to(x.device)
            unpacked = ((self.packed_signs.unsqueeze(2) >> shifts) & 1)
            signs = (unpacked.reshape(self.out_features, -1)[:, :self.in_features].half() * 2 - 1)
            se = self.group_scales.unsqueeze(2).expand(-1, -1, self.group_size)
            sf = se.reshape(self.out_features, -1)[:, :self.in_features]
            w = signs * sf
            return F.linear(x, w, self.bias)

        @staticmethod
        def from_weight(weight_tensor, group_size=128):
            out_f, in_f = weight_tensor.shape
            layer = NativeBitLinear(in_f, out_f, bias=False, group_size=group_size)
            bits = (weight_tensor > 0).to(torch.int32)
            pad = (32 - in_f % 32) % 32
            if pad > 0: bits = F.pad(bits, (0, pad))
            n_ints = bits.shape[1] // 32
            bits = bits.view(out_f, n_ints, 32)
            packed = torch.zeros(out_f, n_ints, dtype=torch.int32)
            for bit in range(32): packed |= bits[:, :, bit] << bit
            layer.packed_signs.copy_(packed)
            pad_s = (group_size - in_f % group_size) % group_size
            w_abs = F.pad(weight_tensor.abs(), (0, pad_s)) if pad_s > 0 else weight_tensor.abs()
            n_groups = w_abs.shape[1] // group_size
            layer.group_scales.data.copy_(w_abs.view(out_f, n_groups, group_size).mean(dim=2).half())
            return layer

    # ─── Load Model ───
    print("\n[1/4] Loading Bonsai 8B...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("prism-ml/Bonsai-8B-unpacked", token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Critical for batched generation on decoder-only
    model = AutoModelForCausalLM.from_pretrained(
        "prism-ml/Bonsai-8B-unpacked", dtype=torch.float16, device_map="cpu",
        low_cpu_mem_usage=True, token=hf_token
    )
    print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

    # Replace with NativeBitLinear
    print("  Replacing linears...", flush=True)
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear): continue
        if "layers." not in name or module.weight.dim() != 2: continue
        parts = name.split('.')
        parent = model
        for p in parts[:-1]: parent = getattr(parent, p)
        native = NativeBitLinear.from_weight(module.weight.data.float(), GROUP_SIZE)
        setattr(parent, parts[-1], native)
        replaced += 1
        del module
        if replaced % 50 == 0: gc.collect()
    gc.collect()
    model = model.to(DEVICE)
    model.eval()
    torch.cuda.empty_cache()
    print(f"  {replaced} linears replaced, VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)

    # ─── Load Saved Scale Tables ───
    print("\n[2/4] Loading saved scale tables...", flush=True)
    checkpoint = torch.load("/checkpoints/scale_personalities_8profiles.pt", map_location="cpu")
    original_scales = checkpoint['original_scales']
    profile_scales = checkpoint['profile_scales']
    profiles = checkpoint['metadata']['profiles']
    print(f"  Loaded {len(profiles)} profiles: {profiles}", flush=True)

    def apply_scales(scale_table):
        for lin_name, scales in scale_table.items():
            parts = lin_name.split('.')
            module = model
            for p in parts: module = getattr(module, p)
            if hasattr(module, 'group_scales'):
                module.group_scales.data.copy_(scales.to(DEVICE))

    def generate(prompt, max_tokens=256):
        model.eval()
        inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_tokens, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)

    def generate_batch(prompts, max_tokens=256):
        """Batch generation — process multiple prompts at once."""
        model.eval()
        inputs = tokenizer(prompts, return_tensors="pt", truncation=True,
                          max_length=512, padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
        results = []
        for i, out in enumerate(outputs):
            input_len = inputs["input_ids"][i].ne(tokenizer.pad_token_id).sum()
            text = tokenizer.decode(out[input_len:], skip_special_tokens=True)
            results.append(text)
        return results

    # ─── Benchmark Functions ───

    BATCH_SIZE = 8

    def bench_gsm8k(n=25):
        """GSM8K: extract final number, check if correct. BATCHED."""
        print(f"\n  GSM8K ({n} problems, batch={BATCH_SIZE})...", flush=True)
        ds = load_dataset("openai/gsm8k", "main", split="test")
        correct = 0
        total = 0

        # Collect all prompts and ground truths
        prompts = []
        gts = []
        for i, ex in enumerate(ds):
            if i >= n: break
            gt_match = re.search(r'####\s*([\-\d,]+)', ex["answer"])
            if not gt_match: continue
            gt = gt_match.group(1).replace(",", "").strip()
            prompts.append(f"Solve step by step.\nQ: {ex['question']}\nA:")
            gts.append(gt)

        # Process in batches
        for b in range(0, len(prompts), BATCH_SIZE):
            batch_prompts = prompts[b:b+BATCH_SIZE]
            batch_gts = gts[b:b+BATCH_SIZE]
            try:
                responses = generate_batch(batch_prompts, max_tokens=200)
            except RuntimeError:
                # OOM fallback to single
                torch.cuda.empty_cache()
                responses = [generate(p, max_tokens=200) for p in batch_prompts]

            for response, gt in zip(responses, batch_gts):
                numbers = re.findall(r'[\-\d,]+', response.replace(",", ""))
                pred = numbers[-1] if numbers else ""
                if pred == gt:
                    correct += 1
                total += 1
                if total <= 3:
                    print(f"    GT: {gt} | Pred: {pred} | {'✓' if pred == gt else '✗'}")
            torch.cuda.empty_cache()

        acc = correct / max(total, 1)
        print(f"  GSM8K: {correct}/{total} = {acc:.1%}", flush=True)
        return {"correct": correct, "total": total, "accuracy": acc}

    def bench_mmlu(n=25):
        """MMLU: multiple choice accuracy. BATCHED."""
        print(f"\n  MMLU ({n} questions, batch={BATCH_SIZE})...", flush=True)
        ds = load_dataset("cais/mmlu", "all", split="test", streaming=True)
        correct = 0
        total = 0
        choices = ["A", "B", "C", "D"]

        # Collect prompts and answers
        prompts = []
        gt_letters = []
        for i, ex in enumerate(ds):
            if i >= n: break
            question = ex.get("question", "")
            answer_idx = ex.get("answer", -1)
            options = ex.get("choices", [])
            if not question or answer_idx < 0 or not options: continue
            opts_str = "\n".join(f"({choices[j]}) {o}" for j, o in enumerate(options[:4]))
            prompts.append(f"Answer with just the letter.\nQ: {question}\n{opts_str}\nAnswer: (")
            gt_letters.append(choices[answer_idx] if answer_idx < len(choices) else "")

        # Process in batches
        for b in range(0, len(prompts), BATCH_SIZE):
            batch_prompts = prompts[b:b+BATCH_SIZE]
            batch_gts = gt_letters[b:b+BATCH_SIZE]
            try:
                responses = generate_batch(batch_prompts, max_tokens=5)
            except RuntimeError:
                torch.cuda.empty_cache()
                responses = [generate(p, max_tokens=5) for p in batch_prompts]

            for response, gt in zip(responses, batch_gts):
                pred = response.strip()[:1].upper()
                if pred == gt:
                    correct += 1
                total += 1
            torch.cuda.empty_cache()

        acc = correct / max(total, 1)
        print(f"  MMLU: {correct}/{total} = {acc:.1%}", flush=True)
        return {"correct": correct, "total": total, "accuracy": acc}

    def bench_code(n=20):
        """MBPP: real code generation benchmark from Google."""
        print(f"\n  MBPP Code ({n} problems)...", flush=True)
        try:
            ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        except:
            ds = load_dataset("mbpp", split="test")

        correct = 0
        total = 0
        for i, ex in enumerate(ds):
            if i >= n: break
            prompt = ex.get("prompt", ex.get("text", ""))
            test_list = ex.get("test_list", [])
            if not prompt or not test_list: continue

            response = generate(f"Write a Python function.\n{prompt}\n\n```python\n", max_tokens=150)
            # Extract code block
            code = response.split("```")[0].strip()
            full_code = code

            try:
                exec_globals = {}
                exec(full_code, exec_globals)
                passed = True
                for test in test_list[:3]:
                    try:
                        exec(test, exec_globals)
                    except:
                        passed = False
                        break
                if passed:
                    correct += 1
            except:
                pass

            total += 1
            if total <= 3:
                print(f"    Problem: {prompt[:80]}...")
                print(f"    Result: {'✓' if total <= correct else '✗'}")
            torch.cuda.empty_cache()

        acc = correct / max(total, 1)
        print(f"  MBPP: {correct}/{total} = {acc:.1%}", flush=True)
        return {"correct": correct, "total": total, "accuracy": acc}

    def bench_qa(n=25):
        """TriviaQA: real factual QA benchmark."""
        print(f"\n  TriviaQA ({n} questions)...", flush=True)
        ds = load_dataset("trivia_qa", "unfiltered.nocontext", split="validation",
                          streaming=True)

        correct = 0
        total = 0
        for i, ex in enumerate(ds):
            if i >= n: break
            question = ex.get("question", "")
            answer_obj = ex.get("answer", {})
            aliases = answer_obj.get("aliases", []) + [answer_obj.get("value", "")]
            aliases = [a.lower().strip() for a in aliases if a]
            if not question or not aliases: continue

            response = generate(f"Q: {question}\nA:", max_tokens=30).lower().strip()
            hit = any(a in response for a in aliases)
            if hit: correct += 1
            total += 1
            if total <= 3:
                print(f"    Q: {question[:80]}")
                print(f"    A: {response[:60]} | Expected: {aliases[0]} | {'✓' if hit else '✗'}")
            torch.cuda.empty_cache()

        acc = correct / max(total, 1)
        print(f"  TriviaQA: {correct}/{total} = {acc:.1%}", flush=True)
        return {"correct": correct, "total": total, "accuracy": acc}

    # ─── Run All Benchmarks for this table ───
    print(f"\n[3/4] Running benchmarks for '{table_name}'...", flush=True)

    # Apply the right scales
    if table_name == "original":
        apply_scales(original_scales)
    else:
        apply_scales(profile_scales[table_name])

    benchmarks = [
        ("gsm8k", bench_gsm8k, 25),
        ("mmlu", bench_mmlu, 25),
        ("code", bench_code, 20),
        ("qa", bench_qa, 25),
    ]

    results = {"table": table_name}
    for bname, bench_fn, n in benchmarks:
        result = bench_fn(n)
        results[bname] = result

    # Print summary for this table
    print(f"\n{'='*70}")
    print(f"  {table_name}: ", end="")
    for bname, _, _ in benchmarks:
        print(f" {bname}={results[bname]['accuracy']:.1%}", end="")
    print(f"\n{'='*70}", flush=True)

    # Save individual result
    with open(f"/checkpoints/bench_{table_name}.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    vol.commit()

    return results


@app.local_entrypoint()
def main():
    import json

    tables = ["original", "reasoning", "compliance", "structured", "tool_use"]

    # Launch ALL 5 in parallel — each gets its own T4
    print(f"Launching {len(tables)} benchmark jobs in parallel...")
    results_list = list(benchmark.map(tables))

    # Combine and print summary
    print(f"\n{'='*70}")
    print("COMBINED RESULTS")
    print(f"{'='*70}")
    print(f"  {'Scales':<15} {'gsm8k':>8} {'mmlu':>8} {'code':>8} {'qa':>8}")
    print(f"  {'-'*47}")

    all_results = {}
    for r in results_list:
        tname = r["table"]
        all_results[tname] = r
        row = f"  {tname:<15}"
        for bname in ["gsm8k", "mmlu", "code", "qa"]:
            row += f" {r[bname]['accuracy']:>7.1%}"
        print(row)

    # Best per benchmark
    print(f"\n  BEST PROFILE PER BENCHMARK:")
    for bname in ["gsm8k", "mmlu", "code", "qa"]:
        best = max([r for r in results_list if r["table"] != "original"],
                   key=lambda r: r[bname]["accuracy"])
        orig = [r for r in results_list if r["table"] == "original"][0]
        delta = best[bname]["accuracy"] - orig[bname]["accuracy"]
        print(f"    {bname:>8}: {best['table']} ({best[bname]['accuracy']:.1%}, "
              f"{'+' if delta >= 0 else ''}{delta:.1%} vs original)")

    # Save combined
    with open("benchmark_results_combined.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\n  Results saved locally.", flush=True)
