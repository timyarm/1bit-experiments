"""Bonsai 8B Comprehensive Scale Personalities — 8 Agentic Profiles + Benchmarks

Train 8 agentic thinking modes on Bonsai 8B native 1-bit (3.46GB VRAM).
Save scale tables to Modal volume. Run hard benchmarks before/after.

Profiles:
  1. Tool Use        — function calling, structured API interaction
  2. Reasoning       — multi-step chain-of-thought decomposition
  3. Structured      — JSON, SQL, code, schemas
  4. Retrieval       — context-grounded answers from documents
  5. Verification    — self-correction, error catching
  6. Planning        — task decomposition, sequencing
  7. Compliance      — precise instruction following
  8. Creative        — synthesis, brainstorming, novel combinations
"""
import modal
import os

app = modal.App("bonsai8b-comprehensive")

vol = modal.Volume.from_name("scale-personalities-8b", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "accelerate", "numpy",
                 "huggingface_hub", "safetensors", "datasets")
)


@app.function(image=image, gpu="T4", timeout=21600,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/checkpoints": vol})
def train_and_eval():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import random, time, math, gc, json
    from math import ceil
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset

    DEVICE = "cuda"
    GROUP_SIZE = 128
    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("BONSAI 8B: 8-PROFILE SCALE PERSONALITY TRAINING + BENCHMARKS")
    print("=" * 70, flush=True)

    hf_token = (os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                or os.environ.get("HUGGINGFACE_TOKEN")
                or None)
    # Set for huggingface_hub to pick up automatically
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    print(f"  HF token set ({hf_token[:8]}...)", flush=True)

    # ─── NativeBitLinear ───
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
            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
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
    print("\n[1/5] Loading Bonsai 8B...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("prism-ml/Bonsai-8B-unpacked", token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "prism-ml/Bonsai-8B-unpacked", dtype=torch.float16, device_map="cpu",
        low_cpu_mem_usage=True, token=hf_token
    )
    print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

    # ─── Replace with NativeBitLinear ───
    print("  Replacing linears...", flush=True)
    replaced = 0
    original_scales = {}
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear): continue
        if "layers." not in name: continue
        if module.weight.dim() != 2: continue
        parts = name.split('.')
        parent = model
        for p in parts[:-1]: parent = getattr(parent, p)
        native = NativeBitLinear.from_weight(module.weight.data.float(), GROUP_SIZE)
        original_scales[name] = native.group_scales.data.clone()
        setattr(parent, parts[-1], native)
        replaced += 1
        del module
        if replaced % 50 == 0: gc.collect()
    gc.collect()
    print(f"  Replaced {replaced} linears", flush=True)

    model = model.to(DEVICE)
    model.eval()
    torch.cuda.empty_cache()
    print(f"  VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)

    # ─── Load Training Datasets ───
    print("\n[2/5] Loading agentic training data...", flush=True)

    def load_profile_data(profile_name, max_examples=250):
        """Load and format training data for each profile."""
        texts = []
        try:
            if profile_name == "tool_use":
                # Glaive function calling v2: 113K examples, Apache 2.0, NOT gated
                ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train",
                                  streaming=True)
                for i, ex in enumerate(ds):
                    if i >= max_examples: break
                    # Format: system prompt with tools + user query + assistant function call
                    system = ex.get("system", "")
                    chat = ex.get("chat", "")
                    if system and chat:
                        text = f"{system[:250]}\n{chat[:350]}"
                    elif chat:
                        text = str(chat)[:600]
                    else:
                        continue
                    if len(text) > 50: texts.append(text[:600])

            elif profile_name == "reasoning":
                # NuminaMath-CoT: problem + step-by-step solution
                ds = load_dataset("AI-MO/NuminaMath-CoT", split="train",
                                  streaming=True)
                for i, ex in enumerate(ds):
                    if i >= max_examples: break
                    problem = ex.get("problem", ex.get("question", ""))
                    solution = ex.get("solution", ex.get("text", ""))
                    if problem and solution:
                        text = f"Problem: {str(problem)[:200]}\nSolution: {str(solution)[:400]}"
                    elif solution:
                        text = str(solution)
                    else:
                        continue
                    if len(text) > 50: texts.append(text[:600])

            elif profile_name == "structured":
                # Text-to-SQL: schema context + natural language + SQL output
                ds = load_dataset("gretelai/synthetic_text_to_sql", split="train",
                                  streaming=True)
                for i, ex in enumerate(ds):
                    if i >= max_examples: break
                    sql = ex.get("sql", ex.get("query", ex.get("sql_query", "")))
                    context = ex.get("sql_context", ex.get("context", ex.get("create_statement", "")))
                    prompt = ex.get("sql_prompt", ex.get("input", ex.get("question", "")))
                    text = f"Schema: {str(context)[:200]}\nQuestion: {str(prompt)[:150]}\nSQL: {str(sql)[:250]}"
                    if len(text) > 50: texts.append(text[:600])

            elif profile_name == "retrieval":
                # Natural Questions style: passage + question + answer grounded in text
                # Build from wiki paragraphs as retrieval passages
                ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
                wiki_paragraphs = [t["text"] for t in ds if len(t["text"].strip()) > 200]
                for para in wiki_paragraphs[:max_examples]:
                    # Format as retrieval: passage is the context, model learns to ground in it
                    sentences = para.split('. ')
                    if len(sentences) > 3:
                        context = '. '.join(sentences[:3]) + '.'
                        answer_part = '. '.join(sentences[3:])
                        text = f"Context: {context[:300]}\nBased on the above: {answer_part[:300]}"
                    else:
                        text = f"Context: {para[:600]}"
                    texts.append(text[:600])

            elif profile_name == "verification":
                # UltraFeedback: instruction + completions with quality ratings
                ds = load_dataset("openbmb/UltraFeedback", split="train",
                                  streaming=True)
                for i, ex in enumerate(ds):
                    if i >= max_examples: break
                    instruction = ex.get("instruction", "")
                    completions = ex.get("completions", [])
                    if instruction and completions and len(completions) >= 2:
                        # Show the model a good vs bad response pattern
                        best = completions[0]
                        response = best.get("response", "") if isinstance(best, dict) else str(best)
                        text = f"Task: {str(instruction)[:200]}\nResponse: {str(response)[:200]}\nVerify: Check this response for accuracy and completeness."
                    elif instruction:
                        text = f"Verify the following: {str(instruction)[:400]}"
                    else:
                        continue
                    if len(text) > 50: texts.append(text[:600])

            elif profile_name == "planning":
                # GSM8K: multi-step decomposition with step-by-step execution
                ds = load_dataset("openai/gsm8k", "main", split="train")
                for ex in ds:
                    if len(texts) >= max_examples: break
                    text = f"Plan the solution step by step.\nQ: {ex['question']}\nStep-by-step:\n{ex['answer']}"
                    texts.append(text[:600])

            elif profile_name == "compliance":
                # MMLU-Pro: precise format, must follow exact structure
                ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test",
                                  streaming=True)
                for i, ex in enumerate(ds):
                    if i >= max_examples: break
                    q = ex.get("question", "")
                    opts = ex.get("options", [])
                    ans = ex.get("answer", "")
                    # Format with strict structure the model must follow
                    opts_str = "\n".join(f"  ({chr(65+j)}) {o}" for j, o in enumerate(opts[:6])) if isinstance(opts, list) else str(opts)
                    text = f"Follow the format exactly.\nQuestion: {str(q)[:200]}\nOptions:\n{opts_str[:200]}\nCorrect Answer: {str(ans)}"
                    if len(text) > 50: texts.append(text[:600])

            elif profile_name == "creative":
                # UltraChat: open-ended multi-turn creative conversations
                ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft",
                                  streaming=True)
                for i, ex in enumerate(ds):
                    if i >= max_examples: break
                    msgs = ex.get("messages", [])
                    if msgs and len(msgs) >= 2:
                        # Combine user prompt + assistant creative response
                        parts = []
                        for m in msgs[:4]:
                            role = m.get("role", "")
                            content = m.get("content", "")
                            if role and content:
                                parts.append(f"{role}: {content[:150]}")
                        text = "\n".join(parts)
                    elif msgs:
                        text = str(msgs[0].get("content", ""))
                    else:
                        continue
                    if len(text) > 50: texts.append(text[:600])

        except Exception as e:
            print(f"    WARN: Failed loading {profile_name}: {e}", flush=True)

        # Fallback: if dataset failed, use wiki
        if len(texts) < 20:
            print(f"    Fallback to wiki for {profile_name} ({len(texts)} loaded)", flush=True)
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            for ex in ds:
                if len(texts) >= max_examples: break
                text = ex["text"]
                if isinstance(text, str) and len(text.strip()) > 100:
                    texts.append(text[:512])

        print(f"    {profile_name}: {len(texts)} examples loaded", flush=True)
        return texts

    PROFILES = ["tool_use", "reasoning", "structured", "retrieval",
                "verification", "planning", "compliance", "creative"]

    profile_data = {}
    for pname in PROFILES:
        profile_data[pname] = load_profile_data(pname, max_examples=250)

    # ─── Benchmark: PPL + Generation ───
    def compute_ppl(texts, n=30):
        model.eval()
        tl, tt = 0, 0
        for text in texts[:n]:
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).input_ids.to(DEVICE)
            if ids.shape[1] < 10: continue
            with torch.no_grad():
                out = model(ids, labels=ids)
                if out.loss is not None and not torch.isnan(out.loss):
                    tl += out.loss.item() * ids.shape[1]
                    tt += ids.shape[1]
        return math.exp(tl / max(tt, 1))

    def apply_scales(scale_table):
        for lin_name, scales in scale_table.items():
            parts = lin_name.split('.')
            module = model
            for p in parts: module = getattr(module, p)
            if hasattr(module, 'group_scales'):
                module.group_scales.data.copy_(scales.to(DEVICE))

    def gen(prompt, max_tokens=80):
        model.eval()
        inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_tokens, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)[:200]

    # ─── Baseline PPL ───
    print("\n[3/5] Baseline PPL across all domains...", flush=True)
    baseline_ppl = {}
    for pname in PROFILES:
        ppl = compute_ppl(profile_data[pname])
        baseline_ppl[pname] = ppl
        print(f"  {pname:>15}: {ppl:.2f}", flush=True)

    # ─── Train All Profiles ───
    print("\n[4/5] Training 8 profiles...", flush=True)

    def train_personality(name, texts, epochs=15, lr=0.01, examples_per_epoch=50,
                          max_len=128, accum_steps=4):
        print(f"\n  ── Training {name} ({epochs} ep, {examples_per_epoch} ex/ep, "
              f"{max_len} tok, accum={accum_steps}) ──", flush=True)
        t_start = time.time()

        # Reset scales
        for lin_name, orig in original_scales.items():
            parts = lin_name.split('.')
            module = model
            for p in parts: module = getattr(module, p)
            if hasattr(module, 'group_scales'):
                module.group_scales.data.copy_(orig.to(DEVICE))

        # Collect scale params
        scale_params = []
        for lin_name in original_scales:
            parts = lin_name.split('.')
            module = model
            for p in parts: module = getattr(module, p)
            if hasattr(module, 'group_scales'):
                scale_params.append(module.group_scales)

        optimizer = torch.optim.SGD(scale_params, lr=lr)

        # Cosine LR decay (from Trinity — decay to lr/10 by end of training)
        total_steps = epochs * (examples_per_epoch // accum_steps)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=lr / 10
        )

        model.gradient_checkpointing_enable()

        best_loss = float('inf')
        for epoch in range(epochs):
            total_loss, n = 0, 0
            optimizer.zero_grad()
            random.shuffle(texts)
            for step, text in enumerate(texts[:examples_per_epoch]):
                ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(DEVICE)
                if ids.shape[1] < 5: continue
                torch.cuda.empty_cache()
                model.train()
                try:
                    out = model(ids, labels=ids)
                    if out.loss is None or torch.isnan(out.loss): continue
                    # Scale loss by accumulation steps for correct averaging
                    loss = out.loss / accum_steps
                    loss.backward()
                    total_loss += out.loss.item()
                    n += 1

                    # Update weights every accum_steps
                    if (step + 1) % accum_steps == 0 or step == examples_per_epoch - 1:
                        torch.nn.utils.clip_grad_norm_(scale_params, 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        with torch.no_grad():
                            for p in scale_params:
                                p.data.clamp_(min=1e-6)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        continue
                    raise
                del ids, out
                torch.cuda.empty_cache()

            avg_loss = total_loss / max(n, 1)
            if avg_loss < best_loss:
                best_loss = avg_loss
            if epoch % 3 == 0 or epoch == epochs - 1:
                cur_lr = scheduler.get_last_lr()[0]
                print(f"    Epoch {epoch+1}/{epochs}: CE={avg_loss:.4f} (best={best_loss:.4f}, "
                      f"lr={cur_lr:.5f}, {n} ex)", flush=True)

        # Save trained scales
        result = {}
        for lin_name in original_scales:
            parts = lin_name.split('.')
            module = model
            for p in parts: module = getattr(module, p)
            if hasattr(module, 'group_scales'):
                result[lin_name] = module.group_scales.detach().cpu().clone()
        del optimizer
        torch.cuda.empty_cache()
        print(f"  ── {name} done ({time.time()-t_start:.0f}s) ──", flush=True)
        return result

    # Resume from checkpoint if available
    profile_scales = {}
    resume_checkpoint = None
    try:
        resume_checkpoint = torch.load("/checkpoints/checkpoint_after_creative.pt", map_location="cpu")
    except:
        for pname in reversed(PROFILES):
            try:
                resume_checkpoint = torch.load(f"/checkpoints/checkpoint_after_{pname}.pt", map_location="cpu")
                break
            except:
                continue

    completed = []
    if resume_checkpoint and 'profiles_completed' in resume_checkpoint:
        completed = resume_checkpoint['profiles_completed']
        profile_scales = resume_checkpoint.get('profile_scales', {})
        # Convert tensors back
        for p in profile_scales:
            profile_scales[p] = {k: v for k, v in profile_scales[p].items()}
        print(f"  Resuming from checkpoint. Already completed: {completed}", flush=True)

    for i, pname in enumerate(PROFILES):
        if pname in completed:
            print(f"\n  ── {pname.upper()} already done (from checkpoint) ──", flush=True)
            continue
        profile_scales[pname] = train_personality(pname.upper(), profile_data[pname])

        # Intermediate checkpoint after each profile
        checkpoint = {
            'original_scales': {k: v.cpu() for k, v in original_scales.items()},
            'profile_scales': {p: {k: v.cpu() for k, v in s.items()}
                              for p, s in profile_scales.items()},
            'profiles_completed': list(profile_scales.keys()),
            'metadata': {
                'model': 'Bonsai-8B-unpacked',
                'group_size': GROUP_SIZE,
                'checkpoint_after': pname,
            },
        }
        torch.save(checkpoint, f"/checkpoints/checkpoint_after_{pname}.pt")
        vol.commit()
        print(f"  Checkpoint saved ({i+1}/{len(PROFILES)} profiles done)", flush=True)

    # ─── Full Evaluation Matrix ───
    print(f"\n{'='*70}")
    print("[5/5] FULL EVALUATION MATRIX (8×8)")
    print(f"{'='*70}", flush=True)

    all_tables = {"original": original_scales}
    all_tables.update(profile_scales)

    # PPL matrix: each profile's scales evaluated on each domain's data
    results = {}
    header = f"  {'Scales':<15}"
    for pname in PROFILES:
        header += f" {pname[:8]:>9}"
    print(header)
    print(f"  {'-'*15}" + f" {'-'*9}" * len(PROFILES))

    for tname, table in all_tables.items():
        apply_scales(table)
        results[tname] = {}
        row = f"  {tname:<15}"
        for pname in PROFILES:
            ppl = compute_ppl(profile_data[pname])
            results[tname][pname] = ppl
            row += f" {ppl:>9.2f}"
        print(row, flush=True)

    # ─── Diagonal Dominance Check ───
    print(f"\n  DIAGONAL DOMINANCE CHECK:")
    wins = 0
    for pname in PROFILES:
        own_ppl = results[pname][pname]
        others = [results[other][pname] for other in PROFILES if other != pname]
        is_best = all(own_ppl < o for o in others)
        beats_original = own_ppl < results["original"][pname]
        status = "BEST" if is_best else ("IMPROVED" if beats_original else "no")
        print(f"    {pname:>15} on own domain: {own_ppl:.2f} (original: {results['original'][pname]:.2f}) → {status}")
        if is_best: wins += 1

    print(f"\n  DIAGONAL WINS: {wins}/{len(PROFILES)}")
    if wins >= 6:
        print(f"  STRONG DIAGONAL DOMINANCE!")
    elif wins >= 4:
        print(f"  MODERATE DIFFERENTIATION")
    else:
        print(f"  WEAK DIFFERENTIATION — needs more data/epochs")

    # ─── Improvement Summary ───
    print(f"\n  IMPROVEMENT vs ORIGINAL:")
    for pname in PROFILES:
        orig_ppl = results["original"][pname]
        own_ppl = results[pname][pname]
        pct = (orig_ppl - own_ppl) / orig_ppl * 100
        arrow = "↑" if pct > 0 else "↓"
        print(f"    {pname:>15}: {orig_ppl:.2f} → {own_ppl:.2f} ({arrow} {abs(pct):.1f}%)")

    # ─── Generation Quality ───
    print(f"\n{'='*70}")
    print("GENERATION QUALITY (selected profiles)")
    print(f"{'='*70}", flush=True)

    gen_prompts = [
        ("Call the book_load API with load_id L-4521 and rate 2850", "tool_use"),
        ("What is 15 * 27 + 33? Think step by step.", "reasoning"),
        ("Write a SQL query to find the top 5 customers by revenue", "structured"),
        ("Based on the context, what is the capital of France?", "retrieval"),
        ("Write a creative analogy for how neural networks learn", "creative"),
        ("Break down the task: deploy a web application to production", "planning"),
    ]

    for tname in ["original", "tool_use", "reasoning", "structured", "creative"]:
        table = all_tables[tname]
        apply_scales(table)
        print(f"\n  --- {tname} scales ---")
        for prompt, ptype in gen_prompts[:3]:
            answer = gen(prompt, max_tokens=60)
            print(f"    [{ptype}] {prompt[:60]}")
            print(f"           → {answer[:120]}")

    # ─── Save Scale Tables ───
    print(f"\n{'='*70}")
    print("SAVING SCALE TABLES TO VOLUME")
    print(f"{'='*70}", flush=True)

    save_data = {
        'original_scales': {k: v.cpu() for k, v in original_scales.items()},
        'profile_scales': {pname: {k: v.cpu() for k, v in scales.items()}
                          for pname, scales in profile_scales.items()},
        'metadata': {
            'model': 'Bonsai-8B-unpacked',
            'n_profiles': len(PROFILES),
            'profiles': PROFILES,
            'group_size': GROUP_SIZE,
            'training': {
                'optimizer': 'SGD',
                'lr': 0.01,
                'epochs': 10,
                'examples_per_epoch': 25,
                'max_length': 64,
                'seed': SEED,
            },
        },
        'results': {
            'baseline_ppl': baseline_ppl,
            'eval_matrix': results,
        },
    }

    torch.save(save_data, "/checkpoints/scale_personalities_8profiles.pt")
    vol.commit()
    print(f"  Saved to /checkpoints/scale_personalities_8profiles.pt", flush=True)

    # Also save as JSON for easy inspection
    json_results = {
        'profiles': PROFILES,
        'baseline_ppl': baseline_ppl,
        'eval_matrix': {t: {p: float(v) for p, v in ppls.items()}
                       for t, ppls in results.items()},
        'diagonal_wins': wins,
        'total_profiles': len(PROFILES),
    }
    with open("/checkpoints/results_8profiles.json", "w") as f:
        json.dump(json_results, f, indent=2)
    vol.commit()
    print(f"  Saved results JSON", flush=True)

    # ─── Final Summary ───
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Model: Bonsai 8B (native 1-bit, {replaced} linears)")
    print(f"  VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
    print(f"  Profiles trained: {len(PROFILES)}")
    print(f"  Diagonal wins: {wins}/{len(PROFILES)}")
    print(f"  Scale tables saved: YES")
    print(f"  Total scale data: {sum(sum(v.numel() for v in s.values()) for s in profile_scales.values()) * 2 / 1e6:.1f}MB")
    print(f"{'='*70}", flush=True)

    return json_results


@app.local_entrypoint()
def main():
    results = train_and_eval.remote()
    print("\n\nResults returned:")
    print(json.dumps(results, indent=2))
