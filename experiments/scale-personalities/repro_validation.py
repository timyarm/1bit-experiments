"""Bonsai 8B Scale Personality — TRUE Native 1-bit (~1.1GB on GPU)

Replace ALL nn.Linear with NativeBitLinear that stores:
  - packed_signs: int32 (1 bit per weight, 32 weights per int)
  - group_scales: fp16 (1 per 128 weights)

8B model: 16GB in fp16 → 1.1GB in native 1-bit
Leaves 14GB free on T4 for training ALL layers.
"""
import modal
import os

app = modal.App("bonsai8b-native")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "accelerate", "numpy",
                 "huggingface_hub", "safetensors", "datasets")
)


@app.function(image=image, gpu="T4", timeout=5400,
              secrets=[modal.Secret.from_name("huggingface-secret")])
def test():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import random, time, math, gc, json
    from math import ceil
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset

    DEVICE = "cuda"
    GROUP_SIZE = 128
    ALPHA = 0.05
    random.seed(123)
    torch.manual_seed(123)

    print("=" * 60)
    print("BONSAI 8B: TRUE Native 1-bit (~1.1GB GPU)")
    print("=" * 60, flush=True)

    hf_token = os.environ.get("HF_TOKEN")

    # ─── NativeBitLinear: stores 1 bit per weight ───
    class NativeBitLinear(nn.Module):
        def __init__(self, in_features, out_features, bias=False, group_size=128):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.group_size = group_size
            n_ints = ceil(in_features / 32)
            n_groups = ceil(in_features / group_size)
            self.register_buffer('packed_signs', torch.zeros(out_features, n_ints, dtype=torch.int32))
            # Use Parameter (not buffer) so gradients flow through scales
            self.group_scales = nn.Parameter(torch.zeros(out_features, n_groups, dtype=torch.float16))
            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.bias = None

        def cache_signs(self):
            """Call once after loading. Caches unpacked signs — never recompute."""
            shifts = torch.arange(32, device=self.packed_signs.device, dtype=torch.int32)
            unpacked = ((self.packed_signs.unsqueeze(2) >> shifts) & 1)
            self._cached_signs = (unpacked.reshape(self.out_features, -1)[:, :self.in_features].half() * 2 - 1)

        def forward(self, x):
            # Vectorized unpack every call (caching OOMs on 8B)
            if not hasattr(self, '_bit_shifts'):
                self._bit_shifts = torch.arange(32, device=x.device, dtype=torch.int32)
            shifts = self._bit_shifts.to(x.device)
            unpacked = ((self.packed_signs.unsqueeze(2) >> shifts) & 1)
            signs = (unpacked.reshape(self.out_features, -1)[:, :self.in_features].half() * 2 - 1)

            # Scales in graph for gradient flow
            se = self.group_scales.unsqueeze(2).expand(-1, -1, self.group_size)
            sf = se.reshape(self.out_features, -1)[:, :self.in_features]
            w = signs * sf
            return F.linear(x, w, self.bias)

        @staticmethod
        def from_weight(weight_tensor, group_size=128):
            """Create from fp16 weight tensor (Bonsai dequantized weights)."""
            out_f, in_f = weight_tensor.shape
            layer = NativeBitLinear(in_f, out_f, bias=False, group_size=group_size)
            # Pack signs
            bits = (weight_tensor > 0).to(torch.int32)
            pad = (32 - in_f % 32) % 32
            if pad > 0: bits = F.pad(bits, (0, pad))
            n_ints = bits.shape[1] // 32
            bits = bits.view(out_f, n_ints, 32)
            packed = torch.zeros(out_f, n_ints, dtype=torch.int32)
            for bit in range(32): packed |= bits[:, :, bit] << bit
            layer.packed_signs.copy_(packed)
            # Compute scales
            pad_s = (group_size - in_f % group_size) % group_size
            w_abs = F.pad(weight_tensor.abs(), (0, pad_s)) if pad_s > 0 else weight_tensor.abs()
            n_groups = w_abs.shape[1] // group_size
            layer.group_scales.data.copy_(w_abs.view(out_f, n_groups, group_size).mean(dim=2).half())
            return layer

    # ─── Load Bonsai 8B directly (ONE download) ───
    print("\nLoading Bonsai 8B directly (single download)...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("prism-ml/Bonsai-8B-unpacked", token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "prism-ml/Bonsai-8B-unpacked", dtype=torch.float16, device_map="cpu",
        low_cpu_mem_usage=True, token=hf_token
    )
    n_layers = len(model.model.layers)
    print(f"  Loaded in {time.time()-t0:.1f}s ({n_layers} layers)", flush=True)

    # ─── Replace nn.Linear with NativeBitLinear ───
    print("  Replacing linears with NativeBitLinear...", flush=True)
    replaced = 0
    original_scales = {}

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear): continue
        if "layers." not in name: continue
        if module.weight.dim() != 2: continue

        parts = name.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        child_name = parts[-1]

        native = NativeBitLinear.from_weight(module.weight.data.float(), GROUP_SIZE)
        original_scales[name] = native.group_scales.data.clone()
        setattr(parent, child_name, native)
        replaced += 1
        del module
        if replaced % 50 == 0:
            gc.collect()

    gc.collect()
    print(f"  Replaced {replaced} linears", flush=True)

    # ─── Move to GPU ───
    print("  Moving to GPU...", flush=True)
    gc.collect()
    model = model.to(DEVICE)
    model.eval()

    torch.cuda.empty_cache()

    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"  VRAM: {vram:.2f}GB (vs ~16GB in fp16)", flush=True)

    # ─── PPL ───
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    wiki_texts = [t for t in wiki["text"] if len(t.strip()) > 100]
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    math_texts = [f"Q: {gsm[i]['question']}\nA: {gsm[i]['answer']}" for i in range(200)]

    def compute_ppl(texts, n=20):
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

    print(f"\n{'='*60}")
    print("BASELINE")
    print(f"{'='*60}", flush=True)
    base_m = compute_ppl(math_texts)
    base_l = compute_ppl(wiki_texts)
    print(f"  Math PPL: {base_m:.2f}  Lang PPL: {base_l:.2f}", flush=True)

    # ─── Train personality with LoRA-style scale deltas ───
    def train_personality(name, texts, epochs=10, lr=0.01):
        print(f"\n  Training {name} (epochs={epochs}, lr={lr})...", flush=True)

        # Reset scales to original before training
        for lin_name, orig in original_scales.items():
            parts = lin_name.split('.')
            module = model
            for p in parts:
                module = getattr(module, p)
            if hasattr(module, 'group_scales'):
                module.group_scales.data.copy_(orig.to(DEVICE))

        # Collect scale parameters (they're nn.Parameter now, gradients flow directly)
        scale_params = []
        for lin_name in original_scales:
            parts = lin_name.split('.')
            module = model
            for p in parts:
                module = getattr(module, p)
            if hasattr(module, 'group_scales'):
                scale_params.append(module.group_scales)

        # SGD — no momentum/variance states, saves ~2x VRAM vs Adam
        optimizer = torch.optim.SGD(scale_params, lr=lr)

        # Enable gradient checkpointing to save VRAM
        model.gradient_checkpointing_enable()

        # Quick gradient check on first example
        test_ids = tokenizer(texts[0], return_tensors="pt", truncation=True, max_length=48).input_ids.to(DEVICE)
        model.train()
        test_out = model(test_ids, labels=test_ids)
        if test_out.loss is not None and not torch.isnan(test_out.loss):
            test_out.loss.backward()
            n_has_grad = sum(1 for p in scale_params if p.grad is not None and p.grad.abs().max() > 0)
            max_g = max((p.grad.abs().max().item() for p in scale_params if p.grad is not None), default=0)
            print(f"    GRAD CHECK: loss={test_out.loss.item():.4f}, {n_has_grad}/{len(scale_params)} have grad, max_grad={max_g:.2e}", flush=True)
            optimizer.zero_grad()
        else:
            print(f"    GRAD CHECK: loss is None or NaN!", flush=True)
        del test_ids, test_out
        torch.cuda.empty_cache()

        for epoch in range(epochs):
            total_loss, n = 0, 0
            random.shuffle(texts)
            for text in texts[:20]:
                ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=48).input_ids.to(DEVICE)
                if ids.shape[1] < 5: continue
                torch.cuda.empty_cache()

                model.train()
                try:
                    out = model(ids, labels=ids)
                    if out.loss is None or torch.isnan(out.loss): continue

                    optimizer.zero_grad()
                    out.loss.backward()

                    # DEBUG: check if any scale has non-None grad
                    if epoch == 0 and n == 0:
                        has_grad = sum(1 for p in scale_params if p.grad is not None)
                        max_grad = max((p.grad.abs().max().item() for p in scale_params if p.grad is not None), default=0)
                        print(f"      DEBUG: {has_grad}/{len(scale_params)} params have grad, max={max_grad:.2e}", flush=True)

                    torch.nn.utils.clip_grad_norm_(scale_params, 1.0)
                    optimizer.step()

                    # Clamp scales positive
                    with torch.no_grad():
                        for p in scale_params:
                            p.data.clamp_(min=1e-6)

                    total_loss += out.loss.item()
                    n += 1
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        continue
                    raise
                del ids, out
                torch.cuda.empty_cache()

            print(f"    Epoch {epoch+1}: CE={total_loss/max(n,1):.8f}", flush=True)

        # Save trained scales
        result = {}
        for lin_name in original_scales:
            parts = lin_name.split('.')
            module = model
            for p in parts:
                module = getattr(module, p)
            if hasattr(module, 'group_scales'):
                result[lin_name] = module.group_scales.detach().cpu().clone()
        del optimizer
        torch.cuda.empty_cache()
        return result

    # ─── Train ───
    print(f"\n{'='*60}")
    print("MATH PERSONALITY")
    print(f"{'='*60}", flush=True)
    math_scales = train_personality("MATH", math_texts)

    print(f"\n{'='*60}")
    print("LANG PERSONALITY")
    print(f"{'='*60}", flush=True)
    lang_scales = train_personality("LANG", wiki_texts)

    # ─── SWAP TEST ───
    print(f"\n{'='*60}")
    print("SWAP TEST")
    print(f"{'='*60}", flush=True)

    def apply_scales(scale_table):
        for lin_name, scales in scale_table.items():
            parts = lin_name.split('.')
            module = model
            for p in parts:
                module = getattr(module, p)
            if hasattr(module, 'group_scales'):
                module.group_scales.data.copy_(scales.to(DEVICE))

    # Original
    apply_scales(original_scales)
    om = compute_ppl(math_texts)
    ol = compute_ppl(wiki_texts)
    print(f"  Original: Math={om:.2f}  Lang={ol:.2f}", flush=True)

    # Math
    apply_scales(math_scales)
    mm = compute_ppl(math_texts)
    ml = compute_ppl(wiki_texts)
    print(f"  Math:     Math={mm:.2f}  Lang={ml:.2f}", flush=True)

    # Lang
    apply_scales(lang_scales)
    lm = compute_ppl(math_texts)
    ll = compute_ppl(wiki_texts)
    print(f"  Lang:     Math={lm:.2f}  Lang={ll:.2f}", flush=True)

    # ─── Results ───
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  {'Table':<12} {'Math PPL':>10} {'Lang PPL':>10}")
    print(f"  {'-'*34}")
    print(f"  {'Original':<12} {om:>10.2f} {ol:>10.2f}")
    print(f"  {'Math':<12} {mm:>10.2f} {ml:>10.2f}")
    print(f"  {'Lang':<12} {lm:>10.2f} {ll:>10.2f}")

    math_wins = mm < lm
    lang_wins = ll < ml
    math_improves = mm < om
    lang_improves = ll < ol

    print(f"\n  Math beats lang on math: {'YES' if math_wins else 'NO'}")
    print(f"  Lang beats math on lang: {'YES' if lang_wins else 'NO'}")
    print(f"  Math improves original:  {'YES' if math_improves else 'NO'}")
    print(f"  Lang improves original:  {'YES' if lang_improves else 'NO'}")

    if math_wins and lang_wins:
        print(f"\n  SCALE PERSONALITIES WORK ON 8B!")
    elif math_wins or lang_wins:
        print(f"\n  PARTIAL DIFFERENTIATION")
    else:
        print(f"\n  NO DIFFERENTIATION")

    print(f"\n  GPU VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

    # ─── Generation quality test ───
    print(f"\n{'='*60}")
    print("GENERATION QUALITY (side-by-side)")
    print(f"{'='*60}", flush=True)

    def gen(prompt, max_tokens=60):
        model.eval()
        inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=96).to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_tokens, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)[:100]

    test_prompts = [
        ("What is 15 + 27?", "MATH"),
        ("What is the square root of 144?", "MATH"),
        ("What is the capital of France?", "LANG"),
        ("Write one sentence about the ocean.", "LANG"),
    ]

    for table_name, table in [("Original", original_scales), ("Math", math_scales), ("Lang", lang_scales)]:
        apply_scales(table)
        print(f"\n  --- {table_name} personality ---")
        for prompt, ptype in test_prompts:
            answer = gen(prompt)
            print(f"    [{ptype}] {prompt}")
            print(f"           → {answer}")


@app.local_entrypoint()
def main():
    test.remote()
