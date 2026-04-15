"""Diagnostic: test 1-bit Qwen3-0.6B at multiple stages.

1. Teacher (bf16) — baseline, should be coherent
2. Naive PTQ (sign + mean-abs scale, NO training) — can 1-bit work at all?
3. QAT student pre-bake (STE forward pass) — is training helping?
4. QAT student post-bake (exact binary) — where does collapse happen?

This tells us EXACTLY where the problem is.
"""
import modal
import json

app = modal.App("1bit-qwen-diagnostic")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.51.0",
        "accelerate>=1.5.0",
        "huggingface_hub>=0.28.0",
    )
)

vol = modal.Volume.from_name("1bit-checkpoints", create_if_missing=True)

TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain gravity in one sentence.",
    "1 + 1 =",
    "The color of the sky is",
    "List three animals:",
]


@app.function(
    image=image,
    gpu="T4:1",
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=32768,
    volumes={"/data": vol},
)
def run_diagnostic():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import os
    import gc

    hf_token = os.environ.get("HF_TOKEN")
    model_id = "Qwen/Qwen3-0.6B"
    device = "cuda"

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

    def test_model(model, label):
        print(f"\n{'=' * 60}")
        print(f"TEST: {label}")
        print(f"{'=' * 60}", flush=True)
        model.eval()
        for prompt in TEST_PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=50, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen = output[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(gen, skip_special_tokens=True)
            print(f"  Q: {prompt}")
            print(f"  A: {text[:150]}", flush=True)

    # ══════════════════════════════════════════════════════════
    # Test 1: Teacher (bf16 baseline)
    # ══════════════════════════════════════════════════════════
    print("Loading teacher (bf16)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cpu", token=hf_token,
    )
    model.to(device)
    test_model(model, "TEACHER (bf16)")

    # ══════════════════════════════════════════════════════════
    # Test 2: Naive PTQ (sign + mean-abs scale, NO training)
    # ══════════════════════════════════════════════════════════
    print("\nApplying naive 1-bit PTQ...", flush=True)
    ptq_count = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or module.weight.numel() < 128:
            continue
        with torch.no_grad():
            w = module.weight
            flat = w.reshape(-1)
            gs = 128
            rem = flat.numel() % gs
            if rem:
                flat = F.pad(flat, (0, gs - rem))
            groups = flat.reshape(-1, gs)
            signs = groups.sign()
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            scales = groups.abs().mean(dim=1, keepdim=True)
            baked = (signs * scales).reshape(-1)[:w.numel()].reshape(w.shape)
            module.weight.copy_(baked)
        ptq_count += 1
    print(f"  Naive PTQ applied to {ptq_count} layers", flush=True)
    test_model(model, "NAIVE PTQ (no training)")

    # ══════════════════════════════════════════════════════════
    # Test 3: Load QAT checkpoint (epoch 5 — good forensics)
    # ══════════════════════════════════════════════════════════
    ckpt_path = "/data/qwen06b_1bit_epoch5.pt"
    if os.path.exists(ckpt_path):
        print(f"\nLoading QAT checkpoint: {ckpt_path}...", flush=True)
        # Reload fresh model
        del model
        gc.collect()
        torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cpu", token=hf_token,
        )

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        applied = 0
        for name, module in model.named_modules():
            if name in ckpt and isinstance(module, nn.Linear):
                with torch.no_grad():
                    w = ckpt[name]["weight"]
                    scales = ckpt[name]["scales"]
                    # Bake: sign(w) * scale
                    flat = w.reshape(-1)
                    gs = 128
                    rem = flat.numel() % gs
                    if rem:
                        flat = F.pad(flat, (0, gs - rem))
                    groups = flat.reshape(-1, gs)
                    signs = groups.sign()
                    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
                    n = groups.shape[0]
                    s = scales[:n].abs().clamp(min=1e-6).unsqueeze(1)
                    baked = (signs * s).reshape(-1)[:module.weight.numel()]
                    baked = baked.reshape(module.weight.shape).to(module.weight.dtype)
                    module.weight.copy_(baked)
                applied += 1
        print(f"  Applied QAT weights to {applied} layers", flush=True)

        model.to(device)
        test_model(model, "QAT EPOCH 5 (baked)")

        # Also test epoch 1
        ckpt1_path = "/data/qwen06b_1bit_epoch1.pt"
        if os.path.exists(ckpt1_path):
            del model
            gc.collect()
            torch.cuda.empty_cache()

            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map="cpu", token=hf_token,
            )
            ckpt1 = torch.load(ckpt1_path, map_location="cpu", weights_only=True)
            for name, module in model.named_modules():
                if name in ckpt1 and isinstance(module, nn.Linear):
                    with torch.no_grad():
                        w = ckpt1[name]["weight"]
                        scales = ckpt1[name]["scales"]
                        flat = w.reshape(-1)
                        gs = 128
                        rem = flat.numel() % gs
                        if rem:
                            flat = F.pad(flat, (0, gs - rem))
                        groups = flat.reshape(-1, gs)
                        signs = groups.sign()
                        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
                        n = groups.shape[0]
                        s = scales[:n].abs().clamp(min=1e-6).unsqueeze(1)
                        baked = (signs * s).reshape(-1)[:module.weight.numel()]
                        baked = baked.reshape(module.weight.shape).to(module.weight.dtype)
                        module.weight.copy_(baked)
            model.to(device)
            test_model(model, "QAT EPOCH 1 (baked)")
    else:
        print(f"\n  No QAT checkpoint found at {ckpt_path}", flush=True)

    print(f"\n{'=' * 60}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'=' * 60}", flush=True)


@app.local_entrypoint()
def main():
    print("1-bit Diagnostic — Qwen3-0.6B\n")
    run_diagnostic.remote()
