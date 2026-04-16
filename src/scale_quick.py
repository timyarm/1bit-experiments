"""Minimal scale training — ONE domain, ONE eval, fast iteration.

No llama.cpp. No multi-domain. Just: train scales on math, measure PPL before/after.
Target: under 5 minutes total.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S",
                    handlers=[logging.FileHandler("/tmp/scale_training.log", mode='w'),
                              logging.StreamHandler(sys.stdout)])
log = logging.getLogger("s")

DEVICE = "cuda"
GROUP_SIZE = 128

def main():
    log.info("=== SCALE QUICK TEST ===")
    t_start = time.time()

    # ─── Load model with PackedBitLinear (signs as frozen fp16 buffer + trainable scales) ───
    log.info("Loading Bonsai 1.7B (PackedBitLinear: frozen signs + trainable scales)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, os.path.dirname(__file__))
    from packed_bitlinear import PackedBitLinear, convert_model

    tokenizer = AutoTokenizer.from_pretrained("prism-ml/Bonsai-1.7B-unpacked", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "prism-ml/Bonsai-1.7B-unpacked", dtype=torch.float16, trust_remote_code=True, device_map="cpu"
    )
    convert_model(model)
    model = model.to(DEVICE)
    model.gradient_checkpointing_enable()
    log.info(f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # ─── Prepare data (hardcoded, no HF download) ───
    math_texts = [
        "Question: What is 15 + 27?\nAnswer: 15 + 27 = 42. The answer is 42.",
        "Question: A store has 48 apples. If 13 are sold, how many remain?\nAnswer: 48 - 13 = 35. The answer is 35.",
        "Question: If a car travels 60 miles per hour for 3 hours, how far does it go?\nAnswer: 60 * 3 = 180 miles. The answer is 180.",
        "Question: What is 8 times 7?\nAnswer: 8 * 7 = 56. The answer is 56.",
        "Question: A train has 120 passengers. At the next stop, 35 get off and 22 get on. How many passengers now?\nAnswer: 120 - 35 + 22 = 107. The answer is 107.",
        "Question: What is 144 divided by 12?\nAnswer: 144 / 12 = 12. The answer is 12.",
        "Question: If you have 3 boxes with 24 items each, how many items total?\nAnswer: 3 * 24 = 72. The answer is 72.",
        "Question: A rectangle has length 8 and width 5. What is its area?\nAnswer: 8 * 5 = 40. The answer is 40.",
        "Question: What is 25% of 200?\nAnswer: 200 * 0.25 = 50. The answer is 50.",
        "Question: If 5x = 35, what is x?\nAnswer: x = 35 / 5 = 7. The answer is 7.",
        "Question: A pizza is cut into 8 slices. If 3 people each eat 2 slices, how many are left?\nAnswer: 8 - (3 * 2) = 8 - 6 = 2. The answer is 2.",
        "Question: What is the square root of 81?\nAnswer: sqrt(81) = 9. The answer is 9.",
        "Question: If a book costs $12.50 and you buy 4, how much do you pay?\nAnswer: 12.50 * 4 = 50. The answer is $50.",
        "Question: What is 1000 minus 387?\nAnswer: 1000 - 387 = 613. The answer is 613.",
        "Question: A garden is 15 meters long and 8 meters wide. What is its perimeter?\nAnswer: 2 * (15 + 8) = 2 * 23 = 46 meters. The answer is 46.",
        "Question: If you save $5 per day for 30 days, how much do you save?\nAnswer: 5 * 30 = 150. The answer is $150.",
        "Question: What is 7 squared?\nAnswer: 7^2 = 49. The answer is 49.",
        "Question: A bag has 36 marbles. 1/4 are red. How many red marbles?\nAnswer: 36 / 4 = 9. The answer is 9.",
        "Question: What is 15% of 80?\nAnswer: 80 * 0.15 = 12. The answer is 12.",
        "Question: If a triangle has sides 3, 4, and 5, what is its perimeter?\nAnswer: 3 + 4 + 5 = 12. The answer is 12.",
    ]

    wiki_texts = [
        "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
        "Paris is the capital and most populous city of France. With a population of over 2 million residents.",
        "Water is a transparent, tasteless, odorless, and nearly colorless chemical substance.",
        "The Amazon rainforest produces approximately 20 percent of the world's oxygen.",
        "Albert Einstein developed the theory of relativity, one of the two pillars of modern physics.",
        "The Great Wall of China is a series of fortifications built along the historical northern borders.",
        "DNA carries the genetic instructions used in the growth and functioning of all living organisms.",
        "The Pacific Ocean is the largest and deepest ocean on Earth, covering more than 63 million square miles.",
        "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
        "The human brain contains approximately 86 billion neurons, each connected to thousands of others.",
    ]

    # ─── Baseline PPL ───
    log.info("\n--- BASELINE PPL ---")
    math_ppl_base = eval_ppl(model, tokenizer, math_texts)
    wiki_ppl_base = eval_ppl(model, tokenizer, wiki_texts)
    log.info(f"  Math PPL: {math_ppl_base:.2f}")
    log.info(f"  Wiki PPL: {wiki_ppl_base:.2f}")

    # ─── Train math scales ───
    log.info("\n--- TRAINING MATH SCALES ---")

    # Freeze everything, unfreeze only PackedBitLinear scales
    for p in model.parameters():
        p.requires_grad = False

    scale_params = []
    scale_originals = {}
    for name, module in model.named_modules():
        if isinstance(module, PackedBitLinear):
            module.scales.requires_grad = True
            scale_params.append(module.scales)
            scale_originals[name] = module.scales.data.clone()

    trainable = sum(p.numel() for p in scale_params)
    log.info(f"  Trainable params: {trainable:,} (group scales)")
    log.info(f"  VRAM for grads: {trainable * 4 / 1e6:.1f}MB")

    optimizer = torch.optim.SGD(scale_params, lr=0.01)
    model.train()

    for epoch in range(10):
        total_loss = 0
        n = 0
        for text in math_texts:
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(DEVICE)
            if tokens.input_ids.shape[1] < 5:
                continue
            with torch.amp.autocast('cuda', dtype=torch.float16):
                loss = model(**tokens, labels=tokens.input_ids).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            n += 1
        log.info(f"  Epoch {epoch+1}/10: loss={total_loss/max(n,1):.4f} VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB")

    # ─── Post-training PPL (math scales still active via hooks) ───
    log.info("\n--- POST-TRAINING PPL (math scales) ---")
    model.eval()
    torch.cuda.empty_cache()
    math_ppl_after = eval_ppl(model, tokenizer, math_texts)
    wiki_ppl_after = eval_ppl(model, tokenizer, wiki_texts)
    log.info(f"  Math PPL: {math_ppl_base:.2f} -> {math_ppl_after:.2f} ({(1-math_ppl_after/math_ppl_base)*100:+.1f}%)")
    log.info(f"  Wiki PPL: {wiki_ppl_base:.2f} -> {wiki_ppl_after:.2f} ({(1-wiki_ppl_after/wiki_ppl_base)*100:+.1f}%)")

    # ─── Diagonal check ───
    diagonal = math_ppl_after < math_ppl_base
    log.info(f"\n  Math improved: {'YES' if diagonal else 'NO'}")

    # Save math scales, restore originals, train wiki
    math_scales = {name: module.scales.data.clone() for name, module in model.named_modules()
                   if isinstance(module, PackedBitLinear)}

    log.info("\n--- TRAINING WIKI SCALES ---")
    for name, module in model.named_modules():
        if isinstance(module, PackedBitLinear) and name in scale_originals:
            module.scales.data.copy_(scale_originals[name])

    optimizer = torch.optim.SGD(scale_params, lr=0.01)
    model.train()
    for epoch in range(10):
        total_loss = 0
        n = 0
        for text in wiki_texts:
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(DEVICE)
            if tokens.input_ids.shape[1] < 5:
                continue
            with torch.amp.autocast('cuda', dtype=torch.float16):
                loss = model(**tokens, labels=tokens.input_ids).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            n += 1
        log.info(f"  Epoch {epoch+1}/10: loss={total_loss/max(n,1):.4f} VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB")

    wiki_scales = {name: module.scales.data.clone() for name, module in model.named_modules()
                   if isinstance(module, PackedBitLinear)}

    log.info("\n--- POST-TRAINING PPL (wiki scales) ---")
    model.eval()
    torch.cuda.empty_cache()
    math_ppl_wiki = eval_ppl(model, tokenizer, math_texts)
    wiki_ppl_wiki = eval_ppl(model, tokenizer, wiki_texts)
    log.info(f"  Math PPL: {math_ppl_base:.2f} -> {math_ppl_wiki:.2f}")
    log.info(f"  Wiki PPL: {wiki_ppl_base:.2f} -> {wiki_ppl_wiki:.2f}")

    # Restore math scales and verify
    log.info("\n--- PPL WITH MATH SCALES (restored) ---")
    for name, module in model.named_modules():
        if isinstance(module, PackedBitLinear) and name in math_scales:
            module.scales.data.copy_(math_scales[name])
    math_ppl_math2 = eval_ppl(model, tokenizer, math_texts)
    wiki_ppl_math2 = eval_ppl(model, tokenizer, wiki_texts)
    log.info(f"  Math PPL: {math_ppl_math2:.2f}")
    log.info(f"  Wiki PPL: {wiki_ppl_math2:.2f}")

    # Save scale tables for GGUF patching
    torch.save(math_scales, "checkpoints/math_scales.pt")
    torch.save(wiki_scales, "checkpoints/wiki_scales.pt")
    torch.save(scale_originals, "checkpoints/original_scales.pt")
    log.info("Saved scale tables to checkpoints/")

    # ─── Summary ───
    log.info(f"\n{'='*50}")
    log.info("DIAGONAL DOMINANCE CHECK")
    log.info(f"{'='*50}")
    log.info(f"{'Scales':<15} {'Math PPL':>10} {'Wiki PPL':>10}")
    log.info(f"{'-'*37}")
    log.info(f"{'baseline':<15} {math_ppl_base:>10.2f} {wiki_ppl_base:>10.2f}")
    log.info(f"{'math':<15} {math_ppl_after:>10.2f} {wiki_ppl_after:>10.2f}")
    log.info(f"{'wiki':<15} {math_ppl_wiki:>10.2f} {wiki_ppl_wiki:>10.2f}")

    math_diag = math_ppl_after < math_ppl_wiki  # math scales should be best on math
    wiki_diag = wiki_ppl_wiki < wiki_ppl_after   # wiki scales should be best on wiki
    log.info(f"\nMath diagonal: {'YES' if math_diag else 'NO'}")
    log.info(f"Wiki diagonal: {'YES' if wiki_diag else 'NO'}")

    elapsed = time.time() - t_start
    log.info(f"\nTotal time: {elapsed:.0f}s")


def eval_ppl(model, tokenizer, texts):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
            if tokens.input_ids.shape[1] < 5:
                continue
            with torch.amp.autocast('cuda', dtype=torch.float16):
                out = model(**tokens, labels=tokens.input_ids)
            total_loss += out.loss.item() * tokens.input_ids.shape[1]
            total_tokens += tokens.input_ids.shape[1]
    return np.exp(total_loss / max(total_tokens, 1))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"CRASHED: {e}", exc_info=True)
    finally:
        log.info("DONE")
