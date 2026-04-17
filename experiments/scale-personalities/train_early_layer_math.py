"""Corrected #12: Targeted early-layer math scale training.

Exp 14 finding: math specialization lives in EARLY layers (0-11), not late.
This script tests whether targeted early-layer training matches the full-model
28% GSM8K with fewer trainable parameters.

Two conditions (run sequentially):
  A. early_all   — all scales in layers 0-11 trainable (~40% of params)
  B. early_ffn   — only ffn_up/gate in layers 0-11 (~14% of params)

Full-model baseline for comparison: 28.0% GSM8K (from scale_v2_proper.py run).

Same v2 recipe throughout:
  AdamW lr=1e-4, Rho-1 top-60% token weighting, elastic band λ=0.1,
  3 epochs, 150 examples (70% math + 30% wikitext), seq_len=256.

Monitor: ssh -p 2222 timyarm@100.110.173.110 "tail -f /tmp/early_layer_train.log"
"""
import sys
import os
import time
import json
import logging
import shutil
import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler("/tmp/early_layer_train.log", mode='w'),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("early")

sys.path.insert(0, os.path.dirname(__file__))

CKPT = os.path.expanduser("~/freigent/apps/trucksim/checkpoints")
GGUF_DIR = os.path.expanduser("~/freigent/apps/trucksim/data/llm")
GGUF_BASE = f"{GGUF_DIR}/Bonsai-1.7B.gguf"
GGUF_OUT = f"{GGUF_DIR}/Bonsai-1.7B-early.gguf"

DEVICE = "cuda"
N_LAYERS = 24
EARLY_CUTOFF = 12        # layers 0-11 = early, 12-23 = late
MAX_LEN = 256
TRAIN_EXAMPLES = 150
LR = 1e-4
EPOCHS = 3
REG_LAMBDA = 0.1
TOKEN_SELECT_RATIO = 0.6
DOMAIN_MIX = 0.7
N_GSM8K_EVAL = 100

FULL_MODEL_BASELINE = 0.280   # 28.0% GSM8K from scale_v2_proper.py

FFN_EARLY_TYPES = {"mlp.gate_proj", "mlp.up_proj"}   # ffn_up/gate


def get_layer_idx(name):
    if "model.layers." not in name:
        return None
    try:
        return int(name.split("model.layers.")[1].split(".")[0])
    except (IndexError, ValueError):
        return None


def is_early_layer(name):
    idx = get_layer_idx(name)
    return idx is not None and idx < EARLY_CUTOFF


def is_early_ffn(name):
    if not is_early_layer(name):
        return False
    return any(t in name for t in FFN_EARLY_TYPES)


def select_scale_params(model, condition):
    """Return (scale_params list, trainable_count) for the given condition."""
    from packed_bitlinear import PackedBitLinear
    for p in model.parameters():
        p.requires_grad = False

    scale_params = []
    for name, module in model.named_modules():
        if not isinstance(module, PackedBitLinear):
            continue
        if condition == "early_all" and is_early_layer(name):
            module.scales.requires_grad = True
            scale_params.append(module.scales)
        elif condition == "early_ffn" and is_early_ffn(name):
            module.scales.requires_grad = True
            scale_params.append(module.scales)
        elif condition == "full":
            module.scales.requires_grad = True
            scale_params.append(module.scales)

    trainable = sum(p.numel() for p in scale_params)
    return scale_params, trainable


def get_math_data():
    from datasets import load_dataset
    math_texts, diverse_texts = [], []
    ds = load_dataset("gsm8k", "main", split="train", streaming=True)
    for ex in ds:
        math_texts.append(f"Question: {ex['question']}\nAnswer: {ex['answer']}")
        if len(math_texts) >= TRAIN_EXAMPLES * 2:
            break
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
    for ex in ds:
        if len(ex['text'].strip()) > 200:
            diverse_texts.append(ex['text'])
        if len(diverse_texts) >= TRAIN_EXAMPLES:
            break
    return math_texts, diverse_texts


def mix_data(math, diverse, n_total, seed=42):
    rng = np.random.default_rng(seed)
    n_math = int(n_total * DOMAIN_MIX)
    n_div = n_total - n_math
    mixed = math[:n_math] + diverse[:n_div]
    return [mixed[i] for i in rng.permutation(len(mixed))]


def train(model, tokenizer, train_texts, orig_snapshots, scale_params):
    optimizer = torch.optim.AdamW(scale_params, lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, EPOCHS * len(train_texts)))
    model.train()

    from packed_bitlinear import PackedBitLinear
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        n = 0
        for text in train_texts:
            tokens = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=MAX_LEN).to(DEVICE)
            if tokens.input_ids.shape[1] < 10:
                continue
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(**tokens, labels=tokens.input_ids)
                student_logits = outputs.logits[:, :-1]
                labels = tokens.input_ids[:, 1:]
                ce_per_token = F.cross_entropy(
                    student_logits.reshape(-1, student_logits.size(-1)).float(),
                    labels.reshape(-1), reduction='none')
                threshold = torch.quantile(ce_per_token, 1.0 - TOKEN_SELECT_RATIO)
                mask = (ce_per_token >= threshold).float()
                task_loss = (ce_per_token * mask).sum() / (mask.sum() + 1e-8)
                reg_loss = torch.tensor(0.0, device=DEVICE)
                for name, module in model.named_modules():
                    if isinstance(module, PackedBitLinear) and name in orig_snapshots:
                        if module.scales.requires_grad:
                            reg_loss = reg_loss + F.mse_loss(module.scales, orig_snapshots[name])
                loss = task_loss + REG_LAMBDA * reg_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scale_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += task_loss.item()
            n += 1
        log.info(f"    epoch {epoch+1}/{EPOCHS}  loss={epoch_loss/max(n,1):.4f}")


def patch_gguf(trained_scales, output_path):
    from gguf import GGUFReader
    shutil.copy2(GGUF_BASE, output_path)
    reader = GGUFReader(GGUF_BASE)
    data = np.memmap(output_path, dtype=np.uint8, mode='r+')
    BLOCK = 18

    def map_name(pt_name):
        n = pt_name.replace('model.layers.', 'blk.')
        n = n.replace('.self_attn.q_proj', '.attn_q').replace('.self_attn.k_proj', '.attn_k')
        n = n.replace('.self_attn.v_proj', '.attn_v').replace('.self_attn.o_proj', '.attn_output')
        n = n.replace('.mlp.gate_proj', '.ffn_gate').replace('.mlp.up_proj', '.ffn_up')
        n = n.replace('.mlp.down_proj', '.ffn_down')
        return n + '.weight'

    name_to_scales = {map_name(k): v for k, v in trained_scales.items()}
    patched = 0
    for t in reader.tensors:
        tname = str(t.name)
        if tname not in name_to_scales:
            continue
        scales = name_to_scales[tname].cpu().numpy().astype(np.float16).flatten()
        offset = t.data_offset
        n_groups = t.n_bytes // BLOCK
        if len(scales) != n_groups:
            continue
        blocks = data[offset:offset + t.n_bytes].reshape(n_groups, BLOCK)
        blocks[:, :2] = scales.view(np.uint8).reshape(n_groups, 2)
        patched += 1
    data.flush()
    return patched


def eval_gsm8k(gguf_path, n=N_GSM8K_EVAL):
    from llama_fast_eval import LlamaEval
    from datasets import load_dataset
    import re
    ev = LlamaEval(gguf_path, n_ctx=1024)
    ev.start()
    try:
        ds = load_dataset("gsm8k", "main", split="test", streaming=True)
        correct = total = 0
        for ex in ds:
            if total >= n:
                break
            resp = ev.generate(f"Question: {ex['question']}\nAnswer:", max_tokens=200)
            resp = resp if isinstance(resp, str) else ""
            pred = re.findall(r'-?\d+\.?\d*', resp.replace(',', ''))
            gold = re.findall(r'-?\d+\.?\d*', ex['answer'].split('####')[-1].replace(',', ''))
            try:
                if pred and gold and abs(float(pred[-1]) - float(gold[-1])) < 0.01:
                    correct += 1
            except ValueError:
                pass
            total += 1
        return correct / max(total, 1)
    finally:
        ev.stop()


ALL_CONDITIONS = ["early_all", "early_ffn"]
# Run one condition per process to avoid GPU state corruption between trainings.
# Set EARLY_CONDITION env var to select which one. Shell loop handles sequencing.
CONDITION = os.environ.get("EARLY_CONDITION", "early_all")


def main():
    t0 = time.time()
    log.info("EARLY-LAYER MATH SCALE TRAINING")
    log.info(f"Hypothesis: early layers 0-{EARLY_CUTOFF-1} carry math — target them to match 28% full-model GSM8K")
    log.info(f"Full-model baseline: {FULL_MODEL_BASELINE:.1%} GSM8K")
    log.info(f"Running condition: {CONDITION} (set EARLY_CONDITION env var to change)")

    log.info("\n--- DATA ---")
    math_texts, diverse_texts = get_math_data()
    train_texts = mix_data(math_texts, diverse_texts, TRAIN_EXAMPLES)
    log.info(f"  {len(train_texts)} examples ({int(DOMAIN_MIX*100)}% math)")

    log.info("\n--- MODEL ---")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from packed_bitlinear import PackedBitLinear, convert_model
    tokenizer = AutoTokenizer.from_pretrained(
        "prism-ml/Bonsai-1.7B-unpacked", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "prism-ml/Bonsai-1.7B-unpacked", dtype=torch.float16,
        trust_remote_code=True, device_map="cpu")
    convert_model(model)
    model = model.to(DEVICE)
    model.gradient_checkpointing_enable()
    log.info(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    orig_scales = {}
    for name, module in model.named_modules():
        if isinstance(module, PackedBitLinear):
            orig_scales[name] = module.scales.data.clone()

    total_scales = sum(p.numel() for p in orig_scales.values())
    log.info(f"  Total scale params: {total_scales:,}")

    # Load any existing results so conditions accumulate across processes
    out_path = f"{CKPT}/early_layer_results.json"
    results = {}
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                results = json.load(f)
        except Exception:
            results = {}

    for condition in [CONDITION]:
        log.info(f"\n{'='*55}")
        log.info(f"CONDITION: {condition}")
        log.info(f"{'='*55}")

        # Reset to original scales
        for name, module in model.named_modules():
            if isinstance(module, PackedBitLinear) and name in orig_scales:
                module.scales.data.copy_(orig_scales[name])

        scale_params, trainable = select_scale_params(model, condition)
        log.info(f"  Trainable: {trainable:,} / {total_scales:,} ({trainable/total_scales:.1%})")

        # Log which layers are being trained
        trained_layers = sorted(set(
            get_layer_idx(n) for n in orig_scales
            if any(p is model.state_dict().get(n + '.scales', None)
                   for p in scale_params)
        ) - {None})
        # Simpler: just log the layer range
        trained_names = [name for name, module in model.named_modules()
                         if isinstance(module, PackedBitLinear) and module.scales.requires_grad]
        layer_idxs = sorted(set(get_layer_idx(n) for n in trained_names) - {None})
        if layer_idxs:
            log.info(f"  Layers: {min(layer_idxs)}–{max(layer_idxs)} ({len(layer_idxs)} layers)")
        log.info(f"  Modules: {len(trained_names)}")

        t_train = time.time()
        train(model, tokenizer, train_texts, orig_scales, scale_params)
        train_time = time.time() - t_train
        log.info(f"  Trained in {train_time:.0f}s")

        # Snapshot trained scales (only trained ones; rest stay original)
        trained_scales = {}
        for name, module in model.named_modules():
            if isinstance(module, PackedBitLinear):
                trained_scales[name] = module.scales.data.clone().cpu()

        torch.save(trained_scales, f"{CKPT}/early_math_{condition}_scales.pt")

        # Free GPU for llama.cpp eval
        log.info("  Patching GGUF and evaluating...")
        n_patched = patch_gguf(trained_scales, GGUF_OUT)
        log.info(f"  Patched {n_patched} tensors")

        # Temporarily free model from GPU
        model.cpu()
        torch.cuda.empty_cache()

        gsm = eval_gsm8k(GGUF_OUT)
        delta = gsm - FULL_MODEL_BASELINE
        log.info(f"  GSM8K: {gsm:.1%}  (full-model: {FULL_MODEL_BASELINE:.1%}, delta: {delta:+.1%})")

        results[condition] = {
            "gsm8k": gsm,
            "trainable_params": trainable,
            "pct_of_total": trainable / total_scales,
            "train_time": train_time,
        }

        # Restore model to GPU for next condition
        model.to(DEVICE)

    # Summary
    log.info(f"\n{'='*55}")
    log.info("SUMMARY — EARLY-LAYER TARGETED TRAINING")
    log.info(f"{'='*55}")
    log.info(f"Full-model v2 baseline: {FULL_MODEL_BASELINE:.1%} GSM8K (100% scale params)")
    log.info(f"{'Condition':<15}{'GSM8K':<10}{'Params':<12}{'% of total':<12}{'vs full-model'}")
    for cond, r in results.items():
        delta = r['gsm8k'] - FULL_MODEL_BASELINE
        reached = "MATCHES" if abs(delta) < 0.03 else ("EXCEEDS" if delta > 0.03 else "below")
        log.info(f"{cond:<15}{r['gsm8k']:<10.1%}{r['trainable_params']:<12,}{r['pct_of_total']:<12.1%}"
                 f"{delta:+.1%}  {reached}")

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved: {out_path}")
    log.info(f"Total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"CRASHED: {e}", exc_info=True)
    finally:
        log.info("DONE")
