"""Exp 17: Two-region cooperative scale training.

Hypothesis: early layers (0-11) handle math encoding; late layers (12-23) handle
knowledge retrieval. If we train each region on its natural domain and combine the
resulting scale tables, the composite should exceed the full-model math baseline
(28.0% GSM8K) because each region is doing what it's actually built for.

Procedure:
  Region A: early-ffn (mlp.gate_proj/up_proj, layers 0-11) trained on math (GSM8K)
  Region B: late-ffn  (mlp.gate_proj/up_proj, layers 12-23) trained on knowledge (TriviaQA)
  Combine:  early-ffn scales from A + late-ffn scales from B + baseline for everything else
  Eval:     GSM8K n=100 + MMLU n=50

Subprocess isolation between regions (same CUDA fix as Exp 16).

Evidence from prior runs:
  - early_ffn alone (math): 22.0% GSM8K  (21.4% params)
  - full model (math):       28.0% GSM8K (100% params)
  - flat_0.7 blend:          40.0% GSM8K (no training, just math+knowledge mix)
  Target: beat 28.0% GSM8K while also recovering MMLU (flat_0.7 got 41.7%)

Monitor: ssh -p 2222 timyarm@100.110.173.110 "tail -f /tmp/two_region_train.log"
"""
import sys
import os
import subprocess
import time
import json
import logging
import shutil
import numpy as np
import torch
import torch.nn.functional as F

LOG_PATH = "/tmp/two_region_train.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_PATH, mode='w'),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("coop")

sys.path.insert(0, os.path.dirname(__file__))

CKPT = os.path.expanduser("~/freigent/apps/trucksim/checkpoints")
GGUF_DIR = os.path.expanduser("~/freigent/apps/trucksim/data/llm")
GGUF_BASE = f"{GGUF_DIR}/Bonsai-1.7B.gguf"
GGUF_OUT  = f"{GGUF_DIR}/Bonsai-1.7B-coop.gguf"

DEVICE = "cuda"
N_LAYERS = 24
EARLY_CUTOFF = 12
MAX_LEN = 256
TRAIN_EXAMPLES = 150
LR = 1e-4
EPOCHS = 3
REG_LAMBDA = 0.1
TOKEN_SELECT_RATIO = 0.6

FULL_MATH_BASELINE  = 0.280   # 28.0% GSM8K — full model math training
EARLY_FFN_BASELINE  = 0.220   # 22.0% GSM8K — early-ffn only (Exp 16)
FLAT_BLEND_BASELINE = 0.400   # 40.0% GSM8K — flat_0.7 blend (no training)

N_GSM8K_EVAL = 100
N_MMLU_EVAL  = 50

FFN_TYPES = {"mlp.gate_proj", "mlp.up_proj"}

REGION = os.environ.get("COOP_REGION", "")   # "early_math" | "late_knowledge"


# ── helpers ──────────────────────────────────────────────────────────────────

def get_layer_idx(name):
    if "model.layers." not in name:
        return None
    try:
        return int(name.split("model.layers.")[1].split(".")[0])
    except (IndexError, ValueError):
        return None


def is_early_ffn(name):
    idx = get_layer_idx(name)
    return idx is not None and idx < EARLY_CUTOFF and any(t in name for t in FFN_TYPES)


def is_late_ffn(name):
    idx = get_layer_idx(name)
    return idx is not None and idx >= EARLY_CUTOFF and any(t in name for t in FFN_TYPES)


def select_params(model, region):
    from packed_bitlinear import PackedBitLinear
    for p in model.parameters():
        p.requires_grad = False
    scale_params = []
    for name, module in model.named_modules():
        if not isinstance(module, PackedBitLinear):
            continue
        if region == "early_math" and is_early_ffn(name):
            module.scales.requires_grad = True
            scale_params.append(module.scales)
        elif region == "late_knowledge" and is_late_ffn(name):
            module.scales.requires_grad = True
            scale_params.append(module.scales)
    return scale_params


# ── data ─────────────────────────────────────────────────────────────────────

def get_math_data(n):
    from datasets import load_dataset
    texts = []
    ds = load_dataset("gsm8k", "main", split="train", streaming=True)
    for ex in ds:
        texts.append(f"Question: {ex['question']}\nAnswer: {ex['answer']}")
        if len(texts) >= n:
            break
    return texts


def get_knowledge_data(n):
    from datasets import load_dataset
    texts = []
    ds = load_dataset("trivia_qa", "rc", split="train", streaming=True)
    for ex in ds:
        q = ex.get("question", "")
        a = ex.get("answer", {})
        ans = a.get("value", "") if isinstance(a, dict) else str(a)
        if q and ans:
            texts.append(f"Question: {q}\nAnswer: {ans}")
        if len(texts) >= n:
            break
    return texts


# ── train ─────────────────────────────────────────────────────────────────────

def train(model, tokenizer, texts, orig_scales, scale_params):
    optimizer = torch.optim.AdamW(scale_params, lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, EPOCHS * len(texts)))
    model.train()
    from packed_bitlinear import PackedBitLinear
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        n = 0
        for text in texts:
            tokens = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=MAX_LEN).to(DEVICE)
            if tokens.input_ids.shape[1] < 10:
                continue
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(**tokens)   # no labels — we compute CE ourselves
                logits = outputs.logits[:, :-1]
                labels = tokens.input_ids[:, 1:]
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    labels.reshape(-1), reduction='none')
                threshold = torch.quantile(ce, 1.0 - TOKEN_SELECT_RATIO)
                mask = (ce >= threshold).float()
                task_loss = (ce * mask).sum() / (mask.sum() + 1e-8)
                reg_loss = torch.tensor(0.0, device=DEVICE)
                for name, module in model.named_modules():
                    if isinstance(module, PackedBitLinear) and name in orig_scales:
                        if module.scales.requires_grad:
                            reg_loss = reg_loss + F.mse_loss(module.scales, orig_scales[name])
                loss = task_loss + REG_LAMBDA * reg_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scale_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += task_loss.item()
            n += 1
        log.info(f"    epoch {epoch+1}/{EPOCHS}  loss={epoch_loss/max(n,1):.4f}")


# ── GGUF patch ────────────────────────────────────────────────────────────────

def patch_gguf(combined_scales, output_path):
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

    name_to_scales = {map_name(k): v for k, v in combined_scales.items()}
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


# ── eval ──────────────────────────────────────────────────────────────────────

MMLU_SUBJECTS = ["high_school_world_history", "world_religions", "nutrition",
                 "global_facts", "miscellaneous", "sociology"]


def eval_gsm8k(ev, n=N_GSM8K_EVAL):
    from datasets import load_dataset
    import re
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


def eval_mmlu(ev, n=N_MMLU_EVAL):
    from datasets import load_dataset
    per_subj = max(1, n // len(MMLU_SUBJECTS))
    correct = total = 0
    for subj in MMLU_SUBJECTS:
        if total >= n:
            break
        try:
            ds = load_dataset("cais/mmlu", subj, split="test", streaming=True)
        except Exception:
            continue
        taken = 0
        for ex in ds:
            if taken >= per_subj or total >= n:
                break
            options = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(ex['choices']))
            resp = ev.generate(f"Question: {ex['question']}\n{options}\nAnswer:", max_tokens=3)
            resp = resp if isinstance(resp, str) else ""
            if resp.strip().upper()[:1] == chr(65 + ex['answer']):
                correct += 1
            total += 1
            taken += 1
    return correct / max(total, 1)


# ── region worker (runs in subprocess) ───────────────────────────────────────

def run_region(region):
    """Train one region, save its scale table. Called via subprocess."""
    log.info(f"\n{'='*55}")
    log.info(f"REGION: {region}")
    log.info(f"{'='*55}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from packed_bitlinear import PackedBitLinear, convert_model

    tokenizer = AutoTokenizer.from_pretrained(
        "prism-ml/Bonsai-1.7B-unpacked", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "prism-ml/Bonsai-1.7B-unpacked", dtype=torch.float16,
        trust_remote_code=True, device_map="cpu")
    convert_model(model)
    model = model.to(DEVICE)

    # Warm up the GPU driver before heavy training (avoids WDDM TDR on WSL2
    # when a fresh CUDA context starts immediately after a previous process exits)
    _dummy = torch.zeros(1024, 1024, device=DEVICE, dtype=torch.float16)
    (_dummy @ _dummy.T).sum().item()
    del _dummy
    torch.cuda.empty_cache()
    time.sleep(3)

    model.gradient_checkpointing_enable()

    orig_scales = {}
    for name, module in model.named_modules():
        if isinstance(module, PackedBitLinear):
            orig_scales[name] = module.scales.data.clone()

    total_scales = sum(p.numel() for p in orig_scales.values())

    scale_params = select_params(model, region)
    trainable = sum(p.numel() for p in scale_params)
    log.info(f"  Trainable: {trainable:,} / {total_scales:,} ({trainable/total_scales:.1%})")

    trained_names = [name for name, module in model.named_modules()
                     if isinstance(module, torch.nn.Module) and
                     hasattr(module, 'scales') and module.scales.requires_grad]
    layer_idxs = sorted(set(get_layer_idx(n) for n in trained_names) - {None})
    if layer_idxs:
        log.info(f"  Layers: {min(layer_idxs)}–{max(layer_idxs)}")

    if region == "early_math":
        texts = get_math_data(TRAIN_EXAMPLES)
        log.info(f"  Data: {len(texts)} GSM8K math examples")
    else:
        texts = get_knowledge_data(TRAIN_EXAMPLES)
        log.info(f"  Data: {len(texts)} TriviaQA knowledge examples")

    t0 = time.time()
    train(model, tokenizer, texts, orig_scales, scale_params)
    log.info(f"  Trained in {time.time()-t0:.0f}s")

    out = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Module) and hasattr(module, 'scales'):
            out[name] = module.scales.data.clone().cpu()

    save_path = f"{CKPT}/coop_{region}_scales.pt"
    torch.save(out, save_path)
    log.info(f"  Saved: {save_path}")


# ── combine + eval ────────────────────────────────────────────────────────────

def combine_and_eval():
    log.info(f"\n{'='*55}")
    log.info("COMBINING REGIONS + EVAL")
    log.info(f"{'='*55}")

    early_scales = torch.load(f"{CKPT}/coop_early_math_scales.pt", weights_only=False)
    late_scales  = torch.load(f"{CKPT}/coop_late_knowledge_scales.pt", weights_only=False)

    # Start from early_math scales (has all tensors), overlay late-ffn from late_knowledge
    combined = {k: v.clone() for k, v in early_scales.items()}
    swapped = 0
    for name in combined:
        if is_late_ffn(name) and name in late_scales:
            combined[name] = late_scales[name].clone()
            swapped += 1
    log.info(f"  Combined: {swapped} late-ffn tensors swapped to knowledge scales")

    n_patched = patch_gguf(combined, GGUF_OUT)
    log.info(f"  Patched {n_patched} tensors into GGUF")

    from llama_fast_eval import LlamaEval
    ev = LlamaEval(GGUF_OUT, n_ctx=1024)
    ev.start()
    gsm = eval_gsm8k(ev)
    mmlu = eval_mmlu(ev)
    ev.stop()

    log.info(f"\n{'='*55}")
    log.info("RESULTS — TWO-REGION COOPERATIVE TRAINING")
    log.info(f"{'='*55}")
    log.info(f"  GSM8K n={N_GSM8K_EVAL}:  {gsm:.1%}  (full-model math baseline: {FULL_MATH_BASELINE:.1%})")
    log.info(f"  MMLU  n={N_MMLU_EVAL}:   {mmlu:.1%}")
    log.info(f"")
    log.info(f"  Baselines for context:")
    log.info(f"    early_ffn only (math):  {EARLY_FFN_BASELINE:.1%} GSM8K")
    log.info(f"    full model (math):      {FULL_MATH_BASELINE:.1%} GSM8K")
    log.info(f"    flat_0.7 blend:         {FLAT_BLEND_BASELINE:.1%} GSM8K  (no training)")

    gsm_delta = gsm - FULL_MATH_BASELINE
    verdict = "EXCEEDS full-model" if gsm_delta > 0.02 else \
              "MATCHES full-model" if abs(gsm_delta) <= 0.02 else "BELOW full-model"
    log.info(f"\n  Verdict: {verdict} ({gsm_delta:+.1%})")

    results = {"gsm8k": gsm, "mmlu": mmlu, "gsm_delta_vs_full": gsm_delta}
    with open(f"{CKPT}/coop_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"  Saved: {CKPT}/coop_results.json")
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log.info("TWO-REGION COOPERATIVE SCALE TRAINING (Exp 17)")
    log.info("Hypothesis: early-ffn=math + late-ffn=knowledge > full-model math training")

    if REGION:
        # Running as subprocess for one region
        run_region(REGION)
        return

    # Orchestrator: spawn two subprocesses sequentially, then combine+eval
    python = sys.executable
    script = os.path.abspath(__file__)
    env = os.environ.copy()

    for region in ["early_math", "late_knowledge"]:
        log.info(f"\nLaunching subprocess for region: {region}")
        env["COOP_REGION"] = region
        result = subprocess.run(
            [python, script], env=env, capture_output=False)
        if result.returncode != 0:
            log.error(f"Subprocess failed for region {region} (exit {result.returncode})")
            return

    combine_and_eval()
    log.info(f"\nTotal: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"CRASHED: {e}", exc_info=True)
        sys.exit(1)
    finally:
        log.info("DONE")
