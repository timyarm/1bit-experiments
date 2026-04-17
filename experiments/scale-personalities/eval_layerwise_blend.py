"""Per-layer α blend: math ↔ knowledge scales with depth-varying α.

Hypothesis: the activation probe found late layers carry most signal
(ffn_up/gate 44.7% non-redundant, late layers only 2.1% redundant).
If math specialization lives in late layers, then:
  - Giving late layers α≈1 (math) + early layers α≈0 (knowledge) should
    BEAT the best flat α=0.7 on GSM8K.
  - The inverted profile (math early, knowledge late) should be WORSE.

Tests 5 profiles + 1 flat control, evaluates GSM8K(n=50) + MMLU(n=50) each.
Runtime: ~40 min local GPU. No new training — uses existing scale tables.

Profiles:
  flat_0.7      — uniform α=0.7 (interpolation curve winner, control)
  linear_late   — α ramps 0→1 from layer 0 → last (knowledge early, math late)
  linear_early  — α ramps 1→0 (math early, knowledge late) — inverted
  step_late     — bottom half α=0, top half α=1
  step_early    — bottom half α=1, top half α=0 (inverted step)
  ffn_late      — only ffn_up/gate in top-half layers get α=1; all else α=0.5
"""
import sys
import os
import time
import json
import logging
import shutil
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler("/tmp/layerblend_eval.log", mode='w'),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("layerblend")

sys.path.insert(0, os.path.dirname(__file__))
from llama_fast_eval import LlamaEval

CKPT = os.path.expanduser("~/freigent/apps/trucksim/checkpoints")
GGUF_DIR = os.path.expanduser("~/freigent/apps/trucksim/data/llm")
GGUF_BASE = f"{GGUF_DIR}/Bonsai-1.7B.gguf"
GGUF_OUT = f"{GGUF_DIR}/Bonsai-1.7B-layerblend.gguf"

N_LAYERS = 24   # Bonsai 1.7B
N_PER_BENCH = 50

FFN_KEYS = {"ffn_gate", "ffn_up", "ffn_down"}


def get_layer_idx(pt_name):
    """Extract layer index from param name. Returns None for non-layer params."""
    if "model.layers." not in pt_name:
        return None
    try:
        return int(pt_name.split("model.layers.")[1].split(".")[0])
    except (IndexError, ValueError):
        return None


def get_tensor_type(pt_name):
    """Return coarse tensor category for per-type α rules."""
    if "mlp.gate_proj" in pt_name or "mlp.up_proj" in pt_name:
        return "ffn_gate_up"
    if "mlp.down_proj" in pt_name:
        return "ffn_down"
    return "attn"


def build_profile_alphas(profile_name, math_scales):
    """Return dict: param_name → α value for blending math (α=1) vs knowledge (α=0)."""
    alphas = {}
    for name in math_scales:
        idx = get_layer_idx(name)

        if profile_name == "flat_0.7":
            alphas[name] = 0.7

        elif profile_name == "linear_late":
            # α ramps linearly from 0 (layer 0) to 1 (last layer)
            if idx is None:
                alphas[name] = 0.5
            else:
                alphas[name] = idx / (N_LAYERS - 1)

        elif profile_name == "linear_early":
            # Inverted: math early, knowledge late
            if idx is None:
                alphas[name] = 0.5
            else:
                alphas[name] = 1.0 - idx / (N_LAYERS - 1)

        elif profile_name == "step_late":
            # Bottom half knowledge (0), top half math (1)
            if idx is None:
                alphas[name] = 0.5
            else:
                alphas[name] = 1.0 if idx >= N_LAYERS // 2 else 0.0

        elif profile_name == "step_early":
            # Inverted step
            if idx is None:
                alphas[name] = 0.5
            else:
                alphas[name] = 0.0 if idx >= N_LAYERS // 2 else 1.0

        elif profile_name == "ffn_late":
            # Only ffn_up/gate in top-half layers get math scales; all else neutral
            if idx is None:
                alphas[name] = 0.5
            elif idx >= N_LAYERS // 2 and get_tensor_type(name) == "ffn_gate_up":
                alphas[name] = 1.0
            else:
                alphas[name] = 0.5

        else:
            alphas[name] = 0.5

    return alphas


def blend_scales(math_scales, knowledge_scales, alphas):
    """Apply per-param α blending."""
    blended = {}
    for name in math_scales:
        if name not in knowledge_scales:
            continue
        a = alphas.get(name, 0.5)
        blended[name] = a * math_scales[name] + (1 - a) * knowledge_scales[name]
    return blended


def patch_gguf(blended_scales, output_path):
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

    name_to_scales = {map_name(k): v for k, v in blended_scales.items()}
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


def eval_gsm8k(ev, n=N_PER_BENCH):
    from datasets import load_dataset
    import re
    ds = load_dataset("gsm8k", "main", split="test", streaming=True)
    correct = total = 0
    for ex in ds:
        if total >= n:
            break
        resp = ev.generate(f"Question: {ex['question']}\nAnswer:", max_tokens=200)
        resp = resp if isinstance(resp, str) else ""
        pred_nums = re.findall(r'-?\d+\.?\d*', resp.replace(',', ''))
        gold_nums = re.findall(r'-?\d+\.?\d*', ex['answer'].split('####')[-1].replace(',', ''))
        try:
            if pred_nums and gold_nums and abs(float(pred_nums[-1]) - float(gold_nums[-1])) < 0.01:
                correct += 1
        except ValueError:
            pass
        total += 1
    return correct / max(total, 1)


MMLU_SUBJECTS = ["high_school_world_history", "world_religions", "nutrition",
                 "global_facts", "miscellaneous", "sociology"]


def eval_mmlu(ev, n=N_PER_BENCH):
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
            resp_clean = resp.strip().upper()
            if resp_clean and resp_clean[0] == chr(65 + ex['answer']):
                correct += 1
            total += 1
            taken += 1
    return correct / max(total, 1)


PROFILES = ["flat_0.7", "linear_late", "linear_early", "step_late", "step_early", "ffn_late"]


def main():
    t0 = time.time()
    log.info("PER-LAYER BLEND SWEEP")
    log.info(f"Profiles: {PROFILES}")
    log.info(f"Eval: GSM8K(n={N_PER_BENCH}) + MMLU(n={N_PER_BENCH}) per profile")
    log.info(f"Hypothesis: late-layer math > flat α on GSM8K")

    math_scales = torch.load(f"{CKPT}/math_scales_v2.pt", weights_only=False)
    knowledge_scales = torch.load(f"{CKPT}/knowledge_scales_v2.pt", weights_only=False)
    log.info(f"Loaded: math={len(math_scales)} layers, knowledge={len(knowledge_scales)} layers")

    # Spot-check α distribution for linear_late
    alphas_sample = build_profile_alphas("linear_late", math_scales)
    layer_alphas = [(get_layer_idx(k), v) for k, v in alphas_sample.items() if get_layer_idx(k) is not None]
    by_layer = {}
    for li, a in layer_alphas:
        by_layer.setdefault(li, []).append(a)
    log.info("linear_late α by layer (mean): " +
             " ".join(f"L{li}:{sum(v)/len(v):.2f}" for li, v in sorted(by_layer.items())[::4]))

    results = {}
    for profile in PROFILES:
        log.info(f"\n--- {profile} ---")
        alphas = build_profile_alphas(profile, math_scales)
        blended = blend_scales(math_scales, knowledge_scales, alphas)

        n_patched = patch_gguf(blended, GGUF_OUT)
        log.info(f"  patched {n_patched} tensors")

        ev = LlamaEval(GGUF_OUT, n_ctx=1024)
        ev.start()
        gsm = eval_gsm8k(ev)
        mmlu = eval_mmlu(ev)
        ev.stop()
        time.sleep(1)

        log.info(f"  GSM8K={gsm:.1%}  MMLU={mmlu:.1%}")
        results[profile] = {"gsm8k": gsm, "mmlu": mmlu}

    # Summary table
    log.info(f"\n{'='*55}")
    log.info("PER-LAYER BLEND RESULTS")
    log.info(f"{'='*55}")
    log.info(f"{'Profile':<18}{'GSM8K':<10}{'MMLU':<10}{'Avg':<10}")
    for p, r in results.items():
        avg = (r['gsm8k'] + r['mmlu']) / 2
        marker = " ← WINS" if r['gsm8k'] == max(v['gsm8k'] for v in results.values()) else ""
        log.info(f"{p:<18}{r['gsm8k']:<10.1%}{r['mmlu']:<10.1%}{avg:<10.1%}{marker}")

    flat = results.get("flat_0.7", {})
    late = results.get("linear_late", {})
    if flat and late:
        delta = late['gsm8k'] - flat['gsm8k']
        log.info(f"\nlinear_late vs flat_0.7: GSM8K {delta:+.1%}")
        if delta > 0.02:
            log.info("  → DEPTH STRUCTURE CONFIRMED: late layers carry math")
        elif delta < -0.02:
            log.info("  → DEPTH STRUCTURE INVERTED: early layers carry math")
        else:
            log.info("  → FLAT: scale specialization does not vary by depth")

    out_path = f"{CKPT}/layerblend_results.json"
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
