"""Forensic analysis: compare Bonsai-1.7B-unpacked vs Qwen3-1.7B base.

Reproduces Archie Sengupta's reverse-engineered metrics:
1. bit_match — fraction of signs matching base (target: 0.70-0.79)
2. ratio_med — scale inflation vs base mean-abs (target: ffn_up=3.53x at 1.7B)
3. adv_ratio_med — adjusted ratio (target: ffn_up=2.34 at 1.7B)
4. kurtosis — weight distribution shape (target: -1.8 to -1.9 for FFN)
5. scale_std — scale variability (should compress with model size)
6. depth-scale correlation — scales should grow with layer depth

Ground truth: Bonsai IS the working 1-bit model. We measure it directly.
"""
import modal
import json

app = modal.App("1bit-bonsai-forensics")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.51.0",
        "accelerate>=1.5.0",
        "huggingface_hub>=0.28.0",
        "safetensors>=0.4.0",
        "scipy>=1.12.0",
    )
)

vol = modal.Volume.from_name("1bit-checkpoints", create_if_missing=True)

# Archie's category classification for Qwen architecture
def classify_layer(name: str) -> str:
    """Classify a linear layer into Archie's category hierarchy."""
    if "embed" in name or "lm_head" in name:
        return "token_embd"
    if "q_proj" in name:
        return "attn_q"
    if "k_proj" in name:
        return "attn_k"
    if "v_proj" in name:
        return "attn_v"
    if "o_proj" in name:
        return "attn_o"
    if "up_proj" in name:
        return "ffn_up"
    if "gate_proj" in name:
        return "ffn_gate"
    if "down_proj" in name:
        return "ffn_down"
    return "other"


def get_layer_depth(name: str) -> int:
    """Extract layer index from name like 'model.layers.15.mlp.up_proj'."""
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return -1


@app.function(
    image=image,
    gpu="T4:1",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=32768,
    volumes={"/data": vol},
)
def run_forensics():
    import torch
    import os
    import numpy as np
    from scipy import stats as scipy_stats
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM
    from collections import defaultdict

    hf_token = os.environ.get("HF_TOKEN")
    device = "cpu"  # Analysis only, no GPU needed for this
    group_size = 128

    # ══════════════════════════════════════════════════════════
    # Load Bonsai-1.7B-unpacked (safetensors)
    # ══════════════════════════════════════════════════════════
    print("=" * 70)
    print("Loading Bonsai-1.7B-unpacked from HuggingFace...")
    print("=" * 70, flush=True)

    from huggingface_hub import hf_hub_download, list_repo_files

    bonsai_repo = "prism-ml/Bonsai-1.7B-unpacked"

    # List files in the repo to find safetensors
    files = list_repo_files(bonsai_repo, token=hf_token)
    safetensor_files = [f for f in files if f.endswith(".safetensors")]
    print(f"  Found {len(safetensor_files)} safetensor files: {safetensor_files}", flush=True)

    # Download and load all safetensor files
    bonsai_tensors = {}
    for sf_file in safetensor_files:
        path = hf_hub_download(bonsai_repo, sf_file, token=hf_token)
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                bonsai_tensors[key] = f.get_tensor(key)
    print(f"  Loaded {len(bonsai_tensors)} tensors from Bonsai", flush=True)

    # Print some tensor names to understand structure
    print("\n  Sample Bonsai tensor names:")
    for i, name in enumerate(sorted(bonsai_tensors.keys())):
        if i < 30:
            t = bonsai_tensors[name]
            print(f"    {name}: shape={list(t.shape)}, dtype={t.dtype}, "
                  f"min={t.min().item():.4f}, max={t.max().item():.4f}, "
                  f"unique={min(t.unique().numel(), 10)}", flush=True)

    # ══════════════════════════════════════════════════════════
    # Load Qwen3-1.7B base (teacher)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Loading Qwen3-1.7B base teacher...")
    print("=" * 70, flush=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        torch_dtype=torch.float32,
        device_map="cpu",
        token=hf_token,
    )

    base_tensors = {}
    for name, param in base_model.named_parameters():
        base_tensors[name] = param.data.clone()
    print(f"  Loaded {len(base_tensors)} parameters from Qwen3-1.7B", flush=True)

    # Print base tensor names for mapping
    print("\n  Sample base tensor names:")
    for i, name in enumerate(sorted(base_tensors.keys())):
        if i < 30:
            t = base_tensors[name]
            print(f"    {name}: shape={list(t.shape)}, dtype={t.dtype}", flush=True)

    # Free the model
    del base_model
    import gc
    gc.collect()

    # ══════════════════════════════════════════════════════════
    # Map tensor names between Bonsai and Qwen3
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Mapping tensor names between Bonsai and Qwen3...")
    print("=" * 70, flush=True)

    # Find matching tensors by shape and name similarity
    matched = {}
    unmatched_bonsai = []

    for bname in sorted(bonsai_tensors.keys()):
        bt = bonsai_tensors[bname]
        # Try direct name match first
        if bname in base_tensors and base_tensors[bname].shape == bt.shape:
            matched[bname] = bname
            continue

        # Try common transformations
        found = False
        for qname in base_tensors:
            if base_tensors[qname].shape == bt.shape:
                # Check if names are similar (same layer structure)
                bparts = bname.replace("model.", "").split(".")
                qparts = qname.replace("model.", "").split(".")
                if bparts == qparts:
                    matched[bname] = qname
                    found = True
                    break
        if not found:
            # Shape match as fallback
            for qname in base_tensors:
                if base_tensors[qname].shape == bt.shape and qname not in matched.values():
                    # Only match weight tensors to weight tensors
                    if ("weight" in bname) == ("weight" in qname):
                        if get_layer_depth(bname) == get_layer_depth(qname):
                            if classify_layer(bname) == classify_layer(qname):
                                matched[bname] = qname
                                found = True
                                break
        if not found:
            unmatched_bonsai.append(bname)

    print(f"  Matched: {len(matched)} tensors")
    print(f"  Unmatched Bonsai tensors: {len(unmatched_bonsai)}")
    if unmatched_bonsai:
        for n in unmatched_bonsai[:10]:
            print(f"    {n}: shape={list(bonsai_tensors[n].shape)}")

    # ══════════════════════════════════════════════════════════
    # Compute forensic metrics per tensor
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Computing Archie's forensic metrics...")
    print("=" * 70, flush=True)

    results = {}
    category_metrics = defaultdict(lambda: {
        "bit_match": [], "ratio_med": [], "adv_ratio_med": [],
        "kurtosis": [], "scale_std": [], "depths": [], "scale_means": [],
        "count": 0,
    })

    for bname, qname in sorted(matched.items()):
        bt = bonsai_tensors[bname].float()
        qt = base_tensors[qname].float()

        # Skip non-weight tensors (biases, norms, etc.)
        if bt.dim() < 2:
            continue

        cat = classify_layer(bname)
        depth = get_layer_depth(bname)

        # ── Decompose Bonsai into signs and scales ──
        flat_b = bt.reshape(-1)
        flat_q = qt.reshape(-1)

        gs = group_size
        rem = flat_b.numel() % gs
        if rem:
            flat_b = torch.nn.functional.pad(flat_b, (0, gs - rem))
            flat_q = torch.nn.functional.pad(flat_q, (0, gs - rem))

        groups_b = flat_b.reshape(-1, gs)
        groups_q = flat_q.reshape(-1, gs)

        # Bonsai signs (the actual binary weights)
        bonsai_signs = groups_b.sign()
        bonsai_signs = torch.where(bonsai_signs == 0, torch.ones_like(bonsai_signs), bonsai_signs)

        # Bonsai scales (mean absolute value per group)
        bonsai_scales = groups_b.abs().mean(dim=1)

        # Base signs
        base_signs = groups_q.sign()
        base_signs = torch.where(base_signs == 0, torch.ones_like(base_signs), base_signs)

        # Base scales (mean absolute value per group)
        base_scales = groups_q.abs().mean(dim=1)

        # ── Metric 1: bit_match (fraction of signs matching base) ──
        sign_match = (bonsai_signs == base_signs).float().mean().item()

        # ── Metric 2: ratio_med (scale inflation) ──
        # ratio = bonsai_scale / base_scale per group
        safe_base = base_scales.clamp(min=1e-8)
        ratios = bonsai_scales / safe_base
        ratio_med = ratios.median().item()

        # ── Metric 3: adv_ratio_med ──
        # Only compute for groups where signs were flipped
        flipped_mask = (bonsai_signs != base_signs).any(dim=1)
        if flipped_mask.sum() > 0:
            adv_ratios = ratios[flipped_mask]
            adv_ratio_med = adv_ratios.median().item()
        else:
            adv_ratio_med = ratio_med

        # ── Metric 4: kurtosis ──
        # Kurtosis of the bonsai weight values
        flat_vals = bt.reshape(-1).numpy()
        if len(flat_vals) > 100:
            kurt = float(scipy_stats.kurtosis(flat_vals, fisher=True))
        else:
            kurt = 0.0

        # ── Metric 5: scale_std ──
        scale_std = bonsai_scales.std().item()

        # ── Metric 6: depth-scale mean ──
        scale_mean = bonsai_scales.mean().item()

        results[bname] = {
            "category": cat,
            "depth": depth,
            "shape": list(bt.shape),
            "bit_match": round(sign_match, 4),
            "ratio_med": round(ratio_med, 4),
            "adv_ratio_med": round(adv_ratio_med, 4),
            "kurtosis": round(kurt, 4),
            "scale_std": round(scale_std, 6),
            "scale_mean": round(scale_mean, 6),
            "n_groups": int(groups_b.shape[0]),
            "pct_flipped": round(1.0 - sign_match, 4),
        }

        # Accumulate per-category
        cm = category_metrics[cat]
        cm["bit_match"].append(sign_match)
        cm["ratio_med"].append(ratio_med)
        cm["adv_ratio_med"].append(adv_ratio_med)
        cm["kurtosis"].append(kurt)
        cm["scale_std"].append(scale_std)
        cm["depths"].append(depth)
        cm["scale_means"].append(scale_mean)
        cm["count"] += 1

    # ══════════════════════════════════════════════════════════
    # Print per-category summary (Archie's format)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PER-CATEGORY FORENSIC SUMMARY")
    print("=" * 70)

    cat_order = ["ffn_up", "ffn_gate", "attn_v", "ffn_down", "attn_o", "attn_k", "attn_q", "token_embd", "other"]

    print(f"\n{'Category':<12} {'N':>4} {'bit_match':>10} {'ratio_med':>10} {'adv_ratio':>10} "
          f"{'kurtosis':>10} {'scale_std':>10} {'%flipped':>10}")
    print("-" * 82)

    cat_summary = {}
    for cat in cat_order:
        if cat not in category_metrics:
            continue
        cm = category_metrics[cat]
        n = cm["count"]
        bm = np.median(cm["bit_match"])
        rm = np.median(cm["ratio_med"])
        ar = np.median(cm["adv_ratio_med"])
        ku = np.median(cm["kurtosis"])
        ss = np.median(cm["scale_std"])
        flipped = 1.0 - bm

        print(f"{cat:<12} {n:>4} {bm:>10.4f} {rm:>10.4f} {ar:>10.4f} "
              f"{ku:>10.4f} {ss:>10.6f} {flipped:>10.4f}")

        cat_summary[cat] = {
            "count": n,
            "bit_match_median": round(float(bm), 4),
            "ratio_med_median": round(float(rm), 4),
            "adv_ratio_med_median": round(float(ar), 4),
            "kurtosis_median": round(float(ku), 4),
            "scale_std_median": round(float(ss), 6),
            "pct_flipped_median": round(float(flipped), 4),
        }

    # ══════════════════════════════════════════════════════════
    # Depth-scale correlation (Archie's metric 6)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("DEPTH-SCALE CORRELATION")
    print("=" * 70)

    print(f"\n{'Category':<12} {'Correlation':>12} {'p-value':>12}")
    print("-" * 40)

    depth_corr = {}
    for cat in cat_order:
        if cat not in category_metrics:
            continue
        cm = category_metrics[cat]
        if len(cm["depths"]) < 3:
            continue
        depths = np.array(cm["depths"])
        scales = np.array(cm["scale_means"])
        if depths.std() > 0 and scales.std() > 0:
            corr, pval = scipy_stats.pearsonr(depths, scales)
            print(f"{cat:<12} {corr:>12.4f} {pval:>12.6f}")
            depth_corr[cat] = {"correlation": round(corr, 4), "p_value": round(pval, 6)}
        else:
            print(f"{cat:<12} {'N/A':>12} {'N/A':>12}")

    # ══════════════════════════════════════════════════════════
    # Compare with Archie's published targets
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("COMPARISON WITH ARCHIE'S PUBLISHED TARGETS (1.7B)")
    print("=" * 70)

    archie_targets = {
        "ffn_up": {"ratio_med": 3.53, "adv_ratio_med": 2.34, "depth_corr": 0.877},
        "ffn_gate": {"ratio_med": None, "adv_ratio_med": None},
        "ffn_down": {"depth_corr": 0.922},
        "bit_match_range": (0.70, 0.79),
        "kurtosis_ffn": (-1.9, -1.8),
    }

    print(f"\n{'Metric':<30} {'Archie Target':>15} {'Our Measurement':>15} {'Match?':>8}")
    print("-" * 72)

    # bit_match range
    if "ffn_up" in cat_summary:
        bm = cat_summary["ffn_up"]["bit_match_median"]
        target = "0.70-0.79"
        match = "YES" if 0.70 <= bm <= 0.79 else "NO"
        print(f"{'bit_match (ffn_up)':<30} {target:>15} {bm:>15.4f} {match:>8}")

    # ratio_med ffn_up
    if "ffn_up" in cat_summary:
        rm = cat_summary["ffn_up"]["ratio_med_median"]
        print(f"{'ratio_med (ffn_up)':<30} {'3.53':>15} {rm:>15.4f} {'~' if abs(rm - 3.53) < 1.0 else 'NO':>8}")

    # adv_ratio_med ffn_up
    if "ffn_up" in cat_summary:
        ar = cat_summary["ffn_up"]["adv_ratio_med_median"]
        print(f"{'adv_ratio_med (ffn_up)':<30} {'2.34':>15} {ar:>15.4f} {'~' if abs(ar - 2.34) < 1.0 else 'NO':>8}")

    # kurtosis FFN
    for cat in ["ffn_up", "ffn_gate", "ffn_down"]:
        if cat in cat_summary:
            ku = cat_summary[cat]["kurtosis_median"]
            print(f"{'kurtosis (' + cat + ')':<30} {'-1.8 to -1.9':>15} {ku:>15.4f} "
                  f"{'YES' if -2.0 <= ku <= -1.7 else 'NO':>8}")

    # depth-scale correlation
    if "ffn_up" in depth_corr:
        dc = depth_corr["ffn_up"]["correlation"]
        print(f"{'depth-scale corr (ffn_up)':<30} {'0.877':>15} {dc:>15.4f} "
              f"{'YES' if abs(dc - 0.877) < 0.1 else '~':>8}")
    if "ffn_down" in depth_corr:
        dc = depth_corr["ffn_down"]["correlation"]
        print(f"{'depth-scale corr (ffn_down)':<30} {'0.922':>15} {dc:>15.4f} "
              f"{'YES' if abs(dc - 0.922) < 0.1 else '~':>8}")

    # ══════════════════════════════════════════════════════════
    # Optimization pressure hierarchy check
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("OPTIMIZATION PRESSURE HIERARCHY")
    print("Archie's order: ffn_up > ffn_gate > attn_v > ffn_down > attn_o > attn_k > attn_q > token_embd")
    print("=" * 70)

    # Pressure = amount of deviation from base = 1 - bit_match + ratio_med inflation
    pressure = {}
    for cat in cat_order:
        if cat in cat_summary:
            cs = cat_summary[cat]
            # Combined pressure metric: sign flip rate + scale inflation
            p = cs["pct_flipped_median"] * 10 + max(0, cs["ratio_med_median"] - 1.0)
            pressure[cat] = p

    sorted_pressure = sorted(pressure.items(), key=lambda x: -x[1])
    print(f"\n{'Rank':<6} {'Category':<12} {'Pressure':>10} {'%Flipped':>10} {'ScaleInflation':>15}")
    print("-" * 55)
    for i, (cat, p) in enumerate(sorted_pressure):
        cs = cat_summary[cat]
        print(f"{i+1:<6} {cat:<12} {p:>10.4f} {cs['pct_flipped_median']:>10.4f} "
              f"{max(0, cs['ratio_med_median'] - 1.0):>15.4f}")

    # ══════════════════════════════════════════════════════════
    # Per-layer detail (first 5 layers)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PER-LAYER DETAIL (layers 0-4)")
    print("=" * 70)

    print(f"\n{'Layer':<45} {'bit_match':>10} {'ratio_med':>10} {'kurtosis':>10} {'scale_mean':>10}")
    print("-" * 90)
    for bname in sorted(results.keys()):
        r = results[bname]
        if r["depth"] <= 4 and r["depth"] >= 0:
            short = bname.replace("model.", "")
            print(f"{short:<45} {r['bit_match']:>10.4f} {r['ratio_med']:>10.4f} "
                  f"{r['kurtosis']:>10.4f} {r['scale_mean']:>10.6f}")

    # ══════════════════════════════════════════════════════════
    # Check if Bonsai is truly binary
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("BINARY CHECK: Are Bonsai weights truly {-scale, +scale}?")
    print("=" * 70)

    binary_check = {}
    for bname in sorted(bonsai_tensors.keys()):
        bt = bonsai_tensors[bname].float()
        if bt.dim() < 2:
            continue
        flat = bt.reshape(-1)
        gs = group_size
        rem = flat.numel() % gs
        if rem:
            flat = torch.nn.functional.pad(flat, (0, gs - rem))
        groups = flat.reshape(-1, gs)

        # For each group, check if all values are ±scale
        scales_per_group = groups.abs().mean(dim=1, keepdim=True)
        normalized = groups / scales_per_group.clamp(min=1e-8)
        # If truly binary, normalized values should be very close to ±1
        deviation = (normalized.abs() - 1.0).abs()
        max_dev = deviation.max().item()
        mean_dev = deviation.mean().item()

        cat = classify_layer(bname)
        depth = get_layer_depth(bname)
        if depth <= 2 and depth >= 0:
            print(f"  {bname}: max_dev={max_dev:.6f}, mean_dev={mean_dev:.6f} "
                  f"{'BINARY' if max_dev < 0.01 else 'NOT BINARY'}")

        binary_check[bname] = {
            "max_deviation": max_dev,
            "mean_deviation": mean_dev,
            "is_binary": max_dev < 0.01,
        }

    n_binary = sum(1 for v in binary_check.values() if v["is_binary"])
    n_total = len(binary_check)
    print(f"\n  Binary layers: {n_binary}/{n_total} "
          f"({'ALL BINARY' if n_binary == n_total else 'MIXED'})")

    # ══════════════════════════════════════════════════════════
    # Save full results
    # ══════════════════════════════════════════════════════════
    output = {
        "model": "prism-ml/Bonsai-1.7B-unpacked",
        "base": "Qwen/Qwen3-1.7B",
        "group_size": group_size,
        "per_tensor": results,
        "per_category": cat_summary,
        "depth_correlation": depth_corr,
        "binary_check_summary": {
            "n_binary": n_binary,
            "n_total": n_total,
            "all_binary": n_binary == n_total,
        },
    }

    save_path = "/data/bonsai_1.7b_forensics.json"
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to {save_path}")
    vol.commit()

    print(f"\n{'=' * 70}")
    print("FORENSIC ANALYSIS COMPLETE")
    print(f"{'=' * 70}", flush=True)

    return output


@app.local_entrypoint()
def main():
    print("Bonsai-1.7B Forensic Analysis\n")
    result = run_forensics.remote()
    # Print summary
    print("\n\nSUMMARY (returned):")
    if result and "per_category" in result:
        for cat, metrics in result["per_category"].items():
            print(f"  {cat}: bit_match={metrics['bit_match_median']:.4f}, "
                  f"ratio_med={metrics['ratio_med_median']:.4f}, "
                  f"kurtosis={metrics['kurtosis_median']:.4f}")
