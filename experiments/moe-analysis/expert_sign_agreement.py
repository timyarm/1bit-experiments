"""MoE → 1-bit Delta Encoding: Convert GLM-4.7-Flash MoE to our architecture.

Streams model layer by layer. Never loads full model into memory.

Output:
  - consensus_signs/   (packed int32, one file per layer)
  - expert_deltas/     (sparse XOR masks, per expert per layer)
  - expert_scales/     (fp16 group scales, per expert per layer)
  - shared_layers/     (attention Q/K/V/O + norms + embeddings in 1-bit)
  - router_weights/    (fp16, kept as-is)
  - metadata.json      (architecture info, expert mapping)
"""

import os
import sys
import json
import time
import numpy as np
import torch
from pathlib import Path
from safetensors import safe_open


def load_tensor(shard_path, key):
    """Load single tensor from safetensors shard, convert bf16→float32→numpy."""
    with safe_open(shard_path, framework="pt") as f:
        if key in f.keys():
            return f.get_tensor(key).float().numpy()
    return None


def get_keys(shard_path):
    """Get all tensor names from a shard."""
    with safe_open(shard_path, framework="pt") as f:
        return list(f.keys())

# Output directory
OUTPUT_DIR = Path("./data/glm47flash_1bit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "consensus_signs").mkdir(exist_ok=True)
(OUTPUT_DIR / "expert_deltas").mkdir(exist_ok=True)
(OUTPUT_DIR / "expert_scales").mkdir(exist_ok=True)
(OUTPUT_DIR / "shared_layers").mkdir(exist_ok=True)
(OUTPUT_DIR / "router_weights").mkdir(exist_ok=True)

MODEL_ID = "zai-org/GLM-4.7-Flash"
GROUP_SIZE = 128
HF_TOKEN = os.environ.get("HF_TOKEN")

def pack_signs_np(weights):
    """Pack sign bits into int32 arrays using numpy."""
    signs = (weights > 0).astype(np.int32)
    out_features, in_features = signs.shape
    pad = (32 - in_features % 32) % 32
    if pad > 0:
        signs = np.pad(signs, ((0, 0), (0, pad)))
    n_ints = signs.shape[1] // 32
    signs = signs.reshape(out_features, n_ints, 32)
    packed = np.zeros((out_features, n_ints), dtype=np.int32)
    for bit in range(32):
        packed |= signs[:, :, bit] << bit
    return packed


def compute_group_scales_np(weights, group_size=128):
    """Compute group-wise mean absolute value as scales."""
    out_features, in_features = weights.shape
    pad = (group_size - in_features % group_size) % group_size
    if pad > 0:
        w = np.pad(np.abs(weights), ((0, 0), (0, pad)))
    else:
        w = np.abs(weights)
    n_groups = w.shape[1] // group_size
    return w.reshape(out_features, n_groups, group_size).mean(axis=2).astype(np.float16)


def sparse_delta(consensus_packed, expert_packed):
    """Compute sparse XOR delta between consensus and expert signs."""
    xor = consensus_packed ^ expert_packed
    # Count actual BITS that differ (not int32 words)
    total_bits = consensus_packed.size * 32
    differing_bits = 0
    temp = xor.copy()
    for _ in range(32):
        differing_bits += (temp & 1).sum()
        temp >>= 1
    bit_agreement = 1.0 - float(differing_bits) / total_bits

    # Find non-zero int32 words for sparse storage
    nonzero_mask = xor != 0
    if not nonzero_mask.any():
        return {"indices": [], "values": [], "shape": consensus_packed.shape,
                "nnz": 0, "bit_agreement": 1.0, "bits_differing": 0}

    indices = np.argwhere(nonzero_mask)
    values = xor[nonzero_mask]
    return {
        "indices": indices.tolist(),
        "values": values.tolist(),
        "shape": list(consensus_packed.shape),
        "nnz": len(values),
        "bit_agreement": bit_agreement,
        "bits_differing": int(differing_bits),
        "total_bits": total_bits,
    }


def main():
    print("=" * 70)
    print("MoE → 1-bit Delta Encoding: GLM-4.7-Flash")
    print("=" * 70, flush=True)

    t_start = time.time()

    # Use safetensors index to discover architecture
    print("\n[1/4] Fetching model index...", flush=True)

    try:
        from huggingface_hub import hf_hub_download, HfApi

        # Download config first
        config_path = hf_hub_download(MODEL_ID, "config.json", token=HF_TOKEN)
        with open(config_path) as f:
            config = json.load(f)

        print(f"  Model config loaded:", flush=True)
        print(f"    hidden_size: {config.get('hidden_size', '?')}", flush=True)
        print(f"    num_layers: {config.get('num_hidden_layers', '?')}", flush=True)
        print(f"    num_experts: {config.get('num_experts', config.get('n_routed_experts', '?'))}", flush=True)
        print(f"    num_experts_per_tok: {config.get('num_experts_per_tok', config.get('num_selected_experts', '?'))}", flush=True)
        print(f"    intermediate_size: {config.get('intermediate_size', '?')}", flush=True)
        print(f"    moe_intermediate_size: {config.get('moe_intermediate_size', '?')}", flush=True)

        # Save config
        with open(OUTPUT_DIR / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Download safetensors index
        try:
            index_path = hf_hub_download(MODEL_ID, "model.safetensors.index.json", token=HF_TOKEN)
            with open(index_path) as f:
                index = json.load(f)
            weight_map = index["weight_map"]
            total_size = index.get("metadata", {}).get("total_size", "unknown")
            print(f"  Total model size: {total_size}", flush=True)
        except Exception:
            print("  No index file — model might be single safetensors", flush=True)
            weight_map = None

    except Exception as e:
        print(f"  Error fetching config: {e}", flush=True)
        sys.exit(1)

    # Discover model structure from weight names
    print("\n[2/4] Analyzing model structure...", flush=True)

    if weight_map:
        all_keys = list(weight_map.keys())
    else:
        # Single file — load and get keys
        model_path = hf_hub_download(MODEL_ID, "model.safetensors", token=HF_TOKEN)
        all_keys = get_keys(model_path)

    # Categorize weights
    expert_keys = [k for k in all_keys if "experts" in k]
    router_keys = [k for k in all_keys if "gate" in k or "router" in k or "e_score" in k]
    attention_keys = [k for k in all_keys if any(x in k for x in ["q_proj", "k_proj", "v_proj", "o_proj"])]
    norm_keys = [k for k in all_keys if "norm" in k or "layernorm" in k]
    embed_keys = [k for k in all_keys if "embed" in k]
    other_keys = [k for k in all_keys if k not in expert_keys + router_keys + attention_keys + norm_keys + embed_keys]

    print(f"  Expert weights: {len(expert_keys)}", flush=True)
    print(f"  Router weights: {len(router_keys)}", flush=True)
    print(f"  Attention weights: {len(attention_keys)}", flush=True)
    print(f"  Norm weights: {len(norm_keys)}", flush=True)
    print(f"  Embedding weights: {len(embed_keys)}", flush=True)
    print(f"  Other weights: {len(other_keys)}", flush=True)

    # Parse layer/expert structure
    n_layers = config.get("num_hidden_layers", 0)
    n_experts = config.get("num_experts", config.get("n_routed_experts", 0))

    # Find expert weight pattern
    sample_expert_keys = [k for k in expert_keys if ".0." in k or "experts.0" in k][:5]
    print(f"\n  Sample expert keys: {sample_expert_keys[:3]}", flush=True)

    print(f"\n  Architecture: {n_layers} layers, {n_experts} experts", flush=True)

    # Save metadata
    metadata = {
        "model_id": MODEL_ID,
        "n_layers": n_layers,
        "n_experts": n_experts,
        "hidden_size": config.get("hidden_size"),
        "group_size": GROUP_SIZE,
        "expert_keys_sample": sample_expert_keys[:5],
        "router_keys": router_keys[:5],
        "n_expert_weights": len(expert_keys),
        "n_attention_weights": len(attention_keys),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # [3/4] Process shared layers (attention, norms, embeddings)
    print("\n[3/4] Processing shared layers...", flush=True)

    from huggingface_hub import hf_hub_download
    shared_keys = attention_keys + norm_keys + embed_keys
    files_needed = set()
    if weight_map:
        for k in shared_keys + router_keys:
            if k in weight_map:
                files_needed.add(weight_map[k])

    shared_stats = {"total_params": 0, "quantized_params": 0, "fp16_params": 0}

    # Process shared weights file by file
    processed_files = set()
    for shard_file in sorted(files_needed) if files_needed else ["model.safetensors"]:
        if shard_file in processed_files:
            continue
        processed_files.add(shard_file)

        print(f"  Loading shard: {shard_file}...", flush=True)
        try:
            shard_path = hf_hub_download(MODEL_ID, shard_file, token=HF_TOKEN)
        except Exception as e:
            print(f"    WARN: Failed to download {shard_file}: {e}", flush=True)
            continue

        shard_keys = get_keys(shard_path)

        for key in shard_keys:
            # Router weights — save as-is in fp16
            if key in router_keys:
                tensor = load_tensor(shard_path, key)
                if tensor is None: continue
                save_path = OUTPUT_DIR / "router_weights" / f"{key.replace('/', '_')}.npy"
                np.save(save_path, tensor)
                shared_stats["fp16_params"] += tensor.size
                continue

            # Shared layers (attention, norms, embeddings)
            if key in shared_keys:
                tensor = load_tensor(shard_path, key)
                if tensor is None: continue

                if tensor.ndim == 2 and tensor.shape[0] > 1 and tensor.shape[1] > 1:
                    # Quantize 2D weight matrices to 1-bit
                    packed = pack_signs_np(tensor.astype(np.float32))
                    scales = compute_group_scales_np(tensor.astype(np.float32), GROUP_SIZE)
                    save_path = OUTPUT_DIR / "shared_layers" / f"{key.replace('/', '_')}"
                    np.save(f"{save_path}_signs.npy", packed)
                    np.save(f"{save_path}_scales.npy", scales)
                    shared_stats["quantized_params"] += tensor.size
                else:
                    # Keep 1D (biases, norms) in fp16
                    save_path = OUTPUT_DIR / "shared_layers" / f"{key.replace('/', '_')}.npy"
                    np.save(save_path, tensor)
                    shared_stats["fp16_params"] += tensor.size

                shared_stats["total_params"] += tensor.size

    print(f"  Shared layers: {shared_stats['total_params']/1e6:.1f}M params", flush=True)
    print(f"    Quantized to 1-bit: {shared_stats['quantized_params']/1e6:.1f}M", flush=True)
    print(f"    Kept fp16: {shared_stats['fp16_params']/1e6:.1f}M", flush=True)

    # [4/4] Process MoE experts — the main event
    print("\n[4/4] Processing MoE experts (layer by layer)...", flush=True)

    # Figure out which shard files contain expert weights
    expert_files = set()
    if weight_map:
        for k in expert_keys:
            expert_files.add(weight_map[k])

    # Group expert keys by layer
    layer_expert_keys = {}
    for k in expert_keys:
        # Parse layer number from key like "model.layers.0.block_sparse_moe.experts.0.w1.weight"
        parts = k.split(".")
        layer_idx = None
        expert_idx = None
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try: layer_idx = int(parts[i + 1])
                except: pass
            if p == "experts" and i + 1 < len(parts):
                try: expert_idx = int(parts[i + 1])
                except: pass

        if layer_idx is not None:
            if layer_idx not in layer_expert_keys:
                layer_expert_keys[layer_idx] = {}
            if expert_idx is not None:
                if expert_idx not in layer_expert_keys[layer_idx]:
                    layer_expert_keys[layer_idx][expert_idx] = []
                layer_expert_keys[layer_idx][expert_idx].append(k)

    print(f"  Found {len(layer_expert_keys)} MoE layers", flush=True)
    if layer_expert_keys:
        first_layer = min(layer_expert_keys.keys())
        n_experts_found = len(layer_expert_keys[first_layer])
        print(f"  Experts per layer: {n_experts_found}", flush=True)

    total_expert_params = 0
    total_delta_size = 0
    total_agreement = 0
    n_layers_processed = 0

    # Process each MoE layer
    for layer_idx in sorted(layer_expert_keys.keys()):
        layer_t = time.time()
        experts_data = layer_expert_keys[layer_idx]
        n_exp = len(experts_data)

        print(f"\n  Layer {layer_idx} ({n_exp} experts)...", flush=True)

        # For each weight matrix type (w1, w2, w3), process all experts
        # Collect weight matrix names from first expert
        first_expert_keys = experts_data[min(experts_data.keys())]
        weight_types = set()
        for k in first_expert_keys:
            # Extract the weight type (e.g., "w1.weight", "w2.weight")
            parts = k.split(".")
            for i, p in enumerate(parts):
                if p in ["w1", "w2", "w3", "gate_proj", "up_proj", "down_proj"]:
                    wtype = p
                    if i + 1 < len(parts) and parts[i+1] == "weight":
                        wtype += ".weight"
                    weight_types.add(wtype)

        for wtype in sorted(weight_types):
            # Find the key for this weight type across all experts
            expert_packed_signs = {}
            expert_scales = {}

            for exp_idx in sorted(experts_data.keys()):
                # Find the matching key
                matching_key = None
                for k in experts_data[exp_idx]:
                    if wtype.split(".")[0] in k and (wtype.endswith(".weight") == k.endswith("weight")):
                        matching_key = k
                        break

                if matching_key is None:
                    continue

                # Load this expert's weight
                if weight_map and matching_key in weight_map:
                    shard = weight_map[matching_key]
                    shard_path = hf_hub_download(MODEL_ID, shard, token=HF_TOKEN)
                else:
                    shard_path = hf_hub_download(MODEL_ID, "model.safetensors", token=HF_TOKEN)

                w = load_tensor(shard_path, matching_key)
                if w is None:
                    continue

                expert_packed_signs[exp_idx] = pack_signs_np(w)
                expert_scales[exp_idx] = compute_group_scales_np(w, GROUP_SIZE)
                total_expert_params += w.size

                del w  # Free memory

            if not expert_packed_signs:
                continue

            # Compute consensus (majority vote across experts)
            exp_indices = sorted(expert_packed_signs.keys())
            n_exp_loaded = len(exp_indices)

            # For majority vote: count positive signs per position
            # Unpack, count, repack consensus
            first_shape = expert_packed_signs[exp_indices[0]].shape

            # Simple majority: for each int32, count bit-by-bit across experts
            consensus = np.zeros(first_shape, dtype=np.int32)
            for bit in range(32):
                bit_counts = np.zeros(first_shape, dtype=np.int32)
                for exp_idx in exp_indices:
                    bit_counts += (expert_packed_signs[exp_idx] >> bit) & 1
                # Majority: if more than half the experts have this bit set
                majority = (bit_counts > n_exp_loaded // 2).astype(np.int32)
                consensus |= majority << bit

            # Save consensus
            np.save(OUTPUT_DIR / "consensus_signs" / f"layer{layer_idx}_{wtype.replace('.', '_')}.npy", consensus)

            # Compute and save deltas + scales per expert
            layer_agreement = 0
            for exp_idx in exp_indices:
                delta = sparse_delta(consensus, expert_packed_signs[exp_idx])

                # Save sparse delta
                delta_path = OUTPUT_DIR / "expert_deltas" / f"layer{layer_idx}_{wtype.replace('.', '_')}_expert{exp_idx}.json"
                with open(delta_path, "w") as f:
                    json.dump(delta, f)

                total_delta_size += len(json.dumps(delta))
                layer_agreement += delta.get("bit_agreement", 0)

                # Save scales
                np.save(
                    OUTPUT_DIR / "expert_scales" / f"layer{layer_idx}_{wtype.replace('.', '_')}_expert{exp_idx}.npy",
                    expert_scales[exp_idx]
                )

            avg_agreement = layer_agreement / n_exp_loaded if n_exp_loaded > 0 else 0
            total_agreement += avg_agreement
            n_layers_processed += 1

            print(f"    {wtype}: {n_exp_loaded} experts, {avg_agreement:.1%} sign agreement with consensus", flush=True)

            # Free expert data for this weight type
            del expert_packed_signs, expert_scales

        elapsed = time.time() - layer_t
        print(f"    Layer {layer_idx} done in {elapsed:.1f}s", flush=True)

    # Final summary
    total_time = time.time() - t_start
    avg_agreement_overall = total_agreement / max(n_layers_processed, 1)

    # Compute output sizes
    consensus_size = sum(f.stat().st_size for f in (OUTPUT_DIR / "consensus_signs").glob("*.npy"))
    delta_size = sum(f.stat().st_size for f in (OUTPUT_DIR / "expert_deltas").glob("*.json"))
    scales_size = sum(f.stat().st_size for f in (OUTPUT_DIR / "expert_scales").glob("*.npy"))
    shared_size = sum(f.stat().st_size for f in (OUTPUT_DIR / "shared_layers").glob("*.npy"))
    router_size = sum(f.stat().st_size for f in (OUTPUT_DIR / "router_weights").glob("*.npy"))
    total_output = consensus_size + delta_size + scales_size + shared_size + router_size

    print(f"\n{'='*70}")
    print("CONVERSION COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time: {total_time/3600:.1f} hours ({total_time:.0f}s)")
    print(f"  Expert params processed: {total_expert_params/1e9:.2f}B")
    print(f"  Average sign agreement: {avg_agreement_overall:.1%}")
    print(f"")
    print(f"  Output sizes:")
    print(f"    Consensus signs:  {consensus_size/1e6:.1f} MB")
    print(f"    Expert deltas:    {delta_size/1e6:.1f} MB")
    print(f"    Expert scales:    {scales_size/1e6:.1f} MB")
    print(f"    Shared layers:    {shared_size/1e6:.1f} MB")
    print(f"    Router weights:   {router_size/1e6:.1f} MB")
    print(f"    TOTAL:            {total_output/1e6:.1f} MB")
    print(f"")
    print(f"  Compression: {total_expert_params * 2 / max(total_output, 1):.1f}x from fp16 experts")
    print(f"  Output dir: {OUTPUT_DIR}")

    # Save final metadata
    metadata["total_time_seconds"] = total_time
    metadata["total_expert_params"] = total_expert_params
    metadata["average_sign_agreement"] = avg_agreement_overall
    metadata["output_sizes"] = {
        "consensus_mb": consensus_size / 1e6,
        "deltas_mb": delta_size / 1e6,
        "scales_mb": scales_size / 1e6,
        "shared_mb": shared_size / 1e6,
        "router_mb": router_size / 1e6,
        "total_mb": total_output / 1e6,
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone!", flush=True)


if __name__ == "__main__":
    main()
