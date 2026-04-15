"""MoE → 1-bit Delta Encoding: GLM-5.1 → B2 Backblaze

Same as moe_to_1bit.py but:
  - Targets GLM-5.1 (754B, 256 experts)
  - Uploads each layer's output to B2 immediately
  - Deletes local files after upload
  - Peak disk: ~5 GB (one layer buffer + upload queue)

Run overnight. Estimated: 12-20 hours.
"""

import os
import sys
import json
import time
import numpy as np
import torch
from pathlib import Path
from safetensors import safe_open
from b2sdk.v2 import InMemoryAccountInfo, B2Api

# B2 Config
B2_KEY_ID = os.environ.get("B2_KEY_ID")
B2_APP_KEY = os.environ.get("B2_APP_KEY")
B2_BUCKET = "my-bucket"
B2_PREFIX = "models/1bit/glm51"

# Model config
MODEL_ID = "zai-org/GLM-5.1-FP8"
GROUP_SIZE = 128
HF_TOKEN = os.environ.get("HF_TOKEN")

# Local temp directory (small, files uploaded then deleted)
TEMP_DIR = Path("/tmp/glm51_1bit_temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# B2 setup
print("Connecting to B2...", flush=True)
b2_info = InMemoryAccountInfo()
b2_api = B2Api(b2_info)
b2_api.authorize_account('production', B2_KEY_ID, B2_APP_KEY)
b2_bucket = b2_api.get_bucket_by_name(B2_BUCKET)
print(f"  B2 bucket: {B2_BUCKET}/{B2_PREFIX}", flush=True)


def upload_and_delete(local_path, b2_path):
    """Upload file to B2 and delete local copy."""
    full_b2_path = f"{B2_PREFIX}/{b2_path}"
    b2_bucket.upload_local_file(str(local_path), full_b2_path)
    os.unlink(local_path)
    return full_b2_path


def load_tensor(shard_path, key):
    """Load single tensor, convert bf16/fp8→float32→numpy."""
    with safe_open(shard_path, framework="pt") as f:
        if key in f.keys():
            return f.get_tensor(key).float().numpy()
    return None


def get_keys(shard_path):
    with safe_open(shard_path, framework="pt") as f:
        return list(f.keys())


def pack_signs_np(weights):
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
    out_features, in_features = weights.shape
    pad = (group_size - in_features % group_size) % group_size
    if pad > 0:
        w = np.pad(np.abs(weights), ((0, 0), (0, pad)))
    else:
        w = np.abs(weights)
    n_groups = w.shape[1] // group_size
    return w.reshape(out_features, n_groups, group_size).mean(axis=2).astype(np.float16)


def sparse_delta(consensus_packed, expert_packed):
    xor = consensus_packed ^ expert_packed
    total_bits = consensus_packed.size * 32
    differing_bits = 0
    temp = xor.copy()
    for _ in range(32):
        differing_bits += (temp & 1).sum()
        temp >>= 1
    bit_agreement = 1.0 - float(differing_bits) / total_bits

    nonzero_mask = xor != 0
    if not nonzero_mask.any():
        return {"nnz": 0, "bit_agreement": 1.0, "bits_differing": 0,
                "total_bits": total_bits, "shape": list(consensus_packed.shape)}

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
    print("GLM-5.1 → 1-bit Delta Encoding → B2 Backblaze")
    print("=" * 70, flush=True)

    t_start = time.time()

    from huggingface_hub import hf_hub_download

    # Get config
    print("\n[1/4] Fetching model config...", flush=True)
    config_path = hf_hub_download(MODEL_ID, "config.json", token=HF_TOKEN)
    with open(config_path) as f:
        config = json.load(f)

    n_layers = config.get("num_hidden_layers", 0)
    n_experts = config.get("num_experts", config.get("n_routed_experts", 0))
    hidden_size = config.get("hidden_size", 0)
    print(f"  Layers: {n_layers}, Experts: {n_experts}, Hidden: {hidden_size}", flush=True)

    # Upload config to B2
    config_local = TEMP_DIR / "config.json"
    with open(config_local, "w") as f:
        json.dump(config, f, indent=2)
    upload_and_delete(config_local, "config.json")
    print("  Config uploaded to B2", flush=True)

    # Get weight map
    print("\n[2/4] Loading weight index...", flush=True)
    index_path = hf_hub_download(MODEL_ID, "model.safetensors.index.json", token=HF_TOKEN)
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    all_keys = list(weight_map.keys())

    # Categorize
    expert_keys = [k for k in all_keys if "experts" in k]
    router_keys = [k for k in all_keys if "gate" in k or "router" in k or "e_score" in k]
    attention_keys = [k for k in all_keys if any(x in k for x in ["q_proj", "k_proj", "v_proj", "o_proj"])]
    norm_keys = [k for k in all_keys if "norm" in k or "layernorm" in k]
    embed_keys = [k for k in all_keys if "embed" in k]

    print(f"  Expert: {len(expert_keys)}, Router: {len(router_keys)}, "
          f"Attention: {len(attention_keys)}, Norm: {len(norm_keys)}", flush=True)

    # [3/4] Process shared layers
    print("\n[3/4] Processing shared layers → B2...", flush=True)
    shared_keys = attention_keys + norm_keys + embed_keys

    files_needed = set()
    for k in shared_keys + router_keys:
        if k in weight_map:
            files_needed.add(weight_map[k])

    processed_files = set()
    shared_count = 0
    for shard_file in sorted(files_needed):
        if shard_file in processed_files:
            continue
        processed_files.add(shard_file)
        print(f"  Shard: {shard_file}...", flush=True)

        shard_path = hf_hub_download(MODEL_ID, shard_file, token=HF_TOKEN)
        shard_keys = get_keys(shard_path)

        for key in shard_keys:
            if key in router_keys:
                tensor = load_tensor(shard_path, key)
                if tensor is None: continue
                local = TEMP_DIR / f"{key.replace('/', '_')}.npy"
                np.save(local, tensor)
                upload_and_delete(local, f"router/{key.replace('/', '_')}.npy")
                shared_count += 1

            elif key in shared_keys:
                tensor = load_tensor(shard_path, key)
                if tensor is None: continue

                if tensor.ndim == 2 and tensor.shape[0] > 1 and tensor.shape[1] > 1:
                    packed = pack_signs_np(tensor)
                    scales = compute_group_scales_np(tensor, GROUP_SIZE)
                    signs_local = TEMP_DIR / f"{key.replace('/', '_')}_signs.npy"
                    scales_local = TEMP_DIR / f"{key.replace('/', '_')}_scales.npy"
                    np.save(signs_local, packed)
                    np.save(scales_local, scales)
                    upload_and_delete(signs_local, f"shared/{key.replace('/', '_')}_signs.npy")
                    upload_and_delete(scales_local, f"shared/{key.replace('/', '_')}_scales.npy")
                else:
                    local = TEMP_DIR / f"{key.replace('/', '_')}.npy"
                    np.save(local, tensor)
                    upload_and_delete(local, f"shared/{key.replace('/', '_')}.npy")

                shared_count += 1

    print(f"  Shared layers: {shared_count} tensors uploaded to B2", flush=True)

    # [4/4] Process MoE experts
    print("\n[4/4] Processing MoE experts → B2...", flush=True)

    # Group expert keys by layer
    layer_expert_keys = {}
    for k in expert_keys:
        parts = k.split(".")
        layer_idx = expert_idx = None
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try: layer_idx = int(parts[i + 1])
                except: pass
            if p == "experts" and i + 1 < len(parts):
                try: expert_idx = int(parts[i + 1])
                except: pass
        if layer_idx is not None and expert_idx is not None:
            if layer_idx not in layer_expert_keys:
                layer_expert_keys[layer_idx] = {}
            if expert_idx not in layer_expert_keys[layer_idx]:
                layer_expert_keys[layer_idx][expert_idx] = []
            layer_expert_keys[layer_idx][expert_idx].append(k)

    print(f"  MoE layers: {len(layer_expert_keys)}", flush=True)

    total_agreement = 0
    n_processed = 0

    for layer_idx in sorted(layer_expert_keys.keys()):
        layer_t = time.time()
        experts_data = layer_expert_keys[layer_idx]
        n_exp = len(experts_data)

        # Get weight types from first expert
        first_expert_keys = experts_data[min(experts_data.keys())]
        weight_types = set()
        for k in first_expert_keys:
            for p in k.split("."):
                if p in ["w1", "w2", "w3", "gate_proj", "up_proj", "down_proj"]:
                    weight_types.add(p)
                    break

        for wtype in sorted(weight_types):
            expert_packed = {}
            expert_scales = {}

            for exp_idx in sorted(experts_data.keys()):
                matching_key = None
                for k in experts_data[exp_idx]:
                    if wtype in k:
                        matching_key = k
                        break
                if not matching_key: continue

                shard = weight_map.get(matching_key)
                if not shard: continue
                shard_path = hf_hub_download(MODEL_ID, shard, token=HF_TOKEN)
                w = load_tensor(shard_path, matching_key)
                if w is None: continue

                expert_packed[exp_idx] = pack_signs_np(w)
                expert_scales[exp_idx] = compute_group_scales_np(w, GROUP_SIZE)
                del w

            if not expert_packed: continue

            # Consensus
            exp_indices = sorted(expert_packed.keys())
            first_shape = expert_packed[exp_indices[0]].shape

            consensus = np.zeros(first_shape, dtype=np.int32)
            for bit in range(32):
                bit_counts = np.zeros(first_shape, dtype=np.int32)
                for exp_idx in exp_indices:
                    bit_counts += (expert_packed[exp_idx] >> bit) & 1
                majority = (bit_counts > len(exp_indices) // 2).astype(np.int32)
                consensus |= majority << bit

            # Save consensus → B2
            local = TEMP_DIR / f"consensus_layer{layer_idx}_{wtype}.npy"
            np.save(local, consensus)
            upload_and_delete(local, f"consensus/layer{layer_idx}_{wtype}.npy")

            # Save deltas + scales per expert → B2
            layer_agreement = 0
            for exp_idx in exp_indices:
                delta = sparse_delta(consensus, expert_packed[exp_idx])
                layer_agreement += delta.get("bit_agreement", 0)

                delta_local = TEMP_DIR / f"delta_l{layer_idx}_{wtype}_e{exp_idx}.json"
                with open(delta_local, "w") as f:
                    json.dump(delta, f)
                upload_and_delete(delta_local, f"deltas/layer{layer_idx}_{wtype}_expert{exp_idx}.json")

                scales_local = TEMP_DIR / f"scales_l{layer_idx}_{wtype}_e{exp_idx}.npy"
                np.save(scales_local, expert_scales[exp_idx])
                upload_and_delete(scales_local, f"scales/layer{layer_idx}_{wtype}_expert{exp_idx}.npy")

            avg_agreement = layer_agreement / len(exp_indices) if exp_indices else 0
            total_agreement += avg_agreement
            n_processed += 1

            print(f"    L{layer_idx} {wtype}: {len(exp_indices)} experts, "
                  f"{avg_agreement:.1%} sign agreement → B2", flush=True)

            del expert_packed, expert_scales

        elapsed = time.time() - layer_t
        print(f"  Layer {layer_idx} done ({elapsed:.0f}s)", flush=True)

    # Summary
    total_time = time.time() - t_start
    avg_agreement = total_agreement / max(n_processed, 1)

    summary = {
        "model": MODEL_ID,
        "n_layers": len(layer_expert_keys),
        "n_experts": n_experts,
        "avg_sign_agreement": avg_agreement,
        "total_time_hours": total_time / 3600,
        "b2_path": f"{B2_BUCKET}/{B2_PREFIX}",
    }
    summary_local = TEMP_DIR / "summary.json"
    with open(summary_local, "w") as f:
        json.dump(summary, f, indent=2)
    upload_and_delete(summary_local, "summary.json")

    print(f"\n{'='*70}")
    print(f"DONE — GLM-5.1 → 1-bit → B2")
    print(f"  Time: {total_time/3600:.1f} hours")
    print(f"  Avg sign agreement: {avg_agreement:.1%}")
    print(f"  B2 path: {B2_BUCKET}/{B2_PREFIX}/")
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
