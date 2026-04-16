"""Patch GGUF Q1_0_g128 scale values in-place.

Format per group of 128 weights: [16 bytes signs][2 bytes fp16 scale]
Total per group: 18 bytes.

This lets us modify scales and eval at llama.cpp speed (340 tok/s)
without touching signs.
"""

import numpy as np
import struct
import shutil
from pathlib import Path
from gguf import GGUFReader


GROUP_SIZE = 128
SCALE_BYTES = 2  # fp16 — comes FIRST in the block
SIGNS_BYTES = GROUP_SIZE // 8  # 16 — packed sign bits after scale
BLOCK_BYTES = SCALE_BYTES + SIGNS_BYTES  # 18


def read_scales(gguf_path, tensor_name):
    """Read all group scales from a Q1_0_g128 tensor."""
    reader = GGUFReader(gguf_path)
    for t in reader.tensors:
        if str(t.name) == tensor_name:
            data = bytes(t.data)
            n_groups = len(data) // BLOCK_BYTES
            scales = []
            for g in range(n_groups):
                offset = g * BLOCK_BYTES  # scale is at the START of each block
                scale_bytes = data[offset:offset + SCALE_BYTES]
                scale = struct.unpack('<e', scale_bytes)[0]  # fp16 little-endian
                scales.append(scale)
            return np.array(scales, dtype=np.float16)
    raise ValueError(f"Tensor {tensor_name} not found")


def patch_scales(gguf_path, output_path, scale_patches):
    """Patch group scales in a GGUF file.

    Args:
        gguf_path: source GGUF
        output_path: patched GGUF (can be same file for in-place)
        scale_patches: dict of {tensor_name: np.array of new scales (fp16)}
    """
    if gguf_path != output_path:
        shutil.copy2(gguf_path, output_path)

    reader = GGUFReader(output_path)

    # Read entire file as numpy array for fast manipulation
    file_data = np.fromfile(output_path, dtype=np.uint8)

    for t in reader.tensors:
        name = str(t.name)
        if name not in scale_patches:
            continue

        new_scales = np.array(scale_patches[name], dtype=np.float16)
        data_offset = t.data.ctypes.data - reader.data.ctypes.data
        n_groups = t.n_bytes // BLOCK_BYTES

        if len(new_scales) != n_groups:
            print(f"  WARNING: {name}: expected {n_groups} scales, got {len(new_scales)}, skipping")
            continue

        # Vectorized: view the tensor data as structured blocks
        tensor_data = file_data[data_offset:data_offset + t.n_bytes]
        # Reshape into [n_groups, BLOCK_BYTES] — each row is [2B scale][16B signs]
        blocks = tensor_data.reshape(n_groups, BLOCK_BYTES)
        # Replace first 2 bytes (scale) of each block
        new_scale_bytes = new_scales.view(np.uint8).reshape(n_groups, SCALE_BYTES)
        blocks[:, :SCALE_BYTES] = new_scale_bytes

        print(f"  Patched {name}: {n_groups} scales")

    file_data.tofile(output_path)
    print(f"  Written {len(file_data)/1e6:.0f}MB")


def apply_multipliers_to_gguf(gguf_path, output_path, multipliers, layer_name_map=None):
    """Apply scale multipliers (from PyTorch training) to GGUF.

    multipliers: dict of {pytorch_layer_name: tensor of per-group multipliers}
    layer_name_map: optional mapping from PyTorch names to GGUF tensor names
    """
    if layer_name_map is None:
        layer_name_map = _default_name_map()

    reader = GGUFReader(gguf_path)
    gguf_names = {str(t.name) for t in reader.tensors}

    patches = {}
    for pt_name, mults in multipliers.items():
        gguf_name = layer_name_map.get(pt_name, _auto_map_name(pt_name))
        if gguf_name not in gguf_names:
            print(f"  WARNING: {pt_name} -> {gguf_name} not found in GGUF, skipping")
            continue

        # Read current scales
        current = read_scales(gguf_path, gguf_name)
        mults_np = mults.detach().cpu().numpy().astype(np.float16)

        if len(mults_np) != len(current):
            print(f"  WARNING: {pt_name}: {len(mults_np)} multipliers vs {len(current)} scales, skipping")
            continue

        # Apply multipliers
        new_scales = current * mults_np
        patches[gguf_name] = new_scales

    if patches:
        patch_scales(gguf_path, output_path, patches)
        print(f"Patched {len(patches)} tensors in {output_path}")
    else:
        print("WARNING: No patches applied")


def _auto_map_name(pt_name):
    """Convert PyTorch layer name to GGUF tensor name.

    model.layers.0.self_attn.q_proj -> blk.0.attn_q.weight
    model.layers.0.mlp.gate_proj    -> blk.0.ffn_gate.weight
    """
    name = pt_name.replace('model.layers.', 'blk.')
    name = name.replace('.self_attn.q_proj', '.attn_q')
    name = name.replace('.self_attn.k_proj', '.attn_k')
    name = name.replace('.self_attn.v_proj', '.attn_v')
    name = name.replace('.self_attn.o_proj', '.attn_output')
    name = name.replace('.mlp.gate_proj', '.ffn_gate')
    name = name.replace('.mlp.up_proj', '.ffn_up')
    name = name.replace('.mlp.down_proj', '.ffn_down')
    return name + '.weight'


def _default_name_map():
    """Build full mapping for 28-layer Qwen/Bonsai model."""
    mapping = {}
    for i in range(28):
        prefix_pt = f'model.layers.{i}'
        prefix_gguf = f'blk.{i}'
        mapping[f'{prefix_pt}.self_attn.q_proj'] = f'{prefix_gguf}.attn_q.weight'
        mapping[f'{prefix_pt}.self_attn.k_proj'] = f'{prefix_gguf}.attn_k.weight'
        mapping[f'{prefix_pt}.self_attn.v_proj'] = f'{prefix_gguf}.attn_v.weight'
        mapping[f'{prefix_pt}.self_attn.o_proj'] = f'{prefix_gguf}.attn_output.weight'
        mapping[f'{prefix_pt}.mlp.gate_proj'] = f'{prefix_gguf}.ffn_gate.weight'
        mapping[f'{prefix_pt}.mlp.up_proj'] = f'{prefix_gguf}.ffn_up.weight'
        mapping[f'{prefix_pt}.mlp.down_proj'] = f'{prefix_gguf}.ffn_down.weight'
    return mapping


if __name__ == "__main__":
    import sys
    gguf = sys.argv[1] if len(sys.argv) > 1 else "/home/timyarm/freigent/apps/trucksim/data/llm/Bonsai-1.7B.gguf"

    # Test: read scales from first layer
    scales = read_scales(gguf, "blk.0.attn_q.weight")
    print(f"blk.0.attn_q scales: {len(scales)} groups")
    print(f"  min={scales.min():.6f}, max={scales.max():.6f}, mean={scales.mean():.6f}")

    scales2 = read_scales(gguf, "blk.0.ffn_gate.weight")
    print(f"blk.0.ffn_gate scales: {len(scales2)} groups")
    print(f"  min={scales2.min():.6f}, max={scales2.max():.6f}, mean={scales2.mean():.6f}")
