"""Triton kernel for packed 1-bit matmul — SM 7.5 compatible.

Unpacks on-the-fly in registers, never materializes full weight in VRAM.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _packed_1bit_mv_kernel(
    x_ptr, w_packed_ptr, scales_ptr, out_ptr,
    N, K,  # N=out_features, K=in_features
    K_packed,  # K // 8
    group_size: tl.constexpr,
    groups_per_row: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_BYTES: tl.constexpr,  # process this many bytes at a time
):
    """Matrix-vector: out[n] = sum_k(sign[n,k] * scale[n,k//gs] * x[k]) for single input."""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for kb in range(0, K_packed, BLOCK_K_BYTES):
        offs_kb = kb + tl.arange(0, BLOCK_K_BYTES)
        kb_mask = offs_kb < K_packed

        # Load packed bytes: [BLOCK_N, BLOCK_K_BYTES]
        w_offs = offs_n[:, None] * K_packed + offs_kb[None, :]
        w_mask = n_mask[:, None] & kb_mask[None, :]
        packed = tl.load(w_packed_ptr + w_offs, mask=w_mask, other=0).to(tl.uint8)

        # Unpack each bit and accumulate
        for bit in tl.static_range(8):
            k_idx = (kb + tl.arange(0, BLOCK_K_BYTES)) * 8 + bit
            k_mask = k_idx < K

            # Extract sign bit → {-1, +1}
            sign = ((packed >> bit) & 1).to(tl.float32) * 2.0 - 1.0  # [BLOCK_N, BLOCK_K_BYTES]

            # Load scales for these positions
            g_idx = offs_n[:, None] * groups_per_row + k_idx[None, :] // group_size
            s_mask = n_mask[:, None] & k_mask[None, :]
            scale = tl.load(scales_ptr + g_idx, mask=s_mask, other=0.0).to(tl.float32)

            # Load x values
            x_vals = tl.load(x_ptr + k_idx, mask=k_mask, other=0.0).to(tl.float32)

            # Accumulate: out[n] += sign[n,k] * scale[n,k//gs] * x[k]
            acc += tl.sum(sign * scale * x_vals[None, :], axis=1)

    tl.store(out_ptr + offs_n, acc.to(tl.float16), mask=n_mask)


def packed_1bit_matvec(x_1d, w_packed, scales, out_features, in_features, group_size=128):
    """Single vector × packed 1-bit matrix. Optimized for autoregressive generation."""
    assert x_1d.dim() == 1 and x_1d.shape[0] == in_features
    N = out_features
    K = in_features
    K_packed = K // 8
    groups_per_row = K // group_size

    out = torch.empty(N, dtype=torch.float16, device=x_1d.device)

    BLOCK_N = 64
    BLOCK_K_BYTES = min(64, K_packed)

    grid = (triton.cdiv(N, BLOCK_N),)

    _packed_1bit_mv_kernel[grid](
        x_1d, w_packed, scales, out,
        N, K, K_packed,
        group_size=group_size,
        groups_per_row=groups_per_row,
        BLOCK_N=BLOCK_N,
        BLOCK_K_BYTES=BLOCK_K_BYTES,
    )
    return out


class TritonPackedBitLinear(torch.nn.Module):
    """1-bit linear with Triton kernel. ~14x VRAM compression."""

    def __init__(self, weight: torch.Tensor, group_size: int = 128):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.group_size = group_size

        signs = (weight > 0).to(torch.uint8)
        self.register_buffer('packed_signs', self._pack(signs))

        w_flat = weight.reshape(-1, group_size)
        scales = w_flat.abs().mean(dim=1)
        self.scales = torch.nn.Parameter(scales.half())
        self.register_buffer('_original_scales', scales.half().clone())

    def _pack(self, signs):
        flat = signs.reshape(-1)
        pad = (8 - flat.shape[0] % 8) % 8
        if pad:
            flat = torch.nn.functional.pad(flat, (0, pad))
        flat = flat.reshape(-1, 8)
        packed = torch.zeros(flat.shape[0], dtype=torch.uint8, device=flat.device)
        for i in range(8):
            packed |= (flat[:, i] << i)
        return packed.reshape(self.out_features, self.in_features // 8)

    def forward(self, x):
        shape = x.shape
        if x.dim() == 2 and x.shape[0] == 1:
            # Single token — use fast matvec kernel
            out = packed_1bit_matvec(
                x.squeeze(0), self.packed_signs, self.scales,
                self.out_features, self.in_features, self.group_size
            )
            return out.unsqueeze(0)
        elif x.dim() == 3 and x.shape[0] == 1 and x.shape[1] == 1:
            out = packed_1bit_matvec(
                x.reshape(-1), self.packed_signs, self.scales,
                self.out_features, self.in_features, self.group_size
            )
            return out.reshape(1, 1, -1)
        else:
            # Batch/sequence — fall back to unpacked matmul for now
            x_2d = x.reshape(-1, self.in_features)
            # Unpack for batch (prefill)
            bits = []
            for i in range(8):
                bits.append((self.packed_signs >> i) & 1)
            unpacked = torch.stack(bits, dim=2).reshape(self.out_features, -1)
            signs = unpacked[:, :self.in_features].half() * 2 - 1
            scales = self.scales.unsqueeze(1).expand(-1, self.group_size)
            weight = signs * scales.reshape(self.out_features, self.in_features)
            out = torch.nn.functional.linear(x_2d, weight)
            del weight, signs, scales
            return out.reshape(*shape[:-1], self.out_features)


def convert_model_triton(model, group_size=128, skip_names=('lm_head',)):
    """Convert Linear layers to TritonPackedBitLinear."""
    converted = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(s in name for s in skip_names):
            continue
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        parent = dict(model.named_modules())[parent_name] if parent_name else model
        setattr(parent, child_name, TritonPackedBitLinear(module.weight.data, group_size))
        converted += 1

    total_buf = sum(b.numel() * b.element_size() for b in model.buffers())
    total_par = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Converted {converted} layers | Total: {(total_par+total_buf)/1e6:.0f}MB")
    return converted
