"""
Triton kernels for 1-bit native training.

Eliminates the unpack step: reads bits directly from packed int32.
Python version: 42x slower than F.linear. Triton eliminates this gap.

Kernels:
1. binary_matmul_fwd: forward pass reading packed weights
2. grad_x_bwd: backward chain rule from packed weights
3. sign_grad_bwd: accumulate sign(dL/dW) into int8 flip votes
4. flip_weights: XOR packed weights with flip mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BITS': 32}, num_warps=4),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BITS': 32}, num_warps=4),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 16, 'BITS': 32}, num_warps=4),
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BITS': 32}, num_warps=2),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BITS': 32}, num_warps=8),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BITS': 32}, num_warps=4),
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BITS': 32}, num_warps=2),
        ],
        key=['M', 'N', 'K'],
    )
    @triton.jit
    def binary_matmul_fwd_kernel(
        input_ptr, packed_ptr, scales_ptr, output_ptr,
        M, N, K, K32,
        stride_im, stride_ik,
        stride_pn, stride_pk32,
        stride_sn, stride_sg,
        stride_om, stride_on,
        GROUP_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BITS: tl.constexpr,  # always 32
    ):
        """Forward: output[m,n] = sum_k input[m,k] * sign(packed[n,k]) * scale[n,k//G]"""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        mask_m = offs_m < M
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k32_idx in range(K32):
            k_base = k32_idx * BITS

            # Load packed weights: [BLOCK_N] int32 values
            packed_offsets = offs_n * stride_pn + k32_idx * stride_pk32
            packed_vals = tl.load(packed_ptr + packed_offsets, mask=mask_n, other=0)

            # Load scale for this group
            group_idx = k_base // GROUP_SIZE
            scale_offsets = offs_n * stride_sn + group_idx * stride_sg
            scale_vals = tl.load(scales_ptr + scale_offsets, mask=mask_n, other=0.0)

            # Process all 32 bits — mask out-of-bounds with k_mask
            for bit in tl.static_range(BITS):
                k = k_base + bit
                k_valid = k < K

                # Extract sign: bit -> {-1, +1}
                sign = ((packed_vals >> bit) & 1).to(tl.float32) * 2.0 - 1.0

                # Load input column with bounds check
                input_offsets = offs_m * stride_im + k * stride_ik
                k_mask = mask_m & k_valid
                input_vals = tl.load(input_ptr + input_offsets, mask=k_mask, other=0.0).to(tl.float32)

                # Reload scale at group boundaries
                new_group = (k % GROUP_SIZE) == 0
                if bit > 0 and new_group:
                    new_gidx = k // GROUP_SIZE
                    new_soffs = offs_n * stride_sn + new_gidx * stride_sg
                    scale_vals = tl.load(scales_ptr + new_soffs, mask=mask_n, other=0.0)

                # Accumulate outer product
                weighted_sign = sign * scale_vals.to(tl.float32)
                acc += input_vals[:, None] * weighted_sign[None, :]

        # Store output
        out_offsets = offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        out_mask = mask_m[:, None] & mask_n[None, :]
        tl.store(output_ptr + out_offsets, acc.to(tl.float16), mask=out_mask)


    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        ],
        key=['M', 'N', 'K'],
    )
    @triton.jit
    def binary_dot_fwd_kernel(
        input_ptr, packed_ptr, scales_ptr, output_ptr,
        M, N, K, K32,
        stride_im, stride_ik,
        stride_pn, stride_pk32,
        stride_sn, stride_sg,
        stride_om, stride_on,
        GROUP_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,  # 32 — matches packed int32
    ):
        """Tensor-core accelerated binary matmul.

        1. Unpack 32 signs from packed int32 into [BLOCK_N, 32] fp16 tile
        2. Load [BLOCK_M, 32] input tile
        3. tl.dot(input_tile, signs_tile^T) → uses tensor cores!
        4. Multiply by scale, accumulate

        The unpack is scalar but the matmul uses tensor cores.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        mask_m = offs_m < M
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k32_idx in range(K32):
            k_base = k32_idx * BLOCK_K

            # Load packed weights: [BLOCK_N]
            packed_offsets = offs_n * stride_pn + k32_idx * stride_pk32
            packed_vals = tl.load(packed_ptr + packed_offsets, mask=mask_n, other=0)

            # Unpack to sign tile [BLOCK_N, 32] in registers
            # Each row is one output neuron's 32 signs for this chunk
            signs_tile = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float16)
            for bit in tl.static_range(BLOCK_K):
                sign_bit = ((packed_vals >> bit) & 1).to(tl.float16) * 2.0 - 1.0  # [BLOCK_N]
                signs_tile = tl.where(
                    offs_k[None, :] == bit,
                    sign_bit[:, None] * tl.ones((1, BLOCK_K), dtype=tl.float16),
                    signs_tile
                )

            # Load scale for this group
            group_idx = k_base // GROUP_SIZE
            scale_offsets = offs_n * stride_sn + group_idx * stride_sg
            scale_vals = tl.load(scales_ptr + scale_offsets, mask=mask_n, other=0.0).to(tl.float32)

            # Apply scale to signs tile
            signs_scaled = (signs_tile.to(tl.float32) * scale_vals[:, None]).to(tl.float16)

            # Load input tile [BLOCK_M, 32]
            k_offs = k_base + offs_k
            k_mask = k_offs[None, :] < K
            input_offsets = offs_m[:, None] * stride_im + k_offs[None, :] * stride_ik
            input_tile = tl.load(input_ptr + input_offsets,
                                mask=mask_m[:, None] & k_mask,
                                other=0.0).to(tl.float16)

            # TENSOR CORE MATMUL: [BLOCK_M, 32] × [32, BLOCK_N] → [BLOCK_M, BLOCK_N]
            acc += tl.dot(input_tile, tl.trans(signs_scaled))

        out_offsets = offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        out_mask = mask_m[:, None] & mask_n[None, :]
        tl.store(output_ptr + out_offsets, acc.to(tl.float16), mask=out_mask)


    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4),
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=4),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 16}, num_warps=4),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=8),
        ],
        key=['M', 'N', 'K32'],
    )
    @triton.jit
    def binary_xnor_fwd_kernel(
        input_ptr, packed_ptr, scales_ptr, output_ptr,
        M, N, K, K32,
        stride_im, stride_ik,
        stride_pn, stride_pk32,
        stride_sn, stride_sg,
        stride_om, stride_on,
        GROUP_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """FAST approximate forward using XNOR+popcount.

        Instead of extracting 32 individual bits:
        1. Pack input signs into int32
        2. XNOR with packed weights → popcount = sign agreement
        3. Multiply by average input magnitude × scale

        Approximate because it uses avg magnitude instead of per-element.
        But much faster: ONE int32 op per 32 weights instead of 32 scalar ops.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        mask_m = offs_m < M
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k32_idx in range(K32):
            k_base = k32_idx * 32

            # Load packed weights: [BLOCK_N] int32 values
            packed_offsets = offs_n * stride_pn + k32_idx * stride_pk32
            packed_w = tl.load(packed_ptr + packed_offsets, mask=mask_n, other=0)

            # Load scale
            group_idx = k_base // GROUP_SIZE
            scale_offsets = offs_n * stride_sn + group_idx * stride_sg
            scale_vals = tl.load(scales_ptr + scale_offsets, mask=mask_n, other=0.0).to(tl.float32)

            # For each row m: pack input signs + compute avg magnitude
            for m_idx in tl.static_range(BLOCK_M):
                m = pid_m * BLOCK_M + m_idx
                if m >= M:
                    continue

                # Pack 32 input signs into one int32
                packed_x = tl.zeros((), dtype=tl.int32)
                avg_mag = tl.zeros((), dtype=tl.float32)
                n_valid = tl.zeros((), dtype=tl.float32)

                for bit in tl.static_range(32):
                    k = k_base + bit
                    if k < K:
                        val = tl.load(input_ptr + m * stride_im + k * stride_ik).to(tl.float32)
                        # Pack sign: positive → 1, negative → 0
                        is_pos = val > 0.0
                        packed_x = packed_x | (is_pos.to(tl.int32) << bit)
                        avg_mag += tl.abs(val)
                        n_valid += 1.0

                avg_mag = avg_mag / tl.maximum(n_valid, 1.0)

                # XNOR + popcount: count how many signs AGREE
                xnor_result = ~(packed_x ^ packed_w)  # XNOR
                # popcount via Triton — count set bits
                # Triton doesn't have native popcount, use bit manipulation
                count = tl.zeros((BLOCK_N,), dtype=tl.int32)
                temp = xnor_result
                for _ in tl.static_range(32):
                    count += temp & 1
                    temp = temp >> 1

                # dot_product = (2 * agreements - 32) * avg_magnitude * scale
                dot = (2.0 * count.to(tl.float32) - n_valid) * avg_mag * scale_vals
                acc[m_idx, :] += dot

        # Store
        out_offsets = offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        out_mask = mask_m[:, None] & mask_n[None, :]
        tl.store(output_ptr + out_offsets, acc.to(tl.float16), mask=out_mask)


    @triton.jit
    def grad_x_bwd_kernel(
        grad_output_ptr, packed_ptr, scales_ptr, grad_x_ptr,
        M, N, K, K32,
        stride_gm, stride_gn,
        stride_pn, stride_pk32,
        stride_sn, stride_sg,
        stride_xm, stride_xk,
        GROUP_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BITS: tl.constexpr,
    ):
        """Backward: grad_x[m,k] = sum_n grad_output[m,n] * sign(packed[n,k]) * scale[n,k//G]"""
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

        mask_m = offs_m < M
        mask_k = offs_k < K

        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

        for n in range(N):
            # Load grad_output[m, n]
            go_offsets = offs_m * stride_gm + n * stride_gn
            go_vals = tl.load(grad_output_ptr + go_offsets, mask=mask_m, other=0.0).to(tl.float32)

            # For each k in BLOCK_K, extract sign and scale
            for bk in tl.static_range(BLOCK_K):
                k = pid_k * BLOCK_K + bk
                k_valid = k < K

                k32_idx = k // BITS
                bit = k % BITS

                packed_val = tl.load(packed_ptr + n * stride_pn + k32_idx * stride_pk32,
                                     mask=k_valid, other=0)
                sign = ((packed_val >> bit) & 1).to(tl.float32) * 2.0 - 1.0

                group_idx = k // GROUP_SIZE
                scale = tl.load(scales_ptr + n * stride_sn + group_idx * stride_sg,
                               mask=k_valid, other=0.0).to(tl.float32)

                k_mask = mask_m & k_valid
                acc[:, bk] += tl.where(k_mask, go_vals * sign * scale, 0.0)

        out_offsets = offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        out_mask = mask_m[:, None] & mask_k[None, :]
        tl.store(grad_x_ptr + out_offsets, acc.to(tl.float16), mask=out_mask)


    @triton.jit
    def sign_grad_bwd_kernel(
        grad_output_ptr, input_ptr, flip_votes_ptr,
        M, N, K,
        stride_gm, stride_gn,
        stride_im, stride_ik,
        stride_fn, stride_fk,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Accumulate sign(dL/dW[n,k]) into flip_votes[n,k]."""
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

        mask_n = offs_n < N
        mask_k = offs_k < K

        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

        for m in range(M):
            go_offsets = m * stride_gm + offs_n * stride_gn
            go_vals = tl.load(grad_output_ptr + go_offsets, mask=mask_n, other=0.0).to(tl.float32)

            in_offsets = m * stride_im + offs_k * stride_ik
            in_vals = tl.load(input_ptr + in_offsets, mask=mask_k, other=0.0).to(tl.float32)

            acc += go_vals[:, None] * in_vals[None, :]

        sign_acc = tl.where(acc > 0, 1, tl.where(acc < 0, -1, 0)).to(tl.int8)

        vote_offsets = offs_n[:, None] * stride_fn + offs_k[None, :] * stride_fk
        vote_mask = mask_n[:, None] & mask_k[None, :]
        tl.atomic_add(flip_votes_ptr + vote_offsets, sign_acc, mask=vote_mask)


    @triton.jit
    def flip_weights_kernel(packed_ptr, flip_mask_ptr, num_elements,
                            BLOCK: tl.constexpr):
        """XOR packed weights with flip mask."""
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < num_elements
        packed = tl.load(packed_ptr + offsets, mask=mask)
        flips = tl.load(flip_mask_ptr + offsets, mask=mask)
        tl.store(packed_ptr + offsets, packed ^ flips, mask=mask)


# ─── Helper functions ───
def cdiv(a, b):
    return (a + b - 1) // b


def triton_dot_forward(x, packed_signs, group_scales, group_size):
    """Launch tensor-core accelerated kernel."""
    assert HAS_TRITON
    batch_dims = x.shape[:-1]
    M = x.reshape(-1, x.shape[-1]).shape[0]
    K = x.shape[-1]
    N = packed_signs.shape[0]
    K32 = packed_signs.shape[1]
    x_2d = x.reshape(M, K).contiguous()
    output = torch.empty(M, N, dtype=torch.float16, device=x.device)

    def grid(META):
        return (cdiv(M, META['BLOCK_M']), cdiv(N, META['BLOCK_N']))

    binary_dot_fwd_kernel[grid](
        x_2d, packed_signs, group_scales, output,
        M, N, K, K32,
        x_2d.stride(0), x_2d.stride(1),
        packed_signs.stride(0), packed_signs.stride(1),
        group_scales.stride(0), group_scales.stride(1),
        output.stride(0), output.stride(1),
        GROUP_SIZE=group_size,
    )
    return output.reshape(*batch_dims, N)


def triton_xnor_forward(x, packed_signs, group_scales, group_size):
    """Launch FAST approximate XNOR kernel."""
    assert HAS_TRITON
    batch_dims = x.shape[:-1]
    M = x.reshape(-1, x.shape[-1]).shape[0]
    K = x.shape[-1]
    N = packed_signs.shape[0]
    K32 = packed_signs.shape[1]
    x_2d = x.reshape(M, K).contiguous()
    output = torch.empty(M, N, dtype=torch.float16, device=x.device)

    def grid(META):
        return (cdiv(M, META['BLOCK_M']), cdiv(N, META['BLOCK_N']))

    binary_xnor_fwd_kernel[grid](
        x_2d, packed_signs, group_scales, output,
        M, N, K, K32,
        x_2d.stride(0), x_2d.stride(1),
        packed_signs.stride(0), packed_signs.stride(1),
        group_scales.stride(0), group_scales.stride(1),
        output.stride(0), output.stride(1),
        GROUP_SIZE=group_size,
    )
    return output.reshape(*batch_dims, N)


def triton_binary_matmul_forward(x, packed_signs, group_scales, group_size):
    """Launch exact forward kernel."""
    assert HAS_TRITON, "Triton not available"

    batch_dims = x.shape[:-1]
    M = x.reshape(-1, x.shape[-1]).shape[0]
    K = x.shape[-1]
    N = packed_signs.shape[0]
    K32 = packed_signs.shape[1]

    x_2d = x.reshape(M, K).contiguous()
    output = torch.empty(M, N, dtype=torch.float16, device=x.device)

    # Grid uses lambda — autotune picks BLOCK_M/BLOCK_N
    def grid(META):
        return (cdiv(M, META['BLOCK_M']), cdiv(N, META['BLOCK_N']))

    binary_matmul_fwd_kernel[grid](
        x_2d, packed_signs, group_scales, output,
        M, N, K, K32,
        x_2d.stride(0), x_2d.stride(1),
        packed_signs.stride(0), packed_signs.stride(1),
        group_scales.stride(0), group_scales.stride(1),
        output.stride(0), output.stride(1),
        GROUP_SIZE=group_size,
    )

    return output.reshape(*batch_dims, N)


def triton_grad_x(grad_output, packed_signs, group_scales, group_size, K):
    """Launch grad_x backward kernel."""
    assert HAS_TRITON

    batch_dims = grad_output.shape[:-1]
    M = grad_output.reshape(-1, grad_output.shape[-1]).shape[0]
    N = grad_output.shape[-1]
    K32 = packed_signs.shape[1]

    go_2d = grad_output.reshape(M, N).contiguous()
    grad_x = torch.empty(M, K, dtype=torch.float16, device=grad_output.device)

    BLOCK_M, BLOCK_K = 32, 32
    grid = (cdiv(M, BLOCK_M), cdiv(K, BLOCK_K))

    grad_x_bwd_kernel[grid](
        go_2d, packed_signs, group_scales, grad_x,
        M, N, K, K32,
        go_2d.stride(0), go_2d.stride(1),
        packed_signs.stride(0), packed_signs.stride(1),
        group_scales.stride(0), group_scales.stride(1),
        grad_x.stride(0), grad_x.stride(1),
        GROUP_SIZE=group_size,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        BITS=32,
    )

    return grad_x.reshape(*batch_dims, K)


def triton_sign_grad(grad_output, x, flip_votes):
    """Launch sign gradient backward kernel."""
    assert HAS_TRITON

    M = grad_output.reshape(-1, grad_output.shape[-1]).shape[0]
    N = grad_output.shape[-1]
    K = x.shape[-1]

    go_2d = grad_output.reshape(M, N).contiguous()
    x_2d = x.reshape(M, K).contiguous()

    BLOCK_N, BLOCK_K = 32, 32
    grid = (cdiv(N, BLOCK_N), cdiv(K, BLOCK_K))

    sign_grad_bwd_kernel[grid](
        go_2d, x_2d, flip_votes,
        M, N, K,
        go_2d.stride(0), go_2d.stride(1),
        x_2d.stride(0), x_2d.stride(1),
        flip_votes.stride(0), flip_votes.stride(1),
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )


def triton_flip_weights(packed_signs, flip_mask):
    """XOR packed weights with flip mask."""
    assert HAS_TRITON
    n = packed_signs.numel()
    BLOCK = 1024
    grid = (cdiv(n, BLOCK),)
    flip_weights_kernel[grid](
        packed_signs.reshape(-1), flip_mask.reshape(-1), n, BLOCK=BLOCK
    )


# ─── Autograd Function ───
class TritonBitLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed_signs, group_scales, bias, flip_votes,
                group_size, in_features, out_features, training):
        output = triton_binary_matmul_forward(x, packed_signs, group_scales, group_size)
        if bias is not None:
            output = output + bias
        if training:
            ctx.save_for_backward(x, packed_signs, group_scales)
            ctx.flip_votes = flip_votes
            ctx.group_size = group_size
            ctx.in_features = in_features
            ctx.out_features = out_features
            ctx.has_bias = bias is not None
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, packed_signs, group_scales = ctx.saved_tensors
        grad_x = triton_grad_x(grad_output, packed_signs, group_scales,
                                ctx.group_size, ctx.in_features)
        triton_sign_grad(grad_output, x, ctx.flip_votes)
        grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1))) if ctx.has_bias else None
        return grad_x, None, None, grad_bias, None, None, None, None, None


# ─── Module ───
class NativeBitLinearTriton(nn.Module):
    """Drop-in replacement using Triton kernels. Same API as NativeBitLinear."""

    def __init__(self, in_features, out_features, bias=True, group_size=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        n_ints = ceil(in_features / 32)
        self.register_buffer('packed_signs',
                            torch.zeros(out_features, n_ints, dtype=torch.int32))

        n_groups = ceil(in_features / group_size)
        self.group_scales = nn.Parameter(
            torch.ones(out_features, n_groups, dtype=torch.float16) * 0.02)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        self.register_buffer('flip_votes',
                            torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('vote_count', torch.zeros(1, dtype=torch.int16))

    def forward(self, x):
        return TritonBitLinearFn.apply(
            x, self.packed_signs, self.group_scales, self.bias,
            self.flip_votes, self.group_size,
            self.in_features, self.out_features, self.training
        )

    def increment_vote_count(self):
        self.vote_count += 1

    def apply_flips(self, threshold=0.8, max_flip_rate=0.001):
        """Flip weights using packed XOR."""
        count = self.vote_count.item()
        if count == 0:
            return 0
        try:
            from .native_bitlinear import unpack_signs, pack_signs
        except ImportError:
            from native_bitlinear import unpack_signs, pack_signs

        current_signs = unpack_signs(self.packed_signs, self.out_features, self.in_features)
        vote_ratio = self.flip_votes.float() / count

        strong = vote_ratio.abs() > threshold
        same_dir = (vote_ratio.sign() * current_signs) > 0
        should_flip = strong & same_dir

        n_cand = should_flip.sum().item()
        max_flips = int(self.in_features * self.out_features * max_flip_rate)
        if n_cand > max_flips and max_flips > 0:
            flat = (vote_ratio.abs() * should_flip.float()).view(-1)
            _, top_idx = flat.topk(max_flips)
            new_mask = torch.zeros_like(flat, dtype=torch.bool)
            new_mask[top_idx] = True
            should_flip = new_mask.view(should_flip.shape)

        n_flips = should_flip.sum().item()
        if n_flips > 0:
            new_signs = current_signs.clone()
            new_signs[should_flip] *= -1
            self.packed_signs.copy_(pack_signs(new_signs))

        self.flip_votes.zero_()
        self.vote_count.zero_()
        return n_flips

    @classmethod
    def from_linear(cls, linear, group_size=128):
        """Convert nn.Linear to NativeBitLinearTriton."""
        try:
            from .native_bitlinear import pack_signs, compute_group_scales
        except ImportError:
            from native_bitlinear import pack_signs, compute_group_scales

        has_bias = linear.bias is not None
        native = cls(linear.in_features, linear.out_features,
                     bias=has_bias, group_size=group_size)
        w = linear.weight.data.float()
        native.packed_signs.copy_(pack_signs(w))
        native.group_scales.data.copy_(compute_group_scales(w, group_size))
        if has_bias and linear.bias is not None:
            native.bias.data.copy_(linear.bias.data)
        return native
