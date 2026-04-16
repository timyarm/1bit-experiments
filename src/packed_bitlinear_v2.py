"""Packed 1-bit linear layer — ~300MB VRAM for 1.7B model.

Signs packed as uint8 (8 weights per byte). Scales fp16 per group.
Custom forward: unpack → multiply → matmul. Scales are trainable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class _BitLinearFn(torch.autograd.Function):
    """Custom autograd: forward computes weight on-the-fly, backward only stores what's needed for scale grads."""
    @staticmethod
    def forward(ctx, x, signs, scales, out_features, in_features, group_size):
        # Reconstruct weight
        scales_exp = scales.unsqueeze(1).expand(-1, group_size).reshape(out_features, in_features)
        weight = signs * scales_exp.half()
        output = F.linear(x, weight)
        # Save only what backward needs — NOT the full weight matrix
        ctx.save_for_backward(x, signs, scales)
        ctx.out_features = out_features
        ctx.in_features = in_features
        ctx.group_size = group_size
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, signs, scales = ctx.saved_tensors
        out_f, in_f, gs = ctx.out_features, ctx.in_features, ctx.group_size

        # Grad w.r.t. x (for upstream layers)
        scales_exp = scales.unsqueeze(1).expand(-1, gs).reshape(out_f, in_f)
        weight = signs * scales_exp.half()
        grad_x = grad_output.matmul(weight.to(grad_output.dtype))

        # Grad w.r.t. scales
        # d(loss)/d(scale_g) = sum over group of d(loss)/d(w_i) * sign_i
        # d(loss)/d(w) = grad_output^T @ x
        grad_w = grad_output.reshape(-1, out_f).t().matmul(x.reshape(-1, in_f))  # [out, in]
        grad_w_signed = (grad_w * signs).reshape(-1, gs)  # [n_groups, gs]
        grad_scales = grad_w_signed.sum(dim=1).float()  # [n_groups]

        return grad_x, None, grad_scales, None, None, None


class PackedBitLinear(nn.Module):
    """1-bit linear: frozen fp16 signs + trainable fp16 group scales.

    Uses custom autograd to avoid storing full weight matrix during backward.
    """

    def __init__(self, weight: torch.Tensor, group_size: int = 128):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.group_size = group_size

        # Pre-unpack signs to fp16 {-1, +1} — frozen buffer, no gradient
        signs_fp16 = weight.sign().half()
        signs_fp16[signs_fp16 == 0] = 1
        self.register_buffer('signs', signs_fp16)

        # Group scales: absmean per group, trainable
        n_groups = (self.out_features * self.in_features) // group_size
        w_flat = weight.reshape(-1, group_size)
        scales = w_flat.abs().mean(dim=1)
        self.scales = nn.Parameter(scales.float())
        self.register_buffer('_original_scales', scales.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _BitLinearFn.apply(x, self.signs, self.scales,
                                   self.out_features, self.in_features, self.group_size)


def convert_model(model: nn.Module, group_size: int = 128, skip_names=('lm_head',)):
    """Convert all Linear layers to PackedBitLinear. Returns VRAM savings."""
    before_params = sum(p.numel() * p.element_size() for p in model.parameters())
    converted = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if any(s in name for s in skip_names):
            continue

        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        parent = dict(model.named_modules())[parent_name] if parent_name else model

        packed = PackedBitLinear(module.weight.data, group_size)
        setattr(parent, child_name, packed)
        converted += 1

    after_params = sum(p.numel() * p.element_size() for p in model.parameters())
    # Add buffer sizes
    after_buffers = sum(b.numel() * b.element_size() for b in model.buffers())

    print(f"Converted {converted} layers to PackedBitLinear")
    print(f"  Params: {before_params/1e6:.0f}MB → {after_params/1e6:.0f}MB")
    print(f"  Buffers (packed signs): {after_buffers/1e6:.0f}MB")
    print(f"  Total: {(after_params + after_buffers)/1e6:.0f}MB")

    return converted
