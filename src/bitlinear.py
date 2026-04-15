"""
BitLinear — 1-bit linear layer with Straight-Through Estimator (STE).

Drop-in replacement for nn.Linear. Supports two modes:

  binary=True  (default): weights are {-1, +1} with shared FP16 group scale.
      This is what PrismML Bonsai uses (Q1_0_g128 format).
      w_i = s_g * sign(w_i)  — true 1-bit, 1.125 bits/weight effective.
      Enables multiplication-free inference (just additions).

  binary=False: weights are ternary {-1, 0, +1} (1.58-bit, BitNet b1.58).
      Kept for comparison but NOT the target format.

Gradients flow through the quantization via STE.

Used for QAT (Quantization-Aware Training) — train with simulated 1-bit weights,
then export the binary weights for deployment.

Reference: PrismML Bonsai 8B whitepaper (March 2026) — Q1_0_g128 format,
1 sign bit + 1 FP16 scale per 128 weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BitLinear(nn.Module):
    """True 1-bit linear layer with STE gradient flow.

    Default mode (binary=True): weights are {-1, +1} with shared FP16 group scale.
    Matches PrismML Bonsai Q1_0_g128: w_i = s_g * (2*b_i - 1), b_i in {0,1}.
    Effective storage: 1 + 16/group_size bits/weight (1.125 at g128).

    Ternary mode (binary=False): weights are {-1, 0, +1} (1.58-bit, BitNet b1.58).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 group_size: int = 128, binary: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.binary = binary

        # Trainable weights — same as nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Per-group learned scales (one FP16 scale per group, like Bonsai Q1_0_g128)
        n_groups = max(1, (out_features * in_features + group_size - 1) // group_size)
        self.scale = nn.Parameter(torch.ones(n_groups))

        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_bin = self._quantize_ste(self.weight)
        return F.linear(x, w_bin, self.bias)

    def _quantize_ste(self, w: torch.Tensor) -> torch.Tensor:
        """Quantize weights with STE.

        binary=True:  {-1, +1} * scale  (Bonsai Q1_0_g128)
        binary=False: {-1, 0, +1} * scale  (BitNet b1.58 ternary)
        """
        orig_dtype = w.dtype
        flat = w.float().reshape(-1)
        gs = self.group_size

        # Pad to group_size multiple
        remainder = flat.numel() % gs
        if remainder != 0:
            flat = F.pad(flat, (0, gs - remainder))

        groups = flat.reshape(-1, gs)
        n_groups = groups.shape[0]

        # Use learned scale (clamped positive)
        scale = self.scale[:n_groups].abs().clamp(min=1e-8).unsqueeze(1)

        # Normalize by scale
        normalized = groups / scale

        if self.binary:
            # True 1-bit: {-1, +1} — no zero, like Bonsai
            # tanh gives smooth gradient for STE, sign() gives hard binary
            smooth = torch.tanh(normalized)
            hard = normalized.sign()
            # Handle exact zeros (rare but possible) — map to +1
            hard = torch.where(hard == 0, torch.ones_like(hard), hard)
            # STE: forward uses hard sign, backward uses tanh gradient
            binary_w = hard.detach() + smooth - smooth.detach()
            quantized = binary_w * scale
        else:
            # Ternary {-1, 0, +1} — BitNet b1.58 mode
            clipped = normalized.clamp(-1, 1)
            rounded = clipped.round()
            ternary = rounded.detach() + clipped - clipped.detach()
            quantized = ternary * scale

        result = quantized.reshape(-1)[:w.numel()].reshape(w.shape)
        return result.to(orig_dtype)

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = 128,
                    binary: bool = True) -> 'BitLinear':
        """Convert an nn.Linear to BitLinear, copying weights.

        Args:
            linear: Source nn.Linear layer.
            group_size: Weights per group (Bonsai uses 128).
            binary: True for {-1,+1} (Bonsai), False for {-1,0,+1} (BitNet b1.58).
        """
        has_bias = linear.bias is not None
        bit = cls(linear.in_features, linear.out_features, bias=has_bias,
                  group_size=group_size, binary=binary)

        with torch.no_grad():
            bit.weight.copy_(linear.weight)
            if has_bias:
                bit.bias.copy_(linear.bias)

            # Initialize scales from weight magnitude per group
            flat = linear.weight.reshape(-1)
            gs = group_size
            remainder = flat.numel() % gs
            if remainder != 0:
                flat = F.pad(flat, (0, gs - remainder))
            groups = flat.reshape(-1, gs)
            group_scales = groups.abs().mean(dim=1)

            # Resize scale parameter to exact number of groups
            bit.scale = nn.Parameter(group_scales.clone())

        return bit.to(linear.weight.device)

    def extra_repr(self) -> str:
        mode = "binary_1bit" if self.binary else "ternary_1.58bit"
        bpw = f"{1 + 16 / self.group_size:.3f}" if self.binary else "1.585"
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'bias={self.bias is not None}, group_size={self.group_size}, '
                f'mode={mode}, bits_per_weight={bpw}')


def swap_linear_to_bitlinear(model: nn.Module, group_size: int = 128,
                              binary: bool = True, min_params: int = 128,
                              skip_names: list = None):
    """Replace nn.Linear layers in model with BitLinear (Bonsai Q1_0_g128).

    Args:
        model: The model to convert (modified in-place).
        group_size: Weights per group (Bonsai uses 128).
        binary: True for {-1,+1} (Bonsai), False for {-1,0,+1} (BitNet b1.58).
        min_params: Skip layers smaller than this.
        skip_names: List of substrings — skip layers whose name contains any.

    Returns:
        Number of layers swapped.
    """
    skip_names = skip_names or []
    swapped = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        n_params = module.weight.numel()
        if n_params < min_params:
            continue

        if any(s in name for s in skip_names):
            continue

        # Navigate to parent module and replace
        parts = name.split('.')
        parent = model
        for p in parts[:-1]:
            if p.isdigit():
                parent = parent[int(p)]
            else:
                parent = getattr(parent, p)

        attr_name = parts[-1]
        bit_layer = BitLinear.from_linear(module, group_size, binary)
        if attr_name.isdigit():
            parent[int(attr_name)] = bit_layer
        else:
            setattr(parent, attr_name, bit_layer)
        swapped += 1

    return swapped


# ─── Forensic verification (Archie's techniques) ───────────────────────

def forensic_analyze(model: nn.Module, base_model: nn.Module = None) -> dict:
    """Run Archie Sengupta's forensic measurements on a 1-bit model.

    Implements the exact metrics from Archie's Bonsai reverse-engineering
    (April 2, 2026). These are the statistical fingerprints that prove
    whether QAT was applied and how aggressively per category.

    Metrics (matching Archie's definitions exactly):

    1. bit_match: frac of weight signs matching base model.
       Bonsai target: 0.70-0.79 (21-30% signs reconfigured by QAT).
       Pre-QAT baseline: 1.0 (no signs changed yet).

    2. ratio_med: median(|w_bonsai_i| / |w_base_i|).
       Measures magnitude inflation from training. Bonsai ffn_up = 3.53x at 1.7B.
       Naive PTQ baseline: ~1.0-1.2.

    3. adv_ratio_med: ratio_med_bonsai - ratio_med_naive_ptq.
       The extra magnitude boost above what naive PTQ gives.
       Bonsai ffn_up = 2.34 at 1.7B, decaying with model size.

    4. kurtosis: excess kurtosis of dequantized weight distribution.
       Bimodal {-s, +s} → strongly negative. Bonsai target: -1.8 to -1.9.
       Gaussian → 0. Pre-QAT: depends on init.

    5. scale_std: std dev of per-group scales within a tensor.
       Compresses with model size under joint learning. Fingerprint of
       scales being learned parameters, not post-hoc statistics.

    6. depth-scale correlation: Pearson(layer_index, median_group_scale).
       Bonsai ffn_up = 0.877 at 1.7B. Cannot arise from post-hoc procedure.

    Args:
        model: The 1-bit model to analyze (must have BitLinear layers).
        base_model: Original FP16/FP32 model for bit_match + ratio_med.
                    If None, bit_match and ratio_med are skipped.

    Returns:
        Dict with per-layer stats and category/depth aggregates.
    """
    results = {"layers": [], "summary": {}}
    scales_by_category = {}  # category → [(depth, median_scale)]

    for name, module in model.named_modules():
        if not (hasattr(module, 'scale') and hasattr(module, '_quantize_ste')):
            continue

        w = module.weight.detach().float()
        flat = w.reshape(-1)

        # ── Layer depth ──
        depth = None
        for part in name.split('.'):
            if part.isdigit():
                depth = int(part)
                break

        category = _classify_layer_category(name)

        # ── Kurtosis (Archie metric 4) ──
        # Bimodal target: -1.8 to -1.9. Gaussian: 0.
        w_mean = flat.mean()
        w_std = flat.std()
        if w_std > 1e-10:
            kurt = ((flat - w_mean) ** 4).mean() / (w_std ** 4) - 3.0
            kurt = kurt.item() if torch.is_tensor(kurt) else kurt
        else:
            kurt = 0.0

        # ── Scale statistics (Archie metric 5) ──
        scales = module.scale.detach().float()
        median_scale = scales.median().item()
        mean_scale = scales.mean().item()
        scale_std = scales.std().item()

        # ── Naive PTQ baseline for comparison ──
        gs = module.group_size
        remainder = flat.numel() % gs
        flat_padded = F.pad(flat, (0, gs - remainder)) if remainder else flat
        groups = flat_padded.reshape(-1, gs)
        naive_scales = groups.abs().mean(dim=1)  # s_g* = mean(|w_i|)

        # Naive PTQ dequant: sign(w_i) * mean(|w_group|)
        naive_median_scale = naive_scales.median().item()

        # ── Metrics requiring base model (Archie metrics 1, 2, 3) ──
        bit_match = None
        ratio_med = None
        adv_ratio_med = None

        if base_model is not None:
            base_module = _find_module(base_model, name)
            if base_module is not None and hasattr(base_module, 'weight'):
                base_w = base_module.weight.detach().float().reshape(-1)
                if base_w.shape == flat.shape:
                    # ── bit_match (Archie metric 1) ──
                    # Fraction of signs preserved vs base model
                    model_signs = flat.sign()
                    base_signs = base_w.sign()
                    mask = base_signs != 0
                    if mask.sum() > 0:
                        bit_match = (model_signs[mask] == base_signs[mask]).float().mean().item()

                    # ── ratio_med (Archie metric 2) ──
                    # median(|w_bonsai_i| / |w_base_i|)
                    # For 1-bit: |w_bonsai_i| = scale of its group
                    # Dequantize: w_hat_i = s_g * sign(w_i)
                    # So |w_hat_i| = s_g for all i in group
                    base_abs = base_w.abs()
                    base_abs_safe = base_abs.clamp(min=1e-10)

                    # Dequantized magnitudes = group scale for each weight
                    learned_s = scales[:groups.shape[0]].abs().clamp(min=1e-8)
                    deq_abs = learned_s.repeat_interleave(gs)[:flat.numel()]
                    ratio_per_weight = deq_abs / base_abs_safe
                    ratio_med = ratio_per_weight.median().item()

                    # ── adv_ratio_med (Archie metric 3) ──
                    # ratio_med_bonsai - ratio_med_naive_ptq
                    naive_s = naive_scales
                    naive_deq_abs = naive_s.repeat_interleave(gs)[:flat.numel()]
                    naive_ratio = naive_deq_abs / base_abs_safe
                    naive_ratio_med = naive_ratio.median().item()
                    adv_ratio_med = ratio_med - naive_ratio_med

        layer_result = {
            "name": name,
            "category": category,
            "depth": depth,
            "params": flat.numel(),
            # Archie's 6 metrics
            "bit_match": round(bit_match, 4) if bit_match is not None else None,
            "ratio_med": round(ratio_med, 4) if ratio_med is not None else None,
            "adv_ratio_med": round(adv_ratio_med, 4) if adv_ratio_med is not None else None,
            "kurtosis": round(kurt, 4),
            "scale_std": round(scale_std, 6),
            "median_scale": round(median_scale, 6),
        }
        results["layers"].append(layer_result)

        if depth is not None:
            scales_by_category.setdefault(category, []).append((depth, median_scale))

    # ── Depth-scale correlation per category (Archie metric 6) ──
    correlations = {}
    for cat, pairs in scales_by_category.items():
        if len(pairs) < 3:
            continue
        depths = torch.tensor([p[0] for p in pairs], dtype=torch.float)
        scs = torch.tensor([p[1] for p in pairs], dtype=torch.float)
        d_mean, s_mean = depths.mean(), scs.mean()
        d_std, s_std = depths.std(), scs.std()
        if d_std > 1e-10 and s_std > 1e-10:
            corr = ((depths - d_mean) * (scs - s_mean)).mean() / (d_std * s_std)
            correlations[cat] = round(corr.item(), 4)

    # ── Category aggregates ──
    by_category = {}
    for cat in ["ffn_up", "ffn_gate", "ffn_down", "attn_q", "attn_k",
                "attn_v", "attn_o", "embed", "lm_head"]:
        cat_layers = [l for l in results["layers"] if l["category"] == cat]
        if not cat_layers:
            continue
        n = len(cat_layers)

        def _avg(key):
            vals = [l[key] for l in cat_layers if l[key] is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        by_category[cat] = {
            "count": n,
            "avg_bit_match": _avg("bit_match"),
            "avg_ratio_med": _avg("ratio_med"),
            "avg_adv_ratio_med": _avg("adv_ratio_med"),
            "avg_kurtosis": _avg("kurtosis"),
            "avg_scale_std": _avg("scale_std"),
            "avg_median_scale": _avg("median_scale"),
            "depth_scale_corr": correlations.get(cat),
        }

    results["summary"] = {
        "num_layers": len(results["layers"]),
        "total_params": sum(l["params"] for l in results["layers"]),
        "by_category": by_category,
        "depth_scale_correlations": correlations,
        "bonsai_targets": {
            "bit_match": "0.70-0.79 (21-30% signs reconfigured after QAT)",
            "ratio_med_ffn_up": "3.53 (1.7B) — magnitude inflation",
            "adv_ratio_med_ffn_up": "2.34 (1.7B), decaying with model size",
            "kurtosis": "-1.8 to -1.9 (bimodal shaping)",
            "depth_scale_corr_ffn_up": "0.877 (1.7B)",
            "scale_std": "compresses with model size (joint learning fingerprint)",
        },
    }

    return results


def _classify_layer_category(name: str) -> str:
    """Classify layer into Bonsai forensic category."""
    if "up_proj" in name or "ffn_up" in name:
        return "ffn_up"
    elif "down_proj" in name or "ffn_down" in name:
        return "ffn_down"
    elif "gate_proj" in name or "ffn_gate" in name:
        return "ffn_gate"
    elif "q_proj" in name or "attn_q" in name:
        return "attn_q"
    elif "k_proj" in name or "attn_k" in name:
        return "attn_k"
    elif "v_proj" in name or "attn_v" in name:
        return "attn_v"
    elif "o_proj" in name or "attn_o" in name:
        return "attn_o"
    elif "embed" in name:
        return "embed"
    elif "lm_head" in name:
        return "lm_head"
    return "other"


def _find_module(model: nn.Module, name: str):
    """Find a module by dotted name path."""
    parts = name.split('.')
    current = model
    for p in parts:
        if p.isdigit():
            try:
                current = current[int(p)]
            except (IndexError, TypeError):
                return None
        else:
            if hasattr(current, p):
                current = getattr(current, p)
            else:
                return None
    return current


def apply_depth_indexed_scales(model: nn.Module, num_layers: int):
    """Apply Bonsai's depth-indexed scale initialization.

    Deeper layers get larger scales (correlation 0.877 in Bonsai).
    Call AFTER from_linear but BEFORE QAT training.
    """
    for name, module in model.named_modules():
        if not (hasattr(module, 'scale') and hasattr(module, '_quantize_ste')):
            continue

        depth = None
        for part in name.split('.'):
            if part.isdigit():
                depth = int(part)
                break

        if depth is not None:
            # Linear scaling: deeper layers get larger scales
            # Normalized to [0.5, 1.5] based on depth position
            depth_factor = 0.5 + (depth / max(num_layers - 1, 1))
            with torch.no_grad():
                module.scale.mul_(depth_factor)
