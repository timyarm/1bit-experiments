"""Fast 1-bit inference pipeline — no HuggingFace, minimal Python overhead.

Fused Triton kernels for:
- RMSNorm
- Packed 1-bit matmul (signs uint8 + fp16 scales)
- RoPE (inline with attention)
- Attention with KV cache

Architecture: Qwen3/Bonsai 1.7B
  hidden=2048, layers=28, heads=16, kv_heads=8, intermediate=6144
  head_dim=128, group_size=128, vocab=151669
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import math
import time
from pathlib import Path


# ─── Triton Kernels ──────────────────────────────────────────────────────────

@torch.compile(mode="reduce-overhead", fullgraph=True)
def rms_norm(x, weight, eps=1e-6):
    """Compiled RMSNorm."""
    x_f32 = x.float()
    rms = torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + eps)
    return (x_f32 * rms * weight.float()).half()


@triton.jit
def _packed_1bit_mv(
    x_ptr, w_ptr, scales_ptr, out_ptr,
    N, K, K_packed,
    group_size: tl.constexpr,
    groups_per_row: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_KB: tl.constexpr,
):
    """Single-vector × packed 1-bit matrix."""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for kb in range(0, K_packed, BLOCK_KB):
        offs_kb = kb + tl.arange(0, BLOCK_KB)
        kb_mask = offs_kb < K_packed
        w_offs = offs_n[:, None] * K_packed + offs_kb[None, :]
        packed = tl.load(w_ptr + w_offs, mask=n_mask[:, None] & kb_mask[None, :], other=0).to(tl.uint8)

        for bit in tl.static_range(8):
            k_idx = (kb + tl.arange(0, BLOCK_KB)) * 8 + bit
            k_mask = k_idx < K
            sign = ((packed >> bit) & 1).to(tl.float32) * 2.0 - 1.0
            g_idx = offs_n[:, None] * groups_per_row + k_idx[None, :] // group_size
            scale = tl.load(scales_ptr + g_idx, mask=n_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
            x_val = tl.load(x_ptr + k_idx, mask=k_mask, other=0.0).to(tl.float32)
            acc += tl.sum(sign * scale * x_val[None, :], axis=1)

    tl.store(out_ptr + offs_n, acc.to(tl.float16), mask=n_mask)


def packed_matvec(x_1d, w_packed, scales, N, K, group_size=128):
    out = torch.empty(N, dtype=torch.float16, device=x_1d.device)
    K_packed = K // 8
    groups_per_row = K // group_size
    BLOCK_N = 64
    BLOCK_KB = min(32, K_packed)
    _packed_1bit_mv[(triton.cdiv(N, BLOCK_N),)](
        x_1d, w_packed, scales, out,
        N, K, K_packed, group_size=group_size, groups_per_row=groups_per_row,
        BLOCK_N=BLOCK_N, BLOCK_KB=BLOCK_KB,
    )
    return out


# ─── Model Components ────────────────────────────────────────────────────────

class PackedLinear:
    """Packed 1-bit linear — just data, no nn.Module overhead."""
    __slots__ = ['packed_w', 'scales', 'out_features', 'in_features', 'group_size']

    def __init__(self, weight, group_size=128):
        self.out_features, self.in_features = weight.shape
        self.group_size = group_size
        signs = (weight > 0).to(torch.uint8)
        self.packed_w = self._pack(signs, self.out_features, self.in_features)
        w_flat = weight.reshape(-1, group_size)
        self.scales = w_flat.abs().mean(dim=1).half()

    @staticmethod
    def _pack(signs, out_f, in_f):
        flat = signs.reshape(-1)
        pad = (8 - flat.shape[0] % 8) % 8
        if pad:
            flat = F.pad(flat, (0, pad))
        flat = flat.reshape(-1, 8)
        packed = torch.zeros(flat.shape[0], dtype=torch.uint8, device=flat.device)
        for i in range(8):
            packed |= (flat[:, i] << i)
        return packed.reshape(out_f, in_f // 8)

    def to(self, device):
        self.packed_w = self.packed_w.to(device)
        self.scales = self.scales.to(device)
        return self

    def __call__(self, x):
        if x.dim() == 1:
            return packed_matvec(x, self.packed_w, self.scales,
                                 self.out_features, self.in_features, self.group_size)
        elif x.shape[0] == 1:
            out = packed_matvec(x.squeeze(0), self.packed_w, self.scales,
                                self.out_features, self.in_features, self.group_size)
            return out.unsqueeze(0)
        else:
            # Batch: loop over rows (still fast for small batches)
            outs = []
            for i in range(x.shape[0]):
                outs.append(packed_matvec(x[i], self.packed_w, self.scales,
                                          self.out_features, self.in_features, self.group_size))
            return torch.stack(outs)

    def nbytes(self):
        return self.packed_w.numel() + self.scales.numel() * 2


class TransformerLayer:
    """Single transformer layer — no nn.Module, pure function calls."""

    def __init__(self, q_proj, k_proj, v_proj, o_proj, gate, up, down,
                 input_norm_w, post_norm_w, q_norm_w, k_norm_w,
                 num_heads, num_kv_heads, head_dim, eps=1e-6):
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.gate = gate
        self.up = up
        self.down = down
        self.input_norm_w = input_norm_w
        self.post_norm_w = post_norm_w
        self.q_norm_w = q_norm_w
        self.k_norm_w = k_norm_w
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.eps = eps
        self.kv_repeat = num_heads // num_kv_heads

    def to(self, device):
        self.q_proj.to(device)
        self.k_proj.to(device)
        self.v_proj.to(device)
        self.o_proj.to(device)
        self.gate.to(device)
        self.up.to(device)
        self.down.to(device)
        self.input_norm_w = self.input_norm_w.to(device)
        self.post_norm_w = self.post_norm_w.to(device)
        self.q_norm_w = self.q_norm_w.to(device)
        self.k_norm_w = self.k_norm_w.to(device)
        return self

    def forward(self, h, pos, k_cache, v_cache, cos, sin):
        """h: [D], pos: int, k_cache/v_cache: [max_seq, kv_heads, head_dim]"""
        D = h.shape[0]

        # Input norm
        normed = rms_norm(h.unsqueeze(0), self.input_norm_w, self.eps)[0]

        # QKV projections
        q = self.q_proj(normed)  # [num_heads * head_dim]
        k = self.k_proj(normed)  # [num_kv_heads * head_dim]
        v = self.v_proj(normed)  # [num_kv_heads * head_dim]

        # Reshape for heads
        q = q.reshape(self.num_heads, self.head_dim)
        k = k.reshape(self.num_kv_heads, self.head_dim)
        v = v.reshape(self.num_kv_heads, self.head_dim)

        # Q/K norms
        q = _apply_head_norm(q, self.q_norm_w)
        k = _apply_head_norm(k, self.k_norm_w)

        # RoPE
        q = _apply_rope(q, cos[pos], sin[pos])
        k = _apply_rope(k, cos[pos], sin[pos])

        # Update KV cache
        k_cache[pos] = k
        v_cache[pos] = v

        # Attention
        attn_out = _attention(q, k_cache[:pos+1], v_cache[:pos+1], self.kv_repeat)

        # Output projection
        attn_out = self.o_proj(attn_out.reshape(-1))

        # Residual
        h = h + attn_out

        # MLP
        normed2 = rms_norm(h.unsqueeze(0), self.post_norm_w, self.eps)[0]
        gate_out = self.gate(normed2)
        up_out = self.up(normed2)
        mlp_out = F.silu(gate_out) * up_out
        mlp_out = self.down(mlp_out)

        h = h + mlp_out
        return h


def _apply_head_norm(x, w):
    """RMSNorm per head. x: [num_heads, head_dim], w: [head_dim]"""
    rms = x.float().pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
    return (x.float() * rms * w.float()).half()


def _apply_rope(x, cos, sin):
    """Apply rotary embeddings. x: [num_heads, head_dim], cos/sin: [head_dim]"""
    d2 = x.shape[-1] // 2
    x1 = x[..., :d2].float()
    x2 = x[..., d2:].float()
    c = cos[:d2].float()
    s = sin[:d2].float()
    out1 = x1 * c - x2 * s
    out2 = x2 * c + x1 * s
    return torch.cat([out1, out2], dim=-1).half()


def _attention(q, k_cache, v_cache, kv_repeat):
    """Compute attention. q: [H, D], k_cache: [T, KH, D], v_cache: [T, KH, D]"""
    H = q.shape[0]
    KH = k_cache.shape[1]
    D = q.shape[1]
    T = k_cache.shape[0]

    # Repeat KV heads to match Q heads
    if kv_repeat > 1:
        k_cache = k_cache.unsqueeze(2).expand(-1, -1, kv_repeat, -1).reshape(T, H, D)
        v_cache = v_cache.unsqueeze(2).expand(-1, -1, kv_repeat, -1).reshape(T, H, D)

    # scores: [H, T]
    scale = 1.0 / math.sqrt(D)
    scores = torch.einsum('hd,thd->ht', q.float(), k_cache.float()) * scale
    attn = F.softmax(scores, dim=-1)

    # output: [H, D]
    out = torch.einsum('ht,thd->hd', attn, v_cache.float())
    return out.half()


# ─── Full Model ──────────────────────────────────────────────────────────────

class Fast1BitModel:
    """Minimal 1-bit transformer — no nn.Module, no HuggingFace."""

    def __init__(self, embed_w, layers, final_norm_w, lm_head_w, config):
        self.embed = embed_w  # [vocab, hidden] fp16
        self.layers = layers  # list of TransformerLayer
        self.final_norm_w = final_norm_w  # [hidden] fp16
        self.lm_head_w = lm_head_w  # [vocab, hidden] fp16 (tied with embed)
        self.config = config

        # Precompute RoPE
        head_dim = config['head_dim']
        max_seq = config.get('max_seq', 2048)
        theta = config.get('rope_theta', 1000000.0)
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq).float()
        angles = torch.outer(t, freqs)
        self.cos = angles.cos().half()
        self.sin = angles.sin().half()

    def to(self, device):
        self.embed = self.embed.to(device)
        self.final_norm_w = self.final_norm_w.to(device)
        self.lm_head_w = self.lm_head_w.to(device)
        self.cos = self.cos.to(device)
        self.sin = self.sin.to(device)
        for layer in self.layers:
            layer.to(device)
        return self

    def generate(self, input_ids, max_new_tokens=50):
        """Autoregressive generation — optimized for single-token steps."""
        device = self.embed.device
        config = self.config
        max_seq = max(len(input_ids) + max_new_tokens, 512)

        # Init KV caches
        kv_caches = []
        for _ in self.layers:
            k = torch.zeros(max_seq, config['num_kv_heads'], config['head_dim'],
                           dtype=torch.float16, device=device)
            v = torch.zeros_like(k)
            kv_caches.append((k, v))

        # Prefill
        generated = list(input_ids)
        for pos in range(len(input_ids)):
            h = self.embed[generated[pos]]
            for i, layer in enumerate(self.layers):
                h = layer.forward(h, pos, kv_caches[i][0], kv_caches[i][1],
                                  self.cos, self.sin)
            if pos == len(input_ids) - 1:
                h_norm = rms_norm(h.unsqueeze(0), self.final_norm_w).squeeze(0)
                logits = F.linear(h_norm.unsqueeze(0), self.lm_head_w).squeeze(0)
                next_token = logits.argmax().item()
                generated.append(next_token)

        # Generate
        for step in range(max_new_tokens - 1):
            pos = len(generated) - 1
            h = self.embed[generated[-1]]
            for i, layer in enumerate(self.layers):
                h = layer.forward(h, pos, kv_caches[i][0], kv_caches[i][1],
                                  self.cos, self.sin)
            h_norm = rms_norm(h.unsqueeze(0), self.final_norm_w).squeeze(0)
            logits = F.linear(h_norm.unsqueeze(0), self.lm_head_w).squeeze(0)
            next_token = logits.argmax().item()
            generated.append(next_token)

            if next_token == 151643:  # eos
                break

        return generated

    def vram_usage(self):
        total = self.embed.numel() * 2 + self.final_norm_w.numel() * 2
        for layer in self.layers:
            for proj in [layer.q_proj, layer.k_proj, layer.v_proj, layer.o_proj,
                         layer.gate, layer.up, layer.down]:
                total += proj.nbytes()
            total += layer.input_norm_w.numel() * 2
            total += layer.post_norm_w.numel() * 2
        return total


# ─── Load from HuggingFace weights ──────────────────────────────────────────

def load_from_hf(model_id="prism-ml/Bonsai-1.7B-unpacked", device="cuda", group_size=128):
    """Load HF model and convert to fast pipeline."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print("Loading HF weights (CPU)...")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, trust_remote_code=True, device_map="cpu"
    )

    hidden = config.hidden_size
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads
    head_dim = hidden // n_heads
    intermediate = config.intermediate_size

    print(f"Converting {n_layers} layers to packed 1-bit...")

    # Embed + final norm
    embed_w = hf_model.model.embed_tokens.weight.data.half()
    final_norm_w = hf_model.model.norm.weight.data.half()
    # Tied weights
    lm_head_w = embed_w

    layers = []
    for i in range(n_layers):
        hf_layer = hf_model.model.layers[i]
        attn = hf_layer.self_attn
        mlp = hf_layer.mlp

        layer = TransformerLayer(
            q_proj=PackedLinear(attn.q_proj.weight.data, group_size),
            k_proj=PackedLinear(attn.k_proj.weight.data, group_size),
            v_proj=PackedLinear(attn.v_proj.weight.data, group_size),
            o_proj=PackedLinear(attn.o_proj.weight.data, group_size),
            gate=PackedLinear(mlp.gate_proj.weight.data, group_size),
            up=PackedLinear(mlp.up_proj.weight.data, group_size),
            down=PackedLinear(mlp.down_proj.weight.data, group_size),
            input_norm_w=hf_layer.input_layernorm.weight.data.half(),
            post_norm_w=hf_layer.post_attention_layernorm.weight.data.half(),
            q_norm_w=attn.q_norm.weight.data.half(),
            k_norm_w=attn.k_norm.weight.data.half(),
            num_heads=n_heads,
            num_kv_heads=n_kv_heads,
            head_dim=head_dim,
        )
        layers.append(layer)

    del hf_model
    torch.cuda.empty_cache()

    model_config = {
        'hidden_size': hidden,
        'num_layers': n_layers,
        'num_heads': n_heads,
        'num_kv_heads': n_kv_heads,
        'head_dim': head_dim,
        'intermediate_size': intermediate,
        'max_seq': 2048,
        'rope_theta': getattr(config, 'rope_theta', 1000000.0) or 1000000.0,
    }

    model = Fast1BitModel(embed_w, layers, final_norm_w, lm_head_w, model_config)
    print(f"Moving to {device}...")
    model.to(device)

    vram = torch.cuda.memory_allocated() / 1e9
    print(f"VRAM: {vram:.2f} GB")

    return model, tokenizer


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, tokenizer = load_from_hf()

    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt)
    print(f"\nPrompt: {prompt}")
    print(f"Tokens: {len(input_ids)}")

    # Warmup
    model.generate(input_ids, max_new_tokens=5)

    # Bench
    torch.cuda.synchronize()
    t0 = time.time()
    output_ids = model.generate(input_ids, max_new_tokens=50)
    torch.cuda.synchronize()

    n_new = len(output_ids) - len(input_ids)
    elapsed = time.time() - t0
    text = tokenizer.decode(output_ids)
    print(f"\nGenerated {n_new} tokens in {elapsed:.2f}s = {n_new/elapsed:.1f} tok/s")
    print(f"Output: {text[:120]}")
    print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
