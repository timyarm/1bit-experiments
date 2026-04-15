"""Bonsai 8B Activation Probing — Map which weight groups fire per domain

Feeds domain-specific text through the model, records activation magnitude
per group of 128 weights across all NativeBitLinear layers.

Output: activation map classifying every group as DEDICATED / SHARED / REDUNDANT

Runs independently from training. Uses saved original scales only.
"""
import modal
import os

app = modal.App("bonsai8b-probe")

vol = modal.Volume.from_name("scale-personalities-8b", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "accelerate", "numpy",
                 "huggingface_hub", "safetensors", "datasets")
)


@app.function(image=image, gpu="T4", timeout=7200,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/checkpoints": vol})
def probe():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import random, time, math, gc, json
    from math import ceil
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset

    DEVICE = "cuda"
    GROUP_SIZE = 128
    random.seed(42)
    torch.manual_seed(42)

    print("=" * 70)
    print("BONSAI 8B: ACTIVATION PROBING — WEIGHT GROUP DOMAIN MAP")
    print("=" * 70, flush=True)

    hf_token = (os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                or None)
    os.environ["HF_TOKEN"] = hf_token

    # ─── NativeBitLinear (same as training) ───
    class NativeBitLinear(nn.Module):
        def __init__(self, in_features, out_features, bias=False, group_size=128):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.group_size = group_size
            n_ints = ceil(in_features / 32)
            n_groups = ceil(in_features / group_size)
            self.register_buffer('packed_signs', torch.zeros(out_features, n_ints, dtype=torch.int32))
            self.group_scales = nn.Parameter(torch.zeros(out_features, n_groups, dtype=torch.float16))
            self.bias = None
            self._activation_magnitude = None

        def forward(self, x):
            if not hasattr(self, '_bit_shifts'):
                self._bit_shifts = torch.arange(32, device=x.device, dtype=torch.int32)
            shifts = self._bit_shifts.to(x.device)
            unpacked = ((self.packed_signs.unsqueeze(2) >> shifts) & 1)
            signs = (unpacked.reshape(self.out_features, -1)[:, :self.in_features].half() * 2 - 1)
            se = self.group_scales.unsqueeze(2).expand(-1, -1, self.group_size)
            sf = se.reshape(self.out_features, -1)[:, :self.in_features]
            w = signs * sf
            output = F.linear(x, w, self.bias)

            # Record per-group activation magnitude (VRAM-efficient)
            # Use output magnitude as proxy — how much does each group contribute?
            with torch.no_grad():
                n_groups = ceil(self.in_features / self.group_size)
                # Simple proxy: per-group scale magnitude × mean |input| per group
                # This avoids materializing the full outer product
                x_flat = x.reshape(-1, x.shape[-1])  # [tokens, in_features]
                # Mean |input| per group
                x_padded = x_flat[:, :n_groups * self.group_size]
                x_grouped = x_padded.reshape(-1, n_groups, self.group_size)
                input_mag = x_grouped.abs().mean(dim=(0, 2))  # [n_groups]
                # Scale magnitude per group (already computed above as sf)
                scale_mag = sf.abs().mean(dim=0)  # [in_features] → group
                scale_grouped = scale_mag[:n_groups * self.group_size].reshape(n_groups, self.group_size).mean(dim=1)
                # Activation = input_magnitude × scale_magnitude per group
                self._activation_magnitude = (input_mag * scale_grouped).cpu()  # [n_groups]

            return output

        @staticmethod
        def from_weight(weight_tensor, group_size=128):
            out_f, in_f = weight_tensor.shape
            layer = NativeBitLinear(in_f, out_f, bias=False, group_size=group_size)
            bits = (weight_tensor > 0).to(torch.int32)
            pad = (32 - in_f % 32) % 32
            if pad > 0: bits = F.pad(bits, (0, pad))
            n_ints = bits.shape[1] // 32
            bits = bits.view(out_f, n_ints, 32)
            packed = torch.zeros(out_f, n_ints, dtype=torch.int32)
            for bit in range(32): packed |= bits[:, :, bit] << bit
            layer.packed_signs.copy_(packed)
            pad_s = (group_size - in_f % group_size) % group_size
            w_abs = F.pad(weight_tensor.abs(), (0, pad_s)) if pad_s > 0 else weight_tensor.abs()
            n_groups = w_abs.shape[1] // group_size
            layer.group_scales.data.copy_(w_abs.view(out_f, n_groups, group_size).mean(dim=2).half())
            return layer

    # ─── Load Model ───
    print("\n[1/4] Loading Bonsai 8B...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("prism-ml/Bonsai-8B-unpacked", token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "prism-ml/Bonsai-8B-unpacked", dtype=torch.float16, device_map="cpu",
        low_cpu_mem_usage=True, token=hf_token
    )
    print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

    print("  Replacing linears...", flush=True)
    replaced = 0
    layer_names = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear): continue
        if "layers." not in name or module.weight.dim() != 2: continue
        parts = name.split('.')
        parent = model
        for p in parts[:-1]: parent = getattr(parent, p)
        native = NativeBitLinear.from_weight(module.weight.data.float(), GROUP_SIZE)
        setattr(parent, parts[-1], native)
        layer_names.append(name)
        replaced += 1
        del module
        if replaced % 50 == 0: gc.collect()
    gc.collect()
    model = model.to(DEVICE)
    model.eval()
    torch.cuda.empty_cache()
    print(f"  {replaced} linears, VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)

    # ─── Load Domain Data ───
    print("\n[2/4] Loading probe data...", flush=True)

    domains = {}

    # Math/reasoning
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    domains["math"] = [f"Q: {gsm[i]['question']}\nA: {gsm[i]['answer']}" for i in range(50)]

    # Code
    domains["code"] = [
        "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item): self.items.append(item)",
        "SELECT u.name, COUNT(o.id) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.name",
        "async def fetch(url):\n    async with aiohttp.ClientSession() as s:\n        return await s.get(url)",
        "def binary_search(arr, t):\n    lo, hi = 0, len(arr)-1\n    while lo <= hi:\n        mid = (lo+hi)//2\n        if arr[mid] == t: return mid",
    ] * 10

    # Tool calling
    domains["tools"] = [
        '{"function": "book_load", "params": {"load_id": "L-4521", "rate": 2850}}',
        '{"function": "check_rate", "params": {"origin": "CHI", "dest": "DEN"}}',
        '{"function": "send_email", "params": {"to": "broker@xyz.com", "subject": "Rate confirmation"}}',
        'Tools: [book_load, check_rate, send_email]\nQuery: Book the load\nCall: {"function": "book_load"}',
        '{"function": "update_status", "params": {"load_id": "L-9982", "status": "in_transit"}}',
    ] * 10

    # Creative/language
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    domains["creative"] = [t["text"] for t in wiki if len(t["text"].strip()) > 100][:50]

    # Retrieval-style
    domains["retrieval"] = []
    for t in wiki:
        text = t["text"]
        if len(text.strip()) > 200:
            sentences = text.split('. ')
            if len(sentences) > 3:
                ctx = '. '.join(sentences[:3])
                ans = '. '.join(sentences[3:])
                domains["retrieval"].append(f"Context: {ctx[:300]}\nBased on the above: {ans[:200]}")
        if len(domains["retrieval"]) >= 50: break

    for dname, texts in domains.items():
        print(f"  {dname}: {len(texts)} examples", flush=True)

    # ─── Probe Activations ───
    print("\n[3/4] Probing activations per domain...", flush=True)

    activation_maps = {}  # {domain: {layer_name: [group_magnitudes]}}

    for dname, texts in domains.items():
        print(f"\n  Probing {dname}...", flush=True)
        activation_maps[dname] = {}

        # Run 20 examples through model, collect activation magnitudes
        for text in texts[:20]:
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).input_ids.to(DEVICE)
            if ids.shape[1] < 5: continue

            with torch.no_grad():
                model(ids)

            # Collect activation magnitudes from all NativeBitLinear layers
            for name in layer_names:
                parts = name.split('.')
                module = model
                for p in parts: module = getattr(module, p)
                if hasattr(module, '_activation_magnitude') and module._activation_magnitude is not None:
                    if name not in activation_maps[dname]:
                        activation_maps[dname][name] = []
                    activation_maps[dname][name].append(module._activation_magnitude.clone())

            del ids
            torch.cuda.empty_cache()

        # Average across examples
        for name in activation_maps[dname]:
            stacked = torch.stack(activation_maps[dname][name])
            activation_maps[dname][name] = stacked.mean(dim=0)

        print(f"    Collected {len(activation_maps[dname])} layers", flush=True)

    # ─── Classify Groups ───
    print("\n[4/4] Classifying weight groups...", flush=True)

    domain_names = list(domains.keys())
    classification = {"dedicated": {}, "shared": {}, "redundant": {}}
    total_groups = 0
    n_dedicated = 0
    n_shared = 0
    n_redundant = 0

    # Per-layer analysis
    layer_stats = {}
    for name in layer_names:
        # Get activation vector for each domain
        acts = {}
        valid = True
        for dname in domain_names:
            if name in activation_maps[dname]:
                acts[dname] = activation_maps[dname][name]
            else:
                valid = False
                break
        if not valid or len(acts) < 3: continue

        n_groups = acts[domain_names[0]].shape[0]
        total_groups += n_groups

        layer_dedicated = 0
        layer_shared = 0
        layer_redundant = 0

        for g in range(n_groups):
            group_acts = {d: acts[d][g].item() for d in domain_names}
            max_act = max(group_acts.values())
            mean_act = sum(group_acts.values()) / len(group_acts)

            if max_act < 0.01:
                # Barely fires for anything
                n_redundant += 1
                layer_redundant += 1
                if name not in classification["redundant"]:
                    classification["redundant"][name] = []
                classification["redundant"][name].append(g)
            else:
                # Check if dedicated (one domain >> others)
                sorted_acts = sorted(group_acts.values(), reverse=True)
                if sorted_acts[0] > 3 * sorted_acts[1] and sorted_acts[0] > 0.02:
                    # Dominant domain
                    dominant = max(group_acts, key=group_acts.get)
                    n_dedicated += 1
                    layer_dedicated += 1
                    if name not in classification["dedicated"]:
                        classification["dedicated"][name] = []
                    classification["dedicated"][name].append((g, dominant, group_acts))
                else:
                    n_shared += 1
                    layer_shared += 1

        # Determine layer type from ffn position
        layer_type = "other"
        if "up_proj" in name or "gate_proj" in name:
            layer_type = "ffn_up/gate"
        elif "down_proj" in name:
            layer_type = "ffn_down"
        elif "q_proj" in name or "k_proj" in name:
            layer_type = "attn_qk"
        elif "v_proj" in name or "o_proj" in name:
            layer_type = "attn_vo"

        layer_idx = -1
        for part in name.split('.'):
            if part.isdigit():
                layer_idx = int(part)
                break

        layer_stats[name] = {
            "layer_idx": layer_idx,
            "layer_type": layer_type,
            "n_groups": n_groups,
            "dedicated": layer_dedicated,
            "shared": layer_shared,
            "redundant": layer_redundant,
            "redundant_pct": layer_redundant / max(n_groups, 1) * 100,
        }

    # ─── Summary ───
    print(f"\n{'='*70}")
    print("ACTIVATION PROBE RESULTS")
    print(f"{'='*70}")
    print(f"  Total groups analyzed: {total_groups}")
    print(f"  Dedicated (single domain):  {n_dedicated} ({n_dedicated/max(total_groups,1)*100:.1f}%)")
    print(f"  Shared (multi-domain):      {n_shared} ({n_shared/max(total_groups,1)*100:.1f}%)")
    print(f"  Redundant (barely active):  {n_redundant} ({n_redundant/max(total_groups,1)*100:.1f}%)")

    # Redundant by layer depth
    print(f"\n  REDUNDANT GROUPS BY LAYER DEPTH:")
    depth_buckets = {"early (0-10)": 0, "mid (11-20)": 0, "late (21-31)": 0}
    depth_totals = {"early (0-10)": 0, "mid (11-20)": 0, "late (21-31)": 0}
    for name, stats in layer_stats.items():
        idx = stats["layer_idx"]
        if idx <= 10:
            bucket = "early (0-10)"
        elif idx <= 20:
            bucket = "mid (11-20)"
        else:
            bucket = "late (21-31)"
        depth_buckets[bucket] += stats["redundant"]
        depth_totals[bucket] += stats["n_groups"]

    for bucket in depth_buckets:
        total = depth_totals[bucket]
        redundant = depth_buckets[bucket]
        pct = redundant / max(total, 1) * 100
        print(f"    {bucket}: {redundant}/{total} redundant ({pct:.1f}%)")

    # Redundant by layer type
    print(f"\n  REDUNDANT GROUPS BY LAYER TYPE:")
    type_buckets = {}
    type_totals = {}
    for name, stats in layer_stats.items():
        lt = stats["layer_type"]
        type_buckets[lt] = type_buckets.get(lt, 0) + stats["redundant"]
        type_totals[lt] = type_totals.get(lt, 0) + stats["n_groups"]
    for lt in sorted(type_buckets.keys()):
        total = type_totals[lt]
        redundant = type_buckets[lt]
        pct = redundant / max(total, 1) * 100
        print(f"    {lt}: {redundant}/{total} redundant ({pct:.1f}%)")

    # Dedicated groups — which domain owns the most?
    print(f"\n  DEDICATED GROUPS BY DOMAIN:")
    domain_counts = {d: 0 for d in domain_names}
    for name, groups in classification["dedicated"].items():
        for g, dominant, acts in groups:
            domain_counts[dominant] += 1
    for d in domain_names:
        print(f"    {d}: {domain_counts[d]} dedicated groups")

    # Top 10 most redundant layers (candidates for sign flipping)
    print(f"\n  TOP 10 MOST REDUNDANT LAYERS (sign flip candidates):")
    sorted_layers = sorted(layer_stats.items(), key=lambda x: x[1]["redundant_pct"], reverse=True)
    for name, stats in sorted_layers[:10]:
        print(f"    {name}: {stats['redundant']}/{stats['n_groups']} ({stats['redundant_pct']:.1f}%) redundant")

    # ─── Save Results ───
    results = {
        "total_groups": total_groups,
        "dedicated": n_dedicated,
        "shared": n_shared,
        "redundant": n_redundant,
        "dedicated_pct": n_dedicated / max(total_groups, 1) * 100,
        "shared_pct": n_shared / max(total_groups, 1) * 100,
        "redundant_pct": n_redundant / max(total_groups, 1) * 100,
        "domain_dedicated_counts": domain_counts,
        "layer_stats": {k: {kk: vv for kk, vv in v.items()} for k, v in layer_stats.items()},
        "redundant_layers_top10": [(name, stats["redundant_pct"]) for name, stats in sorted_layers[:10]],
    }

    with open("/checkpoints/activation_probe_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    vol.commit()
    print(f"\n  Results saved to /checkpoints/activation_probe_results.json", flush=True)
    print(f"  VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)

    return results


@app.local_entrypoint()
def main():
    import json
    results = probe.remote()
    print("\nProbe Results:")
    print(json.dumps(results, indent=2, default=float))
