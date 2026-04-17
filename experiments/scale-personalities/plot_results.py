"""Plot the interpolation curve and data efficiency curve.

Reads interpolation_results.json and data_efficiency_results.json,
produces PNG figures for the writeup.

Usage:
    python plot_results.py interp   # just the interpolation curve
    python plot_results.py deff     # just the data efficiency curve
    python plot_results.py all      # both (default)
"""
import os
import sys
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CKPT = os.path.expanduser("~/freigent/apps/trucksim/checkpoints")
DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "docs", "figures")
os.makedirs(DOCS, exist_ok=True)


def plot_interpolation(path=f"{CKPT}/interpolation_results.json", out=f"{DOCS}/interpolation_curve.png"):
    if not os.path.exists(path):
        print(f"[skip] no file at {path}")
        return
    with open(path) as f:
        raw = json.load(f)

    alphas = sorted(float(k) for k in raw.keys())
    gsm = [raw[f"{a:.1f}"]["gsm8k"] for a in alphas]
    mmlu = [raw[f"{a:.1f}"]["mmlu"] for a in alphas]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, [g * 100 for g in gsm], "o-", color="#d62728", label="GSM8K", linewidth=2)
    ax.plot(alphas, [m * 100 for m in mmlu], "o-", color="#1f77b4", label="MMLU-Knowledge", linewidth=2)

    # Mark the sweet spot (best average)
    avgs = [(gsm[i] + mmlu[i]) / 2 for i in range(len(alphas))]
    best_i = max(range(len(avgs)), key=lambda i: avgs[i])
    ax.axvline(alphas[best_i], color="gray", linestyle="--", alpha=0.6,
               label=f"best avg (α={alphas[best_i]:.1f})")

    ax.set_xlabel("α (0 = pure knowledge scales, 1 = pure math scales)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Scale interpolation: math ↔ knowledge on Bonsai 1.7B")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig(out, dpi=140)
    print(f"wrote {out}")


def plot_data_efficiency(path=f"{CKPT}/data_efficiency_results.json", out=f"{DOCS}/data_efficiency_curve.png"):
    if not os.path.exists(path):
        print(f"[skip] no file at {path}")
        return
    with open(path) as f:
        raw = json.load(f)

    entries = sorted(raw.values(), key=lambda r: r["n_train"])
    ns = [r["n_train"] for r in entries]
    accs = [r["gsm8k"] * 100 for r in entries]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, accs, "o-", color="#2ca02c", linewidth=2, markersize=8)
    for x, y in zip(ns, accs):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    baseline = accs[0] if ns[0] == 0 else None
    if baseline is not None:
        ax.axhline(baseline, color="gray", linestyle="--", alpha=0.5, label=f"baseline ({baseline:.1f}%)")
        ax.legend()

    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("Training examples (log scale)")
    ax.set_ylabel("GSM8K accuracy (%)")
    ax.set_title("Data efficiency: math scale training on Bonsai 1.7B")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("interp", "all"):
        plot_interpolation()
    if which in ("deff", "all"):
        plot_data_efficiency()
