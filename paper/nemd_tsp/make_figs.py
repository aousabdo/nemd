"""Generate paper figures that are derived from experimental JSON summaries.

Run from the repo root:

    python paper/nemd_tsp/make_figs.py

Outputs PNGs into paper/nemd_tsp/figs/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]
FIGDIR = REPO / "paper" / "nemd_tsp" / "figs"
FIGDIR.mkdir(parents=True, exist_ok=True)

EXP3_SNR = {
    3:  REPO / "paper/figures/phase3_exp3/summary.json",
    10: REPO / "paper/figures/phase3_exp3_snr10/summary.json",
    20: REPO / "paper/figures/phase3_exp3_snr20/summary.json",
}
EXP2 = REPO / "paper/figures/phase3_exp2/summary.json"

# ---------------------------------------------------------------------
# LaTeX-compatible rcParams
# ---------------------------------------------------------------------

mpl.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["STIXGeneral", "Computer Modern", "Times"],
    "mathtext.fontset":   "stix",
    "axes.labelsize":     9,
    "axes.titlesize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "axes.linewidth":     0.7,
    "xtick.major.width":  0.7,
    "ytick.major.width":  0.7,
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
})

# Pipeline display order, labels, colors
PIPELINE_ORDER = [
    ("emd_mlp",          "EMD+MLP",       "#888888"),
    ("vmd_mlp",          "VMD+MLP",       "#4477aa"),
    ("nemd_pretrained",  "N-EMD pre.+MLP",  "#bb5555"),
    ("nemd_scratch",     "N-EMD scratch (e2e)", "#cc3344"),
    ("raw_cnn",          "Raw CNN",        "#338833"),
]

# ---------------------------------------------------------------------
# Figure 1: SNR-sweep accuracy bar chart
# ---------------------------------------------------------------------

def figure_snr_sweep() -> None:
    # Load all three summaries
    snr_levels = sorted(EXP3_SNR.keys())
    data = {snr: json.load(open(p)) for snr, p in EXP3_SNR.items()}
    acc = {snr: {p["pipeline"]: p["test_acc"] for p in d["pipelines"]}
           for snr, d in data.items()}

    # Sanity-check pipeline keys
    for snr in snr_levels:
        missing = {k for k, _, _ in PIPELINE_ORDER} - set(acc[snr])
        if missing:
            raise RuntimeError(f"Missing pipelines at SNR={snr}: {missing}")

    # Bar layout
    n_pipe = len(PIPELINE_ORDER)
    n_snr = len(snr_levels)
    group_w = 0.82
    bar_w = group_w / n_pipe
    group_centers = np.arange(n_snr)

    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    for i, (key, label, color) in enumerate(PIPELINE_ORDER):
        xs = group_centers + (i - (n_pipe - 1) / 2) * bar_w
        ys = [acc[snr][key] for snr in snr_levels]
        ax.bar(xs, ys, bar_w * 0.94, label=label, color=color,
               edgecolor="black", linewidth=0.4)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.008, f"{y*100:.1f}", ha="center", va="bottom",
                    fontsize=6.5)

    ax.set_xticks(group_centers)
    ax.set_xticklabels([f"{s} dB" for s in snr_levels])
    ax.set_xlabel("Signal-to-noise ratio")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.3, 1.05)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    # Chance line (3 classes)
    ax.axhline(1/3, linestyle="--", linewidth=0.6, color="black",
               alpha=0.6)
    ax.text(n_snr - 0.48, 1/3 + 0.008, "chance", fontsize=6.5,
            alpha=0.6, ha="right")

    ax.legend(loc="lower right", ncol=5, frameon=False,
              handlelength=1.2, columnspacing=1.0, handletextpad=0.4,
              bbox_to_anchor=(1.0, -0.33))

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    out = FIGDIR / "snr_sweep_accuracy.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


# ---------------------------------------------------------------------
# Figure 2: Generalization across test distributions (mode mixing, ortho,
# energy) — three mini panels
# ---------------------------------------------------------------------

SET_ORDER = [
    ("A_in_dist",    "A: in-dist."),
    ("B_high_freq",  "B: high-freq."),
    ("C_noisy_5dB",  "C: noisy 5 dB"),
    ("D_damped",     "D: damped"),
    ("E_K_mismatch", "E: K mismatch"),
]
METHOD_COLORS = {
    "N-EMD": "#cc3344",
    "VMD":   "#4477aa",
    "EMD":   "#888888",
}


def figure_generalization() -> None:
    d = json.load(open(EXP2))
    metrics = [
        ("mode_mix_mean", "Mode mixing\n(lower is better)", (0.0, 0.60)),
        ("ortho_mean",    "Orthogonality index\n(lower is better)", (0.0, 0.15)),
        ("energy_mean",   "Energy ratio\n(closer to 1 is better)", (0.55, 1.05)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    n_sets = len(SET_ORDER)
    n_methods = 3
    group_w = 0.76
    bar_w = group_w / n_methods
    centers = np.arange(n_sets)

    for ax, (metric, ylabel, ylim) in zip(axes, metrics):
        for i, (method, color) in enumerate(METHOD_COLORS.items()):
            xs = centers + (i - 1) * bar_w
            ys = [d[s][method][metric] for s, _ in SET_ORDER]
            ax.bar(xs, ys, bar_w * 0.92, color=color,
                   edgecolor="black", linewidth=0.3,
                   label=method if ax is axes[0] else None)
        ax.set_xticks(centers)
        ax.set_xticklabels([lbl for _, lbl in SET_ORDER],
                           rotation=35, ha="right", fontsize=6.5)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if metric == "energy_mean":
            ax.axhline(1.0, linestyle="--", linewidth=0.5, color="black",
                       alpha=0.5)

    axes[0].legend(loc="upper right", frameon=False, fontsize=7,
                   handlelength=1.2)

    fig.tight_layout()
    out = FIGDIR / "generalization_metrics.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


# ---------------------------------------------------------------------
# Figure 3: Per-signal inference time across methods
# ---------------------------------------------------------------------

def figure_inference_time() -> None:
    d = json.load(open(EXP2))
    # Take the in-distribution set for the representative number
    methods = ["EMD", "VMD", "N-EMD"]
    times = [d["A_in_dist"][m]["per_signal_ms"] for m in methods]
    # Note that VMD's timing is constant per signal (no amortisation)
    # whereas N-EMD is a single batched forward pass at inference; we
    # show the single-signal CPU time for fair comparison.
    colors = [METHOD_COLORS[m] for m in methods]

    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    bars = ax.bar(np.arange(len(methods)), times,
                  color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods)
    ax.set_ylabel("Per-signal inference (ms)")
    for b, t in zip(bars, times):
        ax.text(b.get_x() + b.get_width() / 2,
                t + 0.15, f"{t:.1f}",
                ha="center", va="bottom", fontsize=7)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    out = FIGDIR / "inference_time.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


# ---------------------------------------------------------------------

if __name__ == "__main__":
    figure_snr_sweep()
    figure_generalization()
    figure_inference_time()
