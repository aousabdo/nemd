"""Build paper figures from the Pass-B multi-seed JSON outputs.

Reads:
  paper/figures/phase3_exp3_passB/sweep_results.json      (cheap baselines, 5 seeds)
  paper/figures/phase3_exp3_passB/nemd_sweep_results.json (N-EMD, 3 seeds)
  paper/figures/phase3_exp3_passB/physics_ablation.json   (physics-lambda sweep)

Writes:
  paper/nemd_tsp/figs/snr_sweep_accuracy_ci.png
  paper/nemd_tsp/figs/physics_ablation.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO  = Path(__file__).resolve().parents[2]
PASSB = REPO / "paper" / "figures" / "phase3_exp3_passB"
FIGS  = REPO / "paper" / "nemd_tsp" / "figs"

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
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
})


PIPELINE_ORDER = [
    ("emd_mlp",         "EMD+MLP",                "#888888"),
    ("vmd_mlp",         "VMD+MLP",                "#4477aa"),
    ("vmd_mlp_big",     "VMD+MLP (big)",          "#225599"),
    ("mel_fb_mlp",      "Mel-FB+MLP",             "#aa8833"),
    ("sincnet_mlp",     "SincNet+MLP",            "#cc9933"),
    ("nemd_pretrained", "N-EMD pre.\\ (e2e)",     "#bb5555"),
    ("nemd_scratch",    "N-EMD scratch (e2e)",    "#cc3344"),
]


def aggregate(sweep: dict, nemd_sweep: dict | None = None) -> dict:
    """Combine cheap and N-EMD sweeps into per-pipeline (snr, mean, std) arrays."""
    agg: dict = {}
    for snr_key, seeds in sweep["snr_results"].items():
        snr = int(snr_key.split("_")[1])
        for seed_key, pipes in seeds.items():
            for pname, pd in pipes.items():
                agg.setdefault(pname, {}).setdefault(snr, []).append(pd["test_acc"])
    if nemd_sweep is not None:
        for snr_key, seeds in nemd_sweep["snr_results"].items():
            snr = int(snr_key.split("_")[1])
            for seed_key, pipes in seeds.items():
                for pname, pd in pipes.items():
                    agg.setdefault(pname, {}).setdefault(snr, []).append(pd["test_acc"])
    return agg


def figure_snr_sweep_ci(agg: dict) -> None:
    """Grouped bar chart over SNR with error bars from multi-seed runs."""
    snrs = sorted({s for pname in agg for s in agg[pname]})
    pipelines = [(k, lbl, col) for (k, lbl, col) in PIPELINE_ORDER if k in agg]

    n_pipe = len(pipelines)
    n_snr  = len(snrs)
    group_w = 0.86
    bar_w   = group_w / n_pipe
    centers = np.arange(n_snr)

    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    for i, (key, label, color) in enumerate(pipelines):
        means = np.array([np.mean(agg[key].get(s, [np.nan])) for s in snrs])
        stds  = np.array([np.std(agg[key].get(s, [0.0]))   for s in snrs])
        xs = centers + (i - (n_pipe - 1) / 2) * bar_w
        ax.bar(
            xs, means, bar_w * 0.94, yerr=stds, capsize=1.5,
            label=label, color=color, edgecolor="black", linewidth=0.35,
            error_kw={"linewidth": 0.6, "ecolor": "black"},
        )

    ax.set_xticks(centers)
    ax.set_xticklabels([f"{s} dB" for s in snrs])
    ax.set_xlabel("Signal-to-noise ratio")
    ax.set_ylabel("Test accuracy (mean $\\pm$ std)")
    ax.set_ylim(0.3, 1.05)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
    ax.axhline(1/3, ls="--", lw=0.6, color="black", alpha=0.6)
    ax.text(n_snr - 0.48, 1/3 + 0.008, "chance", fontsize=6.5,
            alpha=0.6, ha="right")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center", ncol=4, frameon=False, fontsize=7,
        bbox_to_anchor=(0.5, -0.18), handlelength=1.2,
        columnspacing=1.0, handletextpad=0.4,
    )
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    out = FIGS / "snr_sweep_accuracy_ci.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


def figure_physics_ablation(ablation: dict) -> None:
    lambdas = [float(k.split("_")[-1]) for k in ablation["results"].keys()]
    accs_mean = []
    accs_std  = []
    for lk in ablation["results"]:
        accs = [seed_data["test_acc"]
                for seed_data in ablation["results"][lk].values()]
        accs_mean.append(np.mean(accs))
        accs_std.append(np.std(accs))
    lambdas = np.array(lambdas)
    means   = np.array(accs_mean)
    stds    = np.array(accs_std)

    order = np.argsort(lambdas)
    lambdas = lambdas[order]; means = means[order]; stds = stds[order]

    # Replace exact 0 with a small value for log-display
    disp_x = np.where(lambdas == 0, 1e-4, lambdas)
    labels = ["0" if l == 0 else f"{l:g}" for l in lambdas]

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.errorbar(
        disp_x, means, yerr=stds, marker="o", capsize=3,
        color="#cc3344", linewidth=1.2, markersize=5,
    )
    ax.set_xscale("log")
    ax.set_xticks(disp_x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlabel("Physics regulariser weight $\\lambda_{\\mathrm{phys}}$")
    ax.set_ylabel("Test acc. (SNR 10 dB)")
    ax.set_ylim(0.3, 1.0)
    ax.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    out = FIGS / "physics_ablation.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    sweep = json.load(open(PASSB / "sweep_results.json"))
    nemd_sweep = None
    nemd_path = PASSB / "nemd_sweep_results.json"
    if nemd_path.exists():
        nemd_sweep = json.load(open(nemd_path))
    agg = aggregate(sweep, nemd_sweep)
    figure_snr_sweep_ci(agg)

    ablation_path = PASSB / "physics_ablation.json"
    if ablation_path.exists():
        ablation = json.load(open(ablation_path))
        figure_physics_ablation(ablation)
