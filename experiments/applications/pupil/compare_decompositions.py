"""Step 5: Compare EMD/VMD/N-EMD decomposition on real pupil data.

Runs each method on 50 clean epochs (balanced across conditions) and
reports standard metrics plus per-IMF band occupancy — the fraction of
each IMF's power that falls in each physiological band.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from nemd.classical import ClassicalEMD, VMD
from nemd.model import NEMD
from nemd.train import TrainConfig
from nemd.utils import (
    energy_ratio,
    mode_mixing_index,
    orthogonality_index,
    to_numpy,
)
from nemd.data.pupil_loader import stream_pupil_data, load_events
from nemd.data.pupil_preprocessing import PupilPreprocessor, PupilPreprocessConfig
from experiments.applications.pupil.synthetic_pupil import PUPIL_BANDS, BAND_NAMES


def load_nemd(ckpt_path: str, sample_rate: float = 100.0) -> NEMD:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config: TrainConfig = blob["config"]
    model = NEMD(
        num_imfs=config.num_imfs,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        kernel_size=config.kernel_size,
        sample_rate=sample_rate,
        temperature=0.5,
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def band_occupancy(imfs: np.ndarray, fs: float, bands: list[tuple] | None = None):
    """Compute per-IMF fraction of power in each physiological band.

    Returns (K, n_bands) array where entry [k, b] is the fraction of
    IMF k's total power that falls in band b.
    """
    if bands is None:
        bands = PUPIL_BANDS
    K, T = imfs.shape
    freqs = np.fft.rfftfreq(T, d=1 / fs)
    occ = np.zeros((K, len(bands)))
    for k in range(K):
        psd = np.abs(np.fft.rfft(imfs[k])) ** 2
        total = psd.sum() + 1e-12
        for b, (lo, hi) in enumerate(bands):
            in_band = psd[(freqs >= lo) & (freqs <= hi)].sum()
            occ[k, b] = in_band / total
    return occ


def run_methods(epoch: np.ndarray, fs: float, nemd_model: NEMD, K: int = 4):
    """Decompose one epoch with EMD, VMD, and N-EMD. Return dict of IMF arrays."""
    T = len(epoch)
    t = np.arange(T) / fs
    results = {}

    # EMD
    emd = ClassicalEMD(max_imfs=K + 1)
    imfs_emd = emd.decompose(epoch, t)
    if imfs_emd.shape[0] > K:
        imfs_emd = imfs_emd[:K]
    elif imfs_emd.shape[0] < K:
        pad = np.zeros((K - imfs_emd.shape[0], T))
        imfs_emd = np.vstack([imfs_emd, pad])
    results["EMD"] = imfs_emd

    # VMD
    vmd = VMD(n_modes=K)
    results["VMD"] = vmd.decompose(epoch)

    # N-EMD
    x = torch.from_numpy(epoch).float().unsqueeze(0)
    with torch.no_grad():
        imfs, _, _ = nemd_model(x, temperature=0.5, sort_by_centroid=True)
    results["N-EMD"] = to_numpy(imfs.squeeze(0))

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="N-EMD checkpoint path")
    parser.add_argument("--n-epochs-per-cond", type=int, default=8,
                        help="Epochs per condition to sample (8 × 6 = 48 total)")
    parser.add_argument("--out-dir", type=str, default="paper/figures/phase4")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nemd_model = load_nemd(args.ckpt, sample_rate=100.0)
    K = nemd_model.num_imfs
    fs = 100.0
    rng = np.random.default_rng(args.seed)

    # Collect epochs from multiple subjects
    cache = Path("data/ds003838_cache")
    repo = Path("data/ds003838")
    parquets = sorted(cache.glob("*_eye0.parquet"))

    proc = PupilPreprocessor(PupilPreprocessConfig())
    all_epochs = []
    all_conds = []
    all_subs = []

    print("Collecting epochs from all subjects...")
    for pq in parquets:
        sub = pq.stem.split("_")[0]
        try:
            import pandas as pd
            df = pd.read_parquet(pq)
            events = load_events(str(repo), sub)
            result = proc.preprocess_subject(df, events)
            for ep, cond in zip(result["epochs"], result["conditions"]):
                all_epochs.append(ep)
                all_conds.append(cond)
                all_subs.append(sub)
        except Exception as e:
            print(f"  {sub}: skip ({e})")

    print(f"Total epochs collected: {len(all_epochs)}")

    # Sample balanced subset
    from collections import Counter
    cond_counts = Counter(all_conds)
    print(f"Conditions: {dict(cond_counts)}")

    sampled_idx = []
    for cond in sorted(set(all_conds)):
        cond_idx = [i for i, c in enumerate(all_conds) if c == cond]
        n_sample = min(args.n_epochs_per_cond, len(cond_idx))
        chosen = rng.choice(cond_idx, size=n_sample, replace=False)
        sampled_idx.extend(chosen.tolist())

    rng.shuffle(sampled_idx)
    print(f"Sampled {len(sampled_idx)} epochs for comparison")

    # Run comparison
    methods = ["EMD", "VMD", "N-EMD"]
    all_metrics = {m: [] for m in methods}
    all_occupancy = {m: [] for m in methods}
    all_timings = {m: [] for m in methods}

    for i, idx in enumerate(sampled_idx):
        epoch = all_epochs[idx]
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing epoch {i + 1}/{len(sampled_idx)}...")

        decomps = run_methods(epoch, fs, nemd_model, K=K)
        for method_name, imfs in decomps.items():
            full = np.vstack([imfs, np.zeros((1, imfs.shape[-1]))])
            all_metrics[method_name].append({
                "ortho": orthogonality_index(imfs),
                "energy": energy_ratio(epoch, full),
            })
            all_occupancy[method_name].append(band_occupancy(imfs, fs))

    # Aggregate
    print(f"\n{'Method':<8} | {'Ortho':>13} | {'Energy':>13}")
    print("-" * 45)
    summary = {}
    for m in methods:
        orthos = [x["ortho"] for x in all_metrics[m]]
        energies = [x["energy"] for x in all_metrics[m]]
        print(f"{m:<8} | {np.mean(orthos):>5.3f} ± {np.std(orthos):>5.3f} | "
              f"{np.mean(energies):>5.3f} ± {np.std(energies):>5.3f}")
        summary[m] = {
            "ortho_mean": float(np.mean(orthos)),
            "ortho_std": float(np.std(orthos)),
            "energy_mean": float(np.mean(energies)),
            "energy_std": float(np.std(energies)),
        }

    # Band occupancy table
    print(f"\n=== Per-IMF Band Occupancy (%) ===")
    for m in methods:
        avg_occ = np.mean(all_occupancy[m], axis=0) * 100  # (K, n_bands)
        print(f"\n{m}:")
        header = f"  {'IMF':>5} | " + " | ".join(f"{bn:>12}" for bn in BAND_NAMES) + " | Out-of-band"
        print(header)
        print("  " + "-" * len(header))
        for k in range(K):
            row = f"  IMF {k + 1} | " + " | ".join(f"{avg_occ[k, b]:>11.1f}%" for b in range(len(BAND_NAMES)))
            out_of_band = 100.0 - avg_occ[k].sum()
            row += f" | {out_of_band:>10.1f}%"
            print(row)
        summary[m]["band_occupancy"] = avg_occ.tolist()

    # Save
    with open(out_dir / "step5_decomposition_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_dir / 'step5_decomposition_summary.json'}")

    # Plot: band occupancy heatmap
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, m in zip(axes, methods):
        avg_occ = np.mean(all_occupancy[m], axis=0) * 100
        im = ax.imshow(avg_occ, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
        ax.set_xticks(range(len(BAND_NAMES)))
        ax.set_xticklabels(BAND_NAMES, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(K))
        ax.set_yticklabels([f"IMF {k + 1}" for k in range(K)])
        ax.set_title(m, fontweight="bold")
        for i in range(K):
            for j in range(len(BAND_NAMES)):
                ax.text(j, i, f"{avg_occ[i, j]:.0f}%", ha="center", va="center",
                        fontsize=8, color="white" if avg_occ[i, j] > 50 else "black")
    fig.colorbar(im, ax=axes[-1], label="% power in band")
    fig.suptitle("Per-IMF band occupancy (real pupil data)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "step5_band_occupancy.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {out_dir / 'step5_band_occupancy.png'}")

    # Plot: example decomposition for one epoch
    idx0 = sampled_idx[0]
    epoch0 = all_epochs[idx0]
    cond0 = all_conds[idx0]
    decomps0 = run_methods(epoch0, fs, nemd_model, K=K)
    t = np.arange(len(epoch0)) / fs

    fig2, axes2 = plt.subplots(K + 1, 3, figsize=(14, 2 * (K + 1)), sharex=True)
    for col, m in enumerate(methods):
        axes2[0, col].plot(t, epoch0, "k", linewidth=0.6)
        axes2[0, col].set_title(m, fontweight="bold")
        if col == 0:
            axes2[0, col].set_ylabel("Signal")
        imfs = decomps0[m]
        for k in range(K):
            axes2[k + 1, col].plot(t, imfs[k], linewidth=0.6)
            if col == 0:
                axes2[k + 1, col].set_ylabel(f"IMF {k + 1}")
    axes2[-1, 1].set_xlabel("Time (s)")
    fig2.suptitle(f"Decomposition comparison — {cond0}", fontweight="bold", y=1.002)
    fig2.tight_layout()
    fig2.savefig(out_dir / "step5_example_decomposition.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {out_dir / 'step5_example_decomposition.png'}")


if __name__ == "__main__":
    main()
