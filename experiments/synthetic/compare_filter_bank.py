"""Comparison: EMD vs VMD vs old NEMD-sifting vs filter-bank NEMD.

Includes a figure of the learned filter responses for the filter-bank model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal as scisignal

from nemd.classical import ClassicalEMD, VMD
from nemd.model import NEMD, NEMDSifting
from nemd.train import TrainConfig
from nemd.utils import (
    energy_ratio,
    generate_synthetic_signal,
    mode_mixing_index,
    orthogonality_index,
    reconstruction_error,
    to_numpy,
)


matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "figure.dpi": 120,
})


def compute_metrics(signal_np, imfs, residual, components):
    if residual is not None:
        full = np.vstack([imfs, residual[np.newaxis, :]])
    else:
        full = imfs
    return {
        "ortho": orthogonality_index(imfs),
        "energy_ratio": energy_ratio(signal_np, full),
        "mode_mix": mode_mixing_index(components, full),
        "recon_err": reconstruction_error(signal_np, full),
        "n_imfs": imfs.shape[0],
    }


def run_nemd_filter_bank(signal_np, ckpt_path, temperature=0.3):
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config: TrainConfig = blob["config"]
    model = NEMD(
        num_imfs=config.num_imfs,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        kernel_size=config.kernel_size,
        sample_rate=config.sample_rate,
        temperature=temperature,
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    x = torch.from_numpy(signal_np).float().unsqueeze(0)
    with torch.no_grad():
        # sort_by_centroid → IMFs ordered by descending signal-weighted centroid
        imfs, residual, metadata = model(
            x, temperature=temperature, sort_by_centroid=True,
        )
    return (
        to_numpy(imfs.squeeze(0)),
        to_numpy(residual.squeeze(0)),
        to_numpy(metadata["filters"].squeeze(0)),
        to_numpy(metadata["centroids"].squeeze(0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("paper/figures"))
    parser.add_argument("--prefix", type=str, default="phase25b")
    parser.add_argument("--n_samples", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    n = args.n_samples
    fs = float(n)
    t, signal, components = generate_synthetic_signal(
        n_samples=n, duration=1.0,
        components=[
            {"f0": 50.0, "f_mod": 2.0, "a_mod": 0.5},
            {"f0": 15.0, "f_mod": 0.5, "a_mod": 0.3},
            {"f0": 3.0,  "f_mod": 0.1, "a_mod": 0.2},
        ],
        noise_std=0.05, seed=42,
    )

    # ---- Run each method ----
    emd = ClassicalEMD(max_imfs=4)
    emd_imfs = emd.decompose(signal, t)
    if emd_imfs.shape[0] >= 2:
        emd_res = emd_imfs[-1]
        emd_imfs = emd_imfs[:-1]
    else:
        emd_res = None

    vmd = VMD(n_modes=3)
    vmd_imfs = vmd.decompose(signal)
    vmd_res = signal - vmd_imfs.sum(axis=0)

    nemd_imfs, nemd_res, nemd_filters, nemd_centroids = run_nemd_filter_bank(
        signal, args.ckpt, temperature=args.temperature,
    )

    methods = {
        "EMD": (emd_imfs, emd_res),
        "VMD": (vmd_imfs, vmd_res),
        "N-EMD": (nemd_imfs, nemd_res),
    }

    # ---- Metrics table ----
    print(f"\n{'Method':<10} | {'Ortho':>6} | {'Energy':>7} | {'ModeMix':>7} | "
          f"{'ReconErr':>8} | {'# IMFs':>6}")
    print("-" * 68)
    for name, (imfs, res) in methods.items():
        m = compute_metrics(signal, imfs, res, components)
        print(f"{name:<10} | {m['ortho']:>6.3f} | {m['energy_ratio']:>7.3f} | "
              f"{m['mode_mix']:>7.3f} | {m['recon_err']:>8.5f} | {m['n_imfs']:>6}")

    # Signal-weighted centroids for N-EMD (where signal energy actually lands)
    X = np.fft.rfft(signal)
    psd = X.real ** 2 + X.imag ** 2
    freqs_hz = np.linspace(0, fs / 2, psd.shape[-1])
    weighted_centroids = np.array([
        (nemd_filters[k] * psd * freqs_hz).sum() / ((nemd_filters[k] * psd).sum() + 1e-12)
        for k in range(nemd_filters.shape[0])
    ])
    print(f"\nN-EMD filter-only centroids (Hz): {nemd_centroids}")
    print(f"N-EMD signal-weighted   (Hz):     {weighted_centroids}")
    print(f"True components (Hz):             [50.0, 15.0, 3.0]")

    # ---- Plot 1: time-domain IMFs side by side ----
    max_k = 3
    n_rows = max_k + 1
    n_cols = 1 + len(methods)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 1.3 * n_rows), sharex=True,
    )
    axes[0, 0].plot(t, signal, "k", linewidth=0.7)
    axes[0, 0].set_title("Input signal + truth", fontweight="bold")
    axes[0, 0].set_ylabel("Signal")
    for k in range(3):
        axes[k + 1, 0].plot(t, components[k], linewidth=0.7, color="C2")
        axes[k + 1, 0].set_ylabel(f"True {k + 1}\n({[50, 15, 3][k]} Hz)")
    axes[-1, 0].set_xlabel("Time (s)")

    for col, (name, (imfs, res)) in enumerate(methods.items(), start=1):
        for k in range(n_rows - 1):
            ax = axes[k, col]
            if k < imfs.shape[0]:
                ax.plot(t, imfs[k], linewidth=0.7)
            if k == 0:
                ax.set_title(name, fontweight="bold")
        ax = axes[-1, col]
        if res is not None:
            ax.plot(t, res, color="gray", linewidth=0.7)
        ax.set_xlabel("Time (s)")
    for row in axes:
        for ax in row:
            ax.tick_params(labelsize=8)
    fig.suptitle(
        "EMD vs VMD vs Filter-Bank N-EMD — 3-component AM-FM (50/15/3 Hz)",
        fontweight="bold", y=1.002,
    )
    fig.tight_layout()
    out1 = args.out_dir / f"{args.prefix}_compare.png"
    fig.savefig(out1, bbox_inches="tight", dpi=150)
    print(f"\nSaved: {out1}")

    # ---- Plot 2: N-EMD learned filter responses + input spectrum ----
    fig2, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 5), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    colors = ["C0", "C1", "C2"]
    for k in range(nemd_filters.shape[0]):
        ax_top.plot(
            freqs_hz, nemd_filters[k], color=colors[k],
            linewidth=1.2, label=f"Filter {k + 1} (c={nemd_centroids[k]:.1f} Hz)",
        )
    ax_top.set_ylabel("Filter response H_k(f)")
    ax_top.set_title("Learned filter bank (partition of unity)", fontweight="bold")
    ax_top.legend(loc="upper right", fontsize=9)
    ax_top.grid(alpha=0.3)

    ax_bot.semilogy(freqs_hz, psd, "k", linewidth=0.8, alpha=0.7)
    for f0, color in zip([50, 15, 3], colors):
        ax_bot.axvline(f0, color=color, linestyle="--", alpha=0.6, linewidth=1)
    ax_bot.set_xlabel("Frequency (Hz)")
    ax_bot.set_ylabel("Signal PSD")
    ax_bot.set_title("Input signal spectrum (dashed = true carriers)", fontweight="bold")
    ax_bot.grid(alpha=0.3)
    ax_bot.set_xlim(0, fs / 2)

    fig2.tight_layout()
    out2 = args.out_dir / f"{args.prefix}_filters.png"
    fig2.savefig(out2, bbox_inches="tight", dpi=150)
    print(f"Saved: {out2}")

    # ---- Plot 3: spectrograms ----
    fig3, axes3 = plt.subplots(
        3, 3, figsize=(11, 7), sharex=True, sharey=True,
    )
    for col, (name, (imfs, _)) in enumerate(methods.items()):
        for k in range(3):
            ax = axes3[k, col]
            if k < imfs.shape[0]:
                f, tt, Sxx = scisignal.spectrogram(
                    imfs[k], fs=fs, nperseg=min(128, n // 4),
                    noverlap=min(128, n // 4) - 8, window="hann",
                )
                ax.pcolormesh(
                    tt, f, 10 * np.log10(Sxx + 1e-12),
                    shading="gouraud", cmap="viridis", vmin=-60, vmax=-10,
                )
                ax.set_ylim(0, 100)
            if col == 0:
                ax.set_ylabel(f"IMF {k + 1}\nFreq (Hz)")
            if k == 0:
                ax.set_title(name, fontweight="bold")
            if k == 2:
                ax.set_xlabel("Time (s)")
    fig3.suptitle("Per-IMF spectrograms (dB)", fontweight="bold")
    fig3.tight_layout()
    out3 = args.out_dir / f"{args.prefix}_spectrograms.png"
    fig3.savefig(out3, bbox_inches="tight", dpi=150)
    print(f"Saved: {out3}")


if __name__ == "__main__":
    main()
