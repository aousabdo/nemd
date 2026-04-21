"""Compare EMD vs VMD vs N-EMD on the canonical 3-component AM-FM signal.

Generates a comparison figure with time-domain IMFs and per-IMF spectrograms
for all three methods, plus a metrics table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
from scipy import signal as scisignal

from nemd.classical import ClassicalEMD, VMD
from nemd.model import NEMD, SiftNetConfig
from nemd.train import TrainConfig
from nemd.utils import (
    energy_ratio,
    generate_synthetic_signal,
    mode_mixing_index,
    monotonicity_score,
    orthogonality_index,
    reconstruction_error,
    to_numpy,
)


matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "figure.dpi": 120,
})


def spectral_centroids(imfs: np.ndarray, nyquist: float) -> np.ndarray:
    """Return the spectral centroid (Hz) of each IMF row."""
    X = np.fft.rfft(imfs, axis=-1)
    psd = X.real ** 2 + X.imag ** 2
    freqs = np.linspace(0, nyquist, psd.shape[-1])
    return (psd * freqs).sum(axis=-1) / (psd.sum(axis=-1) + 1e-12)


def run_nemd(signal_np: np.ndarray, ckpt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a trained N-EMD model and decompose the given signal."""
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config: TrainConfig = blob["config"]
    sift_cfg = config.to_sift_config()
    model = NEMD(max_imfs=config.max_imfs, sift_config=sift_cfg)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    x = torch.from_numpy(signal_np).float().unsqueeze(0)
    with torch.no_grad():
        imfs, residual = model(x, num_imfs=3)
    return to_numpy(imfs.squeeze(0)), to_numpy(residual.squeeze(0))


def compute_metrics(
    signal_np: np.ndarray,
    imfs: np.ndarray,
    residual: np.ndarray | None,
    components: list,
) -> dict:
    """Compute the full metric suite."""
    if residual is None:
        full = imfs
    else:
        full = np.vstack([imfs, residual[np.newaxis, :]])
    return {
        "ortho": orthogonality_index(imfs),
        "energy_ratio": energy_ratio(signal_np, full),
        "mode_mix": mode_mixing_index(components, full),
        "recon_err": reconstruction_error(signal_np, full),
        "n_imfs": imfs.shape[0],
    }


def plot_decomposition(
    ax_grid: np.ndarray,
    t: np.ndarray,
    imfs: np.ndarray,
    residual: np.ndarray | None,
    title: str,
    col: int,
) -> None:
    """Plot (n_imfs + 1) rows in column ``col`` of a shared axes grid."""
    for k in range(imfs.shape[0]):
        ax = ax_grid[k, col]
        ax.plot(t, imfs[k], linewidth=0.7)
        ax.set_ylabel(f"IMF {k + 1}")
        if k == 0:
            ax.set_title(title, fontweight="bold")
        ax.set_xlim(t[0], t[-1])

    # Residual row
    ax = ax_grid[-1, col]
    if residual is not None:
        ax.plot(t, residual, linewidth=0.7, color="gray")
    ax.set_ylabel("Residual")
    ax.set_xlabel("Time (s)")
    ax.set_xlim(t[0], t[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=Path,
        default=Path("checkpoints_p25/final.pt"),
        help="Path to trained N-EMD checkpoint (final.pt)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("paper/figures/phase25_compare.png"),
    )
    parser.add_argument(
        "--n_samples", type=int, default=512,
    )
    args = parser.parse_args()

    # Canonical 3-component test signal
    n = args.n_samples
    fs = float(n)  # duration = 1.0 s
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

    nemd_imfs, nemd_res = run_nemd(signal, args.ckpt)

    # ---- Metrics ----
    methods = {
        "EMD":    (emd_imfs,  emd_res),
        "VMD":    (vmd_imfs,  vmd_res),
        "N-EMD":  (nemd_imfs, nemd_res),
    }

    print(f"\n{'Method':<10} | {'Ortho':>6} | {'Energy':>7} | {'ModeMix':>7} | "
          f"{'ReconErr':>8} | {'# IMFs':>6}")
    print("-" * 68)
    for name, (imfs, res) in methods.items():
        m = compute_metrics(signal, imfs, res, components)
        print(f"{name:<10} | {m['ortho']:>6.3f} | {m['energy_ratio']:>7.3f} | "
              f"{m['mode_mix']:>7.3f} | {m['recon_err']:>8.5f} | {m['n_imfs']:>6}")

    # ---- Centroids (for N-EMD ordering check) ----
    nyquist = fs / 2
    print("\nN-EMD centroids (Hz):", spectral_centroids(nemd_imfs, nyquist))
    print("True centroids (approx): 50.0, 15.0, 3.0 Hz")

    # ---- Plot: time-domain IMFs for each method (side-by-side) ----
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Determine number of IMF rows (max across methods) + residual row
    max_k = max(imfs.shape[0] for imfs, _ in methods.values())
    n_rows = max_k + 1
    n_cols = 1 + len(methods)  # signal column + 3 methods

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 1.3 * n_rows), sharex=True,
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Column 0: original signal at the top, true components below
    axes[0, 0].plot(t, signal, "k", linewidth=0.7)
    axes[0, 0].set_title("Input signal + truth", fontweight="bold")
    axes[0, 0].set_ylabel("Signal")
    for k in range(min(3, n_rows - 1)):
        axes[k + 1, 0].plot(t, components[k], linewidth=0.7, color="C2")
        axes[k + 1, 0].set_ylabel(f"True {k + 1}\n({[50, 15, 3][k]} Hz)")
    axes[-1, 0].set_xlabel("Time (s)")

    # Columns 1..: decompositions
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
        "EMD vs VMD vs N-EMD — 3-component AM-FM (50/15/3 Hz) + SNR≈26 dB",
        fontweight="bold", y=1.002,
    )
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", dpi=150)
    print(f"\nSaved figure: {args.out}")

    # ---- Second figure: spectrograms for N-EMD IMFs ----
    fig2, axes2 = plt.subplots(
        3, 3, figsize=(11, 7), sharex=True, sharey=True,
    )
    for col, (name, (imfs, _)) in enumerate(methods.items()):
        for k in range(3):
            ax = axes2[k, col]
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
    fig2.suptitle("Per-IMF spectrograms (dB)", fontweight="bold")
    fig2.tight_layout()
    out2 = args.out.with_name(args.out.stem + "_spectrograms.png")
    fig2.savefig(out2, bbox_inches="tight", dpi=150)
    print(f"Saved figure: {out2}")


if __name__ == "__main__":
    main()
