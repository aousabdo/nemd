"""Phase 3 Exp 1 comparison: EMD vs VMD vs N-EMD on nonstationary signals.

For each signal kind, runs each method and reports:
  - Orthogonality, energy ratio, mode mixing (usual suite)
  - IF tracking RMSE (new metric for nonstationary)
  - Wall-clock time

Saves time-domain, spectrogram, and IF-trace figures per kind.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal as scisignal

from nemd.classical import ClassicalEMD, VMD
from nemd.model import NEMD
from nemd.train import TrainConfig
from nemd.utils import (
    energy_ratio,
    generate_nonstationary_signal,
    if_tracking_error,
    mode_mixing_index,
    orthogonality_index,
    to_numpy,
)


matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "figure.dpi": 120,
})


TEST_KINDS = [
    "stationary",       # control — still should work
    "chirp_trio",       # one chirp + two constants
    "crossing_chirps",  # two chirps crossing
    "widening_am",      # time-varying AM
    "piecewise",        # abrupt frequency jumps
]


def load_nemd(ckpt_path: Path) -> tuple[NEMD, TrainConfig]:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config: TrainConfig = blob["config"]
    model = NEMD(
        num_imfs=config.num_imfs,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        kernel_size=config.kernel_size,
        sample_rate=config.sample_rate,
        temperature=config.tau_end,
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, config


def run_nemd(model: NEMD, signal_np: np.ndarray, temperature: float = 0.3):
    x = torch.from_numpy(signal_np).float().unsqueeze(0)
    t0 = time.perf_counter()
    with torch.no_grad():
        imfs, _, _ = model(x, temperature=temperature, sort_by_centroid=True)
    elapsed = time.perf_counter() - t0
    return to_numpy(imfs.squeeze(0)), elapsed


def run_emd(signal_np: np.ndarray, t: np.ndarray):
    emd = ClassicalEMD(max_imfs=4)
    t0 = time.perf_counter()
    imfs = emd.decompose(signal_np, t)
    elapsed = time.perf_counter() - t0
    # Drop residual to keep K=3
    if imfs.shape[0] > 3:
        imfs = imfs[:3]
    elif imfs.shape[0] < 3:
        pad = np.zeros((3 - imfs.shape[0], imfs.shape[1]))
        imfs = np.vstack([imfs, pad])
    return imfs, elapsed


def run_vmd(signal_np: np.ndarray):
    vmd = VMD(n_modes=3)
    t0 = time.perf_counter()
    modes = vmd.decompose(signal_np)
    elapsed = time.perf_counter() - t0
    return modes, elapsed


def full_metrics(
    signal_np: np.ndarray,
    imfs: np.ndarray,
    true_comps: list[np.ndarray],
    true_ifs: list[np.ndarray],
    fs: float,
) -> dict:
    full = np.vstack([imfs, np.zeros((1, imfs.shape[-1]))])
    if_result = if_tracking_error(true_ifs, imfs, fs=fs, edge_trim=20)
    return {
        "ortho": orthogonality_index(imfs),
        "energy_ratio": energy_ratio(signal_np, full),
        "mode_mix": mode_mixing_index(true_comps, full),
        "if_rmse_mean": if_result["mean_rmse"],
        "if_rmse_max": if_result["max_rmse"],
        "if_rmse_per": if_result["per_component_rmse"],
    }


def plot_if_traces(t, true_ifs, methods, kind, out_path):
    """Plot true IF vs each method's IMF IF, per component."""
    from nemd.utils import _inst_freq_from_signal
    fs = 1.0 / (t[1] - t[0])

    fig, axes = plt.subplots(
        3, len(methods), figsize=(3.5 * len(methods), 7), sharex=True, sharey=True,
    )
    if len(methods) == 1:
        axes = axes[:, None]

    for col, (name, (imfs, _)) in enumerate(methods.items()):
        est_ifs = np.stack([_inst_freq_from_signal(imfs[k], fs) for k in range(imfs.shape[0])])
        for row, true_if in enumerate(true_ifs):
            ax = axes[row, col]
            # True IF
            ax.plot(t, true_if, "k", linewidth=1.8, alpha=0.8, label="True")
            # Best-matched estimated IMF
            best_rmse = float("inf")
            best_k = 0
            for k in range(est_ifs.shape[0]):
                mid = slice(20, -20)
                rmse = np.sqrt(np.mean((est_ifs[k, mid] - true_if[mid]) ** 2))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_k = k
            ax.plot(t, est_ifs[best_k], "r", linewidth=1.0, alpha=0.8,
                    label=f"IMF {best_k+1} (RMSE {best_rmse:.1f})")
            if row == 0:
                ax.set_title(name, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"Component {row+1}\nFreq (Hz)")
            if row == 2:
                ax.set_xlabel("Time (s)")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(alpha=0.3)
            ax.set_ylim(0, max(100.0, float(true_if.max()) * 1.2))

    fig.suptitle(f"Instantaneous frequency tracking — {kind}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_time_domain(t, signal, components, methods, kind, out_path):
    n_rows = 4
    n_cols = 1 + len(methods)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 1.2 * n_rows), sharex=True)
    axes[0, 0].plot(t, signal, "k", linewidth=0.7)
    axes[0, 0].set_title("Input + truth", fontweight="bold")
    axes[0, 0].set_ylabel("Signal")
    for k in range(3):
        axes[k + 1, 0].plot(t, components[k], linewidth=0.7, color="C2")
        axes[k + 1, 0].set_ylabel(f"True {k + 1}")
    axes[-1, 0].set_xlabel("Time (s)")
    for col, (name, (imfs, _)) in enumerate(methods.items(), start=1):
        for k in range(3):
            ax = axes[k, col]
            if k < imfs.shape[0]:
                ax.plot(t, imfs[k], linewidth=0.7)
            if k == 0:
                ax.set_title(name, fontweight="bold")
        axes[-1, col].plot(t, signal - imfs.sum(axis=0), "gray", linewidth=0.6)
        axes[-1, col].set_xlabel("Time (s)")
    fig.suptitle(f"Decomposition comparison — {kind}", fontweight="bold", y=1.002)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True,
                        help="Path to N-EMD checkpoint (final.pt)")
    parser.add_argument("--out-dir", type=Path, default=Path("paper/figures/phase3_exp1"))
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--n-trials", type=int, default=10,
                        help="Random test signals per kind (for averaging)")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, config = load_nemd(args.ckpt)
    fs = config.sample_rate
    n = args.n_samples
    duration = n / fs
    rng = np.random.default_rng(args.seed)

    all_results: dict[str, dict] = {}

    for kind in TEST_KINDS:
        print(f"\n=== {kind} ===")
        per_trial = {name: [] for name in ["EMD", "VMD", "N-EMD"]}
        timings = {name: [] for name in ["EMD", "VMD", "N-EMD"]}

        for trial in range(args.n_trials):
            trial_seed = int(rng.integers(0, 1_000_000))
            t, signal, comps, ifs = generate_nonstationary_signal(
                n_samples=n, duration=duration, kind=kind, seed=trial_seed,
                noise_std=0.0,
            )
            # Add noise to ~20 dB SNR
            sig_power = float(np.mean(signal ** 2) + 1e-12)
            noise = np.random.default_rng(trial_seed + 1).normal(
                0, np.sqrt(sig_power / 100.0), size=n,
            )
            signal_noisy = signal + noise

            emd_imfs, emd_t = run_emd(signal_noisy, t)
            vmd_imfs, vmd_t = run_vmd(signal_noisy)
            nemd_imfs, nemd_t = run_nemd(model, signal_noisy)

            for name, imfs, wt in [
                ("EMD", emd_imfs, emd_t),
                ("VMD", vmd_imfs, vmd_t),
                ("N-EMD", nemd_imfs, nemd_t),
            ]:
                m = full_metrics(signal_noisy, imfs, comps, ifs, fs)
                per_trial[name].append(m)
                timings[name].append(wt)

            # Save plots from first trial only
            if trial == 0:
                methods_dict = {
                    "EMD": (emd_imfs, None),
                    "VMD": (vmd_imfs, None),
                    "N-EMD": (nemd_imfs, None),
                }
                plot_time_domain(
                    t, signal_noisy, comps, methods_dict, kind,
                    args.out_dir / f"{kind}_timedomain.png",
                )
                plot_if_traces(
                    t, ifs, methods_dict, kind,
                    args.out_dir / f"{kind}_if_traces.png",
                )

        # Aggregate over trials
        summary = {}
        for name in per_trial:
            vals = per_trial[name]
            summary[name] = {
                "ortho":     float(np.mean([v["ortho"] for v in vals])),
                "energy_ratio": float(np.mean([v["energy_ratio"] for v in vals])),
                "mode_mix":  float(np.mean([v["mode_mix"] for v in vals])),
                "if_rmse":   float(np.mean([v["if_rmse_mean"] for v in vals])),
                "if_rmse_std": float(np.std([v["if_rmse_mean"] for v in vals])),
                "time_ms":   float(np.mean(timings[name]) * 1000),
            }
        all_results[kind] = summary

        # Print per-kind table
        print(f"\n  {'Method':<8} | {'Ortho':>6} | {'Energy':>6} | {'ModeMix':>7} | "
              f"{'IF RMSE (Hz)':>13} | {'Time (ms)':>9}")
        print("  " + "-" * 70)
        for name, s in summary.items():
            print(f"  {name:<8} | {s['ortho']:>6.3f} | {s['energy_ratio']:>6.3f} | "
                  f"{s['mode_mix']:>7.3f} | {s['if_rmse']:>6.2f} ± {s['if_rmse_std']:>4.2f} | "
                  f"{s['time_ms']:>9.1f}")

    # Save JSON summary
    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved summary: {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
