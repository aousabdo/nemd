"""Phase 3 Exp 2: generalization + speed study.

Evaluates a *pre-trained* N-EMD model against EMD and VMD on five
held-out test distributions (200 signals each).  The key story is:
N-EMD processes the entire batch in one forward pass; VMD has to run
ADMM per signal.  Wall-clock times should show N-EMD is much faster,
especially at scale.
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

from nemd.classical import ClassicalEMD, VMD
from nemd.model import NEMD
from nemd.train import TrainConfig
from nemd.utils import (
    energy_ratio,
    mode_mixing_index,
    orthogonality_index,
    to_numpy,
)
from experiments.applications.generalization.dataset import TEST_GENERATORS


matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "figure.dpi": 120,
})


def load_nemd(ckpt_path: str, sample_rate: float) -> NEMD:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config: TrainConfig = blob["config"]
    model = NEMD(
        num_imfs=config.num_imfs,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        kernel_size=config.kernel_size,
        sample_rate=sample_rate,
        temperature=0.3,
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Per-method batch runners
# ---------------------------------------------------------------------------

def run_nemd_batch(
    model: NEMD, signals: torch.Tensor, batch_size: int = 64,
) -> tuple[np.ndarray, float]:
    """Returns (imfs_all (N, K, T), total_wall_sec)."""
    all_imfs = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(signals), batch_size):
            x = signals[i:i + batch_size]
            imfs, _, _ = model(x, temperature=0.3, sort_by_centroid=True)
            all_imfs.append(to_numpy(imfs))
    elapsed = time.perf_counter() - t0
    return np.concatenate(all_imfs, axis=0), elapsed


def run_emd_batch(signals: torch.Tensor, fs: float) -> tuple[np.ndarray, float]:
    N, T = signals.shape
    out = np.zeros((N, 3, T), dtype=np.float64)
    emd = ClassicalEMD(max_imfs=4)
    t = np.arange(T) / fs
    t0 = time.perf_counter()
    for i in range(N):
        imfs = emd.decompose(signals[i].cpu().numpy(), t)
        k = min(3, imfs.shape[0])
        out[i, :k] = imfs[:k]
    elapsed = time.perf_counter() - t0
    return out, elapsed


def run_vmd_batch(signals: torch.Tensor) -> tuple[np.ndarray, float]:
    N, T = signals.shape
    out = np.zeros((N, 3, T), dtype=np.float64)
    vmd = VMD(n_modes=3)
    t0 = time.perf_counter()
    for i in range(N):
        out[i] = vmd.decompose(signals[i].cpu().numpy())
    elapsed = time.perf_counter() - t0
    return out, elapsed


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(signals_np: np.ndarray, imfs_all: np.ndarray,
                    comps_list: list) -> dict:
    N = imfs_all.shape[0]
    orthos, energies, mixes = [], [], []
    for i in range(N):
        full = np.vstack([imfs_all[i], np.zeros((1, imfs_all.shape[-1]))])
        orthos.append(orthogonality_index(imfs_all[i]))
        energies.append(energy_ratio(signals_np[i], full))
        mixes.append(mode_mixing_index(comps_list[i], full))
    return {
        "ortho_mean": float(np.mean(orthos)),
        "ortho_std":  float(np.std(orthos)),
        "energy_mean": float(np.mean(energies)),
        "energy_std":  float(np.std(energies)),
        "mode_mix_mean": float(np.mean(mixes)),
        "mode_mix_std":  float(np.std(mixes)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Pretrained N-EMD checkpoint")
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--sample-rate", type=float, default=512.0)
    parser.add_argument("--n-per-set", type=int, default=200)
    parser.add_argument("--out-dir", type=str, default="paper/figures/phase3_exp2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_nemd(args.ckpt, args.sample_rate)
    summary = {}

    for set_name, gen in TEST_GENERATORS.items():
        print(f"\n=== {set_name} ===")
        X, comps_list = gen(
            n=args.n_per_set, n_samples=args.n_samples, fs=args.sample_rate,
            seed=args.seed,
        )
        X_np = X.cpu().numpy()

        print(f"  Generated {len(X)} signals. Running decompositions...")
        # N-EMD (batched)
        nemd_imfs, nemd_t = run_nemd_batch(model, X, batch_size=args.batch_size)
        nemd_metrics = compute_metrics(X_np, nemd_imfs, comps_list)
        print(f"    N-EMD:  {nemd_t:.2f}s total  "
              f"({nemd_t * 1000 / len(X):.2f} ms/signal)")

        # VMD (per-signal)
        vmd_imfs, vmd_t = run_vmd_batch(X)
        vmd_metrics = compute_metrics(X_np, vmd_imfs, comps_list)
        print(f"    VMD:    {vmd_t:.2f}s total  "
              f"({vmd_t * 1000 / len(X):.2f} ms/signal)")

        # EMD (per-signal)
        emd_imfs, emd_t = run_emd_batch(X, args.sample_rate)
        emd_metrics = compute_metrics(X_np, emd_imfs, comps_list)
        print(f"    EMD:    {emd_t:.2f}s total  "
              f"({emd_t * 1000 / len(X):.2f} ms/signal)")

        summary[set_name] = {
            "N-EMD": {**nemd_metrics, "total_time_sec": nemd_t,
                      "per_signal_ms": nemd_t * 1000 / len(X)},
            "VMD":   {**vmd_metrics,  "total_time_sec": vmd_t,
                      "per_signal_ms": vmd_t * 1000 / len(X)},
            "EMD":   {**emd_metrics,  "total_time_sec": emd_t,
                      "per_signal_ms": emd_t * 1000 / len(X)},
        }

        # Print mini-table
        print(f"\n  {'Method':<7} | {'ModeMix':>13} | {'Ortho':>13} | "
              f"{'Energy':>13} | {'Total':>8} | {'Per-sig':>8}")
        print("  " + "-" * 85)
        for method_name, m in summary[set_name].items():
            print(f"  {method_name:<7} | "
                  f"{m['mode_mix_mean']:>5.3f} ± {m['mode_mix_std']:>5.3f} | "
                  f"{m['ortho_mean']:>5.3f} ± {m['ortho_std']:>5.3f} | "
                  f"{m['energy_mean']:>5.3f} ± {m['energy_std']:>5.3f} | "
                  f"{m['total_time_sec']:>7.2f}s | "
                  f"{m['per_signal_ms']:>6.2f}ms")

    # Save
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n\nSaved: {out_dir / 'summary.json'}")

    # Speed bar chart
    fig, ax = plt.subplots(figsize=(9, 4))
    set_names = list(TEST_GENERATORS.keys())
    methods = ["EMD", "VMD", "N-EMD"]
    x = np.arange(len(set_names))
    width = 0.26
    for i, m in enumerate(methods):
        vals = [summary[s][m]["per_signal_ms"] for s in set_names]
        ax.bar(x + (i - 1) * width, vals, width, label=m)
    ax.set_xticks(x); ax.set_xticklabels(set_names, rotation=15, ha="right")
    ax.set_ylabel("Per-signal wall time (ms)")
    ax.set_yscale("log")
    ax.set_title("Generalization benchmark: per-signal inference time",
                 fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "speed_comparison.png", bbox_inches="tight", dpi=150)
    print(f"Saved: {out_dir / 'speed_comparison.png'}")


if __name__ == "__main__":
    main()
