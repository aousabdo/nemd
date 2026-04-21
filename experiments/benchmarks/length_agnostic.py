"""Length-agnostic validation: decomposition quality at varying T.

Reviewer 2, item G: the paper claims the trained analyzer (optimised
at T=512, fs=512 Hz) applies to any other T / fs without retraining.
Demonstrate this with OI and ER numbers across T in {256, 512, 1024,
2048, 4096}.

Generates 200 in-distribution AM-FM composites per length, decomposes
with the released N-EMD checkpoint, and reports OI, ER, and per-signal
inference time.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from nemd.model import NEMD
from nemd.train import TrainConfig
from nemd.utils import (
    generate_synthetic_signal, orthogonality_index, energy_ratio,
)


def make_analyzer(ckpt_path: str, sample_rate: float) -> NEMD:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: TrainConfig = blob["config"]
    model = NEMD(
        num_imfs=cfg.num_imfs, hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers, kernel_size=cfg.kernel_size,
        sample_rate=sample_rate, temperature=0.3,
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def one_signal(t_sec: float, T: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    """Build an AM-FM composite similar to the training distribution."""
    # Three components with centre frequencies scaled to fit <Nyquist
    nyq = fs / 2.0
    f_hi = rng.uniform(0.35, 0.55) * nyq
    f_md = rng.uniform(0.10, 0.25) * nyq
    f_lo = rng.uniform(0.01, 0.08) * nyq
    duration = T / fs
    _, signal, _ = generate_synthetic_signal(
        n_samples=T, duration=duration,
        components=[
            {"type": "am-fm", "f0": f_hi, "f_mod": 2.0, "a_mod": 0.5},
            {"type": "am-fm", "f0": f_md, "f_mod": 0.5, "a_mod": 0.3},
            {"type": "am-fm", "f0": f_lo, "f_mod": 0.1, "a_mod": 0.2},
        ],
        noise_std=0.05,
        seed=int(rng.integers(0, 1_000_000)),
    )
    return signal


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str,
                   default="checkpoints_p25b_v3/final.pt")
    p.add_argument("--lengths", type=str, default="256,512,1024,2048,4096")
    p.add_argument("--fs", type=float, default=512.0,
                   help="fixed sample rate in Hz")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", type=str,
                   default="paper/figures/phase3_exp3_passB/length_agnostic.json")
    args = p.parse_args()

    Ts = [int(t) for t in args.lengths.split(",")]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Device: {device}")

    model = make_analyzer(args.ckpt, args.fs).to(device)

    rng = np.random.default_rng(42)
    results: dict = {"config": {"ckpt": args.ckpt, "fs": args.fs,
                                "n_per_length": args.n, "Ts": Ts},
                     "by_length": {}}

    for T in Ts:
        print(f"\n--- T = {T} samples ({T/args.fs:.2f} s) ---")
        ois, ers, ts = [], [], []
        for i in range(args.n):
            x = one_signal(None, T, args.fs, rng).astype(np.float32)
            x_t = torch.from_numpy(x).unsqueeze(0).to(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                imfs, _, _ = model(x_t, temperature=0.3,
                                   sort_by_centroid=True)
            dt = time.perf_counter() - t0
            imfs_np = imfs.squeeze(0).cpu().numpy()
            ois.append(orthogonality_index(imfs_np))
            ers.append(energy_ratio(x, imfs_np))
            ts.append(dt * 1000)  # ms
        oi_arr = np.array(ois); er_arr = np.array(ers); t_arr = np.array(ts[5:])
        results["by_length"][str(T)] = {
            "oi_mean":  float(oi_arr.mean()),
            "oi_std":   float(oi_arr.std()),
            "er_mean":  float(er_arr.mean()),
            "er_std":   float(er_arr.std()),
            "ms_mean":  float(t_arr.mean()),
            "ms_std":   float(t_arr.std()),
        }
        print(f"  OI = {oi_arr.mean():.4f} +/- {oi_arr.std():.4f}")
        print(f"  ER = {er_arr.mean():.4f} +/- {er_arr.std():.4f}")
        print(f"  per-signal {t_arr.mean():.2f} +/- {t_arr.std():.2f} ms")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
