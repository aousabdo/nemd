"""Phase-C referee-response runs.

Two targeted experiments requested in the second round of reviewer
feedback:

1. **Frontend isolation**: N-EMD (learned filter bank) with a
   capacity-matched MLP head (hidden 320, ~107k head params). If
   this beats VMD+MLP(big) at 3 dB, the learned frontend is doing
   real work beyond the head's capacity. If it does not, the
   'task-aware frontend' claim is weaker than the headline suggests.

2. **lambda_phys = 0 sweep for N-EMD scratch at SNR 10/20 dB**.
   Tests the hypothesis that the physics regulariser over-constrains
   the filter bank at clean SNR, destabilising scratch training.

Output:
    paper/figures/phase3_exp3_passB/phaseC_results.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.applications.classification.dataset import make_splits
from experiments.applications.classification.train_pipelines import (
    train_nemd_end_to_end, seed_everything,
)
from experiments.applications.classification.sweep_passB import pick_device
from nemd.model import NEMD


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str,
                   default="paper/figures/phase3_exp3_passB")
    p.add_argument("--seeds", type=str, default="42,43,44")
    p.add_argument("--n-epochs", type=int, default=50)
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"Device: {device}  seeds: {seeds}")

    fs = 1000.0
    n_samples = 1024

    results: dict = {
        "config": {"seeds": seeds, "n_epochs": args.n_epochs,
                   "fs": fs, "n_samples": n_samples},
        "nemd_big_head": {},      # N-EMD scratch + 320-wide MLP
        "nemd_scratch_lam0": {},  # N-EMD scratch with lambda_phys = 0
    }

    # --------- (1) N-EMD big-head at all three SNRs ---------
    print("\n=== N-EMD scratch + big head (matched capacity) ===")
    for snr in (3.0, 10.0, 20.0):
        snr_key = f"snr_{int(snr)}_db"
        results["nemd_big_head"][snr_key] = {}
        for seed in seeds:
            print(f"  SNR {snr:.0f} dB, seed {seed}")
            splits = make_splits(
                n_train_per_class=1000, n_val_per_class=200,
                n_test_per_class=200, n_samples=n_samples, fs=fs,
                snr_db=snr, seed=seed,
            )
            X_tr, y_tr = splits["train"]
            X_va, y_va = splits["val"]
            X_te, y_te = splits["test"]
            seed_everything(seed)
            nemd = NEMD(
                num_imfs=3, hidden_dim=64, num_layers=3,
                sample_rate=fs, temperature=0.3,
            )
            t0 = time.perf_counter()
            r = train_nemd_end_to_end(
                name=f"nemd_big_head",
                train_signals=X_tr, train_labels=y_tr,
                val_signals=X_va, val_labels=y_va,
                test_signals=X_te, test_labels=y_te,
                nemd_model=nemd, sample_rate=fs,
                n_epochs=args.n_epochs, seed=seed, device=device,
                lr=1e-4, physics_weight=0.1, verbose=False,
                mlp_hidden=320,                 # ≈107 k head params
            )
            results["nemd_big_head"][snr_key][f"seed_{seed}"] = {
                "test_acc":     r.test_acc,
                "n_params":     r.n_params,
                "wallclock_sec": time.perf_counter() - t0,
            }
            print(f"    acc={r.test_acc:.4f}  params={r.n_params}  "
                  f"wall={r.wallclock_sec:.0f}s")
            with open(out_dir / "phaseC_results.json", "w") as f:
                json.dump(results, f, indent=2)

    # --------- (2) lambda_phys = 0 scratch at 10 and 20 dB ---------
    print("\n=== N-EMD scratch, lambda_phys = 0 (at 10 and 20 dB) ===")
    for snr in (10.0, 20.0):
        snr_key = f"snr_{int(snr)}_db"
        results["nemd_scratch_lam0"][snr_key] = {}
        for seed in seeds:
            print(f"  SNR {snr:.0f} dB, seed {seed}")
            splits = make_splits(
                n_train_per_class=1000, n_val_per_class=200,
                n_test_per_class=200, n_samples=n_samples, fs=fs,
                snr_db=snr, seed=seed,
            )
            X_tr, y_tr = splits["train"]
            X_va, y_va = splits["val"]
            X_te, y_te = splits["test"]
            seed_everything(seed)
            nemd = NEMD(
                num_imfs=3, hidden_dim=64, num_layers=3,
                sample_rate=fs, temperature=0.3,
            )
            t0 = time.perf_counter()
            r = train_nemd_end_to_end(
                name="nemd_scratch_lam0",
                train_signals=X_tr, train_labels=y_tr,
                val_signals=X_va, val_labels=y_va,
                test_signals=X_te, test_labels=y_te,
                nemd_model=nemd, sample_rate=fs,
                n_epochs=args.n_epochs, seed=seed, device=device,
                lr=1e-4, physics_weight=0.0,    # <- key difference
                verbose=False,
            )
            results["nemd_scratch_lam0"][snr_key][f"seed_{seed}"] = {
                "test_acc":     r.test_acc,
                "n_params":     r.n_params,
                "wallclock_sec": time.perf_counter() - t0,
            }
            print(f"    acc={r.test_acc:.4f}  "
                  f"wall={r.wallclock_sec:.0f}s")
            with open(out_dir / "phaseC_results.json", "w") as f:
                json.dump(results, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
