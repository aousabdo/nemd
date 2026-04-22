"""Magnitude-only ablation: train NAFB with phase=0 (magnitude input only),
and report classification accuracy at the three SNR levels.

Reviewer 2 / C asked for this as a principled check that phase
carries signal the analyzer actually uses.

Output: paper/figures/phase3_exp3_passB/magonly_results.json
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
    p.add_argument("--out", type=str,
                   default="paper/figures/phase3_exp3_passB/magonly_results.json")
    p.add_argument("--seeds", type=str, default="42,43,44")
    p.add_argument("--n-epochs", type=int, default=50)
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"Device: {device}  seeds: {seeds}")

    fs = 1000.0
    results: dict = {"config": {"seeds": seeds, "n_epochs": args.n_epochs,
                                "use_phase": False},
                     "nemd_magonly": {}}

    for snr in (3.0, 10.0, 20.0):
        for seed in seeds:
            splits = make_splits(
                n_train_per_class=1000, n_val_per_class=200,
                n_test_per_class=200, n_samples=1024, fs=fs,
                snr_db=snr, seed=seed,
            )
            X_tr, y_tr = splits["train"]
            X_va, y_va = splits["val"]
            X_te, y_te = splits["test"]
            snr_key = f"snr_{int(snr)}_db"

            seed_everything(seed)
            nemd = NEMD(num_imfs=3, hidden_dim=64, num_layers=3,
                        sample_rate=fs, temperature=0.3,
                        use_phase=False)       # <- the ablation
            t0 = time.perf_counter()
            r = train_nemd_end_to_end(
                name="nemd_magonly",
                train_signals=X_tr, train_labels=y_tr,
                val_signals=X_va, val_labels=y_va,
                test_signals=X_te, test_labels=y_te,
                nemd_model=nemd, sample_rate=fs,
                n_epochs=args.n_epochs, seed=seed, device=device,
                lr=1e-4, physics_weight=0.1, verbose=False,
                mlp_hidden=320,                # matched-head, default config
            )
            results["nemd_magonly"].setdefault(snr_key, {})[f"seed_{seed}"] = {
                "test_acc": r.test_acc, "n_params": r.n_params,
                "wallclock_sec": time.perf_counter() - t0,
            }
            print(f"  SNR {snr:.0f} seed={seed}: acc={r.test_acc:.4f}")

            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
