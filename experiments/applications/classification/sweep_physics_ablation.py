"""Physics-regulariser ablation for the N-EMD pretrained collapse.

Background: at SNR 10 dB on the synthetic three-class task, the
pretrained N-EMD fine-tune drops from 89.5% (SNR 3 dB) to 72.0%, while
the scratch variant holds up. The hypothesis in the paper is that the
0.1x physics regulariser anchors the filter bank near its
decomposition-optimal configuration, fighting the task gradient in
the clean-signal regime.

This script sweeps the physics regulariser weight to test the hypothesis:

    lambda_physics in {0, 0.001, 0.01, 0.1, 1.0}

Only the pretrained variant at SNR 10 dB is evaluated (the hypothesis
concerns clean-signal anchoring), over 2 seeds.

Output:  paper/figures/phase3_exp3_passB/physics_ablation.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.applications.classification.dataset import make_splits
from experiments.applications.classification.train_pipelines import (
    train_nemd_end_to_end, seed_everything,
)
from experiments.applications.classification.sweep_passB import pick_device
from experiments.applications.classification.sweep_nemd import make_nemd_from_ckpt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snr", type=float, default=10.0)
    p.add_argument("--seeds", type=str, default="42,43")
    p.add_argument("--lambdas", type=str, default="0,0.001,0.01,0.1,1.0")
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--nemd-ckpt", type=str,
                   default="checkpoints_p25b_v3/final.pt")
    p.add_argument("--out-dir", type=str,
                   default="paper/figures/phase3_exp3_passB")
    args = p.parse_args()

    seeds   = [int(s) for s in args.seeds.split(",")]
    lambdas = [float(x) for x in args.lambdas.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"Device: {device}")
    print(f"SNR={args.snr} seeds={seeds} lambdas={lambdas}")

    results: dict = {
        "config": {
            "snr": args.snr, "seeds": seeds, "lambdas": lambdas,
            "n_train": args.n_train, "n_val": args.n_val,
            "n_test": args.n_test, "epochs": args.epochs,
            "nemd_ckpt": args.nemd_ckpt,
            "variant": "pretrained",
        },
        "results": {},
    }

    total_t0 = time.perf_counter()
    for lam in lambdas:
        lam_key = f"lambda_{lam}"
        results["results"][lam_key] = {}
        print(f"\n--- lambda_physics = {lam} ---")

        for seed in seeds:
            splits = make_splits(
                n_train_per_class=args.n_train,
                n_val_per_class=args.n_val,
                n_test_per_class=args.n_test,
                n_samples=1024, fs=1000.0, snr_db=args.snr, seed=seed,
            )
            X_tr, y_tr = splits["train"]
            X_va, y_va = splits["val"]
            X_te, y_te = splits["test"]

            seed_everything(seed)
            nemd = make_nemd_from_ckpt(args.nemd_ckpt, 1000.0)
            t0 = time.perf_counter()
            r = train_nemd_end_to_end(
                name=f"nemd_pretrained_lam{lam}",
                train_signals=X_tr, train_labels=y_tr,
                val_signals=X_va,   val_labels=y_va,
                test_signals=X_te,  test_labels=y_te,
                nemd_model=nemd, sample_rate=1000.0,
                n_epochs=args.epochs, seed=seed, device=device,
                lr=1e-4, physics_weight=lam, verbose=False,
            )
            dt = time.perf_counter() - t0
            results["results"][lam_key][f"seed_{seed}"] = {
                "test_acc": r.test_acc,
                "wallclock_sec": dt,
            }
            print(f"    seed {seed}: acc={r.test_acc:.4f}  wall={dt:.0f}s")

            with open(out_dir / "physics_ablation.json", "w") as f:
                json.dump(results, f, indent=2)

    print(f"\nTotal: {(time.perf_counter() - total_t0)/60:.1f} min")


if __name__ == "__main__":
    main()
