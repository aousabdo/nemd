"""Multi-seed N-EMD scratch + pretrained sweep on MPS.

Complements sweep_passB.py (which covers the cheap/classical baselines
at 5 seeds) by running the compute-heavy N-EMD end-to-end pipelines at
3 seeds across 3 SNR levels.

Output:  paper/figures/phase3_exp3_passB/nemd_sweep_results.json
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
from nemd.model import NEMD


def make_nemd_from_ckpt(ckpt_path: str, sample_rate: float) -> NEMD:
    from nemd.train import TrainConfig
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config: TrainConfig = blob["config"]
    model = NEMD(
        num_imfs=config.num_imfs, hidden_dim=config.hidden_dim,
        num_layers=config.num_layers, kernel_size=config.kernel_size,
        sample_rate=sample_rate, temperature=0.3,
    )
    model.load_state_dict(blob["state_dict"])
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snrs", type=str, default="3,10,20")
    p.add_argument("--seeds", type=str, default="42,43,44")
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--epochs-nemd", type=int, default=50)
    p.add_argument("--sample-rate", type=float, default=1000.0)
    p.add_argument("--n-samples", type=int, default=1024)
    p.add_argument("--nemd-ckpt", type=str,
                   default="checkpoints_p25b_v3/final.pt")
    p.add_argument("--out-dir", type=str,
                   default="paper/figures/phase3_exp3_passB")
    p.add_argument("--variants", type=str, default="scratch,pretrained",
                   help="Comma-separated subset of {scratch,pretrained}")
    args = p.parse_args()

    snr_list = [float(s) for s in args.snrs.split(",")]
    seeds    = [int(s) for s in args.seeds.split(",")]
    variants = [v.strip() for v in args.variants.split(",")]
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"Device: {device}")
    print(f"SNRs: {snr_list}  seeds: {seeds}  variants: {variants}")

    results: dict = {
        "config": {
            "snr_list": snr_list, "seeds": seeds, "variants": variants,
            "n_train": args.n_train, "n_val": args.n_val,
            "n_test": args.n_test, "n_samples": args.n_samples,
            "sample_rate": args.sample_rate,
            "epochs_nemd": args.epochs_nemd,
            "nemd_ckpt": args.nemd_ckpt,
        },
        "snr_results": {},
    }

    total_t0 = time.perf_counter()
    for snr in snr_list:
        snr_key = f"snr_{int(snr)}_db"
        results["snr_results"][snr_key] = {}
        print(f"\n{'='*60}\n  SNR = {snr} dB\n{'='*60}")

        for seed in seeds:
            print(f"\n  --- seed {seed} ---")
            splits = make_splits(
                n_train_per_class=args.n_train,
                n_val_per_class=args.n_val,
                n_test_per_class=args.n_test,
                n_samples=args.n_samples,
                fs=args.sample_rate, snr_db=snr, seed=seed,
            )
            X_tr, y_tr = splits["train"]
            X_va, y_va = splits["val"]
            X_te, y_te = splits["test"]

            seed_key = f"seed_{seed}"
            seed_out: dict = {}

            if "scratch" in variants:
                print("    N-EMD scratch ...")
                seed_everything(seed)
                nemd = NEMD(
                    num_imfs=3, hidden_dim=64, num_layers=3,
                    sample_rate=args.sample_rate, temperature=0.3,
                )
                t0 = time.perf_counter()
                r = train_nemd_end_to_end(
                    name="nemd_scratch",
                    train_signals=X_tr, train_labels=y_tr,
                    val_signals=X_va,   val_labels=y_va,
                    test_signals=X_te,  test_labels=y_te,
                    nemd_model=nemd, sample_rate=args.sample_rate,
                    n_epochs=args.epochs_nemd, seed=seed, device=device,
                    lr=1e-4, physics_weight=0.1, verbose=False,
                )
                seed_out["nemd_scratch"] = {
                    "test_acc": r.test_acc,
                    "n_params": r.n_params,
                    "wallclock_sec": r.wallclock_sec,
                }
                print(f"      acc={r.test_acc:.4f}  "
                      f"wall={r.wallclock_sec:.0f}s")

            if "pretrained" in variants:
                print("    N-EMD pretrained ...")
                seed_everything(seed)
                nemd = make_nemd_from_ckpt(args.nemd_ckpt, args.sample_rate)
                r = train_nemd_end_to_end(
                    name="nemd_pretrained",
                    train_signals=X_tr, train_labels=y_tr,
                    val_signals=X_va,   val_labels=y_va,
                    test_signals=X_te,  test_labels=y_te,
                    nemd_model=nemd, sample_rate=args.sample_rate,
                    n_epochs=args.epochs_nemd, seed=seed, device=device,
                    lr=1e-4, physics_weight=0.1, verbose=False,
                )
                seed_out["nemd_pretrained"] = {
                    "test_acc": r.test_acc,
                    "n_params": r.n_params,
                    "wallclock_sec": r.wallclock_sec,
                }
                print(f"      acc={r.test_acc:.4f}  "
                      f"wall={r.wallclock_sec:.0f}s")

            results["snr_results"][snr_key][seed_key] = seed_out

            # Persist after every seed
            with open(out_dir / "nemd_sweep_results.json", "w") as f:
                json.dump(results, f, indent=2)

    print(f"\nTotal wall-clock: "
          f"{(time.perf_counter() - total_t0)/60:.1f} min")


if __name__ == "__main__":
    main()
