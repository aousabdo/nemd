"""Phase 3 Exp 1: train filter-bank N-EMD on diverse nonstationary signals."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from nemd.train import TrainConfig, train
from nemd.utils import (
    generate_nonstationary_signal,
    if_tracking_error,
    mode_mixing_index,
    orthogonality_index,
    to_numpy,
)
from experiments.applications.nonstationary.dataset import diverse_nonstationary_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-train", type=int, default=2000)
    parser.add_argument("--num-val", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--save-dir", type=str, default="checkpoints_p3_ns")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainConfig(
        signal_length=512,
        sample_rate=512.0,
        batch_size=32,
        num_train_signals=args.num_train,
        num_val_signals=args.num_val,
        num_imfs=3,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        kernel_size=5,
        snr_range_db=(10.0, 35.0),
        lambda_sharp=2.0,
        lambda_order=2.0,
        lambda_ortho=0.1,
        lambda_balance=5.0,
        normalized_margin=0.02,
        sep_w_order=1.0,
        sep_w_repel=0.5,
        sep_w_coverage=0.3,
        tau_start=2.0,
        tau_end=0.3,
        tau_anneal_epochs=40,
        learning_rate=1e-3,
        num_epochs=args.epochs,
        log_interval=5,
        seed=args.seed,
        save_dir=args.save_dir,
    )

    t0 = time.time()
    result = train(config, verbose=True, dataset_fn=diverse_nonstationary_dataset)
    elapsed = time.time() - t0
    model = result["model"]
    print(f"\n=== Training time: {elapsed:.1f}s ({elapsed / 60:.1f} min) ===")

    torch.save(
        {"state_dict": model.state_dict(), "config": config},
        f"{args.save_dir}/final.pt",
    )


if __name__ == "__main__":
    main()
