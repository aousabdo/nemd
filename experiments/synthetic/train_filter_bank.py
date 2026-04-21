"""Run Phase 2.5b filter-bank N-EMD training and report results."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from nemd.train import TrainConfig, train
from nemd.utils import (
    generate_synthetic_signal,
    mode_mixing_index,
    orthogonality_index,
    energy_ratio,
    to_numpy,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-train", type=int, default=2000)
    parser.add_argument("--num-val", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--save-dir", type=str, default="checkpoints_p25b")
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
        # 3-component signals drawn from bands covering the canonical test
        freq_bands=(
            (40.0, 100.0),
            (10.0, 30.0),
            (1.0, 8.0),
        ),
        snr_range_db=(10.0, 40.0),
        lambda_sharp=2.0,
        lambda_order=2.0,
        lambda_ortho=0.1,
        normalized_margin=0.02,
        tau_start=2.0,
        tau_end=0.3,
        tau_anneal_epochs=40,
        learning_rate=1e-3,
        weight_decay=1e-4,
        num_epochs=args.epochs,
        scheduler="cosine",
        grad_clip_norm=1.0,
        log_interval=5,
        seed=args.seed,
        save_dir=args.save_dir,
    )

    t0 = time.time()
    result = train(config, verbose=True)
    elapsed = time.time() - t0
    model = result["model"]
    print(f"\n=== Total training time: {elapsed:.1f}s ({elapsed / 60:.1f} min) ===")

    # Save final.pt with config for the comparison script
    torch.save(
        {"state_dict": model.state_dict(), "config": config},
        f"{args.save_dir}/final.pt",
    )

    # Evaluate on canonical 3-component test signal
    n = config.signal_length
    fs = config.sample_rate
    t_sig, signal, components = generate_synthetic_signal(
        n_samples=n, duration=1.0,
        components=[
            {"f0": 50.0, "f_mod": 2.0, "a_mod": 0.5},
            {"f0": 15.0, "f_mod": 0.5, "a_mod": 0.3},
            {"f0": 3.0,  "f_mod": 0.1, "a_mod": 0.2},
        ],
        noise_std=0.05, seed=42,
    )
    x = torch.from_numpy(signal).float().unsqueeze(0)
    model.eval()
    with torch.no_grad():
        imfs, residual, metadata = model(
            x, temperature=config.tau_end, sort_by_centroid=True,
        )
    imfs_np = to_numpy(imfs.squeeze(0))
    filters = to_numpy(metadata["filters"].squeeze(0))
    centroids_filter = to_numpy(metadata["centroids"].squeeze(0))
    centroids_weighted = to_numpy(metadata["centroids_weighted"].squeeze(0))
    recon = np.sum(imfs_np, axis=0)
    imfs_full = np.vstack([imfs_np, np.zeros((1, n))])

    print()
    print("=== Filter-Bank N-EMD on canonical Phase 1 signal ===")
    print(f"Max recon error:          {np.max(np.abs(signal - recon)):.2e}")
    print(f"Partition (Σ_K=1) dev:    {np.max(np.abs(filters.sum(0) - 1)):.2e}")
    print(f"Filter centroids (Hz):    {centroids_filter}")
    print(f"Signal-weighted cen. (Hz): {centroids_weighted}")
    print(f"True components (Hz):     [50.0, 15.0, 3.0]")
    print(f"Weighted descending:      "
          f"{all(centroids_weighted[i] > centroids_weighted[i+1] for i in range(len(centroids_weighted)-1))}")
    print(f"Ortho index:              {orthogonality_index(imfs_np):.4f}")
    print(f"Energy ratio:             {energy_ratio(signal, imfs_full):.4f}")
    print(f"Mode mixing:              {mode_mixing_index(components, imfs_full):.4f}")
    print(f"IMF energies:             "
          f"{[f'{np.sum(imfs_np[k]**2)/np.sum(signal**2)*100:.1f}%' for k in range(imfs_np.shape[0])]}")
    print()
    print("Baselines (K=3 on same signal):")
    print("  EMD:    ortho=0.06, mode_mix=0.16")
    print("  VMD:    ortho=0.09, mode_mix=0.08")
    print("  P2.5:   ortho=0.46, mode_mix=0.32  (sifting arch)")


if __name__ == "__main__":
    main()
