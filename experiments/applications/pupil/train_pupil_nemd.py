"""Step 4: Train filter-bank N-EMD on synthetic pupil-like signals.

K=4 IMFs for the four physiological bands (VLF, LF, respiratory, hippus).
Signal length 2048 at 100 Hz = 20.48 seconds.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from nemd.train import TrainConfig, train
from nemd.utils import to_numpy
from experiments.applications.pupil.synthetic_pupil import (
    generate_pupil_training_dataset,
    generate_pupil_like_signal,
    BAND_NAMES,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-train", type=int, default=3000)
    parser.add_argument("--num-val", type=int, default=500)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--save-dir", type=str, default="checkpoints_pupil")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainConfig(
        signal_length=2048,
        sample_rate=100.0,
        batch_size=32,
        num_train_signals=args.num_train,
        num_val_signals=args.num_val,
        num_imfs=4,  # VLF, LF, respiratory, hippus
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        kernel_size=5,
        lambda_sharp=2.0,
        lambda_order=2.0,
        lambda_ortho=0.1,
        lambda_balance=5.0,
        normalized_margin=0.02,
        tau_start=2.0,
        tau_end=0.5,
        tau_anneal_epochs=40,
        learning_rate=1e-3,
        num_epochs=args.epochs,
        log_interval=5,
        seed=args.seed,
        save_dir=args.save_dir,
    )

    t0 = time.time()
    result = train(
        config, verbose=True,
        dataset_fn=generate_pupil_training_dataset,
    )
    elapsed = time.time() - t0
    model = result["model"]
    print(f"\n=== Training time: {elapsed:.1f}s ({elapsed / 60:.1f} min) ===")

    # Save for downstream use
    torch.save(
        {"state_dict": model.state_dict(), "config": config},
        f"{args.save_dir}/final.pt",
    )

    # Sanity check: decompose a synthetic validation signal
    print("\n=== Sanity check on synthetic signal ===")
    sig, true_comps, bands = generate_pupil_like_signal(
        n_samples=2048, fs=100.0, seed=999, noise_std=0.1,
    )
    sig_norm = (sig - sig.mean()) / (sig.std() + 1e-8)
    x = torch.from_numpy(sig_norm).float().unsqueeze(0)

    model.eval()
    with torch.no_grad():
        imfs, _, metadata = model(x, temperature=0.5, sort_by_centroid=True)
    imfs_np = to_numpy(imfs.squeeze(0))
    filters = to_numpy(metadata["filters"].squeeze(0))
    centroids = to_numpy(metadata["centroids_weighted"].squeeze(0))

    print(f"IMFs shape: {imfs_np.shape}")
    print(f"Signal-weighted centroids (Hz): {centroids}")
    print(f"True band centers: {[(lo+hi)/2 for lo, hi in bands]}")

    # Check per-IMF band purity
    freqs = np.fft.rfftfreq(2048, d=1 / 100.0)
    for k in range(imfs_np.shape[0]):
        psd = np.abs(np.fft.rfft(imfs_np[k])) ** 2
        total = psd.sum() + 1e-12
        for b, (lo, hi) in enumerate(bands):
            in_band = psd[(freqs >= lo) & (freqs <= hi)].sum()
            pct = in_band / total * 100
            if pct > 10:
                print(f"  IMF {k+1}: {pct:.1f}% in {BAND_NAMES[b]} ({lo:.2f}-{hi:.2f} Hz)")


if __name__ == "__main__":
    main()
