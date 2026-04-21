"""Diverse nonstationary training dataset for Phase 3 Experiment 1.

Mix:
  40% stationary AM-FM   (3 components, 50/15/3 Hz-ish with jitter)
  30% linear chirp trio  (one chirp + two constants)
  20% widening-AM carrier (+ two constants)
  10% piecewise-stationary high-freq (+ two constants)

This variety is what forces the model to learn *signal-adaptive* filters
rather than memorising a fixed frequency layout — the main differentiator
against VMD, whose modes are fixed per signal.
"""

from __future__ import annotations

import numpy as np
import torch

from nemd.utils import generate_nonstationary_signal


MIX_FRACTIONS = {
    "stationary":      0.40,
    "chirp_trio":      0.30,
    "widening_am":     0.20,
    "piecewise":       0.10,
}


def diverse_nonstationary_dataset(
    n_signals: int,
    config,
    seed: int = 0,
) -> torch.Tensor:
    """Build a mixed nonstationary dataset.

    Signature matches the ``dataset_fn`` hook in :func:`nemd.train.train`.

    Parameters
    ----------
    n_signals : int
    config : TrainConfig  — uses ``signal_length``, ``sample_rate``,
        ``snr_range_db``
    seed : int

    Returns
    -------
    signals : (n_signals, signal_length) float32 tensor
    """
    rng = np.random.default_rng(seed)
    T = config.signal_length
    fs = config.sample_rate
    duration = T / fs

    # Assign each signal to a kind by cumulative fractions
    kinds = list(MIX_FRACTIONS.keys())
    weights = np.array([MIX_FRACTIONS[k] for k in kinds])
    assigned = rng.choice(len(kinds), size=n_signals, p=weights / weights.sum())

    out = np.empty((n_signals, T), dtype=np.float64)
    snr_lo, snr_hi = config.snr_range_db

    for i in range(n_signals):
        kind = kinds[int(assigned[i])]
        # Generate clean signal
        _, clean, _, _ = generate_nonstationary_signal(
            n_samples=T, duration=duration, kind=kind, rng=rng, noise_std=0.0,
        )
        # Scale noise to match target SNR
        snr_db = rng.uniform(snr_lo, snr_hi)
        sig_power = float(np.mean(clean ** 2) + 1e-12)
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = rng.normal(0, np.sqrt(noise_power), size=T)
        out[i] = clean + noise

    return torch.from_numpy(out).float()
