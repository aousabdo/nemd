"""3-class synthetic classification dataset for Phase 3 Exp 3.

Each class has three AM-FM components with different energy distributions:
  Class A : high-freq dominant  (carrier 40–60 Hz, amplitude 1.0)
  Class B : mid-freq dominant   (carrier 10–20 Hz, amplitude 1.0)
  Class C : low-freq dominant   (carrier 2–5 Hz,   amplitude 1.0)

Non-dominant bands have amplitudes 0.2–0.3 so the signal always contains
all three components but the discriminative information is "which band
carries most of the energy".  This is exactly what the filter bank
controls — a clean test of whether task-aware training helps.
"""

from __future__ import annotations

import numpy as np
import torch

from nemd.utils import generate_am_fm_component


# Class definitions: (band low, band high, amp when dominant, amp when not).
#
# Amplitudes are intentionally CLOSE (1.0 vs ~0.7) and frequency bands
# OVERLAP slightly so the raw-signal CNN can't win by gross spectral
# energy alone — it has to actually pick up the ~20–30% dominance in
# the right band.  Combined with the low SNR (3 dB) used in the runner,
# this creates a non-trivial task where a smarter decomposition can
# help.  See `generate_classification_dataset`.
CLASS_SPECS = {
    "A": {"bands": [(35.0, 65.0), (8.0, 22.0), (1.5, 6.0)], "dominant_idx": 0,
          "dominant_amp": 1.0, "other_amps": [0.70, 0.60]},
    "B": {"bands": [(35.0, 65.0), (8.0, 22.0), (1.5, 6.0)], "dominant_idx": 1,
          "dominant_amp": 1.0, "other_amps": [0.60, 0.70]},
    "C": {"bands": [(35.0, 65.0), (8.0, 22.0), (1.5, 6.0)], "dominant_idx": 2,
          "dominant_amp": 1.0, "other_amps": [0.70, 0.60]},
}

CLASS_NAMES = ["A", "B", "C"]  # label 0, 1, 2


def _one_signal(
    rng: np.random.Generator,
    class_label: int,
    n_samples: int,
    fs: float,
    snr_db: float,
) -> np.ndarray:
    """Generate a single labelled signal."""
    spec = CLASS_SPECS[CLASS_NAMES[class_label]]
    duration = n_samples / fs
    t = np.linspace(0, duration, n_samples, endpoint=False)

    dominant_idx = spec["dominant_idx"]
    dominant_amp = spec["dominant_amp"]
    other_amps = list(spec["other_amps"])

    # Per-signal amplitude jitter (±10%) so the dominance isn't exactly
    # the same in every example; forces the classifier to generalise
    # rather than memorise a fixed template.
    jitter = lambda a: float(a * rng.uniform(0.9, 1.1))

    components = []
    for i, (lo, hi) in enumerate(spec["bands"]):
        if i == dominant_idx:
            amp = jitter(dominant_amp)
        else:
            amp = jitter(other_amps.pop(0))

        f0 = float(rng.uniform(lo, hi))
        # AM-FM component via the existing helper
        comp = generate_am_fm_component(
            t, f0=f0,
            f_mod=float(rng.uniform(0.2, 2.5)),
            a_mod=float(rng.uniform(0.1, 0.4)),
            phase=float(rng.uniform(0, 2 * np.pi)),
            freq_dev=float(f0 * rng.uniform(0.02, 0.10)),
        )
        components.append(amp * comp)

    clean = np.sum(components, axis=0)
    sig_power = float(np.mean(clean ** 2) + 1e-12)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), size=n_samples)
    return clean + noise


def generate_classification_dataset(
    n_per_class: int = 1000,
    n_samples: int = 1024,
    fs: float = 1000.0,
    snr_db: float = 10.0,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a balanced 3-class dataset.

    Returns
    -------
    signals : (3*n_per_class, n_samples) float32 tensor
    labels  : (3*n_per_class,) long tensor with values in {0, 1, 2}
    """
    rng = np.random.default_rng(seed)
    n_total = 3 * n_per_class
    signals = np.empty((n_total, n_samples), dtype=np.float64)
    labels = np.empty(n_total, dtype=np.int64)

    idx = 0
    for class_label in range(3):
        for _ in range(n_per_class):
            signals[idx] = _one_signal(rng, class_label, n_samples, fs, snr_db)
            labels[idx] = class_label
            idx += 1

    # Shuffle
    perm = rng.permutation(n_total)
    signals = signals[perm]
    labels = labels[perm]

    return torch.from_numpy(signals).float(), torch.from_numpy(labels).long()


def make_splits(
    n_train_per_class: int = 1000,
    n_val_per_class: int = 200,
    n_test_per_class: int = 200,
    n_samples: int = 1024,
    fs: float = 1000.0,
    snr_db: float = 10.0,
    seed: int = 42,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Three-way split, reproducible."""
    train = generate_classification_dataset(
        n_train_per_class, n_samples, fs, snr_db, seed=seed,
    )
    val = generate_classification_dataset(
        n_val_per_class, n_samples, fs, snr_db, seed=seed + 1,
    )
    test = generate_classification_dataset(
        n_test_per_class, n_samples, fs, snr_db, seed=seed + 2,
    )
    return {"train": train, "val": val, "test": test}
