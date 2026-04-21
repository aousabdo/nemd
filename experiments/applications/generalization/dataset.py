"""Generalization test sets for Phase 3 Exp 2.

Training distribution (shared across experiments 2 and 3 for speed):
  3-component AM-FM, carriers in 5–60 Hz, moderate AM/FM, SNR 15–30 dB.

Held-out test distributions, each designed to stress a different aspect:
  A: In-distribution baseline (same as training)
  B: Higher frequencies [60–200 Hz] — frequency extrapolation
  C: Noisy (SNR ≈ 5 dB)
  D: Damped sinusoids (exponential envelope, not AM-FM)
  E: K mismatch (generate 4 components, decompose with K=3)
"""

from __future__ import annotations

import numpy as np
import torch

from nemd.utils import generate_am_fm_component


def _add_noise(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    power = float(np.mean(signal ** 2) + 1e-12)
    noise_power = power / (10 ** (snr_db / 10))
    return signal + rng.normal(0, np.sqrt(noise_power), size=signal.shape)


def _am_fm_signal(
    rng: np.random.Generator,
    f_ranges: list[tuple[float, float]],
    n_samples: int,
    fs: float,
    snr_db: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Generate a multi-component AM-FM signal from the given frequency ranges."""
    duration = n_samples / fs
    t = np.linspace(0, duration, n_samples, endpoint=False)
    comps = []
    for lo, hi in f_ranges:
        f0 = float(rng.uniform(lo, hi))
        comps.append(generate_am_fm_component(
            t, f0=f0,
            f_mod=float(rng.uniform(0.2, 2.5)),
            a_mod=float(rng.uniform(0.1, 0.4)),
            phase=float(rng.uniform(0, 2 * np.pi)),
            freq_dev=float(f0 * rng.uniform(0.02, 0.1)),
        ))
    clean = np.sum(comps, axis=0)
    return _add_noise(clean, snr_db, rng), comps


def _damped_sinusoid_signal(
    rng: np.random.Generator,
    f_ranges: list[tuple[float, float]],
    n_samples: int,
    fs: float,
    snr_db: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Exponentially decaying sinusoids rather than AM-FM carriers."""
    duration = n_samples / fs
    t = np.linspace(0, duration, n_samples, endpoint=False)
    comps = []
    for lo, hi in f_ranges:
        f0 = float(rng.uniform(lo, hi))
        # Damping τ → amplitude goes from 1 to exp(-duration/τ); τ in [0.5, 3.0]s
        tau = float(rng.uniform(0.5, 3.0))
        # Trigger time in first half so the decay is visible
        t0 = float(rng.uniform(0.0, duration * 0.3))
        phase = float(rng.uniform(0, 2 * np.pi))
        envelope = np.where(t >= t0, np.exp(-(t - t0) / tau), 0.0)
        comps.append(envelope * np.cos(2 * np.pi * f0 * t + phase))
    clean = np.sum(comps, axis=0)
    return _add_noise(clean, snr_db, rng), comps


# ---------------------------------------------------------------------------
# Generator signatures match ``dataset_fn(n_signals, config, seed)`` so
# they can be passed to ``nemd.train.train``.
# ---------------------------------------------------------------------------

TRAIN_RANGES = [(40.0, 60.0), (15.0, 30.0), (5.0, 15.0)]  # wide → covers 5-60 Hz


def training_dataset(n_signals: int, config, seed: int = 0) -> torch.Tensor:
    """In-distribution training data for Exp 2."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_signals, config.signal_length), dtype=np.float64)
    for i in range(n_signals):
        snr = rng.uniform(15.0, 30.0)
        sig, _ = _am_fm_signal(
            rng, TRAIN_RANGES, config.signal_length, config.sample_rate, snr,
        )
        out[i] = sig
    return torch.from_numpy(out).float()


# ---------------------------------------------------------------------------
# Held-out test sets
# ---------------------------------------------------------------------------

def generate_test_A(n: int, n_samples: int, fs: float, seed: int = 0):
    """A: In-distribution (same as training)."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, n_samples))
    comps_list = []
    for i in range(n):
        snr = rng.uniform(15.0, 30.0)
        sig, comps = _am_fm_signal(rng, TRAIN_RANGES, n_samples, fs, snr)
        out[i] = sig
        comps_list.append(comps)
    return torch.from_numpy(out).float(), comps_list


def generate_test_B(n: int, n_samples: int, fs: float, seed: int = 0):
    """B: Higher-frequency extrapolation [60-200 Hz]."""
    rng = np.random.default_rng(seed)
    # Clamp top to 0.8 * Nyquist to avoid aliasing
    top = min(200.0, fs / 2 * 0.8)
    ranges = [(120.0, top), (80.0, 120.0), (60.0, 80.0)]
    out = np.empty((n, n_samples))
    comps_list = []
    for i in range(n):
        sig, comps = _am_fm_signal(rng, ranges, n_samples, fs, snr_db=20.0)
        out[i] = sig
        comps_list.append(comps)
    return torch.from_numpy(out).float(), comps_list


def generate_test_C(n: int, n_samples: int, fs: float, seed: int = 0):
    """C: Very noisy (SNR ≈ 5 dB)."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, n_samples))
    comps_list = []
    for i in range(n):
        sig, comps = _am_fm_signal(rng, TRAIN_RANGES, n_samples, fs, snr_db=5.0)
        out[i] = sig
        comps_list.append(comps)
    return torch.from_numpy(out).float(), comps_list


def generate_test_D(n: int, n_samples: int, fs: float, seed: int = 0):
    """D: Damped sinusoids (exponential envelope, not AM-FM)."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, n_samples))
    comps_list = []
    for i in range(n):
        sig, comps = _damped_sinusoid_signal(
            rng, TRAIN_RANGES, n_samples, fs, snr_db=20.0,
        )
        out[i] = sig
        comps_list.append(comps)
    return torch.from_numpy(out).float(), comps_list


def generate_test_E(n: int, n_samples: int, fs: float, seed: int = 0):
    """E: K mismatch — generate 4 components, decompose with K=3."""
    rng = np.random.default_rng(seed)
    ranges = [(45.0, 60.0), (25.0, 40.0), (12.0, 20.0), (3.0, 8.0)]
    out = np.empty((n, n_samples))
    comps_list = []
    for i in range(n):
        sig, comps = _am_fm_signal(rng, ranges, n_samples, fs, snr_db=20.0)
        out[i] = sig
        comps_list.append(comps)
    return torch.from_numpy(out).float(), comps_list


TEST_GENERATORS = {
    "A_in_dist":        generate_test_A,
    "B_high_freq":      generate_test_B,
    "C_noisy_5dB":      generate_test_C,
    "D_damped":         generate_test_D,
    "E_K_mismatch":     generate_test_E,
}
