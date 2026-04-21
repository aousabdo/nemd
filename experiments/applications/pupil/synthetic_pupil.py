"""Synthetic pupil-like signal generator for N-EMD pretraining.

Real pupil signals have 1/f-like broadband spectra with overlapping
physiological modes, NOT sharp AM-FM carriers. This generator produces
signals that match that statistical structure while providing known
ground-truth component decomposition for supervised evaluation.

Each "component" is a band-limited 1/f process with slow amplitude
modulation — broadband, not mono-component. Together they sum to a
realistic pupil-like signal.

Physiological bands:
  VLF:          0.03 – 0.08 Hz  (very low frequency, slow drift)
  LF:           0.08 – 0.15 Hz  (low frequency, sympathetic)
  Respiratory:  0.15 – 0.40 Hz  (parasympathetic, breathing)
  Hippus:       0.40 – 1.00 Hz  (higher autonomic, cardiac influence)
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import signal as scisignal


PUPIL_BANDS = [
    (0.03, 0.08),    # VLF
    (0.08, 0.15),    # LF sympathetic
    (0.15, 0.40),    # respiratory / parasympathetic
    (0.40, 1.00),    # higher autonomic / hippus
]

BAND_NAMES = ["VLF", "LF", "Respiratory", "Hippus"]


def _bandpass_noise(
    n_samples: int,
    fs: float,
    f_low: float,
    f_high: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate band-limited noise with 1/f spectral envelope."""
    # White noise in frequency domain
    n_freqs = n_samples // 2 + 1
    freqs = np.fft.rfftfreq(n_samples, d=1 / fs)

    # Random phase, shaped magnitude
    magnitude = rng.standard_normal(n_freqs)
    phase = rng.uniform(0, 2 * np.pi, n_freqs)

    # 1/f envelope (avoid divide by zero at DC)
    f_safe = np.maximum(freqs, 1e-6)
    one_over_f = 1.0 / np.sqrt(f_safe)

    # Bandpass window (raised cosine edges for smooth rolloff)
    bp = np.zeros(n_freqs)
    bw = f_high - f_low
    rolloff = bw * 0.15  # 15% cosine rolloff at edges

    for i, f in enumerate(freqs):
        if f_low + rolloff <= f <= f_high - rolloff:
            bp[i] = 1.0
        elif f_low <= f < f_low + rolloff:
            bp[i] = 0.5 * (1 - np.cos(np.pi * (f - f_low) / rolloff))
        elif f_high - rolloff < f <= f_high:
            bp[i] = 0.5 * (1 + np.cos(np.pi * (f - (f_high - rolloff)) / rolloff))

    # Combine
    spectrum = magnitude * one_over_f * bp * np.exp(1j * phase)
    spectrum[0] = 0  # remove DC
    return np.fft.irfft(spectrum, n=n_samples)


def _slow_amplitude_modulation(
    n_samples: int,
    fs: float,
    mod_period_range: tuple[float, float] = (5.0, 20.0),
    mod_depth: float = 0.4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Slow amplitude envelope that varies over 5-20 seconds."""
    if rng is None:
        rng = np.random.default_rng()
    t = np.arange(n_samples) / fs
    period = rng.uniform(*mod_period_range)
    phase = rng.uniform(0, 2 * np.pi)
    envelope = 1.0 + mod_depth * np.sin(2 * np.pi * t / period + phase)
    return envelope


def generate_pupil_like_signal(
    n_samples: int = 2048,
    fs: float = 100.0,
    bands: list[tuple[float, float]] | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    noise_std: float = 0.0,
    power_decay: float = 1.5,
) -> tuple[np.ndarray, list[np.ndarray], list[tuple[float, float]]]:
    """Generate a synthetic pupil-like signal with known components.

    Parameters
    ----------
    n_samples : int
    fs : float, sampling rate
    bands : list of (f_low, f_high) tuples, one per component
    seed, rng : for reproducibility
    noise_std : additive Gaussian noise level (after scaling)
    power_decay : how fast higher bands lose power (1/f^decay)

    Returns
    -------
    signal : (n_samples,) — composite signal
    components : list of (n_samples,) — ground-truth per-band components
    bands : the frequency bands used
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    if bands is None:
        bands = list(PUPIL_BANDS)

    K = len(bands)
    components = []

    for k, (f_low, f_high) in enumerate(bands):
        # Band-limited 1/f noise
        comp = _bandpass_noise(n_samples, fs, f_low, f_high, rng)

        # Slow amplitude modulation (nonstationarity)
        envelope = _slow_amplitude_modulation(n_samples, fs, rng=rng)
        comp = comp * envelope

        # Scale: higher bands get less power (1/f^decay overall)
        f_center = (f_low + f_high) / 2
        scale = 1.0 / (f_center ** power_decay + 0.01)

        # Random per-signal amplitude jitter (±30%)
        amp = rng.uniform(0.7, 1.3) * scale
        comp = amp * comp

        # Normalize to unit variance per component (then re-scale)
        std = comp.std()
        if std > 1e-8:
            comp = comp / std * amp

        components.append(comp)

    sig = np.sum(components, axis=0)

    # Add measurement noise
    if noise_std > 0:
        sig_power = np.mean(sig ** 2)
        noise = rng.normal(0, 1, n_samples)
        noise = noise * np.sqrt(sig_power) * noise_std
        sig = sig + noise

    return sig, components, bands


def generate_pupil_training_dataset(
    n_signals: int,
    config,
    seed: int = 0,
) -> torch.Tensor:
    """Dataset generator matching the train.py dataset_fn signature.

    Produces synthetic pupil-like signals for self-supervised N-EMD
    pretraining at pupil-appropriate frequencies.
    """
    rng = np.random.default_rng(seed)
    T = config.signal_length
    fs = config.sample_rate

    out = np.empty((n_signals, T), dtype=np.float64)
    for i in range(n_signals):
        # Slight band jitter so the model doesn't memorize exact boundaries
        bands = []
        for f_low, f_high in PUPIL_BANDS:
            jitter = rng.uniform(-0.01, 0.01)
            bands.append((max(0.02, f_low + jitter), f_high + jitter))

        noise_std = rng.uniform(0.05, 0.3)  # variable SNR
        sig, _, _ = generate_pupil_like_signal(
            n_samples=T, fs=fs, bands=bands,
            rng=rng, noise_std=noise_std,
        )
        # Z-score normalize (like real preprocessing)
        mu, std = sig.mean(), sig.std()
        if std > 1e-8:
            sig = (sig - mu) / std
        out[i] = sig

    return torch.from_numpy(out).float()
