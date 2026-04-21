"""Baseline filter-bank frontends for the classification sweep.

Two additional frontends are provided:

- ``MelFilterBank``: a fixed mel-scale bank of K triangular bandpass
  filters spanning [0, Nyquist]. No learnable parameters.
- ``SincNetFrontend``: a learnable bank of K sinc-windowed bandpass
  filters parameterised by (f_low, f_high) per filter (Ravanelli & Bengio,
  2018).

Both produce filtered time-domain signals of shape (B, K, T), matching
the shape of the N-EMD output so the same per-IMF feature extractor
(log-energy, spectral centroid, bandwidth) can be reused for a clean
head-to-head comparison.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Fixed mel-scale filter bank
# ---------------------------------------------------------------------

def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


class MelFilterBank(nn.Module):
    """Fixed mel-scale triangular filter bank in the frequency domain.

    Produces (B, K, T) time-domain outputs after multiplying the rFFT of
    the input by K triangular filters spanning [f_min, Nyquist] on a mel
    scale, then inverting.  No trainable parameters.
    """

    def __init__(
        self,
        num_filters: int = 3,
        sample_rate: float = 1000.0,
        f_min: float = 0.5,
        n_samples: int = 1024,
    ) -> None:
        super().__init__()
        self.num_filters = num_filters
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.n_freqs = n_samples // 2 + 1

        nyq = sample_rate / 2.0
        mel_edges = np.linspace(
            _hz_to_mel(np.array([f_min]))[0],
            _hz_to_mel(np.array([nyq]))[0],
            num_filters + 2,
        )
        hz_edges = _mel_to_hz(mel_edges)
        freqs = np.linspace(0.0, nyq, self.n_freqs)

        # Triangular filters (K, n_freqs)
        fb = np.zeros((num_filters, self.n_freqs), dtype=np.float32)
        for k in range(num_filters):
            lo, mid, hi = hz_edges[k], hz_edges[k + 1], hz_edges[k + 2]
            # Rising edge lo -> mid
            rising = (freqs - lo) / max(mid - lo, 1e-9)
            rising = np.clip(rising, 0.0, 1.0) * (freqs >= lo) * (freqs <= mid)
            # Falling edge mid -> hi
            falling = (hi - freqs) / max(hi - mid, 1e-9)
            falling = np.clip(falling, 0.0, 1.0) * (freqs >= mid) * (freqs <= hi)
            fb[k] = rising + falling
        self.register_buffer("filters", torch.from_numpy(fb))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) -> (B, K, T)."""
        B, T = x.shape
        X = torch.fft.rfft(x, dim=-1)                       # (B, n_freqs)
        filtered = self.filters.unsqueeze(0) * X.unsqueeze(1)  # (B, K, n_freqs)
        return torch.fft.irfft(filtered, n=T, dim=-1)       # (B, K, T)


# ---------------------------------------------------------------------
# SincNet-style learnable bandpass bank
# ---------------------------------------------------------------------

class SincNetFrontend(nn.Module):
    """Bank of K learnable sinc-windowed bandpass filters.

    Each filter is parameterised by (f_low, f_high) in Hz; a windowed
    sinc difference implements the bandpass in time domain. Initialised
    on a mel scale to break index symmetry. The cutoffs are learnable
    end-to-end with the downstream loss.
    """

    def __init__(
        self,
        num_filters: int = 3,
        sample_rate: float = 1000.0,
        n_taps: int = 101,
        f_min: float = 0.5,
    ) -> None:
        super().__init__()
        if n_taps % 2 == 0:
            n_taps += 1
        self.num_filters = num_filters
        self.sample_rate = sample_rate
        self.n_taps = n_taps
        self.f_min = f_min

        nyq = sample_rate / 2.0
        mel_edges = np.linspace(
            _hz_to_mel(np.array([f_min]))[0],
            _hz_to_mel(np.array([nyq]))[0],
            num_filters + 1,
        )
        hz_edges = _mel_to_hz(mel_edges)
        low  = torch.tensor(hz_edges[:-1], dtype=torch.float32)
        high = torch.tensor(hz_edges[1:],  dtype=torch.float32)
        self.f_low  = nn.Parameter(low)
        self.f_high = nn.Parameter(high)

        # Precompute the Hamming window and the tap centres
        n = torch.arange(n_taps, dtype=torch.float32) - (n_taps - 1) / 2.0
        self.register_buffer("n_taps_arr", n)
        window = 0.54 - 0.46 * torch.cos(
            2.0 * math.pi * torch.arange(n_taps, dtype=torch.float32) / (n_taps - 1)
        )
        self.register_buffer("window", window)

    def _compute_filters(self) -> torch.Tensor:
        """Return (K, 1, n_taps) filter taps."""
        nyq = self.sample_rate / 2.0
        # Clamp cutoffs so they stay in a valid ordering. torch.clamp
        # only accepts scalars or tensors uniformly, so we build a
        # tensor lower-bound for the high cutoff explicitly.
        low = torch.clamp(self.f_low, min=self.f_min, max=nyq - 1e-3)
        high_min = low + 1e-3
        high = torch.maximum(self.f_high, high_min).clamp(max=nyq)
        f_l = (low / self.sample_rate).unsqueeze(-1)   # (K, 1)
        f_h = (high / self.sample_rate).unsqueeze(-1)  # (K, 1)
        n = self.n_taps_arr.unsqueeze(0)               # (1, n_taps)
        # Manual normalised sinc: sin(pi x) / (pi x), with the sinc(0)=1
        # limit handled explicitly. MPS does not yet support torch.special.sinc.
        def _sinc(x: torch.Tensor) -> torch.Tensor:
            eps = 1e-8
            piX = math.pi * x
            return torch.where(
                x.abs() < eps,
                torch.ones_like(x),
                torch.sin(piX) / (piX + eps),
            )

        lp_h = 2.0 * f_h * _sinc(2.0 * f_h * n)
        lp_l = 2.0 * f_l * _sinc(2.0 * f_l * n)
        bp = (lp_h - lp_l) * self.window.unsqueeze(0)  # (K, n_taps)
        return bp.unsqueeze(1)                         # (K, 1, n_taps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) -> (B, K, T) via 1D conv with computed taps."""
        B, T = x.shape
        taps = self._compute_filters()
        pad = self.n_taps // 2
        y = F.conv1d(
            x.unsqueeze(1), taps, padding=pad,
        )  # (B, K, T)
        return y
