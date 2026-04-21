"""Differentiable signal-processing building blocks for N-EMD.

All operations accept and return PyTorch tensors and support batched inputs
of shape ``(B, N)`` or unbatched ``(N,)``.  Gradients flow through every
operation so they can be used inside a training loop.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Differentiable Hilbert transform
# ---------------------------------------------------------------------------

def hilbert_transform(x: torch.Tensor) -> torch.Tensor:
    """Compute the analytic signal via the Hilbert transform (FFT-based).

    Given a real signal x(t), the analytic signal is::

        z(t) = x(t) + j * H[x](t)

    where H is the Hilbert transform.

    Parameters
    ----------
    x : Tensor of shape ``(..., N)``
        Real-valued input signal(s).

    Returns
    -------
    analytic : complex Tensor of shape ``(..., N)``
        The analytic signal.
    """
    N = x.shape[-1]
    X = torch.fft.fft(x, dim=-1)

    # Build the frequency-domain multiplier h:
    #   h[0] = 1, h[N/2] = 1 (if N even), h[1..N/2-1] = 2, rest = 0
    h = torch.zeros(N, device=x.device, dtype=x.dtype)
    h[0] = 1.0
    if N % 2 == 0:
        h[N // 2] = 1.0
        h[1 : N // 2] = 2.0
    else:
        h[1 : (N + 1) // 2] = 2.0

    return torch.fft.ifft(X * h, dim=-1)


def instantaneous_amplitude(x: torch.Tensor) -> torch.Tensor:
    """Instantaneous amplitude (envelope) via the analytic signal.

    Parameters
    ----------
    x : Tensor of shape ``(..., N)``

    Returns
    -------
    env : Tensor of shape ``(..., N)``
    """
    z = hilbert_transform(x)
    return z.abs()


def instantaneous_phase(x: torch.Tensor) -> torch.Tensor:
    """Instantaneous phase via the analytic signal (unwrapped).

    Parameters
    ----------
    x : Tensor of shape ``(..., N)``

    Returns
    -------
    phase : Tensor of shape ``(..., N)``
        Unwrapped instantaneous phase in radians.
    """
    z = hilbert_transform(x)
    phase = torch.atan2(z.imag, z.real)
    return _unwrap_phase(phase)


def instantaneous_frequency(
    x: torch.Tensor, fs: float = 1.0
) -> torch.Tensor:
    """Instantaneous frequency via differentiation of unwrapped phase.

    Parameters
    ----------
    x : Tensor of shape ``(..., N)``
    fs : float
        Sampling frequency.

    Returns
    -------
    freq : Tensor of shape ``(..., N-1)``
        Instantaneous frequency at midpoints.
    """
    phase = instantaneous_phase(x)
    # Central-difference style: diff gives phase increments
    dphase = torch.diff(phase, dim=-1)
    return dphase * fs / (2.0 * np.pi)


def _unwrap_phase(phase: torch.Tensor) -> torch.Tensor:
    """Differentiable phase unwrapping.

    Adjusts phase jumps greater than pi to produce a continuous phase signal.
    """
    dp = torch.diff(phase, dim=-1)
    # Wrap differences to [-pi, pi]
    dp_wrapped = (dp + np.pi) % (2 * np.pi) - np.pi
    # Fix the edge case where dp_wrapped == -pi and dp > 0
    dp_wrapped = torch.where(
        (dp_wrapped == -np.pi) & (dp > 0),
        torch.tensor(np.pi, device=phase.device, dtype=phase.dtype),
        dp_wrapped,
    )
    correction = torch.cumsum(dp_wrapped - dp, dim=-1)
    # Prepend zeros for the first sample
    zeros = torch.zeros(
        *phase.shape[:-1], 1, device=phase.device, dtype=phase.dtype
    )
    return phase + torch.cat([zeros, correction], dim=-1)


# ---------------------------------------------------------------------------
# Differentiable envelope estimation
# ---------------------------------------------------------------------------

def envelope_mean(x: torch.Tensor, window_size: int = 51) -> torch.Tensor:
    """Estimate the mean envelope via a smoothing filter.

    Uses a simple moving average as a differentiable alternative to cubic
    spline envelope fitting.  The filter is applied with reflection padding
    to reduce boundary effects.

    Parameters
    ----------
    x : Tensor of shape ``(..., N)``
    window_size : int
        Smoothing window (must be odd).

    Returns
    -------
    mean_env : Tensor of shape ``(..., N)``
    """
    if window_size % 2 == 0:
        window_size += 1

    # Ensure 3-D for conv1d: (B, 1, N)
    orig_shape = x.shape
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.dim() == 2:
        x = x.unsqueeze(1)  # (B, 1, N)

    pad = window_size // 2
    x_padded = F.pad(x, (pad, pad), mode="reflect")

    kernel = torch.ones(1, 1, window_size, device=x.device, dtype=x.dtype) / window_size
    mean_env = F.conv1d(x_padded, kernel)

    # Restore original shape
    mean_env = mean_env.squeeze(1)
    if len(orig_shape) == 1:
        mean_env = mean_env.squeeze(0)
    return mean_env


def upper_lower_envelopes(
    x: torch.Tensor, window_size: int = 51
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate upper and lower envelopes via max/min pooling + smoothing.

    A differentiable approximation: we use max-pool to get local maxima
    envelope and negated-max-pool for local minima envelope, followed by
    smoothing.

    Parameters
    ----------
    x : Tensor of shape ``(..., N)``
    window_size : int

    Returns
    -------
    upper, lower : Tensors of shape ``(..., N)``
    """
    if window_size % 2 == 0:
        window_size += 1

    orig_shape = x.shape
    if x.dim() == 1:
        x = x.unsqueeze(0)
    # (B, N) -> (B, 1, N) for pooling
    x3 = x.unsqueeze(1)

    pad = window_size // 2
    x_padded = F.pad(x3, (pad, pad), mode="reflect")

    upper = F.max_pool1d(x_padded, kernel_size=window_size, stride=1)
    lower = -F.max_pool1d(-x_padded, kernel_size=window_size, stride=1)

    upper = upper.squeeze(1)
    lower = lower.squeeze(1)

    # Smooth the envelopes
    upper = envelope_mean(upper, window_size=window_size)
    lower = envelope_mean(lower, window_size=window_size)

    if len(orig_shape) == 1:
        upper = upper.squeeze(0)
        lower = lower.squeeze(0)
    return upper, lower


# ---------------------------------------------------------------------------
# Bandwidth / narrow-band measure
# ---------------------------------------------------------------------------

def spectral_bandwidth(x: torch.Tensor, fs: float = 1.0) -> torch.Tensor:
    """Spectral bandwidth of a signal (RMS deviation from spectral centroid).

    A narrow-band (mono-component) signal has low spectral bandwidth.

    Parameters
    ----------
    x : Tensor of shape ``(..., N)``
    fs : float
        Sampling frequency.

    Returns
    -------
    bw : Tensor of shape ``(...,)``
        Spectral bandwidth in Hz.
    """
    N = x.shape[-1]
    X = torch.fft.rfft(x, dim=-1)
    psd = (X.real ** 2 + X.imag ** 2)  # power spectral density

    freqs = torch.linspace(0, fs / 2, psd.shape[-1], device=x.device, dtype=x.dtype)

    total_power = psd.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    centroid = (psd * freqs).sum(dim=-1, keepdim=True) / total_power
    bw_sq = (psd * (freqs - centroid) ** 2).sum(dim=-1) / total_power.squeeze(-1)
    return bw_sq.sqrt()
