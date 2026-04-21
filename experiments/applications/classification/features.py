"""Differentiable per-IMF feature extraction (energy, centroid, bandwidth).

Input:  imfs of shape (B, K, T)
Output: features of shape (B, 3*K) — [energy, centroid, bandwidth] per IMF

All operations use torch so gradients flow end-to-end.  The centroid and
bandwidth are computed from the rFFT of each IMF, as required for
jointly training N-EMD with a downstream task.
"""

from __future__ import annotations

import torch


def imf_features(
    imfs: torch.Tensor,
    sample_rate: float = 1000.0,
    eps: float = 1e-8,
    normalise: bool = True,
) -> torch.Tensor:
    """Compute 3 features per IMF: log-energy, spectral centroid, bandwidth.

    Parameters
    ----------
    imfs : (B, K, T) real tensor
    sample_rate : float
    eps : float
    normalise : bool
        If True, normalise features to roughly ~[-3, 3] range using
        typical signal statistics so the MLP trains well.

    Returns
    -------
    features : (B, 3*K) real tensor
    """
    B, K, T = imfs.shape
    nyquist = sample_rate / 2.0

    # --- Energy (log scale for stability) ---
    energy = (imfs ** 2).sum(dim=-1).clamp(min=eps)           # (B, K)
    log_energy = torch.log(energy)

    # --- FFT for centroid + bandwidth ---
    X = torch.fft.rfft(imfs, dim=-1)                          # (B, K, n_freqs)
    psd = X.real ** 2 + X.imag ** 2                           # (B, K, n_freqs)
    n_freqs = psd.shape[-1]
    freqs = torch.linspace(
        0.0, nyquist, n_freqs, device=imfs.device, dtype=imfs.dtype,
    )

    total_power = psd.sum(dim=-1).clamp(min=eps)              # (B, K)
    centroid = (psd * freqs).sum(dim=-1) / total_power        # (B, K)

    # Bandwidth = sqrt(Σ P(f) · (f - centroid)² / Σ P(f))
    diff_sq = (freqs.unsqueeze(0).unsqueeze(0) - centroid.unsqueeze(-1)) ** 2
    bandwidth = torch.sqrt((psd * diff_sq).sum(dim=-1) / total_power + eps)

    if normalise:
        # Empirical normalisation: centroid/bandwidth ∈ [0, Nyquist]
        centroid = centroid / (nyquist + eps)                 # → [0, 1]
        bandwidth = bandwidth / (nyquist + eps)               # → [0, 1]
        # log_energy varies widely; centre around typical sig-power log
        # (a simple shift+scale is adequate — the MLP picks up the rest)
        log_energy = log_energy / 4.0                         # rough scale

    # Stack features per IMF, then flatten: (B, K, 3) -> (B, 3*K)
    per_imf = torch.stack([log_energy, centroid, bandwidth], dim=-1)  # (B, K, 3)
    return per_imf.reshape(B, 3 * K)
