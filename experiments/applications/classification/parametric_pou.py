"""Parametric partition-of-unity baseline.

Reviewer 2 asked whether the win comes from POU-as-a-constraint or
from the softmax-over-K parameterisation specifically. This module
implements the former: a bank of K Gaussian bumps over frequency
with learnable centre-frequency and bandwidth parameters, normalised
across K at every frequency bin to satisfy the partition-of-unity
identity. This gives the same structural guarantees (exact
reconstruction, non-negativity, energy bound) but without a neural
analyzer and with only 2K learnable parameters.

If this baseline approaches NAFB's classification performance, the
POU constraint is the key ingredient. If it trails noticeably, the
flexibility of the learned analyzer (per-input adaptivity) matters
beyond the constraint.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ParametricPOUFilterBank(nn.Module):
    """Normalised Gaussian bandpass bank with learnable centre / width.

    Parameters ``mu`` and ``log_sigma`` are learnable. The filter
    responses are always normalised to sum to one at every frequency
    bin (partition of unity by construction), so reconstruction,
    non-negativity, and energy boundedness all hold as structural
    identities, exactly as for NAFB's softmax partition.
    """

    def __init__(
        self,
        num_imfs: int = 3,
        sample_rate: float = 1000.0,
        n_samples: int = 1024,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_imfs = num_imfs
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.eps = eps
        nyq = sample_rate / 2.0
        self.n_freqs = n_samples // 2 + 1

        # Mel-like initial placements: log-spaced across [f_min, Nyquist].
        f_min = max(1.0, 0.002 * nyq)
        mel = torch.linspace(
            math.log(f_min), math.log(nyq), num_imfs + 2,
        )
        hz_edges = torch.exp(mel)
        centres = 0.5 * (hz_edges[:-2] + hz_edges[1:-1])
        widths  = 0.5 * (hz_edges[1:-1] - hz_edges[:-2]).clamp(min=1.0)

        self.mu = nn.Parameter(centres.float())
        self.log_sigma = nn.Parameter(widths.log().float())

    def forward(
        self,
        x: torch.Tensor,
        num_imfs: int | None = None,
        temperature: float | None = None,
        sort_by_centroid: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """x: (B, T) -> (imfs (B,K,T), residual (B,T), metadata).

        API mirrors nemd.model.NEMD so the end-to-end classifier
        can be reused verbatim.
        """
        B, T = x.shape
        X = torch.fft.rfft(x, dim=-1)
        n_freqs = X.shape[-1]

        freqs = torch.linspace(
            0.0, self.sample_rate / 2.0, n_freqs,
            device=x.device, dtype=torch.float32,
        )
        sigma = torch.exp(self.log_sigma).clamp(min=1.0)        # (K,)
        # Gaussian bump per filter: (K, n_freqs)
        z = (freqs.unsqueeze(0) - self.mu.unsqueeze(-1)) / sigma.unsqueeze(-1)
        G = torch.exp(-0.5 * z.pow(2))
        # Normalise to sum to one at every frequency bin.
        H = G / (G.sum(dim=0, keepdim=True) + self.eps)          # (K, n_freqs)

        filters = H.unsqueeze(0).expand(B, -1, -1).to(X.real.dtype)  # (B, K, n_freqs)
        imf_spectra = filters * X.unsqueeze(1)                      # (B, K, n_freqs)
        imfs = torch.fft.irfft(imf_spectra, n=T, dim=-1)            # (B, K, T)
        residual = torch.zeros(B, T, device=x.device, dtype=x.dtype)

        # Signal-weighted centroid (for ordering loss / sort)
        signal_power = X.real.pow(2) + X.imag.pow(2)
        weighted = filters * signal_power.unsqueeze(1)
        c = (weighted * freqs).sum(-1) / weighted.sum(-1).clamp(min=self.eps)

        metadata = {
            "filters":            filters,
            "centroids":          c,
            "centroids_weighted": c,
            "signal_power":       signal_power,
            "temperature":        1.0,
            "filter_logits":      H.log().unsqueeze(0).expand(B, -1, -1),
        }
        return imfs, residual, metadata

    # The NEMDClassifier calls .set_temperature() and passes temperature
    # and sort_by_centroid kwargs on forward; mirror those no-ops.
    def set_temperature(self, t: float) -> None:
        pass
