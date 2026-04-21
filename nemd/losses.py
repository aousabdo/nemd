"""Physics-constrained loss functions for N-EMD.

Since reconstruction is exact by construction (architectural guarantee),
the loss focuses entirely on the *quality* of the decomposition:

L = λ_ortho·L_ortho
  + λ_narrow·L_narrow
  + λ_mono·L_mono
  + λ_residual·L_residual
  + λ_energy·L_energy
  + λ_order·L_order
  + λ_concentration·L_concentration
  [+ λ_task·L_task]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import numpy as np

from nemd.layers import spectral_bandwidth


# ---------------------------------------------------------------------------
# Individual loss components
# ---------------------------------------------------------------------------

class OrthogonalityLoss(nn.Module):
    """Penalise cross-correlation between IMFs.

    L_ortho = (1 / K(K-1)) * Σ_{i≠j} (⟨c_i, c_j⟩ / (‖c_i‖·‖c_j‖ + ε))²
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, imfs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        imfs : (B, K, T)

        Returns
        -------
        loss : scalar tensor
        """
        B, K, T = imfs.shape
        if K < 2:
            return torch.tensor(0.0, device=imfs.device, dtype=imfs.dtype)

        # Normalise each IMF: (B, K, T)
        norms = imfs.norm(dim=-1, keepdim=True).clamp(min=self.eps)  # (B, K, 1)
        normed = imfs / norms  # (B, K, T)

        # Gram matrix of normalised IMFs: (B, K, K)
        # Each entry is the cosine similarity ∈ [-1, 1]
        gram = torch.bmm(normed, normed.transpose(1, 2))

        # Extract off-diagonal elements and square them
        mask = ~torch.eye(K, device=imfs.device, dtype=torch.bool).unsqueeze(0)
        off_diag = gram.masked_select(mask).reshape(B, K * (K - 1))
        loss = (off_diag ** 2).mean()
        return loss


class NarrowBandLoss(nn.Module):
    """Penalise broad-band IMFs (non-mono-component signals).

    Uses spectral bandwidth (FFT-based, much cheaper than Hilbert IF)::

        L_narrow_k = spectral_bandwidth(IMF_k) / (fs/2)

    Normalised by Nyquist so the loss is in [0, 1] regardless of sample rate.
    A pure tone has BW ≈ 0; white noise has BW ≈ fs/(2√3).
    """

    def __init__(self, fs: float = 1.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.fs = fs
        self.eps = eps

    def forward(self, imfs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        imfs : (B, K, T)

        Returns
        -------
        loss : scalar tensor
        """
        B, K, T = imfs.shape
        # Reshape to (B*K, T) for batched spectral_bandwidth
        flat = imfs.reshape(B * K, T)
        bw = spectral_bandwidth(flat, fs=self.fs)  # (B*K,)
        # Normalise by Nyquist frequency
        nyquist = self.fs / 2
        return (bw / (nyquist + self.eps)).mean()


class MonotonicResidualLoss(nn.Module):
    """Penalise non-monotonicity in the final residual.

    Detects sign changes in consecutive finite differences and normalises
    by the residual's energy::

        L_mono = mean(ReLU(-r'[t] * r'[t+1])) / (mean(r'²) + ε)
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        residual : (B, T)

        Returns
        -------
        loss : scalar tensor
        """
        dr = torch.diff(residual, dim=-1)  # (B, T-1)
        # Product of consecutive derivatives — negative when sign changes
        prod = dr[:, :-1] * dr[:, 1:]      # (B, T-2)
        # Normalise product by its scale so sigmoid has proper dynamic range
        scale = (dr[:, :-1] ** 2 + dr[:, 1:] ** 2).clamp(min=self.eps) / 2
        prod_norm = prod / scale  # ∈ [-1, 1]: -1 = opposite sign, +1 = same sign
        # Penalise negative normalised products (sign changes)
        # ReLU(-prod_norm) is 0 when same sign, positive when sign changes
        loss = torch.relu(-prod_norm).mean()
        return loss


class ResidualEnergyLoss(nn.Module):
    """Penalise energy remaining in the residual.

    Without this loss, the model can satisfy orthogonality and narrow-band
    constraints trivially by making all IMFs ≈ 0 and leaving the full signal
    in the residual.  This loss forces the model to actually *decompose*::

        L_energy = ‖r‖² / (‖x‖² + ε)

    Minimising this pushes signal energy into the IMFs.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self, residual: torch.Tensor, original: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        residual : (B, T)
        original : (B, T)

        Returns
        -------
        loss : scalar tensor
        """
        res_energy = (residual ** 2).sum(dim=-1)      # (B,)
        sig_energy = (original ** 2).sum(dim=-1)       # (B,)
        ratio = res_energy / (sig_energy + self.eps)   # (B,)
        return ratio.mean()


class EnergyConservationLoss(nn.Module):
    """Penalise violation of Parseval-type energy conservation.

    If the decomposition is truly orthogonal, then::

        ‖x‖² = Σ_k ‖c_k‖² + ‖r‖²

    This loss penalises the deviation::

        L_conserve = | Σ_k ‖c_k‖² + ‖r‖² - ‖x‖² | / (‖x‖² + ε)

    Prevents energy inflation (IMFs creating/cancelling energy) which
    cosine-similarity-based orthogonality alone cannot prevent.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        imfs: torch.Tensor,
        residual: torch.Tensor,
        original: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        imfs : (B, K, T)
        residual : (B, T)
        original : (B, T)

        Returns
        -------
        loss : scalar tensor
        """
        sig_energy = (original ** 2).sum(dim=-1)       # (B,)
        imf_energy = (imfs ** 2).sum(dim=(1, 2))       # (B,)
        res_energy = (residual ** 2).sum(dim=-1)        # (B,)
        total_decomp_energy = imf_energy + res_energy   # (B,)
        # Normalised absolute deviation from energy conservation
        deviation = (total_decomp_energy - sig_energy).abs() / (sig_energy + self.eps)
        return deviation.mean()


class FrequencyOrderingLoss(nn.Module):
    """Enforce descending frequency ordering across IMFs.

    Classical EMD extracts IMFs from highest to lowest frequency.  With
    only cosine-similarity orthogonality, the model has no signal about
    which IMF should capture which band — this loss provides it::

        centroid(c_1) > centroid(c_2) > ... > centroid(c_K)

    A squared-hinge loss with a normalised margin penalises violations::

        L_order = (1/(K-1)) · Σ_{k=1}^{K-1} ReLU(ĉ_{k+1} - ĉ_k + m)²

    where ĉ_k = centroid(c_k) / (fs/2) ∈ [0, 1] and ``m`` is
    ``normalized_margin`` (default 0.02 = 2% of Nyquist).
    """

    def __init__(
        self,
        sample_rate: float = 1.0,
        normalized_margin: float = 0.02,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.normalized_margin = normalized_margin
        self.eps = eps

    def forward(
        self,
        imfs: torch.Tensor | None = None,
        centroids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the ordering loss.

        Parameters
        ----------
        imfs : (B, K, T), optional
            If provided, centroids are computed from the IMF spectra.
        centroids : (B, K), optional
            If provided (e.g. precomputed from filter shapes in the
            filter-bank architecture), this is used directly.  Takes
            precedence over ``imfs`` when both are given.

        Exactly one of ``imfs`` / ``centroids`` must be provided.
        """
        if centroids is None:
            assert imfs is not None, "must provide imfs or centroids"
            B, K, T = imfs.shape
            if K < 2:
                return torch.tensor(0.0, device=imfs.device, dtype=imfs.dtype)
            flat = imfs.reshape(B * K, T)
            X = torch.fft.rfft(flat, dim=-1)
            psd = X.real ** 2 + X.imag ** 2
            nyquist = self.sample_rate / 2.0
            freqs = torch.linspace(
                0.0, nyquist, psd.shape[-1],
                device=imfs.device, dtype=imfs.dtype,
            )
            total_power = psd.sum(dim=-1).clamp(min=self.eps)
            centroids_hz = (psd * freqs).sum(dim=-1) / total_power
            centroids_norm = centroids_hz.reshape(B, K) / (nyquist + self.eps)
        else:
            B, K = centroids.shape
            if K < 2:
                return torch.tensor(
                    0.0, device=centroids.device, dtype=centroids.dtype
                )
            nyquist = self.sample_rate / 2.0
            centroids_norm = centroids / (nyquist + self.eps)

        # Squared hinge loss on consecutive pairs: c_{k+1} should be < c_k
        diff = centroids_norm[:, 1:] - centroids_norm[:, :-1]  # (B, K-1)
        violations = torch.relu(diff + self.normalized_margin)
        loss = (violations ** 2).mean()
        return loss


class CentroidSeparationLoss(nn.Module):
    """Combined ordering + repulsion + coverage loss on centroids.

    Diagnosis behind this loss: a pure ordering hinge (see
    :class:`FrequencyOrderingLoss`) is satisfied the moment centroids are
    merely descending by the margin — it does not push them apart.  On the
    canonical 3-component test, Phase 2.5b v2 produced centroids
    [35.7, 3.2, 3.1] Hz: ordering satisfied but IMFs 2 and 3 collapsed.

    This loss adds two extra terms that reshape the gradient landscape:

    1. **Ordering** (kept from ``FrequencyOrderingLoss``) — hinge on
       consecutive pairs::

           L_order = mean_k ReLU(ĉ_{k+1} - ĉ_k + m)²

    2. **Repulsion** — pairwise log-inverse-distance, pushing ALL
       centroids apart, not just adjacent ones::

           L_repel = -(1/(K(K-1))) · Σ_{i≠j} log(|ĉ_i − ĉ_j| + ε)

       (strong gradient when centroids are close, weak when far).

    3. **Coverage** — negative variance of centroids, which rewards
       spreading them across the Nyquist range::

           L_coverage = -Var(ĉ_k)

    All centroids ``ĉ`` are normalised to Nyquist so the loss is
    scale-invariant.  The three terms are summed with user-supplied
    weights ``w_order``, ``w_repel``, ``w_coverage``.

    The preferred centroid source is the SIGNAL-WEIGHTED centroid
    (``metadata["centroids_weighted"]``), because the filter-only
    centroid can drift to regions of empty signal.
    """

    def __init__(
        self,
        sample_rate: float = 1.0,
        normalized_margin: float = 0.02,
        w_order: float = 1.0,
        w_repel: float = 0.5,
        w_coverage: float = 0.3,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.normalized_margin = normalized_margin
        self.w_order = w_order
        self.w_repel = w_repel
        self.w_coverage = w_coverage
        self.eps = eps

    def forward(self, centroids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        centroids : (B, K) in Hz.

        Returns
        -------
        loss : scalar tensor.  Its sign is not bounded below (the
            coverage/repulsion terms are negative for spread centroids);
            what matters is that decreasing the total value corresponds
            to a better-spread, correctly-ordered decomposition.
        """
        B, K = centroids.shape
        if K < 2:
            return torch.tensor(0.0, device=centroids.device, dtype=centroids.dtype)

        nyquist = self.sample_rate / 2.0
        c = centroids / (nyquist + self.eps)  # normalised to ~[0, 1]

        # 1. Ordering (hinge, squared)
        diff = c[:, 1:] - c[:, :-1]                       # (B, K-1)
        order = (torch.relu(diff + self.normalized_margin) ** 2).mean()

        # 2. Pairwise repulsion — log(|c_i - c_j| + ε), summed over i != j
        # (B, K, K) pairwise distances
        d = (c.unsqueeze(-1) - c.unsqueeze(-2)).abs()       # (B, K, K)
        mask = ~torch.eye(K, device=c.device, dtype=torch.bool).unsqueeze(0)
        dists = d.masked_select(mask).reshape(B, K * (K - 1))
        # Minimising this term (negative log of distance) pushes centroids apart
        repel = -torch.log(dists + self.eps).mean()

        # 3. Coverage — negative variance (maximise spread)
        coverage = -c.var(dim=-1, unbiased=False).mean()

        return (
            self.w_order * order
            + self.w_repel * repel
            + self.w_coverage * coverage
        )


class SpectralConcentrationLoss(nn.Module):
    """Penalise spectral entropy of each IMF.

    A mono-component signal has energy concentrated near a single
    frequency (low entropy); a broad-band signal spreads energy over
    many frequencies (high entropy).  This complements the narrow-band
    loss: bandwidth measures deviation from the centroid, entropy
    measures the shape of the spectral distribution.

    L_concentration = (1/K) · Σ_k H(p_k) / log(N)

    where ``p_k(f) = |FFT(c_k)|² / Σ |FFT(c_k)|²`` is the normalised
    power spectrum and ``N`` is the number of frequency bins.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, imfs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        imfs : (B, K, T)

        Returns
        -------
        loss : scalar tensor in [0, 1]
        """
        B, K, T = imfs.shape
        flat = imfs.reshape(B * K, T)
        X = torch.fft.rfft(flat, dim=-1)
        psd = X.real ** 2 + X.imag ** 2  # (B*K, N)

        total = psd.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        probs = psd / total
        entropy = -(probs * torch.log(probs + self.eps)).sum(dim=-1)  # (B*K,)
        # Normalise by max entropy log(N)
        N = psd.shape[-1]
        max_entropy = math.log(float(N))
        return (entropy / max_entropy).mean()


# ---------------------------------------------------------------------------
# Filter-bank losses
# ---------------------------------------------------------------------------

class FilterBalanceLoss(nn.Module):
    """Prevent filter-collapse degeneracy.

    Without this loss, the softmax partition can collapse to a single
    "dominant" filter where one ``H_k(f) ≈ 1`` everywhere and the others
    are zero.  That solution trivially satisfies orthogonality and gets
    zero entropy contribution from the empty filters, so it looks good
    to the sharpness loss.

    We require each filter to cover at least a ``min_fraction`` share of
    the frequency axis (structural balance, independent of signal)::

        coverage_k = mean_f H_k(f)   ∈ [0, 1]   with   Σ_k coverage_k = 1

        L_balance = mean_k  ReLU(min_fraction - coverage_k)

    Default ``min_fraction = 1/(2K)`` lets the model allocate wider bands
    where the signal has more structure, but forbids collapse.
    """

    def __init__(self, min_fraction: float | None = None, eps: float = 1e-8) -> None:
        super().__init__()
        self.min_fraction = min_fraction
        self.eps = eps

    def forward(self, filters: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        filters : (B, K, n_freqs)

        Returns
        -------
        loss : scalar tensor ≥ 0
        """
        _, K, _ = filters.shape
        coverage = filters.mean(dim=-1)                 # (B, K), Σ_k = 1
        min_frac = self.min_fraction if self.min_fraction is not None \
            else 1.0 / (2 * K)
        return torch.relu(min_frac - coverage).mean()


class FilterSharpnessLoss(nn.Module):
    """Penalise broad / flat bandpass filters.

    Two modes:

    1. **Filter-only** (``signal_power`` is ``None``): measure the
       Shannon entropy of each filter viewed as a distribution over
       frequency bins.  Bounded in ``[0, 1]``, 0 = delta, 1 = uniform.

    2. **Signal-weighted** (``signal_power`` provided): measure the
       entropy of the *filtered signal spectrum* ``H_k(f)·|X(f)|²``.
       This is the correct loss for the filter-bank N-EMD — it rewards
       a filter for isolating a narrow band of **signal** energy, and
       does not penalise broad-but-silent frequency regions.

    The loss is normalised by ``log(n_freqs)`` so it lies in ``[0, 1]``.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        filters: torch.Tensor,
        signal_power: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        filters : (B, K, n_freqs) — non-negative, softmax-partitioned
        signal_power : (B, n_freqs) or None
            ``|X(f)|²`` of the input signal.  If provided, use the
            signal-weighted entropy (recommended for filter-bank N-EMD).

        Returns
        -------
        loss : scalar tensor in [0, 1]
        """
        if signal_power is None:
            weighted = filters
        else:
            # (B, K, n_freqs) * (B, 1, n_freqs) → (B, K, n_freqs)
            weighted = filters * signal_power.unsqueeze(1)

        total = weighted.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        probs = weighted / total
        entropy = -(probs * torch.log(probs + self.eps)).sum(dim=-1)  # (B, K)
        max_entropy = math.log(float(filters.shape[-1]))
        return (entropy / max_entropy).mean()


# ---------------------------------------------------------------------------
# Combined losses
# ---------------------------------------------------------------------------

class NEMDSiftingLoss(nn.Module):
    """[Legacy] 7-term physics loss for the sifting N-EMD (Phase 2 / 2.5).

    Kept for ablation and paper comparisons.  Same API as the previous
    ``NEMDLoss`` — nothing else has changed.

    L = λ_ortho·L_ortho + λ_narrow·L_narrow + λ_mono·L_mono
      + λ_residual·L_residual + λ_energy·L_energy
      + λ_order·L_order + λ_concentration·L_concentration
      [+ λ_task·L_task]

    Parameters
    ----------
    lambda_ortho : float
        Cross-correlation (cosine-similarity) penalty between IMFs.
    lambda_narrow : float
        Spectral-bandwidth penalty (mono-component).
    lambda_mono : float
        Monotonic-residual penalty (sign changes in dr/dt).
    lambda_residual : float
        Penalty on energy left in the residual — prevents the trivial
        all-zero-IMF solution.
    lambda_energy : float
        Parseval-type energy-conservation penalty — prevents IMFs from
        inflating or cancelling each other.
    lambda_order : float
        Frequency-ordering constraint — forces descending spectral
        centroids across IMF indices.  The key fix for mode mixing.
    lambda_concentration : float
        Spectral-entropy penalty — encourages each IMF to concentrate
        energy in a single spectral region.
    lambda_task : float
        Weighting for an optional downstream-task loss.
    sample_rate : float
        Sampling frequency (Hz).  Used by narrow-band and ordering losses.
    normalized_margin : float
        Margin between consecutive IMF centroids (fraction of Nyquist).
    """

    def __init__(
        self,
        lambda_ortho: float = 1.0,
        lambda_narrow: float = 0.1,
        lambda_mono: float = 0.5,
        lambda_residual: float = 0.1,
        lambda_energy: float = 0.1,
        lambda_order: float = 2.0,
        lambda_concentration: float = 0.1,
        lambda_task: float = 0.0,
        sample_rate: float = 1.0,
        normalized_margin: float = 0.02,
    ) -> None:
        super().__init__()
        self.lambda_ortho = lambda_ortho
        self.lambda_narrow = lambda_narrow
        self.lambda_mono = lambda_mono
        self.lambda_residual = lambda_residual
        self.lambda_energy = lambda_energy
        self.lambda_order = lambda_order
        self.lambda_concentration = lambda_concentration
        self.lambda_task = lambda_task

        self.ortho_loss = OrthogonalityLoss()
        self.narrow_loss = NarrowBandLoss(fs=sample_rate)
        self.mono_loss = MonotonicResidualLoss()
        self.residual_loss = ResidualEnergyLoss()
        self.energy_loss = EnergyConservationLoss()
        self.order_loss = FrequencyOrderingLoss(
            sample_rate=sample_rate,
            normalized_margin=normalized_margin,
        )
        self.concentration_loss = SpectralConcentrationLoss()

    def forward(
        self,
        imfs: torch.Tensor,
        residual: torch.Tensor,
        original: torch.Tensor,
        task_loss: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the combined loss.

        Parameters
        ----------
        imfs : (B, K, T)
        residual : (B, T)
        original : (B, T)
        task_loss : scalar tensor or None

        Returns
        -------
        total_loss : scalar tensor
        components : dict mapping loss name → float value (for logging)
        """
        l_ortho = self.ortho_loss(imfs)
        l_narrow = self.narrow_loss(imfs)
        l_mono = self.mono_loss(residual)
        l_residual = self.residual_loss(residual, original)
        l_energy = self.energy_loss(imfs, residual, original)
        l_order = self.order_loss(imfs)
        l_concentration = self.concentration_loss(imfs)

        total = (
            self.lambda_ortho * l_ortho
            + self.lambda_narrow * l_narrow
            + self.lambda_mono * l_mono
            + self.lambda_residual * l_residual
            + self.lambda_energy * l_energy
            + self.lambda_order * l_order
            + self.lambda_concentration * l_concentration
        )

        components = {
            "ortho": l_ortho.item(),
            "narrow": l_narrow.item(),
            "mono": l_mono.item(),
            "residual": l_residual.item(),
            "energy": l_energy.item(),
            "order": l_order.item(),
            "concentration": l_concentration.item(),
        }

        if task_loss is not None and self.lambda_task > 0:
            total = total + self.lambda_task * task_loss
            components["task"] = task_loss.item()

        components["total"] = total.item()
        return total, components


class NEMDLoss(nn.Module):
    """Combined loss for the filter-bank N-EMD (Phase 2.5b onwards).

    The architecture (softmax partition of unity over K bandpass filters)
    provides reconstruction, energy boundedness, and bandpass structure
    for free.  The loss has four terms::

        L = λ_sharp·L_sharp
          + λ_order·L_order
          + λ_ortho·L_ortho
          + λ_balance·L_balance     ← prevents filter-collapse degeneracy
          [+ λ_task·L_task]

    ``L_balance`` is the structural counterweight to ``L_sharp``: without
    it, the model discovers that making ``K-1`` filters zero gives zero
    entropy contribution and thus a low sharp loss.  With it, each filter
    must cover at least a minimum fraction of the frequency axis.
    """

    def __init__(
        self,
        lambda_sharp: float = 1.0,
        lambda_order: float = 1.0,
        lambda_ortho: float = 0.1,
        lambda_balance: float = 5.0,
        lambda_task: float = 0.0,
        sample_rate: float = 1.0,
        normalized_margin: float = 0.02,
        balance_min_fraction: float | None = None,
        # CentroidSeparationLoss sub-weights (applied INSIDE the
        # combined separation term — on top of ``lambda_order``):
        sep_w_order: float = 1.0,
        sep_w_repel: float = 0.5,
        sep_w_coverage: float = 0.3,
    ) -> None:
        super().__init__()
        self.lambda_sharp = lambda_sharp
        self.lambda_order = lambda_order  # outer weight on the combined separation loss
        self.lambda_ortho = lambda_ortho
        self.lambda_balance = lambda_balance
        self.lambda_task = lambda_task

        self.sharp_loss = FilterSharpnessLoss()
        # Replaces the old pure-hinge FrequencyOrderingLoss.  Three sub-terms
        # (ordering, repulsion, coverage) weighted by sep_w_*.
        self.order_loss = CentroidSeparationLoss(
            sample_rate=sample_rate,
            normalized_margin=normalized_margin,
            w_order=sep_w_order,
            w_repel=sep_w_repel,
            w_coverage=sep_w_coverage,
        )
        self.ortho_loss = OrthogonalityLoss()
        self.balance_loss = FilterBalanceLoss(min_fraction=balance_min_fraction)

    def forward(
        self,
        imfs: torch.Tensor,
        metadata: dict,
        task_loss: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the combined filter-bank loss.

        Parameters
        ----------
        imfs : (B, K, T)
        metadata : dict produced by ``NEMD.forward``
            Must contain ``filters`` of shape ``(B, K, n_freqs)`` and
            ``centroids`` of shape ``(B, K)`` (in Hz).
        task_loss : scalar tensor or None

        Returns
        -------
        total_loss : scalar tensor
        components : dict mapping loss name → float value
        """
        filters = metadata["filters"]
        signal_power = metadata.get("signal_power", None)
        # Order on signal-weighted centroids when available (filter-only
        # centroids don't account for where the signal energy actually is).
        centroids_for_order = metadata.get(
            "centroids_weighted", metadata["centroids"]
        )

        l_sharp = self.sharp_loss(filters, signal_power=signal_power)
        # CentroidSeparationLoss takes centroids directly (positional arg)
        l_order = self.order_loss(centroids_for_order)
        l_ortho = self.ortho_loss(imfs)
        l_balance = self.balance_loss(filters)

        total = (
            self.lambda_sharp * l_sharp
            + self.lambda_order * l_order
            + self.lambda_ortho * l_ortho
            + self.lambda_balance * l_balance
        )

        components = {
            "sharp": l_sharp.item(),
            "order": l_order.item(),
            "ortho": l_ortho.item(),
            "balance": l_balance.item(),
        }

        if task_loss is not None and self.lambda_task > 0:
            total = total + self.lambda_task * task_loss
            components["task"] = task_loss.item()

        components["total"] = total.item()
        return total, components
