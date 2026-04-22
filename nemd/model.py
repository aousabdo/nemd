"""N-EMD architectures.

Primary (Phase 2.5b and later)
------------------------------
``NEMD`` — Neural Adaptive Filter Bank (NAFB).  Decomposes signals via K
learned, signal-adaptive bandpass filters that form a softmax partition of
unity in the frequency domain.  Reconstruction is exact by construction and
energy cannot inflate — the architecture makes broad mixed decompositions
impossible, eliminating the failure mode of the Phase 2/2.5 sifting approach.

Legacy (Phase 2 / 2.5)
----------------------
``NEMDSifting`` — iterative FiLM-conditioned U-Net that emulates classical
sifting.  Kept for ablation/comparison in the paper.  New experiments should
use ``NEMD``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SiftNetConfig:
    """Configuration for the SiftNet U-Net architecture."""

    num_levels: int = 3
    channels: list[int] = field(default_factory=lambda: [32, 64, 128])
    bottleneck_channels: int = 256
    kernel_sizes: list[int] = field(default_factory=lambda: [7, 5, 3])
    max_imfs: int = 6
    scale_embed_dim: int = 64
    use_init_filter: bool = True
    init_filter_taps: int = 31

    def __post_init__(self) -> None:
        assert len(self.channels) == self.num_levels
        assert len(self.kernel_sizes) == self.num_levels


# ---------------------------------------------------------------------------
# FiLM (Feature-wise Linear Modulation)
# ---------------------------------------------------------------------------

class FiLMGenerator(nn.Module):
    """Generate (gamma, beta) modulation parameters from a scale embedding.

    Given the IMF index embedding, produces per-channel affine parameters
    for each U-Net level (encoder levels + bottleneck + decoder levels).
    """

    def __init__(self, config: SiftNetConfig) -> None:
        super().__init__()
        d = config.scale_embed_dim
        # Number of FiLM sites: encoder (num_levels) + bottleneck (1) + decoder (num_levels)
        self.num_sites = 2 * config.num_levels + 1
        # Channel dims at each site
        site_channels = (
            list(config.channels)                   # encoder levels
            + [config.bottleneck_channels]           # bottleneck
            + list(reversed(config.channels))        # decoder levels
        )
        # One (gamma, beta) pair per site; gamma and beta each have C dimensions
        self.projections = nn.ModuleList()
        for c in site_channels:
            self.projections.append(nn.Sequential(
                nn.Linear(d, d),
                nn.ReLU(inplace=True),
                nn.Linear(d, 2 * c),
            ))

    def forward(self, embed: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return list of (gamma, beta) pairs, one per FiLM site.

        Parameters
        ----------
        embed : (B, d) scale embedding

        Returns
        -------
        films : list of (gamma, beta) each of shape (B, C, 1)
        """
        films = []
        for proj in self.projections:
            gb = proj(embed)  # (B, 2*C)
            gamma, beta = gb.chunk(2, dim=-1)
            # Reshape for broadcasting with (B, C, T) feature maps
            films.append((gamma.unsqueeze(-1), beta.unsqueeze(-1)))
        return films


# ---------------------------------------------------------------------------
# Encoder / Decoder blocks
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(
        self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (pooled_output, skip_connection)."""
        h = F.relu(self.bn1(self.conv1(x)))
        # FiLM modulation after first conv block
        h = gamma * h + beta
        h = F.relu(self.bn2(self.conv2(h)))
        return self.pool(h), h  # pooled, skip


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, kernel_size: int) -> None:
        super().__init__()
        # in_ch = upsampled channels + skip channels
        self.conv1 = nn.Conv1d(in_ch, mid_ch, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(mid_ch)
        self.conv2 = nn.Conv1d(mid_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_ch)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        # Upsample to match skip size
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        h = F.relu(self.bn1(self.conv1(x)))
        h = gamma * h + beta
        h = F.relu(self.bn2(self.conv2(h)))
        return h


# ---------------------------------------------------------------------------
# Init filter bank — bandpass initialisation to break the frequency-assignment
# symmetry before training begins
# ---------------------------------------------------------------------------

def _windowed_sinc_bandpass(n_taps: int, f_low_norm: float, f_high_norm: float) -> np.ndarray:
    """Windowed-sinc bandpass FIR filter with normalised cutoffs in [0, 1].

    ``f_norm = 1`` corresponds to Nyquist.  Uses a Hamming window.
    """
    n = np.arange(n_taps) - (n_taps - 1) / 2.0
    lp_high = np.sinc(f_high_norm * n) * f_high_norm
    lp_low = np.sinc(f_low_norm * n) * f_low_norm
    bp = lp_high - lp_low
    window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n_taps) / (n_taps - 1))
    return (bp * window).astype(np.float32)


class InitFilterBank(nn.Module):
    """Bank of K learnable 1D filters, initialised as bandpass filters.

    Filter ``k`` is initialised to pass a frequency band centred at
    ``(K - k) / (K + 1)`` (normalised to Nyquist).  This gives IMF 0 a
    high-frequency prior, IMF K-1 a low-frequency prior.  Weights are
    fully learnable after initialisation.
    """

    def __init__(self, max_imfs: int, n_taps: int = 31) -> None:
        super().__init__()
        self.max_imfs = max_imfs
        self.n_taps = n_taps
        pad = n_taps // 2
        self.filters = nn.ModuleList([
            nn.Conv1d(1, 1, kernel_size=n_taps, padding=pad, bias=False)
            for _ in range(max_imfs)
        ])
        self._init_bandpass_weights()

    def _init_bandpass_weights(self) -> None:
        K = self.max_imfs
        bandwidth_norm = 1.0 / (K + 1)
        for k, conv in enumerate(self.filters):
            f_center = (K - k) / (K + 1)
            f_low = max(0.01, f_center - bandwidth_norm)
            f_high = min(0.99, f_center + bandwidth_norm)
            taps = _windowed_sinc_bandpass(self.n_taps, f_low, f_high)
            with torch.no_grad():
                conv.weight.copy_(torch.from_numpy(taps).reshape(1, 1, -1))

    def forward(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """x : (B, 1, T) → (B, 1, T)."""
        return self.filters[k](x)


# ---------------------------------------------------------------------------
# SiftNet: FiLM-conditioned 1D U-Net
# ---------------------------------------------------------------------------

class SiftNet(nn.Module):
    """Neural sifting operator — extracts a single IMF from a residual signal.

    A 1D U-Net with FiLM conditioning on the IMF scale index *k*, so the
    same network can extract IMFs at different frequency scales.
    """

    def __init__(self, config: SiftNetConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = SiftNetConfig()
        self.config = config
        L = config.num_levels
        ch = config.channels
        ks = config.kernel_sizes
        bneck = config.bottleneck_channels

        # Scale embedding: learned lookup table for IMF index
        self.scale_embed = nn.Embedding(config.max_imfs, config.scale_embed_dim)

        # FiLM generator
        self.film_gen = FiLMGenerator(config)

        # Optional init filter bank — bandpass prior per IMF index.
        # Concatenated with the raw signal as an extra input channel.
        if config.use_init_filter:
            self.init_filter_bank = InitFilterBank(
                max_imfs=config.max_imfs,
                n_taps=config.init_filter_taps,
            )
            input_channels = 2
        else:
            self.init_filter_bank = None
            input_channels = 1

        # Encoder
        self.encoders = nn.ModuleList()
        in_ch = input_channels
        for i in range(L):
            self.encoders.append(EncoderBlock(in_ch, ch[i], ks[i]))
            in_ch = ch[i]

        # Bottleneck
        self.bottleneck_conv1 = nn.Conv1d(ch[-1], bneck, 3, padding=1)
        self.bottleneck_bn1 = nn.BatchNorm1d(bneck)
        self.bottleneck_conv2 = nn.Conv1d(bneck, ch[-1], 3, padding=1)
        self.bottleneck_bn2 = nn.BatchNorm1d(ch[-1])

        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(L - 1, -1, -1):
            dec_in = ch[i] + ch[i]  # upsampled + skip
            mid = ch[i]
            out = ch[i - 1] if i > 0 else ch[0]
            self.decoders.append(DecoderBlock(dec_in, mid, out, ks[i]))

        # Final projection to 1 channel
        self.head = nn.Conv1d(ch[0], 1, kernel_size=1)

    def forward(self, residual: torch.Tensor, k: int) -> torch.Tensor:
        """Extract one IMF from the current residual.

        Parameters
        ----------
        residual : (B, T) current residual signal
        k : int, IMF index (0-based)

        Returns
        -------
        imf : (B, T) extracted IMF
        """
        B, T = residual.shape
        L = self.config.num_levels

        # Pad T to be divisible by 2^L
        factor = 2 ** L
        pad_total = (factor - T % factor) % factor
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        x = F.pad(residual.unsqueeze(1), (pad_left, pad_right), mode="reflect")
        # x: (B, 1, T_padded)

        # Scale embedding + FiLM parameters
        k_idx = torch.tensor([k], device=residual.device).expand(B)
        embed = self.scale_embed(k_idx)  # (B, d)
        films = self.film_gen(embed)
        # films[0..L-1] = encoder, films[L] = bottleneck, films[L+1..2L] = decoder

        # Optional bandpass prior — concatenate filtered signal as extra channel
        if self.init_filter_bank is not None:
            filtered = self.init_filter_bank(x, k)  # (B, 1, T_padded)
            h = torch.cat([x, filtered], dim=1)     # (B, 2, T_padded)
        else:
            h = x

        # Encoder
        skips = []
        for i, enc in enumerate(self.encoders):
            h, skip = enc(h, *films[i])
            skips.append(skip)

        # Bottleneck
        g_bn, b_bn = films[L]
        h = F.relu(self.bottleneck_bn1(self.bottleneck_conv1(h)))
        h = g_bn * h + b_bn
        h = F.relu(self.bottleneck_bn2(self.bottleneck_conv2(h)))

        # Decoder (reverse order of skips)
        for i, dec in enumerate(self.decoders):
            skip = skips[L - 1 - i]
            h = dec(h, skip, *films[L + 1 + i])

        # Head
        h = self.head(h)  # (B, 1, T_padded)

        # Remove padding
        if pad_total > 0:
            h = h[:, :, pad_left: pad_left + T]
        return h.squeeze(1)  # (B, T)


# ---------------------------------------------------------------------------
# Legacy: iterative sifting N-EMD (Phase 2 / 2.5).  Kept for ablation.
# ---------------------------------------------------------------------------

class NEMDSifting(nn.Module):
    """[Legacy] Iterative sifting N-EMD via a FiLM-conditioned U-Net.

    Kept for comparison.  New experiments should use :class:`NEMD` (the
    parallel filter bank).
    """

    def __init__(
        self,
        max_imfs: int = 6,
        sift_config: SiftNetConfig | None = None,
    ) -> None:
        super().__init__()
        if sift_config is None:
            sift_config = SiftNetConfig(max_imfs=max_imfs)
        else:
            sift_config.max_imfs = max_imfs
        self.max_imfs = max_imfs
        self.sift_net = SiftNet(sift_config)

    def forward(
        self,
        x: torch.Tensor,
        num_imfs: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompose ``x`` into IMFs + residual (legacy API, no metadata)."""
        K = num_imfs if num_imfs is not None else self.max_imfs
        residual = x
        imfs = []
        for k in range(K):
            imf = self.sift_net(residual, k)
            imfs.append(imf)
            residual = residual - imf
        return torch.stack(imfs, dim=1), residual


# ---------------------------------------------------------------------------
# Primary: Neural Adaptive Filter Bank (NAFB)
# ---------------------------------------------------------------------------

class ResBlock1d(nn.Module):
    """Simple 1D residual block: two Conv1d → BatchNorm → GELU with a skip."""

    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.gelu(h + residual)


class SignalAnalyzer(nn.Module):
    """Spectrum encoder: (magnitude, phase) → K filter log-responses.

    Operates in the frequency domain.  The output ``filter_logits`` of shape
    ``(B, K, n_freqs)`` is consumed by :class:`FrequencyPartition` to produce
    a softmax partition of unity across the K filters.
    """

    def __init__(
        self,
        num_imfs: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.num_imfs = num_imfs
        self.input_conv = nn.Conv1d(2, hidden_dim, kernel_size=7, padding=3)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        self.blocks = nn.ModuleList([
            ResBlock1d(hidden_dim, kernel_size=kernel_size)
            for _ in range(num_layers)
        ])
        self.output = nn.Conv1d(hidden_dim, num_imfs, kernel_size=1)

    def forward(self, spectrum_input: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        spectrum_input : (B, 2, n_freqs)
            Channel 0 = magnitude of rFFT, channel 1 = phase.

        Returns
        -------
        filter_logits : (B, K, n_freqs)
        """
        h = F.gelu(self.input_bn(self.input_conv(spectrum_input)))
        for block in self.blocks:
            h = block(h)
        return self.output(h)


class FrequencyPartition(nn.Module):
    """Softmax over the K dimension → partition-of-unity bandpass filters.

    Given filter log-responses ``s_k(f)`` and temperature τ::

        H_k(f) = softmax_K(s_k(f) / τ)

    so ``Σ_k H_k(f) = 1`` for every frequency bin.  Applying these filters
    to the input spectrum and inverting the FFT gives IMFs whose sum equals
    the input signal exactly.
    """

    def forward(
        self,
        X_complex: torch.Tensor,
        filter_logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        X_complex : (B, n_freqs) complex — rFFT of the input signal
        filter_logits : (B, K, n_freqs) real
        temperature : float

        Returns
        -------
        imf_spectra : (B, K, n_freqs) complex
        filters : (B, K, n_freqs) real, non-negative, Σ_K = 1
        """
        filters = F.softmax(filter_logits / max(temperature, 1e-6), dim=1)
        # Broadcasting: (B, 1, n_freqs) * (B, K, n_freqs) → (B, K, n_freqs)
        imf_spectra = filters * X_complex.unsqueeze(1)
        return imf_spectra, filters


class NEMD(nn.Module):
    """Neural Empirical Mode Decomposition via Adaptive Filter Bank.

    Decomposes a signal ``x`` of length ``T`` into K IMFs by:
      1. rFFT → complex spectrum
      2. Stack (magnitude, phase) as a 2-channel input for the analyzer
      3. Analyzer (1D CNN) predicts K log-filter responses
      4. Softmax over K → partition-of-unity bandpass filters
      5. Multiply each filter with the spectrum, inverse rFFT → IMFs

    Architectural guarantees:
      - ``sum(imfs) == x`` exactly (partition of unity)
      - Each IMF is bandpass (filter is non-negative)
      - Energy cannot inflate (``Σ‖c_k‖² ≤ ‖x‖²`` by Cauchy-Schwarz)

    Parameters
    ----------
    num_imfs : int
        Number of IMFs (fixed, not variable).
    hidden_dim, num_layers, kernel_size : int
        Analyzer network capacity.
    sample_rate : float
        Used only for centroid computation in metadata.
    temperature : float
        Softmax temperature.  Lower → sharper filters, higher → softer overlap.
        Training loop can anneal this; at inference use a fixed value.
    """

    def __init__(
        self,
        num_imfs: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 4,
        kernel_size: int = 5,
        sample_rate: float = 1000.0,
        temperature: float = 1.0,
        use_phase: bool = True,
    ) -> None:
        super().__init__()
        self.num_imfs = num_imfs
        self.sample_rate = sample_rate
        self.temperature = temperature
        self.use_phase = use_phase
        self.analyzer = SignalAnalyzer(
            num_imfs=num_imfs,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
        )
        self.partition = FrequencyPartition()

    def set_temperature(self, temperature: float) -> None:
        """Set the softmax temperature (called by the training loop)."""
        self.temperature = float(temperature)

    def forward(
        self,
        x: torch.Tensor,
        num_imfs: int | None = None,
        temperature: float | None = None,
        sort_by_centroid: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Decompose a batch of signals.

        Parameters
        ----------
        x : (B, T)
        num_imfs : int or None
            If given, must equal ``self.num_imfs`` (the filter bank has a
            fixed K).  Present for API compatibility with ``NEMDSifting``.
        temperature : float or None
            Override the module's current temperature for this forward pass.

        Returns
        -------
        imfs : (B, K, T)
        residual : (B, T) — always zeros (all energy is in the IMFs)
        metadata : dict with keys ``filters``, ``centroids``,
            ``filter_logits``, ``temperature``
        """
        if num_imfs is not None and num_imfs != self.num_imfs:
            raise ValueError(
                f"num_imfs={num_imfs} does not match fixed K={self.num_imfs}"
            )
        tau = float(temperature) if temperature is not None else self.temperature

        B, T = x.shape

        # Spectrum: (B, n_freqs) complex
        X = torch.fft.rfft(x, dim=-1)
        n_freqs = X.shape[-1]

        # Build 2-channel analyzer input. Use torch.atan2 for the phase
        # rather than X.angle() so the graph runs on Apple-Silicon MPS
        # (aten::angle is CPU-only as of torch 2.5).
        mag = X.abs()
        if self.use_phase:
            phase = torch.atan2(X.imag, X.real)
        else:
            # Magnitude-only ablation: zero phase channel, preserve shape.
            phase = torch.zeros_like(mag)
        spectrum_input = torch.stack([mag, phase], dim=1)  # (B, 2, n_freqs)

        # Predict filter log-responses
        filter_logits = self.analyzer(spectrum_input)  # (B, K, n_freqs)

        # Softmax partition → filters, then apply
        imf_spectra, filters = self.partition(X, filter_logits, temperature=tau)

        # Back to time domain, exactly length T
        imfs = torch.fft.irfft(imf_spectra, n=T, dim=-1)  # (B, K, T)

        # Residual is zero by construction
        residual = torch.zeros(B, T, device=x.device, dtype=x.dtype)

        # Metadata: filter centroids in Hz
        freqs = torch.linspace(
            0.0, self.sample_rate / 2.0, n_freqs,
            device=x.device, dtype=filters.dtype,
        )
        centroids = (filters * freqs).sum(-1) / (filters.sum(-1) + 1e-8)  # (B, K)

        # Signal power |X(f)|² — needed by signal-weighted sharpness losses.
        signal_power = X.real ** 2 + X.imag ** 2  # (B, n_freqs)

        # Signal-weighted centroid — where IMF k's energy actually lives.
        # This is the right quantity to order (not the filter-only centroid).
        weighted = filters * signal_power.unsqueeze(1)  # (B, K, n_freqs)
        weighted_sum = weighted.sum(-1).clamp(min=1e-8)
        centroids_weighted = (weighted * freqs).sum(-1) / weighted_sum  # (B, K)

        metadata = {
            "filters": filters,
            "centroids": centroids,                      # filter-only (shape)
            "centroids_weighted": centroids_weighted,    # signal-weighted
            "filter_logits": filter_logits,
            "signal_power": signal_power,
            "temperature": tau,
        }

        # Optional: sort IMFs (and metadata) by descending signal-weighted
        # centroid.  Pure permutation — preserves reconstruction, ortho,
        # partition-of-unity, and energy boundedness exactly.  Recommended
        # at inference time; disabled during training so the ordering loss
        # sees the raw model output.
        if sort_by_centroid:
            # (B, K) sort descending
            order = torch.argsort(centroids_weighted, dim=-1, descending=True)
            imfs = torch.gather(
                imfs, 1, order.unsqueeze(-1).expand(-1, -1, imfs.shape[-1])
            )
            filters = torch.gather(
                filters, 1, order.unsqueeze(-1).expand(-1, -1, filters.shape[-1])
            )
            filter_logits = torch.gather(
                filter_logits, 1,
                order.unsqueeze(-1).expand(-1, -1, filter_logits.shape[-1]),
            )
            centroids = torch.gather(centroids, 1, order)
            centroids_weighted = torch.gather(centroids_weighted, 1, order)
            metadata.update({
                "filters": filters,
                "centroids": centroids,
                "centroids_weighted": centroids_weighted,
                "filter_logits": filter_logits,
                "sort_order": order,
            })

        return imfs, residual, metadata
