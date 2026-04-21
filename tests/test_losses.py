"""Tests for nemd.losses and nemd.layers."""

import numpy as np
import pytest
import torch

from nemd.layers import (
    envelope_mean,
    hilbert_transform,
    instantaneous_amplitude,
    instantaneous_frequency,
    instantaneous_phase,
    spectral_bandwidth,
    upper_lower_envelopes,
)
from nemd.losses import (
    CentroidSeparationLoss,
    EnergyConservationLoss,
    FilterBalanceLoss,
    FilterSharpnessLoss,
    FrequencyOrderingLoss,
    MonotonicResidualLoss,
    NarrowBandLoss,
    NEMDLoss,
    NEMDSiftingLoss,
    OrthogonalityLoss,
    ResidualEnergyLoss,
    SpectralConcentrationLoss,
)


# ===================================================================
# Tests for differentiable layers (Phase 1, kept and expanded)
# ===================================================================


class TestHilbertTransform:
    def test_analytic_signal_shape(self):
        x = torch.randn(256)
        z = hilbert_transform(x)
        assert z.shape == (256,)
        assert z.is_complex()

    def test_analytic_signal_batched(self):
        x = torch.randn(4, 512)
        z = hilbert_transform(x)
        assert z.shape == (4, 512)

    def test_real_part_preserved(self):
        x = torch.randn(256)
        z = hilbert_transform(x)
        torch.testing.assert_close(z.real, x, atol=1e-5, rtol=1e-5)

    def test_pure_cosine_envelope(self):
        t = torch.linspace(0, 1, 1024)
        x = torch.cos(2 * np.pi * 10 * t)
        env = instantaneous_amplitude(x)
        mid = env[50:-50]
        torch.testing.assert_close(
            mid, torch.ones_like(mid), atol=0.05, rtol=0.05
        )

    def test_gradient_flows(self):
        x = torch.randn(128, requires_grad=True)
        z = hilbert_transform(x)
        loss = z.abs().sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestInstantaneousFrequency:
    def test_pure_tone_frequency(self):
        fs = 1000.0
        N = 2048
        t = torch.arange(N, dtype=torch.float32) / fs
        x = torch.cos(2 * np.pi * 20.0 * t)
        freq = instantaneous_frequency(x, fs=fs)
        mid = freq[100:-100]
        assert torch.allclose(mid, torch.full_like(mid, 20.0), atol=1.0)

    def test_if_shape(self):
        x = torch.randn(64, 256)
        freq = instantaneous_frequency(x)
        assert freq.shape == (64, 255)

    def test_gradient_flows(self):
        x = torch.randn(128, requires_grad=True)
        freq = instantaneous_frequency(x)
        freq.sum().backward()
        assert x.grad is not None


class TestEnvelopes:
    def test_envelope_mean_shape(self):
        x = torch.randn(512)
        m = envelope_mean(x, window_size=31)
        assert m.shape == (512,)

    def test_envelope_mean_batched(self):
        x = torch.randn(8, 256)
        m = envelope_mean(x, window_size=21)
        assert m.shape == (8, 256)

    def test_envelope_mean_smoothing(self):
        t = torch.linspace(0, 1, 512)
        x = torch.sin(2 * np.pi * 20 * t) + 0.5 * torch.sin(2 * np.pi * 3 * t)
        m = envelope_mean(x, window_size=51)
        tv_x = torch.diff(x).abs().sum()
        tv_m = torch.diff(m).abs().sum()
        assert tv_m < tv_x

    def test_upper_lower_envelopes_shape(self):
        x = torch.randn(256)
        upper, lower = upper_lower_envelopes(x, window_size=31)
        assert upper.shape == (256,)
        assert lower.shape == (256,)

    def test_upper_above_lower(self):
        t = torch.linspace(0, 1, 512)
        x = torch.sin(2 * np.pi * 10 * t)
        upper, lower = upper_lower_envelopes(x, window_size=31)
        assert torch.all(upper >= lower - 1e-6)

    def test_gradient_flows(self):
        x = torch.randn(128, requires_grad=True)
        m = envelope_mean(x, window_size=15)
        m.sum().backward()
        assert x.grad is not None


class TestSpectralBandwidth:
    def test_pure_tone_narrow(self):
        N = 1024
        fs = 1000.0
        t = torch.arange(N, dtype=torch.float32) / fs
        x = torch.sin(2 * np.pi * 50 * t)
        bw = spectral_bandwidth(x, fs=fs)
        assert bw.item() < 10.0  # narrow relative to Nyquist (500 Hz)

    def test_noise_wider(self):
        N = 2048
        fs = 1000.0
        t = torch.arange(N, dtype=torch.float32) / fs
        tone = torch.sin(2 * np.pi * 50 * t)
        noise = torch.randn(N)
        bw_tone = spectral_bandwidth(tone, fs=fs)
        bw_noise = spectral_bandwidth(noise, fs=fs)
        assert bw_noise > bw_tone

    def test_batched_shape(self):
        x = torch.randn(4, 256)
        bw = spectral_bandwidth(x)
        assert bw.shape == (4,)

    def test_gradient_flows(self):
        x = torch.randn(128, requires_grad=True)
        bw = spectral_bandwidth(x)
        bw.backward()
        assert x.grad is not None


# ===================================================================
# Tests for loss functions (Phase 2)
# ===================================================================


class TestOrthogonalityLoss:
    def test_orthogonal_imfs_low_loss(self):
        """Orthogonal IMFs should produce near-zero loss."""
        B, K, T = 2, 3, 512
        t = torch.linspace(0, 2 * np.pi, T)
        # Three approximately orthogonal sinusoids
        imf1 = torch.sin(t).unsqueeze(0).expand(B, -1)
        imf2 = torch.sin(3 * t).unsqueeze(0).expand(B, -1)
        imf3 = torch.sin(7 * t).unsqueeze(0).expand(B, -1)
        imfs = torch.stack([imf1, imf2, imf3], dim=1)

        loss_fn = OrthogonalityLoss()
        loss = loss_fn(imfs)
        assert loss.item() < 0.01

    def test_identical_imfs_high_loss(self):
        """Identical IMFs should produce high loss."""
        B, K, T = 2, 3, 256
        x = torch.randn(B, T)
        imfs = x.unsqueeze(1).expand(B, K, T)

        loss_fn = OrthogonalityLoss()
        loss = loss_fn(imfs)
        assert loss.item() > 0.1

    def test_single_imf_zero_loss(self):
        imfs = torch.randn(2, 1, 256)
        loss_fn = OrthogonalityLoss()
        loss = loss_fn(imfs)
        assert loss.item() == 0.0

    def test_gradient_flows(self):
        imfs = torch.randn(2, 3, 256, requires_grad=True)
        loss_fn = OrthogonalityLoss()
        loss = loss_fn(imfs)
        loss.backward()
        assert imfs.grad is not None


class TestNarrowBandLoss:
    def test_pure_tone_low_loss(self):
        """Pure tones should have lower narrow-band loss than noise."""
        B, K = 2, 2
        T = 1024
        fs = 1000.0
        t = torch.arange(T, dtype=torch.float32) / fs
        imf1 = torch.sin(2 * np.pi * 20 * t).unsqueeze(0).expand(B, -1)
        imf2 = torch.sin(2 * np.pi * 50 * t).unsqueeze(0).expand(B, -1)
        imfs = torch.stack([imf1, imf2], dim=1)

        loss_fn = NarrowBandLoss(fs=fs)
        loss = loss_fn(imfs)
        assert loss.item() < 0.5  # reasonable for CV^2 of pure tones

    def test_noise_higher_loss(self):
        """Broadband noise should have higher narrow-band loss than tones on average."""
        B, T = 8, 2048  # larger batch + longer signal for statistical stability
        fs = 1000.0
        t = torch.arange(T, dtype=torch.float32) / fs

        tones = torch.stack([
            torch.sin(2 * np.pi * 50 * t).unsqueeze(0).expand(B, -1),
        ], dim=1)
        torch.manual_seed(0)
        noise = torch.randn(B, 1, T)

        loss_fn = NarrowBandLoss(fs=fs)
        loss_tone = loss_fn(tones)
        loss_noise = loss_fn(noise)
        # Noise IF is much more variable than a pure tone
        assert loss_noise > loss_tone

    def test_gradient_flows(self):
        imfs = torch.randn(2, 2, 256, requires_grad=True)
        loss_fn = NarrowBandLoss()
        loss = loss_fn(imfs)
        loss.backward()
        assert imfs.grad is not None


class TestMonotonicResidualLoss:
    def test_monotonic_signal_low_loss(self):
        """A perfectly monotonic signal should have ~0 loss."""
        B = 2
        residual = torch.linspace(0, 1, 256).unsqueeze(0).expand(B, -1)
        loss_fn = MonotonicResidualLoss()
        loss = loss_fn(residual)
        assert loss.item() < 1e-6

    def test_oscillating_signal_high_loss(self):
        """An oscillating signal should have much higher mono loss than monotonic."""
        B = 2
        t = torch.linspace(0, 4 * np.pi, 256)
        residual_osc = torch.sin(t).unsqueeze(0).expand(B, -1)
        residual_mono = torch.linspace(0, 1, 256).unsqueeze(0).expand(B, -1)
        loss_fn = MonotonicResidualLoss()
        loss_osc = loss_fn(residual_osc)
        loss_mono = loss_fn(residual_mono)
        assert loss_osc > loss_mono
        assert loss_osc.item() > 0.001  # detectable sign changes

    def test_gradient_flows(self):
        residual = torch.randn(2, 256, requires_grad=True)
        loss_fn = MonotonicResidualLoss()
        loss = loss_fn(residual)
        loss.backward()
        assert residual.grad is not None


class TestResidualEnergyLoss:
    def test_zero_residual_zero_loss(self):
        residual = torch.zeros(2, 256)
        original = torch.randn(2, 256)
        loss_fn = ResidualEnergyLoss()
        loss = loss_fn(residual, original)
        assert loss.item() < 1e-6

    def test_full_residual_high_loss(self):
        """If residual == original, loss should be ~1.0."""
        original = torch.randn(2, 256)
        loss_fn = ResidualEnergyLoss()
        loss = loss_fn(original, original)
        assert abs(loss.item() - 1.0) < 0.01

    def test_gradient_flows(self):
        residual = torch.randn(2, 256, requires_grad=True)
        original = torch.randn(2, 256)
        loss_fn = ResidualEnergyLoss()
        loss = loss_fn(residual, original)
        loss.backward()
        assert residual.grad is not None


class TestEnergyConservationLoss:
    def test_perfect_conservation(self):
        """Orthogonal decomposition should have ~0 conservation loss."""
        B, T = 2, 512
        t = torch.linspace(0, 2 * np.pi, T)
        c1 = torch.sin(t).unsqueeze(0).expand(B, -1)
        c2 = torch.sin(3 * t).unsqueeze(0).expand(B, -1)
        imfs = torch.stack([c1, c2], dim=1)
        original = c1 + c2
        residual = torch.zeros(B, T)
        loss_fn = EnergyConservationLoss()
        loss = loss_fn(imfs, residual, original)
        assert loss.item() < 0.05  # approximately orthogonal

    def test_energy_inflation_detected(self):
        """If IMFs have more energy than original, loss should be high."""
        B, T = 2, 256
        original = torch.randn(B, T)
        imfs = original.unsqueeze(1).expand(B, 3, T) * 2  # 3 copies, scaled up
        residual = original - imfs.sum(dim=1)
        loss_fn = EnergyConservationLoss()
        loss = loss_fn(imfs, residual, original)
        assert loss.item() > 0.5

    def test_gradient_flows(self):
        imfs = torch.randn(2, 3, 256, requires_grad=True)
        residual = torch.randn(2, 256, requires_grad=True)
        original = torch.randn(2, 256)
        loss_fn = EnergyConservationLoss()
        loss = loss_fn(imfs, residual, original)
        loss.backward()
        assert imfs.grad is not None


class TestFrequencyOrderingLoss:
    def test_correct_order_zero_loss(self):
        """Descending-frequency IMFs should produce ~0 loss."""
        B, T = 2, 1024
        fs = 1000.0
        t = torch.arange(T, dtype=torch.float32) / fs
        imf_hi = torch.sin(2 * np.pi * 100 * t)
        imf_mid = torch.sin(2 * np.pi * 40 * t)
        imf_lo = torch.sin(2 * np.pi * 8 * t)
        imfs = torch.stack([imf_hi, imf_mid, imf_lo]).unsqueeze(0).expand(B, 3, T)
        loss_fn = FrequencyOrderingLoss(sample_rate=fs, normalized_margin=0.02)
        loss = loss_fn(imfs)
        assert loss.item() < 1e-4

    def test_reversed_order_high_loss(self):
        """Ascending-frequency IMFs should produce high loss."""
        B, T = 2, 1024
        fs = 1000.0
        t = torch.arange(T, dtype=torch.float32) / fs
        imf_lo = torch.sin(2 * np.pi * 8 * t)
        imf_mid = torch.sin(2 * np.pi * 40 * t)
        imf_hi = torch.sin(2 * np.pi * 100 * t)
        imfs = torch.stack([imf_lo, imf_mid, imf_hi]).unsqueeze(0).expand(B, 3, T)
        loss_fn = FrequencyOrderingLoss(sample_rate=fs, normalized_margin=0.02)
        loss = loss_fn(imfs)
        assert loss.item() > 0.01

    def test_single_imf_zero_loss(self):
        imfs = torch.randn(2, 1, 256)
        loss_fn = FrequencyOrderingLoss()
        assert loss_fn(imfs).item() == 0.0

    def test_gradient_flows(self):
        imfs = torch.randn(2, 3, 512, requires_grad=True)
        loss_fn = FrequencyOrderingLoss()
        loss_fn(imfs).backward()
        assert imfs.grad is not None
        assert not torch.all(imfs.grad == 0)

    def test_centroids_path_correct_order(self):
        """Direct-centroid API: descending centroids → ~0 loss."""
        centroids = torch.tensor([[100.0, 40.0, 5.0], [80.0, 20.0, 2.0]])
        loss_fn = FrequencyOrderingLoss(sample_rate=1000.0, normalized_margin=0.02)
        loss = loss_fn(centroids=centroids)
        assert loss.item() < 1e-6

    def test_centroids_path_wrong_order(self):
        centroids = torch.tensor([[5.0, 40.0, 100.0]])  # ascending — bad
        loss_fn = FrequencyOrderingLoss(sample_rate=1000.0, normalized_margin=0.02)
        loss = loss_fn(centroids=centroids)
        assert loss.item() > 0.001

    def test_centroids_path_gradient(self):
        centroids = torch.tensor([[5.0, 40.0, 100.0]], requires_grad=True)
        loss_fn = FrequencyOrderingLoss(sample_rate=1000.0)
        loss_fn(centroids=centroids).backward()
        assert centroids.grad is not None


class TestFilterSharpnessLoss:
    def test_uniform_high_loss(self):
        """Uniform filters (max entropy) should give loss = 1.0."""
        # Flat filter = 1/K everywhere (the degenerate softmax solution)
        B, K, N = 2, 3, 64
        filters = torch.full((B, K, N), 1.0 / K)
        loss_fn = FilterSharpnessLoss()
        loss = loss_fn(filters)
        assert abs(loss.item() - 1.0) < 1e-5

    def test_delta_low_loss(self):
        """Concentrated (delta-like) filters should give very low loss."""
        B, K, N = 2, 3, 64
        filters = torch.zeros(B, K, N)
        # Each filter has a single-bin spike
        for b in range(B):
            for k in range(K):
                filters[b, k, 5 + k * 10] = 1.0
        loss_fn = FilterSharpnessLoss()
        loss = loss_fn(filters)
        assert loss.item() < 0.01

    def test_peaked_lower_than_flat(self):
        """A peaked filter should have lower loss than a flat one."""
        flat = torch.full((1, 1, 100), 0.01)
        peaked = torch.zeros(1, 1, 100)
        peaked[0, 0, 50] = 1.0
        loss_fn = FilterSharpnessLoss()
        assert loss_fn(peaked).item() < loss_fn(flat).item()

    def test_gradient_flow(self):
        # Use logits as the leaf tensor; filters derived via softmax is non-leaf.
        logits = torch.randn(2, 3, 64, requires_grad=True)
        filters = logits.softmax(dim=1)
        loss_fn = FilterSharpnessLoss()
        loss_fn(filters).backward()
        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)


class TestSpectralConcentrationLoss:
    def test_pure_tone_low_concentration_loss(self):
        """Pure tone → peaked spectrum → low entropy."""
        T = 1024
        fs = 1000.0
        t = torch.arange(T, dtype=torch.float32) / fs
        tone = torch.sin(2 * np.pi * 50 * t).unsqueeze(0).unsqueeze(0)
        noise = torch.randn(1, 1, T)
        loss_fn = SpectralConcentrationLoss()
        l_tone = loss_fn(tone)
        l_noise = loss_fn(noise)
        assert l_tone < l_noise
        assert l_tone.item() < 0.3

    def test_gradient_flows(self):
        imfs = torch.randn(2, 3, 256, requires_grad=True)
        loss_fn = SpectralConcentrationLoss()
        loss_fn(imfs).backward()
        assert imfs.grad is not None


def _make_fake_metadata(B, K, n_freqs, sample_rate=1000.0, requires_grad=False):
    """Build a synthetic metadata dict for filter-bank loss tests."""
    import torch.nn.functional as F
    logits = torch.randn(B, K, n_freqs, requires_grad=requires_grad)
    filters = F.softmax(logits, dim=1)
    nyquist = sample_rate / 2
    freqs = torch.linspace(0, nyquist, n_freqs)
    centroids = (filters * freqs).sum(-1) / (filters.sum(-1) + 1e-8)
    return {"filters": filters, "centroids": centroids, "filter_logits": logits}


class TestCentroidSeparationLoss:
    def test_well_spread_lower_than_clustered(self):
        """Spread centroids should have a lower loss than clustered ones."""
        # Spread evenly across [0, Nyquist=500]
        spread = torch.tensor([[450.0, 250.0, 50.0]])
        # Two clustered at the low end (the Phase 2.5b v2 failure mode)
        clustered = torch.tensor([[250.0, 10.0, 5.0]])
        loss_fn = CentroidSeparationLoss(sample_rate=1000.0)
        l_spread = loss_fn(spread)
        l_clust = loss_fn(clustered)
        assert l_spread < l_clust

    def test_wrong_order_penalised(self):
        """Ascending centroids must cost more than descending."""
        descending = torch.tensor([[450.0, 250.0, 50.0]])
        ascending = torch.tensor([[50.0, 250.0, 450.0]])
        loss_fn = CentroidSeparationLoss(sample_rate=1000.0)
        assert loss_fn(descending) < loss_fn(ascending)

    def test_repulsion_gradient_pushes_apart(self):
        """Gradient on clustered centroids should push the clustered ones apart."""
        # Two centroids at 5 Hz (clustered), one at 250 Hz.
        centroids = torch.tensor([[250.0, 6.0, 5.0]], requires_grad=True)
        loss_fn = CentroidSeparationLoss(sample_rate=1000.0, w_coverage=0.0)
        loss_fn(centroids).backward()
        # The two clustered centroids should receive gradients pushing them apart
        # d(c[1])/dt should be positive (push up), d(c[2])/dt should be negative (push down)
        # when the repulsion is dominating.
        g = centroids.grad[0]
        assert g[1] < 0  # gradient descent on g[1]<0 increases c[1] (push up)
        assert g[2] > 0  # gradient descent on g[2]>0 decreases c[2] (push down)

    def test_single_imf_zero_loss(self):
        centroids = torch.tensor([[100.0]])
        loss_fn = CentroidSeparationLoss()
        assert loss_fn(centroids).item() == 0.0

    def test_gradient_flow(self):
        # Create the leaf tensor first, then enable grad — `torch.rand * 500`
        # would return a non-leaf tensor and .grad would be None.
        centroids = (torch.rand(2, 4) * 500.0).detach().requires_grad_(True)
        loss_fn = CentroidSeparationLoss(sample_rate=1000.0)
        loss_fn(centroids).backward()
        assert centroids.grad is not None
        assert not torch.all(centroids.grad == 0)


class TestFilterBalanceLoss:
    def test_balanced_filters_zero_loss(self):
        """Uniform coverage (each filter covers 1/K) → loss = 0."""
        B, K, N = 2, 3, 64
        filters = torch.full((B, K, N), 1.0 / K)
        loss_fn = FilterBalanceLoss()
        assert loss_fn(filters).item() == 0.0

    def test_collapsed_filter_high_loss(self):
        """One filter = 1 everywhere, others = 0 → high loss."""
        B, K, N = 2, 3, 64
        filters = torch.zeros(B, K, N)
        filters[:, 0, :] = 1.0
        loss_fn = FilterBalanceLoss()  # default min_fraction = 1/(2K) = 1/6
        loss = loss_fn(filters)
        # Two empty filters (coverage=0); each contributes 1/6 to ReLU(1/6 - 0).
        # mean over (B, K): (0 + 1/6 + 1/6)/3 ≈ 0.111
        assert loss.item() > 0.05

    def test_gradient_flow(self):
        logits = torch.randn(2, 3, 64, requires_grad=True)
        filters = logits.softmax(dim=1)
        loss_fn = FilterBalanceLoss()
        loss_fn(filters).backward()
        assert logits.grad is not None


class TestNEMDLossFilterBank:
    """Tests for the new 3-term NEMDLoss (filter bank)."""

    def test_combined_loss_runs(self):
        B, K, T = 2, 3, 256
        imfs = torch.randn(B, K, T, requires_grad=True)
        metadata = _make_fake_metadata(B, K, T // 2 + 1, requires_grad=True)

        criterion = NEMDLoss(sample_rate=1000.0)
        loss, components = criterion(imfs, metadata)
        assert loss.requires_grad
        for key in ("total", "sharp", "order", "ortho", "balance"):
            assert key in components, f"Missing key: {key}"

    def test_gradient_flow(self):
        B, K, T = 2, 3, 256
        imfs = torch.randn(B, K, T, requires_grad=True)
        metadata = _make_fake_metadata(B, K, T // 2 + 1, requires_grad=True)

        criterion = NEMDLoss(sample_rate=1000.0)
        loss, _ = criterion(imfs, metadata)
        loss.backward()
        # Filter logits should receive gradients (sharp + order depend on filters)
        assert metadata["filter_logits"].grad is not None

    def test_task_loss_integration(self):
        B, K, T = 2, 3, 256
        imfs = torch.randn(B, K, T)
        metadata = _make_fake_metadata(B, K, T // 2 + 1)
        task_loss = torch.tensor(0.5, requires_grad=True)

        criterion = NEMDLoss(lambda_task=1.0, sample_rate=1000.0)
        loss, components = criterion(imfs, metadata, task_loss=task_loss)
        assert "task" in components


class TestNEMDSiftingLoss:
    """Tests for the legacy 7-term NEMDSiftingLoss."""

    def test_combined_loss_runs(self):
        B, K, T = 2, 3, 256
        imfs = torch.randn(B, K, T, requires_grad=True)
        residual = torch.randn(B, T, requires_grad=True)
        original = torch.randn(B, T)

        criterion = NEMDSiftingLoss(sample_rate=1000.0)
        loss, components = criterion(imfs, residual, original)
        assert loss.requires_grad
        for key in ("total", "ortho", "narrow", "mono",
                    "residual", "energy", "order", "concentration"):
            assert key in components

    def test_gradient_flows_to_imfs(self):
        imfs = torch.randn(2, 3, 256, requires_grad=True)
        residual = torch.randn(2, 256, requires_grad=True)
        original = torch.randn(2, 256)
        criterion = NEMDSiftingLoss()
        loss, _ = criterion(imfs, residual, original)
        loss.backward()
        assert imfs.grad is not None
        assert residual.grad is not None
