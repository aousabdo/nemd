"""Tests for nemd.model — filter-bank NEMD, sifting NEMD, and helpers."""

import numpy as np
import pytest
import torch

from nemd.model import (
    NEMD,
    NEMDSifting,
    InitFilterBank,
    SiftNet,
    SiftNetConfig,
    SignalAnalyzer,
    FrequencyPartition,
    ResBlock1d,
)
from nemd.utils import generate_synthetic_signal, reconstruction_error


class TestSiftNet:
    def test_output_shape(self):
        config = SiftNetConfig(max_imfs=4)
        net = SiftNet(config)
        x = torch.randn(2, 512)
        out = net(x, k=0)
        assert out.shape == (2, 512)

    def test_output_shape_odd_length(self):
        """Should handle signal lengths not divisible by 2^num_levels."""
        config = SiftNetConfig(max_imfs=4)
        net = SiftNet(config)
        x = torch.randn(2, 500)  # 500 is not divisible by 8
        out = net(x, k=0)
        assert out.shape == (2, 500)

    def test_different_k_different_output(self):
        """FiLM conditioning on k should produce different outputs."""
        config = SiftNetConfig(max_imfs=4)
        net = SiftNet(config)
        net.eval()
        x = torch.randn(1, 256)
        with torch.no_grad():
            out0 = net(x, k=0)
            out1 = net(x, k=1)
        assert not torch.allclose(out0, out1, atol=1e-6)

    def test_gradient_flow(self):
        config = SiftNetConfig(max_imfs=4)
        net = SiftNet(config)
        x = torch.randn(2, 256, requires_grad=True)
        out = net(x, k=0)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_without_init_filter(self):
        """Should work with init filter disabled."""
        config = SiftNetConfig(max_imfs=3, use_init_filter=False)
        net = SiftNet(config)
        x = torch.randn(2, 256)
        out = net(x, k=0)
        assert out.shape == (2, 256)

    def test_with_init_filter(self):
        """Should work with init filter enabled (default)."""
        config = SiftNetConfig(max_imfs=3, use_init_filter=True)
        net = SiftNet(config)
        assert net.init_filter_bank is not None
        x = torch.randn(2, 256)
        out = net(x, k=0)
        assert out.shape == (2, 256)


class TestInitFilterBank:
    def test_shape(self):
        bank = InitFilterBank(max_imfs=3, n_taps=31)
        x = torch.randn(2, 1, 128)
        out = bank(x, k=0)
        assert out.shape == (2, 1, 128)

    def test_different_k_different_output(self):
        """Different filter indices should produce different outputs."""
        torch.manual_seed(0)
        bank = InitFilterBank(max_imfs=3, n_taps=31)
        x = torch.randn(1, 1, 128)
        out0 = bank(x, k=0)
        out2 = bank(x, k=2)
        assert not torch.allclose(out0, out2)

    def test_frequency_response_ordering(self):
        """IMF 0 filter passes high freq; IMF K-1 filter passes low freq.

        For K=3 the filter bands (normalised to Nyquist) are roughly::

            IMF 0 : 0.5 – 1.0   (upper half)
            IMF 1 : 0.25 – 0.75 (mid)
            IMF 2 : 0.0 – 0.5   (lower half)

        So we pick signals clearly in each band at fs=1000 (Nyquist=500 Hz).
        """
        bank = InitFilterBank(max_imfs=3, n_taps=31)
        fs = 1000.0
        T = 1024
        t = torch.arange(T, dtype=torch.float32) / fs
        # High-freq signal: 0.75 * Nyquist = 375 Hz
        hi = torch.sin(2 * np.pi * 375 * t).unsqueeze(0).unsqueeze(0)
        # Low-freq signal: 0.2 * Nyquist = 100 Hz
        lo = torch.sin(2 * np.pi * 100 * t).unsqueeze(0).unsqueeze(0)

        hi_out0 = bank(hi, k=0).abs().mean().item()
        hi_out2 = bank(hi, k=2).abs().mean().item()
        lo_out0 = bank(lo, k=0).abs().mean().item()
        lo_out2 = bank(lo, k=2).abs().mean().item()

        assert hi_out0 > hi_out2, "IMF 0 filter should amplify high freq more"
        assert lo_out2 > lo_out0, "IMF K-1 filter should amplify low freq more"

    def test_gradient_flows(self):
        bank = InitFilterBank(max_imfs=3)
        x = torch.randn(2, 1, 128, requires_grad=True)
        bank(x, k=1).sum().backward()
        assert x.grad is not None


class TestResBlock1d:
    def test_output_shape(self):
        block = ResBlock1d(channels=16, kernel_size=5)
        x = torch.randn(2, 16, 128)
        out = block(x)
        assert out.shape == x.shape

    def test_gradient_flow(self):
        block = ResBlock1d(channels=8)
        x = torch.randn(2, 8, 64, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None


class TestSignalAnalyzer:
    def test_output_shape(self):
        K = 3
        analyzer = SignalAnalyzer(num_imfs=K, hidden_dim=32, num_layers=2)
        # (batch, 2, n_freqs)
        x = torch.randn(2, 2, 257)
        out = analyzer(x)
        assert out.shape == (2, K, 257)

    def test_gradient_flow(self):
        analyzer = SignalAnalyzer(num_imfs=3, hidden_dim=16, num_layers=1)
        x = torch.randn(2, 2, 129, requires_grad=True)
        analyzer(x).sum().backward()
        assert x.grad is not None


class TestFrequencyPartition:
    def test_partition_of_unity(self):
        """filters should sum to 1 over the K dimension at every frequency."""
        part = FrequencyPartition()
        X = torch.randn(2, 129, dtype=torch.complex64)
        logits = torch.randn(2, 4, 129)
        _, filters = part(X, logits, temperature=1.0)
        sums = filters.sum(dim=1)  # (B, n_freqs)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)

    def test_filters_non_negative(self):
        part = FrequencyPartition()
        X = torch.randn(2, 129, dtype=torch.complex64)
        logits = torch.randn(2, 3, 129)
        _, filters = part(X, logits)
        assert (filters >= 0).all()

    def test_spectra_shape(self):
        part = FrequencyPartition()
        X = torch.randn(2, 129, dtype=torch.complex64)
        logits = torch.randn(2, 3, 129)
        spectra, filters = part(X, logits)
        assert spectra.shape == (2, 3, 129)
        assert spectra.dtype == torch.complex64

    def test_temperature_effect(self):
        """Low temperature → sharper filters (higher max) than high temperature."""
        part = FrequencyPartition()
        X = torch.randn(1, 65, dtype=torch.complex64)
        logits = torch.randn(1, 3, 65)
        _, f_hot = part(X, logits, temperature=5.0)
        _, f_cold = part(X, logits, temperature=0.1)
        # Sharp partition has larger peak values
        assert f_cold.max() > f_hot.max()


class TestNEMDFilterBank:
    """Tests for the primary filter-bank NEMD."""

    def test_output_shapes(self):
        model = NEMD(num_imfs=3, hidden_dim=16, num_layers=2)
        x = torch.randn(4, 512)
        imfs, residual, metadata = model(x)
        assert imfs.shape == (4, 3, 512)
        assert residual.shape == (4, 512)
        assert metadata["filters"].shape == (4, 3, 257)
        assert metadata["centroids"].shape == (4, 3)

    def test_reconstruction_exact(self):
        """sum(imfs) should equal input exactly (architectural)."""
        model = NEMD(num_imfs=3, hidden_dim=16, num_layers=2)
        x = torch.randn(3, 512)
        imfs, _, _ = model(x)
        reconstructed = imfs.sum(dim=1)
        torch.testing.assert_close(reconstructed, x, atol=1e-5, rtol=1e-5)

    def test_reconstruction_various_lengths(self):
        model = NEMD(num_imfs=3, hidden_dim=16, num_layers=2)
        for T in [128, 255, 300, 512, 1000, 1024]:
            x = torch.randn(1, T)
            imfs, _, _ = model(x)
            reconstructed = imfs.sum(dim=1)
            torch.testing.assert_close(
                reconstructed, x, atol=1e-4, rtol=1e-4,
                msg=f"Reconstruction failed for T={T}",
            )

    def test_partition_sums_to_one(self):
        model = NEMD(num_imfs=3, hidden_dim=16, num_layers=2)
        x = torch.randn(2, 512)
        _, _, metadata = model(x)
        sums = metadata["filters"].sum(dim=1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)

    def test_energy_bounded(self):
        """Total IMF energy should be ≤ signal energy (Cauchy-Schwarz)."""
        model = NEMD(num_imfs=3, hidden_dim=16, num_layers=2)
        model.eval()
        x = torch.randn(4, 512)
        with torch.no_grad():
            imfs, _, _ = model(x)
        imf_energy = (imfs ** 2).sum(dim=(1, 2))
        sig_energy = (x ** 2).sum(dim=-1)
        # Allow tiny FP margin
        assert (imf_energy <= sig_energy * 1.001).all()

    def test_gradient_flow_all_params(self):
        model = NEMD(num_imfs=3, hidden_dim=16, num_layers=2)
        x = torch.randn(2, 256)
        imfs, _, metadata = model(x)
        loss = imfs.sum() + metadata["centroids"].sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_temperature_settable(self):
        model = NEMD(num_imfs=3, hidden_dim=8, num_layers=1)
        model.set_temperature(0.5)
        assert model.temperature == 0.5

    def test_temperature_override(self):
        """Passing temperature to forward should override self.temperature."""
        model = NEMD(num_imfs=3, hidden_dim=8, num_layers=1)
        model.eval()
        x = torch.randn(1, 128)
        with torch.no_grad():
            _, _, m_hot = model(x, temperature=5.0)
            _, _, m_cold = model(x, temperature=0.1)
        # Same signal, different temperature → different filter shapes
        assert not torch.allclose(m_hot["filters"], m_cold["filters"])

    def test_num_imfs_mismatch_raises(self):
        model = NEMD(num_imfs=3, hidden_dim=8, num_layers=1)
        with pytest.raises(ValueError):
            model(torch.randn(1, 128), num_imfs=5)

    def test_sort_by_centroid_descending(self):
        """With sort_by_centroid=True, signal-weighted centroids should be descending."""
        torch.manual_seed(0)
        model = NEMD(num_imfs=3, hidden_dim=8, num_layers=1, sample_rate=512.0)
        model.eval()
        x = torch.randn(4, 256)
        with torch.no_grad():
            _, _, meta = model(x, sort_by_centroid=True)
        c = meta["centroids_weighted"]  # (B, K)
        # Descending along K
        diffs = c[:, 1:] - c[:, :-1]
        assert (diffs <= 1e-6).all()  # allow tiny FP slack

    def test_sort_preserves_reconstruction(self):
        """Sorting permutes IMFs; sum must still equal input."""
        torch.manual_seed(0)
        model = NEMD(num_imfs=3, hidden_dim=8, num_layers=1, sample_rate=512.0)
        model.eval()
        x = torch.randn(2, 256)
        with torch.no_grad():
            imfs, _, _ = model(x, sort_by_centroid=True)
        torch.testing.assert_close(imfs.sum(dim=1), x, atol=1e-5, rtol=1e-5)

    def test_sort_preserves_partition(self):
        torch.manual_seed(0)
        model = NEMD(num_imfs=3, hidden_dim=8, num_layers=1, sample_rate=512.0)
        model.eval()
        x = torch.randn(2, 256)
        with torch.no_grad():
            _, _, meta = model(x, sort_by_centroid=True)
        sums = meta["filters"].sum(dim=1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)


class TestNEMDSifting:
    """Legacy sifting N-EMD — kept for comparison."""

    def test_output_shapes(self):
        model = NEMDSifting(max_imfs=4)
        x = torch.randn(3, 512)
        imfs, residual = model(x)
        assert imfs.shape == (3, 4, 512)
        assert residual.shape == (3, 512)

    def test_reconstruction_exact(self):
        model = NEMDSifting(max_imfs=4)
        x = torch.randn(2, 512)
        imfs, residual = model(x)
        reconstructed = imfs.sum(dim=1) + residual
        torch.testing.assert_close(reconstructed, x, atol=1e-5, rtol=1e-5)

    def test_custom_num_imfs(self):
        model = NEMDSifting(max_imfs=6)
        x = torch.randn(2, 256)
        imfs, residual = model(x, num_imfs=3)
        assert imfs.shape == (2, 3, 256)
        assert residual.shape == (2, 256)


class TestClassicalEMDBaseline:
    """Keep the Phase 1 baseline tests working."""

    def test_decompose_runs(self):
        from nemd.classical import ClassicalEMD
        _, signal, _ = generate_synthetic_signal(n_samples=512, seed=0)
        emd = ClassicalEMD()
        imfs = emd.decompose(signal)
        assert imfs.ndim == 2
        assert imfs.shape[1] == 512

    def test_reconstruction(self):
        from nemd.classical import ClassicalEMD
        _, signal, _ = generate_synthetic_signal(n_samples=512, seed=0)
        emd = ClassicalEMD()
        imfs = emd.decompose(signal)
        err = reconstruction_error(signal, imfs)
        assert err < 0.01

    def test_max_imfs(self):
        from nemd.classical import ClassicalEMD
        _, signal, _ = generate_synthetic_signal(n_samples=512, seed=0)
        emd = ClassicalEMD(max_imfs=3)
        imfs = emd.decompose(signal)
        assert imfs.shape[0] <= 3


class TestVMDBaseline:
    def test_decompose_runs(self):
        from nemd.classical import VMD
        _, signal, _ = generate_synthetic_signal(n_samples=512, seed=0)
        vmd = VMD(n_modes=3)
        modes = vmd.decompose(signal)
        assert modes.shape == (3, 512)

    def test_reconstruction(self):
        from nemd.classical import VMD
        _, signal, _ = generate_synthetic_signal(n_samples=512, seed=0)
        vmd = VMD(n_modes=3)
        modes = vmd.decompose(signal)
        err = reconstruction_error(signal, modes)
        assert err < 0.30
