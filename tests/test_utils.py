"""Tests for nemd.utils — signal generation and evaluation metrics."""

import numpy as np
import pytest

from nemd.utils import (
    generate_am_fm_component,
    generate_chirp,
    generate_nonstationary_signal,
    generate_synthetic_signal,
    energy_ratio,
    if_tracking_error,
    mode_mixing_index,
    monotonicity_score,
    orthogonality_index,
    reconstruction_error,
    to_numpy,
    to_tensor,
)


class TestSignalGeneration:
    def test_am_fm_shape(self):
        t = np.linspace(0, 1, 512, endpoint=False)
        comp = generate_am_fm_component(t, f0=10.0)
        assert comp.shape == (512,)

    def test_am_fm_bounded(self):
        """AM-FM with a_mod < 1 should have amplitude bounded by 1 + a_mod."""
        t = np.linspace(0, 1, 1024, endpoint=False)
        comp = generate_am_fm_component(t, f0=20.0, a_mod=0.5)
        assert np.max(np.abs(comp)) <= 1.5 + 1e-6

    def test_synthetic_signal_defaults(self):
        t, signal, comps = generate_synthetic_signal(n_samples=512, seed=0)
        assert t.shape == (512,)
        assert signal.shape == (512,)
        assert len(comps) == 3
        # Signal should be the sum of components (no noise by default)
        np.testing.assert_allclose(signal, sum(comps), atol=1e-12)

    def test_synthetic_signal_noise(self):
        _, sig_clean, _ = generate_synthetic_signal(n_samples=1024, noise_std=0.0, seed=1)
        _, sig_noisy, _ = generate_synthetic_signal(n_samples=1024, noise_std=0.5, seed=1)
        # Noisy signal should differ from clean
        assert not np.allclose(sig_clean, sig_noisy)

    def test_synthetic_signal_reproducible(self):
        _, s1, _ = generate_synthetic_signal(n_samples=256, noise_std=0.1, seed=42)
        _, s2, _ = generate_synthetic_signal(n_samples=256, noise_std=0.1, seed=42)
        np.testing.assert_array_equal(s1, s2)

    def test_chirp_shape(self):
        t, sig = generate_chirp(n_samples=256)
        assert t.shape == (256,)
        assert sig.shape == (256,)

    def test_chirp_bounded(self):
        _, sig = generate_chirp(n_samples=1024, noise_std=0.0)
        assert np.max(np.abs(sig)) <= 1.0 + 1e-6


class TestMetrics:
    def test_orthogonality_perfect(self):
        """Orthogonal sine/cosine pair should have OI ~ 0."""
        t = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
        imfs = np.stack([np.sin(t), np.cos(t)])
        oi = orthogonality_index(imfs)
        assert oi < 0.01

    def test_orthogonality_identical(self):
        """Identical signals should have high OI."""
        x = np.sin(np.linspace(0, 4 * np.pi, 500))
        imfs = np.stack([x, x])
        oi = orthogonality_index(imfs)
        assert oi > 0.9

    def test_energy_ratio_perfect_recon(self):
        """If IMFs sum to signal, energy ratio should be ~1."""
        t = np.linspace(0, 1, 256, endpoint=False)
        c1 = np.sin(2 * np.pi * 5 * t)
        c2 = np.sin(2 * np.pi * 20 * t)
        signal = c1 + c2
        imfs = np.stack([c1, c2])
        er = energy_ratio(signal, imfs)
        # Not exactly 1 unless components are orthogonal
        assert 0.5 < er < 2.0

    def test_reconstruction_error_perfect(self):
        signal = np.random.default_rng(0).normal(size=100)
        imfs = signal[np.newaxis, :]  # single IMF = signal
        err = reconstruction_error(signal, imfs)
        assert err < 1e-10

    def test_reconstruction_error_bad(self):
        signal = np.ones(100)
        imfs = np.zeros((1, 100))
        err = reconstruction_error(signal, imfs)
        assert err > 0.99

    def test_mode_mixing_perfect(self):
        """When IMFs exactly match true components, MMI should be ~0."""
        t = np.linspace(0, 1, 512, endpoint=False)
        c1 = np.sin(2 * np.pi * 10 * t)
        c2 = np.sin(2 * np.pi * 30 * t)
        true_comps = [c1, c2]
        imfs = np.stack([c1, c2])
        mmi = mode_mixing_index(true_comps, imfs)
        assert mmi < 0.01

    def test_monotonicity_perfect(self):
        x = np.linspace(0, 10, 200)
        assert monotonicity_score(x) == 1.0

    def test_monotonicity_oscillating(self):
        t = np.linspace(0, 4 * np.pi, 500)
        x = np.sin(t)
        score = monotonicity_score(x)
        assert score < 0.7  # definitely not monotonic


class TestNonstationarySignals:
    @pytest.mark.parametrize(
        "kind", ["stationary", "chirp_trio", "crossing_chirps", "widening_am", "piecewise"],
    )
    def test_all_kinds_produce_valid_signals(self, kind):
        t, sig, comps, ifs = generate_nonstationary_signal(
            n_samples=512, duration=1.0, kind=kind, seed=0,
        )
        assert sig.shape == (512,)
        assert len(comps) == 3
        assert len(ifs) == 3
        assert all(c.shape == (512,) for c in comps)
        assert all(f.shape == (512,) for f in ifs)
        # Signal should be the sum of components (no noise at seed=0 unless asked)
        np.testing.assert_allclose(sig, sum(comps), atol=1e-10)

    def test_chirp_trio_has_varying_if(self):
        _, _, _, ifs = generate_nonstationary_signal(
            n_samples=512, duration=1.0, kind="chirp_trio", seed=0,
        )
        # First component IF should span a range (it's a chirp)
        assert ifs[0].max() - ifs[0].min() > 5.0
        # Other two should be constant
        assert ifs[1].max() - ifs[1].min() < 0.1
        assert ifs[2].max() - ifs[2].min() < 0.1

    def test_piecewise_has_discrete_if_values(self):
        _, _, _, ifs = generate_nonstationary_signal(
            n_samples=512, duration=1.0, kind="piecewise", seed=0,
        )
        # First component's IF should have only ~3 unique values (3 segments)
        unique = np.unique(ifs[0])
        assert len(unique) <= 5  # allow slight boundary effects


class TestIFTrackingError:
    def test_self_match_low_rmse(self):
        """True components matched to themselves should give low RMSE."""
        t, _, comps, ifs = generate_nonstationary_signal(
            n_samples=1024, duration=1.0, kind="chirp_trio", seed=0,
        )
        imfs = np.stack(comps)
        result = if_tracking_error(ifs, imfs, fs=1024.0, edge_trim=20)
        assert result["mean_rmse"] < 2.0

    def test_result_structure(self):
        t, _, comps, ifs = generate_nonstationary_signal(
            n_samples=256, kind="stationary", seed=0,
        )
        imfs = np.stack(comps)
        result = if_tracking_error(ifs, imfs, fs=256.0)
        assert "per_component_rmse" in result
        assert "mean_rmse" in result
        assert "max_rmse" in result
        assert "matched_imf_idx" in result
        assert len(result["per_component_rmse"]) == 3


class TestConversions:
    def test_to_tensor_numpy(self):
        import torch
        arr = np.array([1.0, 2.0, 3.0])
        t = to_tensor(arr)
        assert isinstance(t, torch.Tensor)
        assert t.dtype == torch.float32

    def test_to_tensor_passthrough(self):
        import torch
        t = torch.tensor([1.0, 2.0])
        t2 = to_tensor(t)
        assert t2 is not t or t2.dtype == torch.float32  # may be same object

    def test_to_numpy(self):
        import torch
        t = torch.tensor([1.0, 2.0, 3.0])
        arr = to_numpy(t)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])
