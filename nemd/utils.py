"""Signal generation, evaluation metrics, and helpers for N-EMD."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Synthetic signal generation
# ---------------------------------------------------------------------------

def generate_am_fm_component(
    t: NDArray[np.float64],
    f0: float = 10.0,
    f_mod: float = 1.0,
    a_mod: float = 0.3,
    phase: float = 0.0,
    freq_dev: float | None = None,
) -> NDArray[np.float64]:
    """Generate a single AM-FM component.

    x(t) = [1 + a_mod * cos(2*pi*f_mod*t)] * cos(2*pi*f0*t + freq_dev*sin(2*pi*f_mod*t) + phase)

    Parameters
    ----------
    t : array
        Time vector.
    f0 : float
        Carrier frequency (Hz).
    f_mod : float
        Modulation frequency (Hz).
    a_mod : float
        Amplitude modulation depth (0 = none, 1 = full).
    phase : float
        Initial phase (radians).
    freq_dev : float or None
        Frequency deviation for FM. If None, defaults to ``f0 * 0.1``.

    Returns
    -------
    component : array
        The AM-FM signal evaluated at ``t``.
    """
    if freq_dev is None:
        freq_dev = f0 * 0.1
    amplitude = 1.0 + a_mod * np.cos(2 * np.pi * f_mod * t)
    inst_phase = 2 * np.pi * f0 * t + freq_dev * np.sin(2 * np.pi * f_mod * t) + phase
    return amplitude * np.cos(inst_phase)


def generate_synthetic_signal(
    n_samples: int = 1024,
    duration: float = 1.0,
    components: list[dict] | None = None,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[NDArray[np.float64]]]:
    """Generate a synthetic multi-component signal with known ground truth.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    duration : float
        Signal duration in seconds.
    components : list of dict, optional
        Each dict specifies one AM-FM component with keys matching
        :func:`generate_am_fm_component` kwargs (``f0``, ``f_mod``,
        ``a_mod``, ``phase``, ``freq_dev``).  The ``"type"`` key is
        accepted but ignored (reserved for future component types).
        Defaults to a 3-component signal if *None*.
    noise_std : float
        Standard deviation of additive Gaussian noise.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    t : array of shape (n_samples,)
        Time vector.
    signal : array of shape (n_samples,)
        The composite signal (sum of components + noise).
    comp_list : list of arrays
        Individual ground-truth components.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    if components is None:
        components = [
            {"f0": 50.0, "f_mod": 2.0, "a_mod": 0.5},
            {"f0": 15.0, "f_mod": 0.5, "a_mod": 0.3},
            {"f0": 3.0,  "f_mod": 0.1, "a_mod": 0.2},
        ]

    comp_list: list[NDArray[np.float64]] = []
    for spec in components:
        kw = {k: v for k, v in spec.items() if k != "type"}
        comp_list.append(generate_am_fm_component(t, **kw))

    signal = np.sum(comp_list, axis=0)
    if noise_std > 0:
        signal = signal + rng.normal(0, noise_std, size=n_samples)

    return t, signal, comp_list


def generate_chirp(
    n_samples: int = 1024,
    duration: float = 1.0,
    f_start: float = 5.0,
    f_end: float = 50.0,
    seed: int | None = None,
    noise_std: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate a linear chirp signal.

    Parameters
    ----------
    n_samples, duration, seed, noise_std
        Same as :func:`generate_synthetic_signal`.
    f_start, f_end : float
        Start and end frequencies (Hz).

    Returns
    -------
    t, signal : arrays
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    chirp_rate = (f_end - f_start) / duration
    phase = 2 * np.pi * (f_start * t + 0.5 * chirp_rate * t ** 2)
    signal = np.cos(phase)
    if noise_std > 0:
        signal = signal + rng.normal(0, noise_std, size=n_samples)
    return t, signal


# ---------------------------------------------------------------------------
# Nonstationary signal generators (Phase 3, Experiment 1)
# ---------------------------------------------------------------------------

def linear_chirp_component(
    t: NDArray[np.float64],
    f_start: float,
    f_end: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Single-component linear chirp with its true instantaneous frequency.

    Returns
    -------
    signal : array (shape (T,))
    inst_freq : array (shape (T,))
        True instantaneous frequency at each time point.
    """
    duration = t[-1] - t[0] + (t[1] - t[0])  # total duration
    chirp_rate = (f_end - f_start) / duration
    phase_vec = 2 * np.pi * (f_start * t + 0.5 * chirp_rate * t ** 2) + phase
    signal = amplitude * np.cos(phase_vec)
    inst_freq = f_start + chirp_rate * t
    return signal, inst_freq


def piecewise_stationary_component(
    t: NDArray[np.float64],
    freqs: list[float],
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Component whose frequency jumps between values in equal-time segments.

    Parameters
    ----------
    freqs : list of floats
        Frequency within each segment.  Segments have equal duration.
    amplitude, phase : float

    Returns
    -------
    signal : (T,)
    inst_freq : (T,)
    """
    T = len(t)
    n_seg = len(freqs)
    seg_len = T // n_seg
    inst_freq = np.empty(T)
    phase_cum = np.zeros(T)
    current_phase = phase
    dt = t[1] - t[0]
    for i, f in enumerate(freqs):
        lo = i * seg_len
        hi = (i + 1) * seg_len if i < n_seg - 1 else T
        for j in range(lo, hi):
            inst_freq[j] = f
            phase_cum[j] = current_phase
            current_phase += 2 * np.pi * f * dt
    signal = amplitude * np.cos(phase_cum)
    return signal, inst_freq


def am_envelope_widening_component(
    t: NDArray[np.float64],
    f0: float,
    mod_rate_start: float = 0.5,
    mod_rate_end: float = 3.0,
    a_mod: float = 0.4,
    phase: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Component with amplitude modulation whose rate grows over time.

    Bandwidth increases monotonically from start → end: tests whether the
    model can adapt to time-varying spectral content.
    """
    duration = t[-1] - t[0] + (t[1] - t[0])
    # Instantaneous modulation rate grows linearly
    rate = mod_rate_start + (mod_rate_end - mod_rate_start) * (t / duration)
    # The AM envelope is cos(integral of 2π·rate)
    # ∫ (rate_start + slope·t) dt = rate_start·t + 0.5·slope·t²
    slope = (mod_rate_end - mod_rate_start) / duration
    mod_phase = 2 * np.pi * (mod_rate_start * t + 0.5 * slope * t ** 2)
    envelope = 1.0 + a_mod * np.cos(mod_phase)
    carrier_phase = 2 * np.pi * f0 * t + phase
    signal = envelope * np.cos(carrier_phase)
    inst_freq = np.full_like(t, f0)   # carrier frequency is constant
    return signal, inst_freq


def generate_nonstationary_signal(
    n_samples: int = 1024,
    duration: float = 1.0,
    kind: str = "chirp",
    seed: int | None = None,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
    **kwargs,
) -> tuple[NDArray, NDArray, list[NDArray], list[NDArray]]:
    """Generate a multi-component nonstationary signal with ground truth.

    Parameters
    ----------
    n_samples, duration, seed, noise_std : standard
    kind : str
        One of:
        - ``"stationary"`` : fixed-frequency AM-FM (3 components)
        - ``"chirp_trio"`` : one linear chirp + two constant-freq components
        - ``"crossing_chirps"`` : two chirps crossing in frequency + one constant low
        - ``"widening_am"``    : constant carriers, widening AM envelope on one
        - ``"piecewise"``      : one piecewise-stationary component + two constants
    rng : optional numpy Generator (takes precedence over ``seed``)

    Returns
    -------
    t : (T,)
    signal : (T,)
    components : list of (T,) arrays — ground-truth components
    inst_freqs : list of (T,) arrays — per-component instantaneous frequencies (Hz)
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    components = []
    inst_freqs = []

    if kind == "stationary":
        # 3-component AM-FM (same shape as the canonical Phase 1 test)
        for f0 in kwargs.get("freqs", [50.0, 15.0, 3.0]):
            comp = generate_am_fm_component(
                t,
                f0=float(f0),
                f_mod=float(rng.uniform(0.2, 3.0)),
                a_mod=float(rng.uniform(0.1, 0.5)),
                phase=float(rng.uniform(0, 2 * np.pi)),
            )
            components.append(comp)
            inst_freqs.append(np.full_like(t, f0))

    elif kind == "chirp_trio":
        # One chirp + two constants (chirp is the high-freq component)
        f_start = kwargs.get("f_start", float(rng.uniform(50, 80)))
        f_end = kwargs.get("f_end", float(rng.uniform(25, 45)))
        c_hi, if_hi = linear_chirp_component(t, f_start, f_end)
        f_mid = kwargs.get("f_mid", float(rng.uniform(12, 22)))
        f_lo = kwargs.get("f_lo", float(rng.uniform(2, 7)))
        c_mid = generate_am_fm_component(t, f0=f_mid, a_mod=0.2, f_mod=0.5)
        c_lo = generate_am_fm_component(t, f0=f_lo, a_mod=0.2, f_mod=0.1)
        components.extend([c_hi, c_mid, c_lo])
        inst_freqs.extend([if_hi, np.full_like(t, f_mid), np.full_like(t, f_lo)])

    elif kind == "crossing_chirps":
        # Two chirps that cross (one up, one down) + one constant low
        f_up_s = kwargs.get("f_up_start", float(rng.uniform(10, 20)))
        f_up_e = kwargs.get("f_up_end", float(rng.uniform(60, 90)))
        f_dn_s = kwargs.get("f_dn_start", float(rng.uniform(60, 90)))
        f_dn_e = kwargs.get("f_dn_end", float(rng.uniform(10, 20)))
        c_up, if_up = linear_chirp_component(t, f_up_s, f_up_e)
        c_dn, if_dn = linear_chirp_component(t, f_dn_s, f_dn_e)
        f_lo = kwargs.get("f_lo", float(rng.uniform(2, 6)))
        c_lo = generate_am_fm_component(t, f0=f_lo, a_mod=0.15, f_mod=0.1)
        components.extend([c_up, c_dn, c_lo])
        inst_freqs.extend([if_up, if_dn, np.full_like(t, f_lo)])

    elif kind == "widening_am":
        # High-freq carrier with widening AM, plus two constants
        f_hi = kwargs.get("f_hi", float(rng.uniform(40, 80)))
        mod_s = kwargs.get("mod_start", float(rng.uniform(0.3, 1.0)))
        mod_e = kwargs.get("mod_end", float(rng.uniform(3.0, 6.0)))
        c_hi, if_hi = am_envelope_widening_component(t, f_hi, mod_s, mod_e)
        f_mid = kwargs.get("f_mid", float(rng.uniform(12, 22)))
        f_lo = kwargs.get("f_lo", float(rng.uniform(2, 7)))
        c_mid = generate_am_fm_component(t, f0=f_mid, a_mod=0.3, f_mod=1.0)
        c_lo = generate_am_fm_component(t, f0=f_lo, a_mod=0.2, f_mod=0.2)
        components.extend([c_hi, c_mid, c_lo])
        inst_freqs.extend([if_hi, np.full_like(t, f_mid), np.full_like(t, f_lo)])

    elif kind == "piecewise":
        # One piecewise-stationary + two constants
        f_seq = kwargs.get("freqs_seq", [
            float(rng.uniform(40, 60)),
            float(rng.uniform(20, 30)),
            float(rng.uniform(60, 80)),
        ])
        c_pw, if_pw = piecewise_stationary_component(t, f_seq)
        f_mid = kwargs.get("f_mid", float(rng.uniform(10, 18)))
        f_lo = kwargs.get("f_lo", float(rng.uniform(2, 6)))
        c_mid = generate_am_fm_component(t, f0=f_mid, a_mod=0.3)
        c_lo = generate_am_fm_component(t, f0=f_lo, a_mod=0.2)
        components.extend([c_pw, c_mid, c_lo])
        inst_freqs.extend([if_pw, np.full_like(t, f_mid), np.full_like(t, f_lo)])

    else:
        raise ValueError(f"Unknown kind: {kind}")

    signal = np.sum(components, axis=0)
    if noise_std > 0:
        signal = signal + rng.normal(0, noise_std, size=n_samples)
    return t, signal, components, inst_freqs


# ---------------------------------------------------------------------------
# Instantaneous-frequency tracking error (Phase 3 metric)
# ---------------------------------------------------------------------------

def _inst_freq_from_signal(
    x: NDArray[np.float64], fs: float,
) -> NDArray[np.float64]:
    """Numpy IF via Hilbert transform + phase unwrap.  Returns shape (T,)."""
    from scipy.signal import hilbert

    z = hilbert(x)
    phase = np.unwrap(np.angle(z))
    inst_freq = np.gradient(phase) * fs / (2 * np.pi)
    return inst_freq


def if_tracking_error(
    true_if_list: list[NDArray],
    estimated_imfs: NDArray,
    fs: float,
    edge_trim: int = 10,
) -> dict:
    """Instantaneous-frequency tracking error between true components and IMFs.

    For each true component, the estimated IMF with lowest IF-RMSE is
    matched, and the RMSE of its instantaneous frequency against the
    true IF trajectory is reported.  Use ``edge_trim`` to ignore the
    boundary samples where Hilbert-based IF is noisy.

    Parameters
    ----------
    true_if_list : list of (T,) true instantaneous frequencies
    estimated_imfs : (K, T) estimated IMFs
    fs : float, sampling rate
    edge_trim : int, samples to ignore at each end

    Returns
    -------
    result : dict with keys
        - ``per_component_rmse`` : list of RMSE (Hz) for each true component
        - ``mean_rmse``          : average over components
        - ``max_rmse``           : worst-component RMSE
        - ``matched_imf_idx``    : which IMF index matched each true component
    """
    K_est = estimated_imfs.shape[0]
    est_ifs = np.stack([
        _inst_freq_from_signal(estimated_imfs[k], fs) for k in range(K_est)
    ])
    per_rmse = []
    matched = []
    for true_if in true_if_list:
        if edge_trim > 0:
            tif = true_if[edge_trim:-edge_trim]
        else:
            tif = true_if
        best_rmse = float("inf")
        best_k = -1
        for k in range(K_est):
            eif = est_ifs[k]
            if edge_trim > 0:
                eif = eif[edge_trim:-edge_trim]
            rmse = float(np.sqrt(np.mean((eif - tif) ** 2)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_k = k
        per_rmse.append(best_rmse)
        matched.append(best_k)
    per_rmse_arr = np.array(per_rmse)
    return {
        "per_component_rmse": per_rmse_arr.tolist(),
        "mean_rmse": float(per_rmse_arr.mean()),
        "max_rmse": float(per_rmse_arr.max()),
        "matched_imf_idx": matched,
    }


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def orthogonality_index(imfs: NDArray[np.float64]) -> float:
    """Compute the orthogonality index (OI) for a set of IMFs.

    OI = sum_{i!=j} |<IMF_i, IMF_j>| / sum_i ||IMF_i||^2

    A value close to 0 indicates near-orthogonal decomposition.

    Parameters
    ----------
    imfs : array of shape (K, N)
        Matrix of K IMFs, each of length N.

    Returns
    -------
    oi : float
    """
    K = imfs.shape[0]
    norms_sq = np.sum(imfs ** 2, axis=1)  # (K,)
    total_energy = np.sum(norms_sq)
    if total_energy < 1e-12:
        return 0.0

    cross = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            cross += np.abs(np.dot(imfs[i], imfs[j]))
    return float(2 * cross / total_energy)


def energy_ratio(signal: NDArray[np.float64], imfs: NDArray[np.float64]) -> float:
    """Energy preservation ratio.

    Ratio of total IMF energy to original signal energy.
    Ideally = 1.0 (perfect energy preservation under Parseval).

    Parameters
    ----------
    signal : array of shape (N,)
    imfs : array of shape (K, N)
        Includes residual as last row if applicable.

    Returns
    -------
    ratio : float
    """
    signal_energy = np.sum(signal ** 2)
    if signal_energy < 1e-12:
        return 1.0
    imf_energy = np.sum(imfs ** 2)
    return float(imf_energy / signal_energy)


def reconstruction_error(
    signal: NDArray[np.float64], imfs: NDArray[np.float64]
) -> float:
    """Normalized reconstruction error.

    ||x - sum(IMFs)|| / ||x||

    Parameters
    ----------
    signal : array of shape (N,)
    imfs : array of shape (K, N)

    Returns
    -------
    error : float
    """
    residual = signal - np.sum(imfs, axis=0)
    signal_norm = np.linalg.norm(signal)
    if signal_norm < 1e-12:
        return 0.0
    return float(np.linalg.norm(residual) / signal_norm)


def mode_mixing_index(
    true_components: list[NDArray[np.float64]],
    estimated_imfs: NDArray[np.float64],
) -> float:
    """Mode mixing index via correlation-based assignment.

    For each true component, find the IMF with highest absolute correlation.
    Mode mixing is indicated when multiple true components map to the same IMF
    or when the best-match correlation is low.

    Returns mean of (1 - |best_correlation|) across true components.
    A value close to 0 means clean separation; close to 1 means heavy mixing.

    Parameters
    ----------
    true_components : list of arrays, each of shape (N,)
    estimated_imfs : array of shape (K, N)

    Returns
    -------
    mmi : float
    """
    scores = []
    for comp in true_components:
        comp_norm = np.linalg.norm(comp)
        if comp_norm < 1e-12:
            continue
        best_corr = 0.0
        for k in range(estimated_imfs.shape[0]):
            imf_norm = np.linalg.norm(estimated_imfs[k])
            if imf_norm < 1e-12:
                continue
            corr = np.abs(np.dot(comp, estimated_imfs[k]) / (comp_norm * imf_norm))
            best_corr = max(best_corr, corr)
        scores.append(1.0 - best_corr)
    return float(np.mean(scores)) if scores else 0.0


def monotonicity_score(residual: NDArray[np.float64]) -> float:
    """Score how monotonic a residual signal is.

    Returns fraction of consecutive differences that have the same sign
    as the majority sign.  1.0 = perfectly monotonic.

    Parameters
    ----------
    residual : array of shape (N,)

    Returns
    -------
    score : float in [0, 1]
    """
    diffs = np.diff(residual)
    if len(diffs) == 0:
        return 1.0
    n_pos = np.sum(diffs > 0)
    n_neg = np.sum(diffs < 0)
    majority = max(n_pos, n_neg)
    total_nonzero = n_pos + n_neg
    if total_nonzero == 0:
        return 1.0
    return float(majority / total_nonzero)


# ---------------------------------------------------------------------------
# Tensor / numpy conversion helpers
# ---------------------------------------------------------------------------

def to_tensor(
    x: NDArray[np.float64] | torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert a numpy array to a PyTorch tensor (no-op if already a tensor)."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.from_numpy(np.asarray(x)).to(dtype)


def to_numpy(x: torch.Tensor | NDArray) -> NDArray:
    """Convert a PyTorch tensor to a numpy array (no-op if already numpy)."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
