"""Classical EMD / EEMD / VMD baselines for comparison.

Thin wrappers around established libraries (EMD-signal, vmdpy) that expose
a uniform interface for benchmarking against N-EMD.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class ClassicalEMD:
    """Wrapper around PyEMD's EMD implementation.

    Parameters
    ----------
    max_imfs : int or None
        Maximum number of IMFs to extract.  *None* = library default.
    spline_kind : str
        Spline type for envelope fitting (``"cubic"``).
    """

    def __init__(
        self,
        max_imfs: int | None = None,
        spline_kind: str = "cubic",
    ) -> None:
        self.max_imfs = max_imfs
        self.spline_kind = spline_kind

    def decompose(
        self, signal: NDArray[np.float64], t: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Run standard EMD and return IMFs (last row is the residual).

        Parameters
        ----------
        signal : array of shape (N,)
        t : array of shape (N,) or None
            Time vector.  If *None*, uses integer indices.

        Returns
        -------
        imfs : array of shape (K, N)
            Rows are IMFs from highest to lowest frequency; the last row
            is the residual (monotonic trend).
        """
        from PyEMD import EMD as _EMD

        emd = _EMD(spline_kind=self.spline_kind)
        if self.max_imfs is not None:
            emd.MAX_ITERATION = 2000
        if t is None:
            t = np.arange(len(signal), dtype=np.float64)
        imfs = emd.emd(signal.astype(np.float64), t)
        if self.max_imfs is not None and imfs.shape[0] > self.max_imfs:
            # Merge excess IMFs into residual
            residual = np.sum(imfs[self.max_imfs - 1 :], axis=0)
            imfs = np.vstack([imfs[: self.max_imfs - 1], residual[np.newaxis, :]])
        return imfs


class EnsembleEMD:
    """Wrapper around PyEMD's EEMD implementation.

    Parameters
    ----------
    n_trials : int
        Number of noise-assisted trials.
    noise_width : float
        Standard deviation of the added white noise.
    max_imfs : int or None
        Maximum number of IMFs.
    """

    def __init__(
        self,
        n_trials: int = 100,
        noise_width: float = 0.05,
        max_imfs: int | None = None,
    ) -> None:
        self.n_trials = n_trials
        self.noise_width = noise_width
        self.max_imfs = max_imfs

    def decompose(
        self, signal: NDArray[np.float64], t: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Run EEMD and return IMFs."""
        from PyEMD import EEMD as _EEMD

        eemd = _EEMD(trials=self.n_trials, noise_width=self.noise_width)
        if t is None:
            t = np.arange(len(signal), dtype=np.float64)
        imfs = eemd.eemd(signal.astype(np.float64), t)
        if self.max_imfs is not None and imfs.shape[0] > self.max_imfs:
            residual = np.sum(imfs[self.max_imfs - 1 :], axis=0)
            imfs = np.vstack([imfs[: self.max_imfs - 1], residual[np.newaxis, :]])
        return imfs


class VMD:
    """Wrapper around vmdpy's VMD implementation.

    Parameters
    ----------
    n_modes : int
        Number of modes (K) to extract.
    alpha : float
        Bandwidth constraint parameter.
    tau : float
        Noise-tolerance (time-step of dual ascent).
    tol : float
        Convergence tolerance.
    """

    def __init__(
        self,
        n_modes: int = 3,
        alpha: float = 2000.0,
        tau: float = 0.0,
        tol: float = 1e-7,
    ) -> None:
        self.n_modes = n_modes
        self.alpha = alpha
        self.tau = tau
        self.tol = tol

    def decompose(
        self, signal: NDArray[np.float64], t: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Run VMD and return modes.

        Note: VMD does not produce a monotonic residual — the returned
        modes sum to the original signal by construction.
        """
        from vmdpy import VMD as _VMD

        u, _, _ = _VMD(
            signal.astype(np.float64),
            self.alpha,
            self.tau,
            self.n_modes,
            0,   # DC — no imposed DC
            1,   # init — uniform center frequencies
            self.tol,
        )
        return np.array(u)
