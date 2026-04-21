"""Regenerate the crossing-chirps comparison figure with synchrosqueezing.

Addresses reviewer 2 item J: the crossing-chirps example in Fig. 3
shows that frequency-partitioning decompositions (EMD, VMD, N-EMD)
split each crossing chirp across two band passes. The paper claims
synchrosqueezing is the right tool for this class of signals; this
script produces the actual synchrosqueezing result side by side so the
claim is supported visually.

Output:  paper/nemd_tsp/figs/crossing_chirps_with_ssq.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from ssqueezepy import ssq_cwt

from nemd.classical import ClassicalEMD, VMD
from nemd.model import NEMD
from nemd.train import TrainConfig
import torch


REPO = Path(__file__).resolve().parents[2]
OUT  = REPO / "paper" / "nemd_tsp" / "figs" / "crossing_chirps_with_ssq.png"

mpl.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["STIXGeneral", "Computer Modern", "Times"],
    "mathtext.fontset": "stix",
    "axes.labelsize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "axes.titlesize":    8.5,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})


def build_crossing_chirps(T: int = 1024, fs: float = 1000.0, seed: int = 0):
    """Two chirps that cross in frequency around t = T/2."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) / fs
    # Chirp 1: 60 Hz -> 10 Hz
    # Chirp 2: 10 Hz -> 60 Hz
    f1 = 60.0 + (10.0 - 60.0) * t / t[-1]
    f2 = 10.0 + (60.0 - 10.0) * t / t[-1]
    # Instantaneous phase = integral of 2*pi*f(t) dt
    phi1 = 2 * np.pi * np.cumsum(f1) / fs
    phi2 = 2 * np.pi * np.cumsum(f2) / fs
    c1 = np.cos(phi1)
    c2 = np.cos(phi2)
    # Low-frequency carrier
    c3 = 0.8 * np.cos(2 * np.pi * 3.0 * t)
    signal = c1 + c2 + c3
    signal += rng.normal(0, 0.05, size=T)
    return t, signal.astype(np.float32), [c1, c2, c3]


def load_nemd(ckpt: str, fs: float) -> NEMD:
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg: TrainConfig = blob["config"]
    m = NEMD(num_imfs=cfg.num_imfs, hidden_dim=cfg.hidden_dim,
             num_layers=cfg.num_layers, kernel_size=cfg.kernel_size,
             sample_rate=fs, temperature=0.3)
    m.load_state_dict(blob["state_dict"])
    return m.eval()


def main() -> None:
    T, fs = 1024, 1000.0
    t, sig, truth = build_crossing_chirps(T=T, fs=fs, seed=0)

    # EMD / VMD / N-EMD decompositions
    emd_imfs = ClassicalEMD(max_imfs=4).decompose(sig, t)[:3]
    vmd_imfs = VMD(n_modes=3).decompose(sig)

    nemd = load_nemd(str(REPO / "checkpoints_p25b_v3" / "final.pt"), fs)
    with torch.no_grad():
        imfs_t, _, _ = nemd(torch.from_numpy(sig).unsqueeze(0),
                            temperature=0.3, sort_by_centroid=True)
    nemd_imfs = imfs_t.squeeze(0).cpu().numpy()

    # Synchrosqueezed CWT. ssqueezepy's ssq_cwt maps low indices to high
    # frequency by default; we pass fs and flip the vertical axis so low
    # frequencies sit at the bottom of the image.
    from ssqueezepy import Wavelet
    wavelet = Wavelet(('gmw', {'beta': 60, 'gamma': 3}))
    ssq_coef, _, ssq_freqs, _ = ssq_cwt(sig, wavelet=wavelet, fs=fs)
    ssq_mag = np.abs(ssq_coef)
    # Log-scale to compress dynamic range; normalise per column
    ssq_log = np.log10(ssq_mag + 1e-8)
    ssq_log -= ssq_log.max()
    ssq_log = np.clip(ssq_log, -3.0, 0.0)

    # Fig: 2x4 grid
    #   Row 1: Signal | EMD IMFs (as spectrogram of summed) ... etc
    # Simpler: 4 columns (truth | EMD | VMD | N-EMD), 3 IMF rows + top signal row
    # Plus one wide row for synchrosqueeze

    fig = plt.figure(figsize=(7.0, 5.0))
    gs = fig.add_gridspec(5, 4, height_ratios=[0.9, 1, 1, 1, 1.3],
                          hspace=0.45, wspace=0.28)

    # Top row: input signal (all 4 cols with same signal for reference)
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(t, sig, color="black", linewidth=0.7)
    ax0.set_xlim(0, t[-1])
    ax0.set_ylabel("signal")
    ax0.set_xticklabels([])
    ax0.set_title(
        "Input: two crossing chirps (60$\\to$10 Hz and 10$\\to$60 Hz) "
        "plus a 3 Hz tone",
        fontsize=8.5,
    )

    # IMF rows (3 rows x 4 columns): truth, EMD, VMD, N-EMD
    sources = [("truth",  truth),
               ("EMD",    emd_imfs),
               ("VMD",    vmd_imfs),
               ("N-EMD",  nemd_imfs)]
    colors = {"truth": "#338833", "EMD": "#888888",
              "VMD": "#4477aa", "N-EMD": "#cc3344"}
    for col, (name, imfs) in enumerate(sources):
        for row in range(3):
            ax = fig.add_subplot(gs[1 + row, col])
            if row < len(imfs):
                y = np.asarray(imfs[row])
                ax.plot(t, y, color=colors[name], linewidth=0.7)
                ax.set_xlim(0, t[-1])
                ymax = np.max(np.abs(y)) * 1.1 + 1e-6
                ax.set_ylim(-ymax, ymax)
            ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(f"IMF {row+1}")
            if row == 0:
                ax.set_title(name, fontweight="bold")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)

    # Bottom row: synchrosqueezed TF picture spans all columns
    ax_ssq = fig.add_subplot(gs[4, :])
    # ssqueezepy returns frequencies sorted high -> low along axis 0,
    # so we filter/flip so the image reads low->high bottom->top.
    order = np.argsort(ssq_freqs)
    f_sorted = ssq_freqs[order]
    mag_sorted = ssq_log[order]
    mask = f_sorted <= 100.0
    ax_ssq.imshow(
        mag_sorted[mask], origin="lower", aspect="auto",
        extent=[0, t[-1], float(f_sorted[mask].min()),
                float(f_sorted[mask].max())],
        cmap="viridis", vmin=-3.0, vmax=0.0,
    )
    ax_ssq.set_xlabel("Time (s)")
    ax_ssq.set_ylabel("Frequency (Hz)")
    ax_ssq.set_title(
        "Synchrosqueezed wavelet transform: the two crossing chirps "
        "appear as single continuous ridges",
        fontsize=8.5,
    )

    fig.savefig(OUT)
    plt.close(fig)
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
