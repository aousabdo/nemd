"""Training loop for N-EMD (filter-bank architecture, Phase 2.5b).

The filter-bank ``NEMD`` model uses a softmax partition of unity in the
frequency domain, so reconstruction and energy boundedness are architectural.
The loss has only three terms: filter sharpness, frequency ordering, and
orthogonality.  The softmax temperature is annealed from soft → sharp over
training.

Legacy: ``train_sifting`` (old iterative sifting loop) is retained for
ablation comparisons; it uses :class:`NEMDSifting` + :class:`NEMDSiftingLoss`.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nemd.losses import NEMDLoss, NEMDSiftingLoss
from nemd.model import NEMD, NEMDSifting, SiftNetConfig
from nemd.utils import (
    energy_ratio,
    generate_synthetic_signal,
    mode_mixing_index,
    monotonicity_score,
    orthogonality_index,
    reconstruction_error,
    to_numpy,
)


# ---------------------------------------------------------------------------
# Training configuration (filter-bank)
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Configuration for filter-bank N-EMD training."""

    # --- Data ---
    signal_length: int = 1024
    sample_rate: float = 1000.0
    batch_size: int = 32
    num_train_signals: int = 2000
    num_val_signals: int = 200
    min_components: int = 2
    max_components: int = 3
    freq_range: tuple[float, float] = (1.0, 200.0)
    snr_range_db: tuple[float, float] = (10.0, 40.0)
    # Optional per-component frequency bands (descending order) — if given,
    # one carrier per band is sampled per signal.
    freq_bands: tuple[tuple[float, float], ...] | None = None

    # --- Model ---
    num_imfs: int = 3
    hidden_dim: int = 128
    num_layers: int = 4
    kernel_size: int = 5

    # --- Loss weights ---
    lambda_sharp: float = 1.0
    lambda_order: float = 1.0
    lambda_ortho: float = 0.1
    lambda_balance: float = 5.0
    lambda_task: float = 0.0
    normalized_margin: float = 0.02
    balance_min_fraction: float | None = None
    # CentroidSeparationLoss sub-weights
    sep_w_order: float = 1.0
    sep_w_repel: float = 0.5
    sep_w_coverage: float = 0.3

    # --- Softmax temperature schedule ---
    tau_start: float = 2.0
    tau_end: float = 0.5
    tau_anneal_epochs: int = 40

    # --- Optimisation ---
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 100
    scheduler: str = "cosine"
    grad_clip_norm: float = 1.0

    # --- Reproducibility / I/O ---
    seed: int = 42
    log_interval: int = 10
    save_dir: str = "checkpoints"


# ---------------------------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------------------------

def generate_training_dataset(
    n_signals: int,
    config: TrainConfig,
    seed: int = 0,
    num_components: int | None = None,
    freq_bands: tuple[tuple[float, float], ...] | None = None,
) -> torch.Tensor:
    """Generate a batch of random multi-component AM-FM signals.

    Parameters
    ----------
    n_signals : int
    config : TrainConfig
    seed : int
    num_components : int or None
        If set, force exactly this many components per signal.  Otherwise
        sample in ``[config.min_components, config.max_components]``.
    freq_bands : tuple of (f_low, f_high), optional
        If set, sample one frequency per band.  Overrides ``config.freq_bands``.

    Returns
    -------
    signals : (n_signals, signal_length) float32 tensor
    """
    rng = np.random.default_rng(seed)
    signals = np.empty((n_signals, config.signal_length), dtype=np.float64)
    nyquist = config.sample_rate / 2
    bands = freq_bands if freq_bands is not None else config.freq_bands

    for i in range(n_signals):
        if num_components is not None:
            n_comp = int(num_components)
        elif bands is not None:
            n_comp = len(bands)
        else:
            n_comp = int(rng.integers(config.min_components, config.max_components + 1))

        if bands is not None:
            assert len(bands) >= n_comp, "Need at least n_comp frequency bands"
            freqs = np.array([
                rng.uniform(max(1.0, lo), min(hi, nyquist * 0.8))
                for lo, hi in bands[:n_comp]
            ])
        else:
            f_low, f_high = config.freq_range
            f_high = min(f_high, nyquist * 0.8)
            freqs = np.sort(rng.uniform(f_low, f_high, size=n_comp))[::-1]

        components = []
        for f0 in freqs:
            components.append({
                "f0": float(f0),
                "f_mod": float(rng.uniform(0.1, 5.0)),
                "a_mod": float(rng.uniform(0.0, 0.6)),
                "phase": float(rng.uniform(0, 2 * np.pi)),
                "freq_dev": float(f0 * rng.uniform(0.01, 0.15)),
            })

        snr_db = rng.uniform(*config.snr_range_db)
        _, clean, _ = generate_synthetic_signal(
            n_samples=config.signal_length,
            duration=config.signal_length / config.sample_rate,
            components=components,
            noise_std=0.0,
            seed=None,
        )
        signal_power = np.mean(clean ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = rng.normal(0, np.sqrt(noise_power), size=config.signal_length)
        signals[i] = clean + noise

    return torch.from_numpy(signals).float()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _temperature_schedule(epoch: int, config: TrainConfig) -> float:
    """Linear anneal from tau_start → tau_end over tau_anneal_epochs."""
    if epoch >= config.tau_anneal_epochs:
        return config.tau_end
    frac = epoch / max(config.tau_anneal_epochs, 1)
    return config.tau_start + frac * (config.tau_end - config.tau_start)


# ---------------------------------------------------------------------------
# Training loop — filter-bank N-EMD
# ---------------------------------------------------------------------------

_METRIC_KEYS = ("sharp", "order", "ortho", "balance")


def train(
    config: TrainConfig | None = None,
    verbose: bool = True,
    dataset_fn=None,
) -> dict:
    """Train a filter-bank N-EMD model from scratch.

    Parameters
    ----------
    config : TrainConfig
    verbose : bool
    dataset_fn : callable or None
        Optional custom dataset generator with signature
        ``dataset_fn(n_signals, config, seed) -> torch.Tensor (n, T)``.
        Overrides the default :func:`generate_training_dataset`.
        Used by Phase 3 experiments to inject nonstationary / multi-kind data.

    Returns a dict with ``model``, ``history``, ``config``.  Also saves
    ``best_model.pt`` and ``final_model.pt`` to ``config.save_dir``.
    """
    if config is None:
        config = TrainConfig()
    if dataset_fn is None:
        dataset_fn = generate_training_dataset

    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Model ----
    model = NEMD(
        num_imfs=config.num_imfs,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        kernel_size=config.kernel_size,
        sample_rate=config.sample_rate,
        temperature=config.tau_start,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Model parameters: {n_params:,}")

    # ---- Loss ----
    criterion = NEMDLoss(
        lambda_sharp=config.lambda_sharp,
        lambda_order=config.lambda_order,
        lambda_ortho=config.lambda_ortho,
        lambda_balance=config.lambda_balance,
        lambda_task=config.lambda_task,
        sample_rate=config.sample_rate,
        normalized_margin=config.normalized_margin,
        balance_min_fraction=config.balance_min_fraction,
        sep_w_order=config.sep_w_order,
        sep_w_repel=config.sep_w_repel,
        sep_w_coverage=config.sep_w_coverage,
    )

    # ---- Optimiser + scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.num_epochs,
        )
        if config.scheduler == "cosine" else None
    )

    # ---- Data ----
    if verbose:
        print("Generating training data ...")
    train_signals = dataset_fn(
        config.num_train_signals, config, seed=config.seed,
    )
    val_signals = dataset_fn(
        config.num_val_signals, config, seed=config.seed + 10_000,
    )
    train_loader = DataLoader(
        TensorDataset(train_signals), batch_size=config.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_signals), batch_size=config.batch_size, shuffle=False,
    )

    # ---- History ----
    history: dict[str, list] = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "lr": [], "tau": [],
    }
    for key in _METRIC_KEYS:
        history[key] = []

    best_val_loss = float("inf")
    best_state = None

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Training for {config.num_epochs} epochs on {device} ...")

    for epoch in range(1, config.num_epochs + 1):
        t0 = time.time()

        # -- Temperature schedule --
        tau = _temperature_schedule(epoch - 1, config)
        model.set_temperature(tau)

        # -- Train --
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            imfs, residual, metadata = model(batch)
            loss, _ = criterion(imfs, metadata)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # -- Validate --
        model.eval()
        val_loss_sum = 0.0
        val_components: dict[str, float] = {k: 0.0 for k in _METRIC_KEYS}
        n_val = 0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                imfs, residual, metadata = model(batch)
                loss, comps = criterion(imfs, metadata)
                val_loss_sum += loss.item()
                for k_ in val_components:
                    val_components[k_] += comps.get(k_, 0.0)
                n_val += 1

        avg_val_loss = val_loss_sum / max(n_val, 1)
        for k_ in val_components:
            val_components[k_] /= max(n_val, 1)

        # -- LR scheduler step --
        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # -- Record --
        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["lr"].append(current_lr)
        history["tau"].append(tau)
        for key in _METRIC_KEYS:
            history[key].append(val_components[key])

        # -- Best model (only after temperature stabilises) --
        if epoch >= config.tau_anneal_epochs and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(model.state_dict())

        # -- Logging --
        elapsed = time.time() - t0
        if verbose and (epoch % config.log_interval == 0 or epoch == 1):
            print(
                f"  Epoch {epoch:3d}/{config.num_epochs} | "
                f"τ={tau:.2f} | train {avg_train_loss:.4f} | "
                f"val {avg_val_loss:.4f} | "
                f"sharp {val_components['sharp']:.3f} | "
                f"order {val_components['order']:.4f} | "
                f"ortho {val_components['ortho']:.4f} | "
                f"bal {val_components['balance']:.4f} | "
                f"lr {current_lr:.2e} | {elapsed:.1f}s"
            )

    # ---- Save final state (last epoch) ----
    torch.save(model.state_dict(), save_dir / "final_model.pt")

    # ---- Restore best model ----
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), save_dir / "best_model.pt")

    with open(save_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2, default=str)
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    if verbose:
        print(f"Training complete.  Best val loss (post-anneal): {best_val_loss:.4f}")
        print(f"Saved to {save_dir}")

    return {"model": model, "history": history, "config": config}


# ---------------------------------------------------------------------------
# Single-signal overfit (sanity check) — filter bank
# ---------------------------------------------------------------------------

def overfit_single_signal(
    signal: torch.Tensor,
    num_imfs: int = 3,
    num_steps: int = 2000,
    lr: float = 1e-3,
    lambda_sharp: float = 1.0,
    lambda_order: float = 1.0,
    lambda_ortho: float = 0.1,
    lambda_balance: float = 5.0,
    normalized_margin: float = 0.02,
    tau_start: float = 2.0,
    tau_end: float = 0.5,
    tau_anneal_steps: int = 1000,
    hidden_dim: int = 128,
    num_layers: int = 4,
    fs: float = 1000.0,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Overfit a filter-bank N-EMD model on a single signal."""
    seed_everything(seed)
    device = signal.device

    model = NEMD(
        num_imfs=num_imfs,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        sample_rate=fs,
        temperature=tau_start,
    ).to(device)
    criterion = NEMDLoss(
        lambda_sharp=lambda_sharp,
        lambda_order=lambda_order,
        lambda_ortho=lambda_ortho,
        lambda_balance=lambda_balance,
        sample_rate=fs,
        normalized_margin=normalized_margin,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x = signal.unsqueeze(0)
    losses_history = []

    for step in range(1, num_steps + 1):
        frac = min(step / tau_anneal_steps, 1.0)
        tau = tau_start + frac * (tau_end - tau_start)
        model.set_temperature(tau)

        model.train()
        imfs, _, metadata = model(x)
        loss, comps = criterion(imfs, metadata)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses_history.append({**comps, "tau": tau})
        if verbose and (step % max(num_steps // 10, 1) == 0 or step == 1):
            print(
                f"  Step {step:4d} | τ={tau:.2f} | total {comps['total']:.4f} | "
                f"sharp {comps['sharp']:.3f} | order {comps['order']:.4f} | "
                f"ortho {comps['ortho']:.4f} | bal {comps['balance']:.4f}"
            )

    model.eval()
    with torch.no_grad():
        imfs, residual, metadata = model(x, temperature=tau_end)

    return {
        "model": model,
        "imfs": imfs.squeeze(0),
        "residual": residual.squeeze(0),
        "filters": metadata["filters"].squeeze(0),
        "centroids": metadata["centroids"].squeeze(0),
        "losses": losses_history,
    }


# ---------------------------------------------------------------------------
# Legacy: sifting-architecture training loop (Phase 2 / 2.5)
# ---------------------------------------------------------------------------

@dataclass
class TrainConfigSifting:
    """[Legacy] Config for the sifting N-EMD training loop (Phase 2 / 2.5).

    Retained for ablation and paper comparisons.  See ``train_sifting``.
    """
    signal_length: int = 1024
    sample_rate: float = 1000.0
    batch_size: int = 32
    num_train_signals: int = 5000
    num_val_signals: int = 500
    min_components: int = 2
    max_components: int = 5
    freq_range: tuple[float, float] = (1.0, 200.0)
    snr_range_db: tuple[float, float] = (5.0, 40.0)
    freq_bands_easy: tuple[tuple[float, float], ...] = (
        (80.0, 120.0), (30.0, 50.0), (5.0, 15.0),
    )
    use_curriculum: bool = True
    curriculum_k_start: int = 2
    curriculum_k_end: int = 3
    curriculum_switch_epoch: int = 20
    easy_data_epochs: int = 30
    max_imfs: int = 6
    num_levels: int = 3
    channels: tuple[int, ...] = (32, 64, 128)
    bottleneck_channels: int = 256
    kernel_sizes: tuple[int, ...] = (7, 5, 3)
    scale_embed_dim: int = 64
    use_init_filter: bool = True
    init_filter_taps: int = 31
    lambda_ortho: float = 1.0
    lambda_narrow: float = 0.1
    lambda_mono: float = 0.5
    lambda_residual: float = 0.1
    lambda_energy: float = 0.1
    lambda_order: float = 2.0
    lambda_concentration: float = 0.1
    lambda_task: float = 0.0
    normalized_margin: float = 0.02
    narrow_anneal_start: float = 0.01
    narrow_anneal_end: float = 0.1
    narrow_anneal_epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 100
    scheduler: str = "cosine"
    grad_clip_norm: float = 1.0
    seed: int = 42
    log_interval: int = 10
    save_dir: str = "checkpoints_sifting"

    def to_sift_config(self) -> SiftNetConfig:
        return SiftNetConfig(
            num_levels=self.num_levels,
            channels=list(self.channels),
            bottleneck_channels=self.bottleneck_channels,
            kernel_sizes=list(self.kernel_sizes),
            max_imfs=self.max_imfs,
            scale_embed_dim=self.scale_embed_dim,
            use_init_filter=self.use_init_filter,
            init_filter_taps=self.init_filter_taps,
        )
