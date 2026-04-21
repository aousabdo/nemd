"""Training / evaluation pipelines for Phase 3 Experiment 3.

Five approaches compared:

  1. raw_cnn         : raw signal → CNN → logits (no decomposition)
  2. emd_mlp         : EMD (fixed) → features → MLP
  3. vmd_mlp         : VMD (fixed) → features → MLP
  4. nemd_pretrained : N-EMD from Phase 2.5b+ checkpoint, fine-tuned end-to-end
  5. nemd_scratch    : N-EMD from random init, trained end-to-end with task loss

All use the same classifier capacity, same optimiser family, same loss
(cross-entropy, plus a light physics regulariser for N-EMD pipelines).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from nemd.classical import ClassicalEMD, VMD
from nemd.losses import NEMDLoss
from nemd.model import NEMD
from nemd.train import TrainConfig
from experiments.applications.classification.features import imf_features
from experiments.applications.classification.models import (
    FeatureMLP, NEMDClassifier, RawSignalCNN,
)


# ---------------------------------------------------------------------------
# Feature caching for fixed-preprocessing pipelines (EMD / VMD)
# ---------------------------------------------------------------------------

def _decompose_dataset_classical(
    signals_np: np.ndarray,
    method: str,
    num_imfs: int,
    fs: float,
) -> np.ndarray:
    """Run EMD or VMD on a batch of signals; return (N, K, T)."""
    N, T = signals_np.shape
    out = np.zeros((N, num_imfs, T), dtype=np.float64)
    if method == "emd":
        decomposer = ClassicalEMD(max_imfs=num_imfs + 1)
        for i in range(N):
            t = np.arange(T) / fs
            imfs = decomposer.decompose(signals_np[i], t)
            # Keep first num_imfs (drop residual)
            k = min(num_imfs, imfs.shape[0])
            out[i, :k] = imfs[:k]
    elif method == "vmd":
        decomposer = VMD(n_modes=num_imfs)
        for i in range(N):
            modes = decomposer.decompose(signals_np[i])
            out[i] = modes
    else:
        raise ValueError(method)
    return out


def cache_features_classical(
    signals: torch.Tensor,
    method: str,
    num_imfs: int,
    fs: float,
    verbose: bool = False,
) -> torch.Tensor:
    """Run EMD/VMD once on the dataset and return the precomputed feature
    tensor (N, 3*K).  Per-signal time is the primary speed metric."""
    if verbose:
        print(f"  Running {method.upper()} on {len(signals)} signals ...")
    t0 = time.perf_counter()
    signals_np = signals.cpu().numpy()
    imfs_np = _decompose_dataset_classical(signals_np, method, num_imfs, fs)
    elapsed = time.perf_counter() - t0
    imfs_tensor = torch.from_numpy(imfs_np).float()
    feats = imf_features(imfs_tensor, sample_rate=fs)
    if verbose:
        print(f"    {method.upper()} total: {elapsed:.2f}s "
              f"({elapsed * 1000 / len(signals):.2f} ms/signal)")
    return feats.detach(), elapsed


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate_features(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    """Accuracy + cross-entropy for a feature-based classifier."""
    model.eval()
    with torch.no_grad():
        logits = model(features.to(device))
        preds = logits.argmax(dim=-1).cpu()
    acc = (preds == labels).float().mean().item()
    loss = F.cross_entropy(logits, labels.to(device)).item()
    return acc, loss


def _evaluate_raw(
    model: nn.Module,
    signals: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int = 128,
) -> tuple[float, float]:
    """Accuracy + cross-entropy for the raw-signal CNN."""
    model.eval()
    all_preds = []
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for i in range(0, len(signals), batch_size):
            x = signals[i:i + batch_size].to(device)
            y = labels[i:i + batch_size].to(device)
            logits = model(x)
            total_loss += F.cross_entropy(logits, y, reduction="sum").item()
            all_preds.append(logits.argmax(dim=-1).cpu())
            n += len(x)
    preds = torch.cat(all_preds)
    acc = (preds == labels).float().mean().item()
    return acc, total_loss / n


def _evaluate_nemd(
    model: NEMDClassifier,
    signals: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[float, float]:
    """Accuracy + CE for the end-to-end N-EMD classifier."""
    model.eval()
    all_preds = []
    total_loss = 0.0
    with torch.no_grad():
        for i in range(0, len(signals), batch_size):
            x = signals[i:i + batch_size].to(device)
            y = labels[i:i + batch_size].to(device)
            logits, _, _ = model(x)
            total_loss += F.cross_entropy(logits, y, reduction="sum").item()
            all_preds.append(logits.argmax(dim=-1).cpu())
    preds = torch.cat(all_preds)
    acc = (preds == labels).float().mean().item()
    return acc, total_loss / len(signals)


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    name: str
    train_acc_curve: list[float] = field(default_factory=list)
    val_acc_curve: list[float] = field(default_factory=list)
    train_loss_curve: list[float] = field(default_factory=list)
    val_loss_curve: list[float] = field(default_factory=list)
    test_acc: float = 0.0
    test_loss: float = 0.0
    wallclock_sec: float = 0.0
    n_params: int = 0
    # Confusion matrix (3x3) on the test set
    confusion: list[list[int]] = field(default_factory=list)
    test_preds: list[int] = field(default_factory=list)
    test_labels: list[int] = field(default_factory=list)


def train_feature_pipeline(
    name: str,
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    val_feats: torch.Tensor,
    val_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    n_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    seed: int = 42,
    device: torch.device | None = None,
    verbose: bool = True,
) -> PipelineResult:
    """Train an MLP on pre-computed features (EMD+MLP or VMD+MLP)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed)

    in_dim = train_feats.shape[-1]
    model = FeatureMLP(in_dim=in_dim, hidden_dim=64, n_classes=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    result = PipelineResult(name=name)
    result.n_params = sum(p.numel() for p in model.parameters())

    loader = DataLoader(
        TensorDataset(train_feats, train_labels),
        batch_size=batch_size, shuffle=True,
    )
    t0 = time.perf_counter()
    best_val_acc = 0.0
    best_state = None
    for epoch in range(1, n_epochs + 1):
        model.train()
        total = 0.0
        correct = 0
        n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(xb)
            correct += (logits.argmax(dim=-1) == yb).sum().item()
            n += len(xb)

        train_loss = total / n
        train_acc = correct / n
        val_acc, val_loss = _evaluate_features(model, val_feats, val_labels, device)

        result.train_loss_curve.append(train_loss)
        result.train_acc_curve.append(train_acc)
        result.val_loss_curve.append(val_loss)
        result.val_acc_curve.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"    [{name}] ep {epoch:3d}/{n_epochs} "
                  f"train_loss {train_loss:.3f} train_acc {train_acc:.3f} "
                  f"val_acc {val_acc:.3f}")

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc, test_loss = _evaluate_features(model, test_feats, test_labels, device)
    result.test_acc = test_acc
    result.test_loss = test_loss
    result.wallclock_sec = time.perf_counter() - t0

    # Confusion
    model.eval()
    with torch.no_grad():
        preds = model(test_feats.to(device)).argmax(dim=-1).cpu()
    result.test_preds = preds.tolist()
    result.test_labels = test_labels.tolist()
    result.confusion = _confusion_matrix(test_labels, preds, n_classes=3)
    return result


def train_raw_cnn(
    name: str,
    train_signals: torch.Tensor, train_labels: torch.Tensor,
    val_signals: torch.Tensor, val_labels: torch.Tensor,
    test_signals: torch.Tensor, test_labels: torch.Tensor,
    n_epochs: int = 50, lr: float = 1e-3, weight_decay: float = 1e-4,
    batch_size: int = 64, seed: int = 42,
    device: torch.device | None = None, verbose: bool = True,
) -> PipelineResult:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed)
    model = RawSignalCNN(n_classes=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    result = PipelineResult(name=name)
    result.n_params = sum(p.numel() for p in model.parameters())

    loader = DataLoader(
        TensorDataset(train_signals, train_labels),
        batch_size=batch_size, shuffle=True,
    )
    t0 = time.perf_counter()
    best_val_acc = 0.0
    best_state = None
    for epoch in range(1, n_epochs + 1):
        model.train()
        total = 0.0; correct = 0; n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(xb)
            correct += (logits.argmax(dim=-1) == yb).sum().item()
            n += len(xb)
        train_loss = total / n
        train_acc = correct / n
        val_acc, val_loss = _evaluate_raw(model, val_signals, val_labels, device)
        result.train_loss_curve.append(train_loss)
        result.train_acc_curve.append(train_acc)
        result.val_loss_curve.append(val_loss)
        result.val_acc_curve.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"    [{name}] ep {epoch:3d}/{n_epochs} train_loss {train_loss:.3f} "
                  f"train_acc {train_acc:.3f} val_acc {val_acc:.3f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc, test_loss = _evaluate_raw(model, test_signals, test_labels, device)
    result.test_acc = test_acc
    result.test_loss = test_loss
    result.wallclock_sec = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        all_preds = []
        for i in range(0, len(test_signals), 128):
            all_preds.append(
                model(test_signals[i:i+128].to(device)).argmax(-1).cpu()
            )
    preds = torch.cat(all_preds)
    result.test_preds = preds.tolist()
    result.test_labels = test_labels.tolist()
    result.confusion = _confusion_matrix(test_labels, preds, n_classes=3)
    return result


def train_nemd_end_to_end(
    name: str,
    train_signals: torch.Tensor, train_labels: torch.Tensor,
    val_signals: torch.Tensor, val_labels: torch.Tensor,
    test_signals: torch.Tensor, test_labels: torch.Tensor,
    nemd_model: NEMD,
    sample_rate: float = 1000.0,
    n_epochs: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    physics_weight: float = 0.1,
    seed: int = 42,
    device: torch.device | None = None,
    verbose: bool = True,
) -> PipelineResult:
    """Train N-EMD + MLP jointly (end-to-end) with CE + physics loss."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed)

    classifier = NEMDClassifier(
        nemd_model=nemd_model, n_classes=3, sample_rate=sample_rate,
    ).to(device)

    physics_loss = NEMDLoss(
        lambda_sharp=1.0, lambda_order=1.0, lambda_ortho=0.1, lambda_balance=5.0,
        sample_rate=sample_rate, normalized_margin=0.02,
    )
    opt = torch.optim.Adam(
        classifier.parameters(), lr=lr, weight_decay=weight_decay,
    )

    result = PipelineResult(name=name)
    result.n_params = sum(p.numel() for p in classifier.parameters())

    loader = DataLoader(
        TensorDataset(train_signals, train_labels),
        batch_size=batch_size, shuffle=True,
    )
    t0 = time.perf_counter()
    best_val_acc = 0.0
    best_state = None
    for epoch in range(1, n_epochs + 1):
        classifier.train()
        total = 0.0; correct = 0; n = 0
        total_phys = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, _, metadata = classifier(xb)
            ce = F.cross_entropy(logits, yb)
            # imfs not returned directly; re-derive from metadata via another forward?
            # Instead compute physics loss on the raw N-EMD output
            # We need imfs for OrthogonalityLoss — easier to call N-EMD directly here
            imfs, _, _ = classifier.nemd(
                xb, temperature=classifier.temperature,
                sort_by_centroid=classifier.sort_by_centroid,
            )
            phys, _ = physics_loss(imfs, metadata)
            loss = ce + physics_weight * phys
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            opt.step()
            total += ce.item() * len(xb)
            total_phys += phys.item() * len(xb)
            correct += (logits.argmax(dim=-1) == yb).sum().item()
            n += len(xb)

        train_loss = total / n
        train_acc = correct / n
        val_acc, val_loss = _evaluate_nemd(
            classifier, val_signals, val_labels, device,
        )
        result.train_loss_curve.append(train_loss)
        result.train_acc_curve.append(train_acc)
        result.val_loss_curve.append(val_loss)
        result.val_acc_curve.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(classifier.state_dict())
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"    [{name}] ep {epoch:3d}/{n_epochs} "
                  f"CE {train_loss:.3f} phys {total_phys / n:.3f} "
                  f"train_acc {train_acc:.3f} val_acc {val_acc:.3f}")

    if best_state is not None:
        classifier.load_state_dict(best_state)
    test_acc, test_loss = _evaluate_nemd(
        classifier, test_signals, test_labels, device,
    )
    result.test_acc = test_acc
    result.test_loss = test_loss
    result.wallclock_sec = time.perf_counter() - t0

    classifier.eval()
    with torch.no_grad():
        all_preds = []
        for i in range(0, len(test_signals), 64):
            logits, _, _ = classifier(test_signals[i:i+64].to(device))
            all_preds.append(logits.argmax(-1).cpu())
    preds = torch.cat(all_preds)
    result.test_preds = preds.tolist()
    result.test_labels = test_labels.tolist()
    result.confusion = _confusion_matrix(test_labels, preds, n_classes=3)

    # Stash the trained NEMDClassifier on the result so callers can pull
    # out filter responses for the final figure.
    result.model = classifier  # type: ignore[attr-defined]
    return result


# ---------------------------------------------------------------------------
# Confusion matrix helper
# ---------------------------------------------------------------------------

def _confusion_matrix(true, pred, n_classes: int) -> list[list[int]]:
    if isinstance(true, torch.Tensor):
        true = true.tolist()
    if isinstance(pred, torch.Tensor):
        pred = pred.tolist()
    cm = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(true, pred):
        cm[int(t)][int(p)] += 1
    return cm


def macro_f1(confusion: list[list[int]]) -> float:
    """Macro-averaged F1 from a confusion matrix (rows=true, cols=pred)."""
    import numpy as np
    cm = np.array(confusion)
    n_classes = cm.shape[0]
    f1s = []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        f1s.append(f1)
    return float(np.mean(f1s))
