"""Classifier architectures for Phase 3 Exp 3.

- :class:`FeatureMLP` takes a feature vector and produces class logits.
- :class:`RawSignalCNN` takes a raw signal and produces class logits
  (no-decomposition baseline).
- :class:`NEMDClassifier` wraps a filter-bank N-EMD model +
  differentiable feature extractor + MLP for end-to-end training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemd.model import NEMD
from experiments.applications.classification.features import imf_features


class FeatureMLP(nn.Module):
    """Simple 2-hidden-layer MLP for (batch, D) → (batch, n_classes)."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, n_classes: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RawSignalCNN(nn.Module):
    """1D CNN baseline: raw signal → class logits (no decomposition)."""

    def __init__(self, n_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(4),                                     # 1024 → 256
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(4),                                     # 256 → 64
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),                             # 64 → 1
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) → (B, 1, T) → CNN → head
        h = self.features(x.unsqueeze(1))
        return self.head(h)


class ModeCNN(nn.Module):
    """1D CNN operating on a K-channel mode stack: (B, K, T) -> class logits.

    Capacity-matched to :class:`RawSignalCNN` so downstream comparisons isolate
    the effect of the decomposition (``n_channels`` differs only in the first
    conv). This is the head used for the ``*_cnn`` pipelines that replace the
    16-stat MLP bottleneck.
    """

    def __init__(self, n_channels: int = 4, n_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, K, T)
        h = self.features(x)
        return self.head(h)


class NEMDClassifierCNN(nn.Module):
    """End-to-end: signal -> N-EMD filter bank -> K-channel stack -> ModeCNN.

    Unlike :class:`NEMDClassifier` this does not summarise modes to 3 stats per
    IMF; the full time-resolved mode stack is forwarded to a CNN head that
    learns its own features. This removes the feature-extraction bottleneck
    so the decomposition quality can actually influence the downstream score.
    """

    def __init__(
        self,
        nemd_model: NEMD,
        n_classes: int = 3,
        sample_rate: float = 100.0,
        temperature: float = 0.5,
        sort_by_centroid: bool = True,
    ) -> None:
        super().__init__()
        self.nemd = nemd_model
        self.sample_rate = sample_rate
        self.temperature = temperature
        self.sort_by_centroid = sort_by_centroid
        self.classifier = ModeCNN(n_channels=nemd_model.num_imfs, n_classes=n_classes)

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        imfs, _, metadata = self.nemd(
            x, temperature=self.temperature, sort_by_centroid=self.sort_by_centroid,
        )
        logits = self.classifier(imfs)
        return logits, imfs, metadata


class NEMDClassifier(nn.Module):
    """End-to-end: signal → N-EMD filter bank → features → MLP → logits.

    All operations are differentiable so the filter bank learns
    decompositions that maximise downstream classification accuracy
    (subject to a light physics regulariser).
    """

    def __init__(
        self,
        nemd_model: NEMD,
        n_classes: int = 3,
        mlp_hidden: int = 64,
        sample_rate: float = 1000.0,
        temperature: float = 0.3,
        sort_by_centroid: bool = True,
    ) -> None:
        super().__init__()
        self.nemd = nemd_model
        self.sample_rate = sample_rate
        self.temperature = temperature
        self.sort_by_centroid = sort_by_centroid
        self.feature_dim = 3 * nemd_model.num_imfs
        self.classifier = FeatureMLP(
            in_dim=self.feature_dim, hidden_dim=mlp_hidden, n_classes=n_classes,
        )

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Return (logits, features, metadata).

        ``metadata`` is the N-EMD forward output (filters, centroids, ...).
        Returning it lets the training loop add physics losses on top.
        """
        imfs, _, metadata = self.nemd(
            x, temperature=self.temperature, sort_by_centroid=self.sort_by_centroid,
        )
        feats = imf_features(imfs, sample_rate=self.sample_rate)
        logits = self.classifier(feats)
        return logits, feats, metadata
