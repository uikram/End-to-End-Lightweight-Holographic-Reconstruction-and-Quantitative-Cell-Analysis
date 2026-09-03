"""Output heads: quantitative phase, cell segmentation, drug condition."""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvBNAct


class PhaseHead(nn.Module):
    """Regresses quantitative phase in radians.

    The output is deliberately unbounded and unnormalised: the downstream
    measurement terms integrate it directly, so any squashing activation here
    would destroy the quantity being measured.
    """

    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.body = ConvBNAct(in_channels, hidden_channels, kernel_size=3)
        self.project = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.body(x))


class SegmentationHead(nn.Module):
    """Per-pixel class logits for background and cell."""

    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int):
        super().__init__()
        self.body = ConvBNAct(in_channels, hidden_channels, kernel_size=3)
        self.project = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.body(x))


class ConditionClassifier(nn.Module):
    """Image-level drug-condition prediction from the encoder bottleneck.

    The label is a property of the dish rather than of an individual cell, so it
    is predicted once per field of view and attributed to every cell segmented
    within it.
    """

    def __init__(self, in_channels: int, hidden: int, num_classes: int, dropout: float):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            # LayerNorm rather than BatchNorm1d: a trailing batch of one image is
            # normal here (small strata, batch_size 1 at full field size), and
            # BatchNorm1d cannot normalise a single sample in training mode.
            nn.LayerNorm(hidden),
            nn.ReLU6(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        return self.net(self.pool(bottleneck))


class AmplitudeHead(nn.Module):
    """Transmitted amplitude at the sample plane, for the forward model.

    A hologram is formed by a complex field, so synthesising one from the
    predicted phase alone assumes the specimen is purely refractive. Live cells
    scatter and absorb a little, and the illumination is never perfectly flat, so
    that assumption puts a floor on how well any forward-model term can fit the
    measurement.

    There is no amplitude ground truth, so this head is trained only by the
    forward-model residual -- exactly as in the self-supervised hologram
    reconstruction literature. It is initialised and biased toward unit
    transmittance so it starts from the pure-phase assumption and departs from it
    only where the measurement demands.
    """

    def __init__(self, in_channels: int, hidden_channels: int, deviation: float = 0.3):
        super().__init__()
        self.body = ConvBNAct(in_channels, hidden_channels, kernel_size=3)
        self.project = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)
        self.deviation = float(deviation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # tanh keeps transmittance in [1 - d, 1 + d]; zero-initialised so the
        # first forward pass returns exactly 1.0 everywhere.
        return 1.0 + self.deviation * torch.tanh(self.project(self.body(x)))
