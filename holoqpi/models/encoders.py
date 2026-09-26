"""Lightweight encoders adapted to single-channel hologram input.

Every encoder exposes the same contract: ``forward`` returns a list of feature
maps ordered shallow to deep, and ``out_channels`` describes their widths. The
first convolution is rebuilt for the requested number of input channels; when
ImageNet weights are used, the RGB filters are averaged across the colour axis
so the pretrained spatial-frequency response is preserved rather than discarded.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ..utils import get_logger

LOGGER = get_logger(__name__)


# Where to look for a locally supplied state dict, before touching the network.
# Set by build_encoder from model.pretrained_dir; a directory, not a file, so
# one setting covers every backbone.
_LOCAL_WEIGHT_DIR: Path | None = None

# torchvision's canonical filenames, so a file downloaded once by hand is found
# without the caller having to name it.
_LOCAL_WEIGHT_FILES = {
    "mobilenet_v2": ("mobilenet_v2-b0353104.pth",),
    "mobilenet_v3_large": ("mobilenet_v3_large-8738ca79.pth",
                           "mobilenet_v3_large-5c1a4163.pth"),
    "resnet18": ("resnet18-f37072fd.pth", "resnet18-5c106cde.pth"),
}


def _local_state_dict(name: str):
    """A state dict from ``model.pretrained_dir``, or None if there isn't one."""
    if _LOCAL_WEIGHT_DIR is None:
        return None
    for filename in _LOCAL_WEIGHT_FILES.get(name, ()):
        candidate = Path(_LOCAL_WEIGHT_DIR) / filename
        if candidate.is_file():
            LOGGER.info("loading pretrained %s from %s", name, candidate)
            return torch.load(candidate, map_location="cpu", weights_only=True)
    LOGGER.warning(
        "model.pretrained_dir is %s but it holds none of %s for %s. Falling back to "
        "the download cache.",
        _LOCAL_WEIGHT_DIR, list(_LOCAL_WEIGHT_FILES.get(name, ())), name,
    )
    return None


def _load_backbone(builder, weights, name: str):
    """Instantiate a torchvision backbone, tolerating an unreachable weight cache.

    THREE ROUTES, IN THIS ORDER, and the order is the point.

    1. A state dict under ``model.pretrained_dir``. On a machine with no
       outbound network -- or behind a proxy that blocks download.pytorch.org --
       this is the only route that works, and it is the one to use when the
       weights have been fetched once and committed alongside the code.
    2. torchvision's own download cache.
    3. Random initialisation, with a warning.

    Route 3 is a silent experiment-ruiner if it is not noticed: every arm then
    trains from scratch and the encoder comparison means something different
    from what the paper says. So it warns loudly, and ``build_encoder`` repeats
    the outcome in its own log line, because that line is the one people read.
    """
    if weights is None:
        return builder(weights=None)

    state = _local_state_dict(name)
    if state is not None:
        model = builder(weights=None)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            LOGGER.warning(
                "%s: loaded local weights with %d missing and %d unexpected keys",
                name, len(missing), len(unexpected),
            )
        return model

    try:
        return builder(weights=weights)
    except Exception as exc:
        LOGGER.warning(
            "could not obtain pretrained weights for %s (%s), and no local copy was "
            "found. FALLING BACK TO RANDOM INITIALISATION -- every arm will train "
            "from scratch, which is not what the configuration asked for. Put "
            "%s under model.pretrained_dir, or set model.pretrained_encoder=false "
            "to make the choice deliberate.",
            name, type(exc).__name__,
            _LOCAL_WEIGHT_FILES.get(name, ("the torchvision checkpoint",))[0],
        )
        return builder(weights=None)


def _adapt_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Rebuild a Conv2d for ``in_channels``, averaging the pretrained filters."""
    if conv.in_channels == in_channels:
        return conv

    adapted = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        mean_filter = conv.weight.data.mean(dim=1, keepdim=True)
        adapted.weight.data = mean_filter.repeat(1, in_channels, 1, 1) / in_channels
        if conv.bias is not None:
            adapted.bias.data = conv.bias.data.clone()
    return adapted


class MobileNetV2Encoder(nn.Module):
    """MobileNetV2 trunk split at the five standard downsampling stages."""

    def __init__(self, in_channels: int, pretrained: bool):
        super().__init__()
        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        features = _load_backbone(mobilenet_v2, weights, "mobilenet_v2").features
        features[0][0] = _adapt_first_conv(features[0][0], in_channels)

        self.stage0 = features[0:2]     # stride 2
        self.stage1 = features[2:4]     # stride 4
        self.stage2 = features[4:7]     # stride 8
        self.stage3 = features[7:14]    # stride 16
        self.stage4 = features[14:]     # stride 32
        self.out_channels = [16, 24, 32, 96, 1280]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f0 = self.stage0(x)
        f1 = self.stage1(f0)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f0, f1, f2, f3, f4]


class MobileNetV3Encoder(nn.Module):
    """MobileNetV3-Large trunk, slightly cheaper at equal depth."""

    def __init__(self, in_channels: int, pretrained: bool):
        super().__init__()
        from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        features = _load_backbone(mobilenet_v3_large, weights, "mobilenet_v3_large").features
        features[0][0] = _adapt_first_conv(features[0][0], in_channels)

        self.stage0 = features[0:2]
        self.stage1 = features[2:4]
        self.stage2 = features[4:7]
        self.stage3 = features[7:13]
        self.stage4 = features[13:]
        self.out_channels = [16, 24, 40, 112, 960]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f0 = self.stage0(x)
        f1 = self.stage1(f0)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f0, f1, f2, f3, f4]


class ResNet18Encoder(nn.Module):
    """Heavier convolutional reference point for the efficiency comparison."""

    def __init__(self, in_channels: int, pretrained: bool):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = _load_backbone(resnet18, weights, "resnet18")
        net.conv1 = _adapt_first_conv(net.conv1, in_channels)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.pool = net.maxpool
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4
        self.out_channels = [64, 64, 128, 256, 512]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f0 = self.stem(x)
        f1 = self.layer1(self.pool(f0))
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f0, f1, f2, f3, f4]


_ENCODERS = {
    "mobilenet_v2": MobileNetV2Encoder,
    "mobilenet_v3_large": MobileNetV3Encoder,
    "resnet18": ResNet18Encoder,
}


def build_encoder(
    name: str, in_channels: int, pretrained: bool, pretrained_dir=None
) -> nn.Module:
    global _LOCAL_WEIGHT_DIR
    if name not in _ENCODERS:
        raise ValueError(f"unknown encoder {name!r}; choose from {sorted(_ENCODERS)}")

    _LOCAL_WEIGHT_DIR = Path(pretrained_dir) if pretrained_dir else None
    if _LOCAL_WEIGHT_DIR is not None and not _LOCAL_WEIGHT_DIR.is_dir():
        LOGGER.warning("model.pretrained_dir %s does not exist", _LOCAL_WEIGHT_DIR)
        _LOCAL_WEIGHT_DIR = None

    encoder = _ENCODERS[name](in_channels=in_channels, pretrained=pretrained)
    # Say where the weights came from, not just whether they were requested.
    # "pretrained=True" in a log next to a random-initialised encoder is how a
    # whole study gets run from scratch without anyone noticing.
    source = "not requested"
    if pretrained:
        source = (
            f"local ({_LOCAL_WEIGHT_DIR})" if _LOCAL_WEIGHT_DIR is not None
            and _local_state_dict(name) is not None else "download cache or random"
        )
    LOGGER.info(
        "encoder=%s in_channels=%d pretrained=%s weights=%s stages=%s",
        name, in_channels, pretrained, source, encoder.out_channels,
    )
    return encoder

