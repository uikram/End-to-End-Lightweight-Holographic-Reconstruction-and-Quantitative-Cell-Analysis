"""Low-rank adaptation of a frozen encoder.

Carried forward from the previous study, where restricting adaptation to a
low-rank subspace was found to preserve rare classes that full fine-tuning
destroyed. Here it remains available as a config-gated adaptation mode: the
encoder is frozen except for its low-rank residuals, while normalisation layers,
the input stem and every decoder and head stay trainable.

Grouped convolutions are deliberately skipped. A depthwise kernel has one filter
per channel, so a shared low-rank factorisation across channels is not defined
for it, and the merge step could not be expressed as a single dense update.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from ..utils import get_logger

LOGGER = get_logger(__name__)

_STRATEGY_PATTERNS = {
    "encoder_only": ("stage", "features", "layer", "blocks", "conv", "stem"),
    "attention_blocks": ("qkv", "q_proj", "v_proj", "query", "value", "self_attn"),
    "bottleneck": ("bottleneck", "neck", "stage4", "layer4"),
}


class LoRALinear(nn.Module):
    """W' = W + (B @ A) * alpha / r, with W frozen."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.merged = False

        self.weight = nn.Parameter(base.weight.data.clone(), requires_grad=False)
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self.lora_a = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.merged:
            return out
        return out + (self.dropout(x) @ self.lora_a.T @ self.lora_b.T) * self.scaling

    @torch.no_grad()
    def merge(self) -> None:
        if not self.merged:
            self.weight.data += (self.lora_b @ self.lora_a) * self.scaling
            self.merged = True


class LoRAConv2d(nn.Module):
    """Low-rank residual for a dense convolution, as a 1x1 then k x k pair."""

    def __init__(self, base: nn.Conv2d, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.stride = base.stride
        self.padding = base.padding
        self.dilation = base.dilation
        self.groups = base.groups
        self.scaling = alpha / rank
        self.merged = False

        self.weight = nn.Parameter(base.weight.data.clone(), requires_grad=False)
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self.lora_a = nn.Conv2d(base.in_channels, rank, kernel_size=1, bias=False)
        self.lora_b = nn.Conv2d(
            rank, base.out_channels, kernel_size=base.kernel_size,
            stride=base.stride, padding=base.padding, dilation=base.dilation, bias=False,
        )
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.conv2d(x, self.weight, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
        if self.merged:
            return out
        return out + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling

    @torch.no_grad()
    def merge(self) -> None:
        if self.merged:
            return
        a = self.lora_a.weight.data.squeeze(-1).squeeze(-1)      # (r, in)
        b = self.lora_b.weight.data                              # (out, r, kh, kw)
        self.weight.data += torch.einsum("orhw,ri->oihw", b, a) * self.scaling
        self.merged = True


def _matches(name: str, module: nn.Module, strategy: str) -> bool:
    if isinstance(module, nn.Conv2d) and module.groups > 1:
        return False
    patterns = _STRATEGY_PATTERNS.get(strategy)
    if patterns is None:
        raise ValueError(f"unknown LoRA strategy {strategy!r}; choose from {sorted(_STRATEGY_PATTERNS)}")
    lowered = name.lower()
    return any(pattern in lowered for pattern in patterns)


def _replace(root: nn.Module, name: str, replacement: nn.Module) -> None:
    """Swap the submodule at a dotted path produced by ``named_modules``.

    Traversal goes through attribute access rather than integer indexing. A
    sliced ``nn.Sequential`` keeps the child keys of the sequence it came from,
    so ``encoder.stage1`` may hold children named "2" and "3"; those names are
    valid attributes but positional indices 2 and 3 are out of range.
    """
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], replacement)


def inject_lora(encoder: nn.Module, cfg: Config) -> int:
    """Freeze the encoder and insert low-rank residuals. Returns the layer count."""
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    targets = [
        (name, module)
        for name, module in encoder.named_modules()
        if isinstance(module, (nn.Linear, nn.Conv2d)) and _matches(name, module, cfg.strategy)
    ]

    injected = 0
    for name, module in targets:
        if isinstance(module, nn.Linear):
            _replace(encoder, name, LoRALinear(module, cfg.rank, cfg.alpha, cfg.dropout))
        else:
            _replace(encoder, name, LoRAConv2d(module, cfg.rank, cfg.alpha, cfg.dropout))
        injected += 1

    if cfg.train_norm_layers:
        for module in encoder.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for parameter in module.parameters():
                    parameter.requires_grad = True

    if cfg.train_input_stem:
        # The hologram stem sees a distribution no pretrained filter has met.
        for module in encoder.modules():
            if isinstance(module, nn.Conv2d) and module.in_channels <= 3:
                for parameter in module.parameters():
                    parameter.requires_grad = True
                break

    LOGGER.info("LoRA: injected %d layers (rank=%d, strategy=%s)", injected, cfg.rank, cfg.strategy)
    return injected


def unfreeze_by_pattern(model: nn.Module, patterns: list[str]) -> None:
    """Keep decoders, heads and classifiers trainable regardless of LoRA."""
    lowered = [p.lower() for p in patterns]
    for name, parameter in model.named_parameters():
        if any(pattern in name.lower() for pattern in lowered):
            parameter.requires_grad = True


def merge_lora(model: nn.Module) -> int:
    """Fold every low-rank residual into its base weight, for export and timing."""
    merged = 0
    for module in model.modules():
        if isinstance(module, (LoRALinear, LoRAConv2d)):
            module.merge()
            merged += 1
    if merged:
        LOGGER.info("LoRA: merged %d layers into their base weights", merged)
    return merged


def has_lora(model: nn.Module) -> bool:
    return any(isinstance(m, (LoRALinear, LoRAConv2d)) for m in model.modules())
