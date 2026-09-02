"""Reusable convolutional building blocks and spatial-size bookkeeping."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )


class SeparableConvBNAct(nn.Sequential):
    """Depthwise-separable convolution: the edge-friendly workhorse."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size // 2,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )


class DecoderBlock(nn.Module):
    """Upsample, concatenate the encoder skip, then fuse."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.fuse = SeparableConvBNAct(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class UNetDecoder(nn.Module):
    """Symmetric decoder consuming a list of encoder feature maps."""

    def __init__(self, encoder_channels: list[int], decoder_channels: list[int]):
        super().__init__()
        # encoder_channels is ordered shallow -> deep; consume it in reverse.
        reversed_skips = list(reversed(encoder_channels[:-1]))
        in_channels = encoder_channels[-1]

        blocks = []
        for stage, out_channels in enumerate(decoder_channels):
            skip = reversed_skips[stage] if stage < len(reversed_skips) else 0
            blocks.append(DecoderBlock(in_channels, skip, out_channels))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = in_channels

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        skips = list(reversed(features[:-1]))
        x = features[-1]
        for stage, block in enumerate(self.blocks):
            x = block(x, skips[stage] if stage < len(skips) else None)
        return x


def pad_to_multiple(x: torch.Tensor, divisor: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Right/bottom-pad so every downsampling stage divides evenly.

    Returns the padded tensor and the amount added, so the caller can undo it.
    The phase field is 900 px, which is not a multiple of 32; padding here keeps
    the network free of resampling that would blur the quantitative signal.
    """
    height, width = x.shape[-2:]
    pad_h = (divisor - height % divisor) % divisor
    pad_w = (divisor - width % divisor) % divisor
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, (pad_h, pad_w)


def unpad(x: torch.Tensor, padding: tuple[int, int]) -> torch.Tensor:
    pad_h, pad_w = padding
    if pad_h:
        x = x[..., :-pad_h, :]
    if pad_w:
        x = x[..., :, :-pad_w]
    return x
