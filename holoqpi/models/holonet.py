"""HoloQPINet: raw hologram in, quantitative phase and cell map out.

A single shared encoder feeds two decoders and one classifier:

    hologram -> [front end] -> encoder -> phase decoder      -> phase   (radians)
                                       -> [amplitude head]   -> transmittance
                                       -> segmentation dec.  -> cell logits
                                       -> bottleneck pool    -> condition logits

The optional amplitude head exists only to complete the complex field for the
forward-model consistency term; it has no supervision of its own and is absent
unless model.amplitude.enabled is set.

Sharing the encoder is the point of the design rather than an economy: the same
latent description of the fringe field has to support both reconstruction and
delineation, which is what ties the segmentation boundary to the optical signal
it will later be used to integrate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from ..utils import get_logger
from .blocks import UNetDecoder, pad_to_multiple, unpad
from .encoders import build_encoder
from .frontend import build_frontend
from .heads import AmplitudeHead, ConditionClassifier, PhaseHead, SegmentationHead
from . import lora as lora_utils

LOGGER = get_logger(__name__)


class HoloQPINet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        model_cfg = cfg.model

        self.frontend, encoder_in_channels = build_frontend(model_cfg.frontend)
        self.size_divisor = model_cfg.size_divisor

        self.encoder = build_encoder(
            name=model_cfg.encoder,
            in_channels=encoder_in_channels,
            pretrained=model_cfg.pretrained_encoder,
        )
        encoder_channels = self.encoder.out_channels
        decoder_channels = list(model_cfg.decoder_channels)

        self.share_decoder = model_cfg.share_decoder
        if self.share_decoder:
            self.shared_decoder = UNetDecoder(encoder_channels, decoder_channels)
            self.phase_decoder = self.segmentation_decoder = None
            trunk_channels = self.shared_decoder.out_channels
        else:
            self.shared_decoder = None
            self.phase_decoder = UNetDecoder(encoder_channels, decoder_channels)
            self.segmentation_decoder = UNetDecoder(encoder_channels, decoder_channels)
            trunk_channels = self.phase_decoder.out_channels

        head_hidden = decoder_channels[-1]
        self.phase_head = PhaseHead(trunk_channels, head_hidden)
        # Only built when the forward model is allowed to fit amplitude; keeping
        # it absent otherwise means the exported graph and the parameter count
        # are unchanged for every configuration that does not use it.
        self.amplitude_head = (
            AmplitudeHead(trunk_channels, head_hidden, model_cfg.amplitude.deviation)
            if model_cfg.amplitude.enabled else None
        )
        self.segmentation_head = SegmentationHead(
            trunk_channels, head_hidden, model_cfg.segmentation_classes
        )
        self.condition_classifier = ConditionClassifier(
            in_channels=encoder_channels[-1],
            hidden=model_cfg.classifier_hidden,
            num_classes=model_cfg.condition_classes,
            dropout=model_cfg.classifier_dropout,
        )

        if model_cfg.lora.enabled:
            lora_utils.inject_lora(self.encoder, model_cfg.lora)
            lora_utils.unfreeze_by_pattern(self, list(model_cfg.lora.always_trainable_patterns))

    # -- forward ----------------------------------------------------------
    def forward(self, hologram: torch.Tensor) -> dict[str, torch.Tensor]:
        original_size = hologram.shape[-2:]

        x = self.frontend(hologram) if self.frontend is not None else hologram
        x, padding = pad_to_multiple(x, self.size_divisor)

        features = self.encoder(x)

        if self.share_decoder:
            trunk = self.shared_decoder(features)
            phase_trunk = segmentation_trunk = trunk
        else:
            phase_trunk = self.phase_decoder(features)
            segmentation_trunk = self.segmentation_decoder(features)

        phase = self._to_input_size(self.phase_head(phase_trunk), padding, original_size)
        segmentation = self._to_input_size(
            self.segmentation_head(segmentation_trunk), padding, original_size
        )
        condition = self.condition_classifier(features[-1])

        outputs = {"phase": phase, "segmentation": segmentation, "condition": condition}
        if self.amplitude_head is not None:
            outputs["amplitude"] = self._to_input_size(
                self.amplitude_head(phase_trunk), padding, original_size
            )
        return outputs

    def _to_input_size(
        self, x: torch.Tensor, padding: tuple[int, int], target: torch.Size
    ) -> torch.Tensor:
        if x.shape[-2:] != target:
            scale_h = (target[0] + padding[0]) / x.shape[-2]
            if abs(scale_h - 1.0) > 1e-6:
                x = F.interpolate(
                    x, size=(target[0] + padding[0], target[1] + padding[1]),
                    mode="bilinear", align_corners=False,
                )
        x = unpad(x, padding)
        if x.shape[-2:] != target:
            x = F.interpolate(x, size=target, mode="bilinear", align_corners=False)
        return x

    # -- deployment helpers ----------------------------------------------
    def merge_lora(self) -> "HoloQPINet":
        lora_utils.merge_lora(self)
        return self

    @property
    def has_lora(self) -> bool:
        return lora_utils.has_lora(self)


class ExportWrapper(nn.Module):
    """Tuple-returning view of the model, for tracers that dislike dicts."""

    def __init__(self, model: HoloQPINet):
        super().__init__()
        self.model = model

    def forward(self, hologram: torch.Tensor):
        out = self.model(hologram)
        return out["phase"], out["segmentation"], out["condition"]


def build_model(cfg: Config) -> HoloQPINet:
    name = cfg.model.name
    if name != "holoqpinet":
        raise ValueError(f"unknown model {name!r}; only 'holoqpinet' is defined")
    model = HoloQPINet(cfg)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info(
        "HoloQPINet built: %.2fM parameters, %.2fM trainable (%.1f%%)",
        total / 1e6, trainable / 1e6, 100.0 * trainable / total,
    )
    return model
