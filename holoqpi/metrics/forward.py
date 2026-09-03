"""Physical plausibility of a reconstruction, scored without ground truth.

Every other metric in this framework compares a prediction with a reference. The
forward-model residual instead asks whether the predicted field could have
produced the hologram that was actually recorded, which is a question the
reference cannot answer and which needs no reference to ask.

Two consequences make it worth reporting separately from the loss.

* It is comparable across ablation arms. An arm trained without the
  forward-model term is still scored on it, so the table shows what optimising
  it buys rather than only that it was optimised.
* It transfers to data with no ground-truth phase at all, which is where a
  measurement-readiness claim eventually has to be defended.

Reported as a correlation-based residual in [0, 2]: zero means the synthesised
hologram matches the measurement perfectly in structure, one means unrelated.
"""

from __future__ import annotations

import numpy as np
import torch


class ForwardModelMetrics:
    """Accumulates the residual between synthesised and measured holograms."""

    def __init__(self, loss_cfg, optics, modality: str):
        self.modality = modality
        self.distance_um = loss_cfg.distance_um
        self.enabled = self.distance_um is not None
        self.wavelength_um = float(optics.wavelength_um)
        self.pitch_x_um = float(optics.pixel_pitch_x_um)
        self.pitch_y_um = float(optics.pixel_pitch_y_um)
        self.feature_um = float(loss_cfg.feature_um)
        self.reference_ratio = float(loss_cfg.reference_ratio)
        self.dc_exclusion_px = int(loss_cfg.dc_exclusion_px)
        self.border_px = loss_cfg.border_px
        self.pad_px = loss_cfg.pad_px
        self.reset()

    def reset(self) -> None:
        self._predicted: list[float] = []
        self._reference: list[float] = []

    def _pad(self, size: int) -> int:
        affordable = max(0, size // 2 - 1)
        if self.pad_px is not None:
            return min(int(self.pad_px), affordable)
        if not self.distance_um:
            return 0
        spread_um = abs(self.wavelength_um * float(self.distance_um)) / self.feature_um
        required = int(np.ceil(spread_um / min(self.pitch_x_um, self.pitch_y_um)))
        return min(required, affordable)

    @staticmethod
    def _standardise(x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(1)
        mean = flat.mean(dim=1).view(-1, 1, 1, 1)
        std = flat.std(dim=1).view(-1, 1, 1, 1).clamp(min=1e-6)
        return (x - mean) / std

    @torch.no_grad()
    def update(
        self,
        phase: torch.Tensor,
        amplitude: torch.Tensor | None,
        hologram: torch.Tensor,
        reference_phase: torch.Tensor | None = None,
    ) -> None:
        """``phase`` and ``hologram`` are (B, 1, H, W) tensors on any device."""
        if not self.enabled:
            return
        from ..physics import estimate_carrier, form_hologram

        pad = self._pad(min(phase.shape[-2], phase.shape[-1]))
        border = int(self.border_px) if self.border_px is not None else pad
        carrier = (
            estimate_carrier(hologram, self.dc_exclusion_px)
            if self.modality == "off_axis" else None
        )

        def residual(field_phase):
            amp = amplitude if amplitude is not None else torch.ones_like(field_phase)
            synthetic = form_hologram(
                field_phase, amp, self.modality,
                self.wavelength_um, self.pitch_x_um, self.pitch_y_um,
                float(self.distance_um), carrier=carrier,
                reference_ratio=self.reference_ratio, pad=pad,
            )
            measured = hologram
            if border > 0 and min(synthetic.shape[-2:]) > 2 * border + 8:
                synthetic = synthetic[..., border:-border, border:-border]
                measured = measured[..., border:-border, border:-border]
            a = self._standardise(synthetic)
            b = self._standardise(measured)
            return (1.0 - (a * b).flatten(1).mean(dim=1)).cpu().numpy()

        self._predicted.extend(residual(phase).tolist())
        if reference_phase is not None:
            # The same residual computed from the ground-truth phase. It is the
            # floor this metric can reach: whatever the reference phase itself
            # fails to explain is model mismatch (amplitude, aberration, an
            # imperfect z) rather than an error the network made.
            self._reference.extend(residual(reference_phase).tolist())

    def compute(self) -> dict:
        if not self._predicted:
            return {}
        predicted = float(np.mean(self._predicted))
        results = {
            "forward_residual": predicted,
            "forward_residual_n": int(len(self._predicted)),
        }
        if self._reference:
            floor = float(np.mean(self._reference))
            results["forward_residual_reference"] = floor
            # How much of the achievable consistency the prediction reaches.
            # 1.0 means it explains the hologram as well as the reference phase
            # does; below zero means worse than an uninformative field.
            results["forward_residual_ratio"] = predicted / floor if floor > 0 else float("nan")
        return results
