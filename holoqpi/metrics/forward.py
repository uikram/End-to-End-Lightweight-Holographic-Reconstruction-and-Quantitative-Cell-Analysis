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

What it reports depends on ``loss.forward_model.criterion``, and the study uses
``l2``: the mean squared residual in units of the measurement's own variance,
i.e. the fraction of that variance the synthesised hologram fails to explain.
Zero is a perfect fit and it is not bounded above. (An earlier version of this
note described a correlation residual in [0, 2], which is the ``correlation``
option and not the configured one.)

Read ``forward_residual_ratio`` rather than the level: the residual the
GROUND-TRUTH phase leaves is the floor this metric can reach, and everything the
forward operator cannot reproduce lands in both.
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

        # The term itself owns the geometry: the wavelength, the pitches, the
        # diffraction pad and the border all live in ForwardModelConsistency and
        # are read from the same config keys. This class used to duplicate five
        # of them as attributes plus a private `_pad` reimplementation of the
        # padding rule -- none of which was ever called, because the term does
        # its own padding. Two copies of one rule is how the metric and the
        # objective drift apart, which is the one thing this class exists to
        # prevent, so the copies are gone.
        from ..losses.terms import ForwardModelConsistency
        self._term = ForwardModelConsistency(loss_cfg, optics) if self.enabled else None
        self.reset()

    def reset(self) -> None:
        self._predicted: list[float] = []
        self._reference: list[float] = []

    @torch.no_grad()
    def update(
        self,
        phase: torch.Tensor,
        amplitude: torch.Tensor | None,
        hologram: torch.Tensor,
        reference_phase: torch.Tensor | None = None,
        aberration: torch.Tensor | None = None,
    ) -> None:
        """``phase`` and ``hologram`` are (B, 1, H, W) tensors on any device."""
        if not self.enabled:
            return

        def residual(field_phase):
            # Scored by the *same* term the loss uses, so the metric and the
            # objective cannot drift apart, and an arm that never optimised the
            # term is still scored on exactly the quantity the others minimised.
            amp = amplitude if amplitude is not None else torch.ones_like(field_phase)
            value = self._term(field_phase, amp, hologram, self.modality,
                               aberration=aberration)
            return value.detach().cpu().numpy()

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
        results["forward_distance_um"] = float(self.distance_um)
        if self._reference:
            floor = float(np.mean(self._reference))
            results["forward_residual_reference"] = floor
            # Predicted residual divided by the residual the GROUND-TRUTH phase
            # leaves, so it is read the same way round as the residual itself:
            # LOWER IS BETTER, 1.0 means the prediction explains the hologram
            # exactly as well as the reference phase does, and above 1.0 means
            # worse than the reference. The previous comment here said 1.0 was
            # the ceiling and "below zero" was bad, which inverts it -- the
            # quantity is a ratio of non-negative residuals and cannot go below
            # zero at all.
            results["forward_residual_ratio"] = predicted / floor if floor > 0 else float("nan")
        return results
