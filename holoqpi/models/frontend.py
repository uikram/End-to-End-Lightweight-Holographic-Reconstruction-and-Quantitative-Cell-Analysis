"""Optional physics front end.

For an off-axis hologram the object field can be recovered analytically by
isolating one first-order sideband in the Fourier plane and shifting it to the
origin. Supplying that demodulated field as extra input channels gives the
network a physically grounded starting point instead of asking it to learn
demodulation from scratch.

The operation is differentiable and runs inside the graph, so it can either be
frozen (``detach: true``) or trained through. It is meaningful only for off-axis
data: an in-line Gabor hologram has no separated sideband, which is precisely
one of the asymmetries the modality comparison is meant to expose.
"""

from __future__ import annotations

import torch
import torch.fft as fft
import torch.nn as nn

from ..config import Config


class AngularSpectrumFrontEnd(nn.Module):
    """Fourier-plane sideband demodulation of an off-axis hologram."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.sideband_radius = cfg.sideband_radius_px
        self.dc_exclusion = cfg.dc_exclusion_px
        self.output = cfg.output
        self.detach = cfg.detach

        if self.output not in ("amplitude_phase", "real_imag"):
            raise ValueError(f"unknown frontend output {self.output!r}")

    @property
    def out_channels(self) -> int:
        """Two demodulated channels are appended to the raw hologram."""
        return 3

    def forward(self, hologram: torch.Tensor) -> torch.Tensor:
        context = torch.no_grad() if self.detach else _NullContext()
        with context:
            field = self._demodulate(hologram)

        if self.output == "amplitude_phase":
            extra = torch.cat([field.abs(), torch.angle(field)], dim=1)
        else:
            extra = torch.cat([field.real, field.imag], dim=1)

        return torch.cat([hologram, extra.to(hologram.dtype)], dim=1)

    def _demodulate(self, hologram: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = hologram.shape
        spectrum = fft.fftshift(fft.fft2(hologram.float()), dim=(-2, -1))

        centre_y, centre_x = height // 2, width // 2
        grid_y = torch.arange(height, device=hologram.device).view(-1, 1)
        grid_x = torch.arange(width, device=hologram.device).view(1, -1)
        radius_from_dc = torch.hypot(
            (grid_y - centre_y).float(), (grid_x - centre_x).float()
        )

        magnitude = spectrum.abs().clone()
        magnitude[:, :, radius_from_dc < self.dc_exclusion] = 0.0

        fields = []
        for item in range(batch):
            flat_index = torch.argmax(magnitude[item, 0])
            peak_y = int(flat_index // width)
            peak_x = int(flat_index % width)

            window = (
                torch.hypot(
                    (grid_y - peak_y).float(), (grid_x - peak_x).float()
                ) < self.sideband_radius
            )
            selected = spectrum[item, 0] * window
            # Move the carrier to the origin: removes the tilt from the reference beam.
            selected = torch.roll(
                selected, shifts=(centre_y - peak_y, centre_x - peak_x), dims=(0, 1)
            )
            fields.append(fft.ifft2(fft.ifftshift(selected)))

        return torch.stack(fields).unsqueeze(1)


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def build_frontend(cfg: Config) -> tuple[nn.Module | None, int]:
    """Return the front end and the number of channels it feeds the encoder."""
    kind = cfg.kind
    if kind == "none":
        return None, 1
    if kind == "angular_spectrum":
        module = AngularSpectrumFrontEnd(cfg)
        return module, module.out_channels
    raise ValueError(f"unknown frontend kind {kind!r}; use none or angular_spectrum")
