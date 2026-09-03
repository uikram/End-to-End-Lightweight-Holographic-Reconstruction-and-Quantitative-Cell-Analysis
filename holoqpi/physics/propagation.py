"""Scalar diffraction: the forward model that connects a field to a hologram.

This module is the physics the study was missing. Everything else in the
framework relates the network's two outputs to each other or to their targets;
here the predicted field is propagated to the sensor and compared with the
measurement that produced it, which is the constraint the reference literature
means by "physics consistency" (Huang et al., Nat. Mach. Intell. 2023;
Galande et al., J. Biomed. Opt.; Lee et al., APL Mach. Learn. 4, 026106).

Angular spectrum propagation over a distance z:

    U(z) = F^-1 { F{u_0} . H },   H = exp( i 2 pi z / lambda sqrt(1 - (lambda fx)^2 - (lambda fy)^2) )

with evanescent components (the square-root argument turning negative) set to
zero rather than allowed to blow up.

The two imaging geometries then form their intensity differently, and the
difference is exactly the asymmetry the study is about:

    in-line Gabor   I = |P_z(o)|^2
                    the unscattered beam is part of o, so object and reference
                    are superposed and the twin image is inseparable

    off-axis        I = |R + P_z(o)|^2
                    a separate tilted reference R puts the object in a sideband
                    that can be isolated, which is why off-axis reconstructs better

Everything here is differentiable and runs on the training device, so the same
code serves the loss term, the conventional baseline and the z calibration.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.fft as fft


# ---------------------------------------------------------------------------
# transfer function
# ---------------------------------------------------------------------------
def angular_spectrum_kernel(
    height: int,
    width: int,
    wavelength_um: float,
    pixel_pitch_x_um: float,
    pixel_pitch_y_um: float,
    distance_um: float | torch.Tensor,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Free-space transfer function H(fx, fy; z), shaped (H, W) or (B, H, W).

    ``distance_um`` may be a scalar or a per-sample tensor; a tensor keeps the
    kernel differentiable with respect to z, which is what allows the
    propagation distance to be refined by gradient descent when it is not known
    from the acquisition.
    """
    # Cycles per micrometre on the sampling grid.
    fx = fft.fftfreq(width, d=pixel_pitch_x_um, device=device, dtype=dtype)
    fy = fft.fftfreq(height, d=pixel_pitch_y_um, device=device, dtype=dtype)
    fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")

    # 1 - (lambda fx)^2 - (lambda fy)^2; negative means evanescent.
    argument = 1.0 - (wavelength_um * fxx) ** 2 - (wavelength_um * fyy) ** 2
    propagating = argument > 0

    kz = torch.zeros_like(argument)
    kz[propagating] = torch.sqrt(argument[propagating])
    decay = torch.zeros_like(argument)
    decay[~propagating] = torch.sqrt(-argument[~propagating])

    if not torch.is_tensor(distance_um):
        distance_um = torch.tensor(float(distance_um), device=device, dtype=dtype)
    distance = distance_um.reshape(-1, 1, 1) if distance_um.ndim else distance_um

    wavenumber = 2.0 * math.pi / wavelength_um
    phase = wavenumber * distance * kz

    # Evanescent components decay rather than vanish. Zeroing them outright
    # would make the kernel a low-pass filter even at z = 0, so propagating by
    # zero would not return the field it was given -- and the in-line hologram
    # of a pure phase object would acquire a contrast it does not physically
    # have. exp(-k |z| sqrt(-arg)) is the correct attenuation and equals one at
    # z = 0, which makes the identity exact.
    attenuation = torch.exp(-wavenumber * distance.abs() * decay)
    return torch.polar(attenuation.expand_as(phase), phase)


def propagate(
    field: torch.Tensor,
    wavelength_um: float,
    pixel_pitch_x_um: float,
    pixel_pitch_y_um: float,
    distance_um: float | torch.Tensor,
) -> torch.Tensor:
    """Propagate a complex field (B, 1, H, W) by ``distance_um``.

    A negative distance back-propagates, which is how the conventional
    reconstruction baseline recovers the sample plane from a recorded hologram.
    """
    height, width = field.shape[-2:]
    kernel = angular_spectrum_kernel(
        height, width, wavelength_um, pixel_pitch_x_um, pixel_pitch_y_um,
        distance_um, device=field.device,
        dtype=torch.float32 if field.dtype == torch.complex64 else torch.float64,
    )
    if kernel.ndim == 3:
        kernel = kernel.unsqueeze(1)
    return fft.ifft2(fft.fft2(field) * kernel)


# ---------------------------------------------------------------------------
# padding, so a training crop does not wrap around
# ---------------------------------------------------------------------------
def diffraction_radius_px(
    wavelength_um: float, distance_um: float, pixel_pitch_um: float
) -> int:
    """How far light spreads laterally over ``distance_um``, in pixels.

    The FFT treats the array as periodic, so light leaving one edge re-enters at
    the other. On a 512 px training crop taken from a 900 px field that is not a
    small effect, and it contaminates exactly the border region. Padding by this
    radius and discarding the same margin from the residual removes it.
    """
    if pixel_pitch_um <= 0:
        return 0
    # Highest spatial frequency the grid samples is 1/(2 dx); the corresponding
    # diffraction angle carries light lambda*z/(2 dx) micrometres sideways.
    spread_um = abs(wavelength_um * distance_um) / (2.0 * pixel_pitch_um)
    return int(math.ceil(spread_um / pixel_pitch_um))


def max_reflect_pad(field: torch.Tensor) -> int:
    """Reflection padding cannot exceed the dimension it mirrors."""
    return max(0, min(field.shape[-2], field.shape[-1]) - 1)


def pad_reflect(field: torch.Tensor, pad: int) -> torch.Tensor:
    """Pad a complex field, mirroring as far as the array allows.

    Reflection is preferred because a hard edge would itself diffract and add a
    ringing artefact to the synthesised hologram. It is limited to one array
    width, so a larger request is mirrored as far as possible and the remainder
    filled with the edge value, which diffracts far less than a zero edge.
    """
    if pad <= 0:
        return field
    limit = max_reflect_pad(field)
    mirrored = min(pad, limit)
    remainder = pad - mirrored

    real, imaginary = field.real, field.imag
    if mirrored > 0:
        real = torch.nn.functional.pad(real, (mirrored,) * 4, mode="reflect")
        imaginary = torch.nn.functional.pad(imaginary, (mirrored,) * 4, mode="reflect")
    if remainder > 0:
        real = torch.nn.functional.pad(real, (remainder,) * 4, mode="replicate")
        imaginary = torch.nn.functional.pad(imaginary, (remainder,) * 4, mode="replicate")
    return torch.complex(real, imaginary)


def unpad(x: torch.Tensor, pad: int) -> torch.Tensor:
    return x if pad <= 0 else x[..., pad:-pad, pad:-pad]


# ---------------------------------------------------------------------------
# carrier estimation for off-axis
# ---------------------------------------------------------------------------
def estimate_carrier(
<<<<<<< Updated upstream
    hologram: torch.Tensor, dc_exclusion_px: int = 60
=======
    hologram: torch.Tensor,
    dc_exclusion_px: int | None = None,
    dc_exclusion_frac: float = 0.13,
>>>>>>> Stashed changes
) -> tuple[torch.Tensor, torch.Tensor]:
    """Locate the off-axis carrier as the brightest non-DC spectral peak.

    Returns normalised frequencies (fy, fx) in cycles per pixel, one pair per
    sample. The reference tilt is a property of the optical setup rather than of
    the specimen, so it is read from the measurement itself and never learned.
<<<<<<< Updated upstream
=======

    Two details are not optional.

    SUB-BIN REFINEMENT. The carrier almost never lands on an integer FFT bin. A
    half-bin error is a phase ramp of pi across the field, which no per-image
    gain or offset can absorb, so the peak is refined by fitting a parabola
    through its immediate neighbours in each axis. On a 128 px grid this is the
    difference between a carrier of 0.1094 and one of 0.1100 -- and between a
    forward-model residual near zero and one near one.

    HALF-PLANE CONVENTION. The two first-order sidebands are conjugates of equal
    magnitude, so which one ``argmax`` returns is arbitrary and can flip between
    images of the same acquisition. The peak in the fy > 0 half-plane is
    returned consistently. Callers that must not depend on the choice at all
    should carry both cross-terms and let a fit decide, as the forward-model
    loss does.

    SCALE-INVARIANT DC EXCLUSION. The radius around DC to ignore is a *fraction*
    of the field, not a pixel count. An absolute radius means something
    different on a 900 px evaluation field and a 512 px training crop -- and on
    a small enough crop it masks out the carrier itself, leaving the search to
    return noise. ``dc_exclusion_px`` still overrides it where an absolute value
    is genuinely wanted.
>>>>>>> Stashed changes
    """
    batch, _, height, width = hologram.shape
    spectrum = fft.fftshift(fft.fft2(hologram.float()), dim=(-2, -1))
    magnitude = spectrum.abs().squeeze(1)

    centre_y, centre_x = height // 2, width // 2
    grid_y = torch.arange(height, device=hologram.device).view(-1, 1)
    grid_x = torch.arange(width, device=hologram.device).view(1, -1)
    radius = torch.hypot((grid_y - centre_y).float(), (grid_x - centre_x).float())
<<<<<<< Updated upstream
    magnitude = magnitude.masked_fill(radius.unsqueeze(0) < dc_exclusion_px, 0.0)

    flat = magnitude.reshape(batch, -1).argmax(dim=1)
    peak_y = (flat // width).float() - centre_y
    peak_x = (flat % width).float() - centre_x
    return peak_y / height, peak_x / width
=======
    exclusion = (
        float(dc_exclusion_px) if dc_exclusion_px is not None
        else dc_exclusion_frac * 0.5 * min(height, width)
    )
    magnitude = magnitude.masked_fill(radius.unsqueeze(0) < exclusion, 0.0)

    # Keep only the upper half-plane so the conjugate pair cannot be chosen at
    # random. The row through the centre is split at the centre column, so the
    # boundary case (fy == 0) is resolved by the sign of fx. The mask selects
    # the peak but must NOT be used for the sub-bin fit below: zeroing a
    # neighbour would corrupt the parabola for any peak sitting near the
    # boundary, which is precisely the case the mask creates.
    upper = (grid_y - centre_y) > 0
    same_row = (grid_y - centre_y) == 0
    keep = upper | (same_row & ((grid_x - centre_x) > 0))
    selectable = magnitude.masked_fill(~keep.unsqueeze(0), 0.0)

    flat = selectable.reshape(batch, -1).argmax(dim=1)
    peak_y = (flat // width).long()
    peak_x = (flat % width).long()

    def refine(index, axis_size, along_rows: bool):
        """Parabolic sub-bin interpolation around the peak."""
        offsets = []
        for item in range(batch):
            y, x = int(peak_y[item]), int(peak_x[item])
            position = y if along_rows else x
            if position <= 0 or position >= axis_size - 1:
                offsets.append(0.0)
                continue
            if along_rows:
                left, centre, right = (magnitude[item, y - 1, x],
                                       magnitude[item, y, x],
                                       magnitude[item, y + 1, x])
            else:
                left, centre, right = (magnitude[item, y, x - 1],
                                       magnitude[item, y, x],
                                       magnitude[item, y, x + 1])
            denominator = float(left - 2 * centre + right)
            offsets.append(
                0.0 if abs(denominator) < 1e-12
                else float(0.5 * (left - right) / denominator)
            )
        return torch.tensor(offsets, device=hologram.device, dtype=torch.float32).clamp(-1, 1)

    fine_y = peak_y.float() + refine(peak_y, height, True) - centre_y
    fine_x = peak_x.float() + refine(peak_x, width, False) - centre_x

    # Final polish: choose the carrier that concentrates the demodulated field
    # into its DC term. The parabolic estimate is accurate to a few hundredths
    # of a bin, and a hundredth of a bin is still several degrees of phase ramp
    # across a 900 px field, which the radiometric fit cannot absorb.
    fine_y, fine_x = _polish_carrier(hologram, fine_y / height, fine_x / width)
    return fine_y, fine_x


def _polish_carrier(
    hologram: torch.Tensor, carrier_y: torch.Tensor, carrier_x: torch.Tensor,
    span: float = 1.5, steps: int = 7, rounds: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refine the carrier by maximising the demodulated DC magnitude.

    Multiplying the hologram by exp(-i 2 pi (fy y + fx x)) and summing gives a
    magnitude that peaks exactly when the trial carrier matches the real fringe
    frequency. A short coordinate descent over a shrinking window is enough, and
    it optimises the quantity the forward model actually depends on rather than
    a proxy read off the spectrum.
    """
    batch, _, height, width = hologram.shape
    signal = hologram.float().squeeze(1)
    y = torch.arange(height, device=hologram.device, dtype=torch.float32).view(1, -1, 1)
    x = torch.arange(width, device=hologram.device, dtype=torch.float32).view(1, 1, -1)

    def score(fy, fx):
        ramp = -2.0 * math.pi * (fy.view(-1, 1, 1) * y + fx.view(-1, 1, 1) * x)
        real = (signal * torch.cos(ramp)).flatten(1).mean(dim=1)
        imaginary = (signal * torch.sin(ramp)).flatten(1).mean(dim=1)
        return torch.hypot(real, imaginary)

    best_y, best_x = carrier_y.clone(), carrier_x.clone()
    best = score(best_y, best_x)
    window_y, window_x = span / height, span / width

    for _ in range(rounds):
        for axis in ("y", "x"):
            window = window_y if axis == "y" else window_x
            for offset in torch.linspace(-window, window, steps, device=hologram.device):
                trial_y = best_y + offset if axis == "y" else best_y
                trial_x = best_x if axis == "y" else best_x + offset
                value = score(trial_y, trial_x)
                better = value > best
                best = torch.where(better, value, best)
                best_y = torch.where(better, trial_y, best_y)
                best_x = torch.where(better, trial_x, best_x)
        window_y /= steps / 2.0
        window_x /= steps / 2.0
    return best_y, best_x
>>>>>>> Stashed changes


def reference_wave(
    height: int,
    width: int,
    carrier_y: torch.Tensor,
    carrier_x: torch.Tensor,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Tilted plane reference R = exp(i 2 pi (fy y + fx x)), shape (B, 1, H, W)."""
    y = torch.arange(height, device=device, dtype=dtype).view(1, -1, 1)
    x = torch.arange(width, device=device, dtype=dtype).view(1, 1, -1)
    phase = 2.0 * math.pi * (
        carrier_y.to(dtype).view(-1, 1, 1) * y + carrier_x.to(dtype).view(-1, 1, 1) * x
    )
    return torch.polar(torch.ones_like(phase), phase).unsqueeze(1)


# ---------------------------------------------------------------------------
# hologram formation
# ---------------------------------------------------------------------------
def form_hologram(
    phase: torch.Tensor,
    amplitude: torch.Tensor,
    modality: str,
    wavelength_um: float,
    pixel_pitch_x_um: float,
    pixel_pitch_y_um: float,
    distance_um: float | torch.Tensor,
    carrier: tuple[torch.Tensor, torch.Tensor] | None = None,
    reference_ratio: float = 1.0,
    pad: int = 0,
) -> torch.Tensor:
    """Predict the intensity a sensor would record from a sample-plane field.

    ``phase`` and ``amplitude`` are (B, 1, H, W) real tensors describing the
    object field o = A exp(i phi) at the sample plane. The returned intensity has
    the same shape.
    """
    field = torch.polar(amplitude.clamp(min=0.0).float(), phase.float())
    field = pad_reflect(field, pad)

    propagated = propagate(
        field, wavelength_um, pixel_pitch_x_um, pixel_pitch_y_um, distance_um
    )

    if modality == "gabor":
        # The unscattered beam travels with the object beam; nothing to add.
        intensity = propagated.abs() ** 2
    elif modality == "off_axis":
        if carrier is None:
            raise ValueError("off-axis hologram formation needs a carrier frequency")
        height, width = propagated.shape[-2:]
        reference = reference_wave(
            height, width, carrier[0], carrier[1],
            device=propagated.device, dtype=torch.float32,
        ) * reference_ratio
        intensity = (reference + propagated).abs() ** 2
    else:
        raise ValueError(f"unknown modality {modality!r}")

    return unpad(intensity, pad)


# ---------------------------------------------------------------------------
# conventional reconstruction, used by the baseline and by z calibration
# ---------------------------------------------------------------------------
def reconstruct_off_axis(
    hologram: torch.Tensor,
    wavelength_um: float,
    pixel_pitch_x_um: float,
    pixel_pitch_y_um: float,
    distance_um: float,
    sideband_radius_px: int = 130,
    dc_exclusion_px: int = 60,
) -> torch.Tensor:
    """Classical off-axis reconstruction: isolate one sideband, then back-propagate.

    This is the textbook pipeline the network is asked to replace, and the
    comparison against it is what establishes whether learning buys anything for
    this geometry.
    """
    batch, _, height, width = hologram.shape
    spectrum = fft.fftshift(fft.fft2(hologram.float()), dim=(-2, -1))

    centre_y, centre_x = height // 2, width // 2
    grid_y = torch.arange(height, device=hologram.device).view(-1, 1)
    grid_x = torch.arange(width, device=hologram.device).view(1, -1)
    from_dc = torch.hypot((grid_y - centre_y).float(), (grid_x - centre_x).float())

    fields = []
    for item in range(batch):
        magnitude = spectrum[item, 0].abs().clone()
        magnitude[from_dc < dc_exclusion_px] = 0.0
        flat = int(torch.argmax(magnitude))
        peak_y, peak_x = flat // width, flat % width

        window = torch.hypot(
            (grid_y - peak_y).float(), (grid_x - peak_x).float()
        ) < sideband_radius_px
        selected = spectrum[item, 0] * window
        # Shift the carrier to the origin: removes the reference tilt.
        centred = torch.roll(
            selected, shifts=(centre_y - int(peak_y), centre_x - int(peak_x)), dims=(0, 1)
        )
        fields.append(fft.ifft2(fft.ifftshift(centred)))

    field = torch.stack(fields).unsqueeze(1)
    if abs(distance_um) > 0:
        field = propagate(
            field, wavelength_um, pixel_pitch_x_um, pixel_pitch_y_um, -distance_um
        )
    return field


def reconstruct_gabor(
    hologram: torch.Tensor,
    wavelength_um: float,
    pixel_pitch_x_um: float,
    pixel_pitch_y_um: float,
    distance_um: float,
    iterations: int = 0,
) -> torch.Tensor:
    """Classical in-line reconstruction: back-propagate the recorded intensity.

    With ``iterations = 0`` this is plain back-propagation and the twin image
    remains superposed on the object, which is the failure mode that motivates
    learning for this geometry. Positive ``iterations`` runs Gerchberg-Saxton
    with a non-negative absorption constraint at the sample plane, giving the
    classical method its fairest chance before the comparison is drawn.
    """
    measured_amplitude = hologram.clamp(min=0.0).sqrt().float()
    field = torch.complex(measured_amplitude, torch.zeros_like(measured_amplitude))

    sample = propagate(
        field, wavelength_um, pixel_pitch_x_um, pixel_pitch_y_um, -distance_um
    )
    for _ in range(max(0, iterations)):
        # Sample-plane constraint: a thin transparent object does not amplify.
        constrained = torch.polar(sample.abs().clamp(max=1.0), torch.angle(sample))
        at_sensor = propagate(
            constrained, wavelength_um, pixel_pitch_x_um, pixel_pitch_y_um, distance_um
        )
        # Sensor-plane constraint: the modulus is measured.
        at_sensor = torch.polar(measured_amplitude, torch.angle(at_sensor))
        sample = propagate(
            at_sensor, wavelength_um, pixel_pitch_x_um, pixel_pitch_y_um, -distance_um
        )
    return sample


def unwrap_phase_2d(wrapped: torch.Tensor) -> torch.Tensor:
    """Two-dimensional phase unwrapping.

    Reconstructed phase is known only modulo 2 pi, and a cell thicker than one
    wavelength of optical path wraps. Dry mass is an integral of the *unwrapped*
    phase, so leaving it wrapped would silently truncate the very quantity the
    study reports.

    Prefers scikit-image's quality-guided unwrapper (Herraez et al. 2002), which
    follows reliable paths first and handles residues properly. The least-squares
    Poisson solve below is the fallback: it never fails, but it distributes the
    error from any residue across the whole field rather than isolating it.
    """
    try:
        from skimage.restoration import unwrap_phase as _skimage_unwrap
    except ImportError:
        return _unwrap_least_squares(wrapped)

    array = wrapped.detach().cpu().numpy()
    flat = array.reshape(-1, *array.shape[-2:])
    unwrapped = np.stack([_skimage_unwrap(plane) for plane in flat])
    return torch.from_numpy(unwrapped.reshape(array.shape)).to(
        device=wrapped.device, dtype=wrapped.dtype
    )


def _unwrap_least_squares(wrapped: torch.Tensor) -> torch.Tensor:
    """Least-squares (Poisson) unwrapping via an FFT-based cosine transform."""
    height, width = wrapped.shape[-2:]

    def laplacian(x):
        gy = torch.diff(x, dim=-2, prepend=x[..., :1, :])
        gx = torch.diff(x, dim=-1, prepend=x[..., :, :1])
        gy = torch.atan2(torch.sin(gy), torch.cos(gy))     # wrap the gradient
        gx = torch.atan2(torch.sin(gx), torch.cos(gx))
        dyy = torch.diff(gy, dim=-2, append=gy[..., -1:, :])
        dxx = torch.diff(gx, dim=-1, append=gx[..., :, -1:])
        return dyy + dxx

    rho = laplacian(wrapped)

    # Solve the Poisson equation with Neumann boundaries by mirroring to 2N and
    # using the FFT, which is equivalent to a DCT-based solve.
    mirrored = torch.cat([rho, rho.flip(-2)], dim=-2)
    mirrored = torch.cat([mirrored, mirrored.flip(-1)], dim=-1)
    spectrum = fft.fft2(mirrored)

    y = torch.arange(2 * height, device=wrapped.device, dtype=torch.float32).view(-1, 1)
    x = torch.arange(2 * width, device=wrapped.device, dtype=torch.float32).view(1, -1)
    denominator = (
        2.0 * torch.cos(2 * math.pi * y / (2 * height))
        + 2.0 * torch.cos(2 * math.pi * x / (2 * width))
        - 4.0
    )
    denominator[0, 0] = 1.0                     # the constant term is unconstrained
    solved = fft.ifft2(spectrum / denominator).real
    return solved[..., :height, :width]
