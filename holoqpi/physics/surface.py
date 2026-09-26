"""Smooth-surface fitting and the off-axis conjugate ambiguity.

Two operations that look unrelated but are the same piece of algebra, and both
of which the rest of the pipeline was getting wrong in the same way.

A low-order polynomial in (x, y) describes what an objective's curvature and a
reference-beam tilt do to the phase across a field. Fitting one is how the
aberration surface is recovered (``scripts/estimate_aberration.py``) and how a
classical reconstruction is flattened before measurement
(``scripts/conventional_baseline.py``).

Removing that surface is also what makes the conjugate ambiguity decidable. An
off-axis hologram carries the object in two conjugate first-order sidebands of
equal magnitude, so which one a spectral ``argmax`` returns is arbitrary and
flips between fields of the same acquisition. Taking the wrong one returns the
conjugate field, whose phase is negated -- a reconstruction that looks entirely
plausible and is anti-correlated with the truth.

The disambiguating fact is physical rather than numerical: cells are optically
denser than their medium, so they add optical path, and a field of sparse cells
on a flat background is right-skewed in phase. That prior needs no mask, no
threshold and no reference, so it can be applied wherever a field is
reconstructed, including at inference time.

It does, however, need the surface removed first, and this is not a detail. On
this dataset the raw unwrapped skewness picks the WRONG sideband on 13 of 13
fields, and the wrapped skewness picks correctly on 6 of 13 -- a coin flip. The
objective's curvature is itself a strongly skewed bowl spanning tens of radians,
and it dwarfs the few radians the cells contribute. Detrending with a plane
(order 1) does not help; the curvature is quadratic. From order 2 the rule is
correct on 13 of 13 with a margin of at least 1.2 in skewness units. Hence
``detrend_order`` defaults to nothing and must be supplied from configuration,
and anything below 2 is refused.
"""

from __future__ import annotations

import torch


def polynomial_basis(
    height: int,
    width: int,
    order: int,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """(H*W, K) design matrix for a 2-D polynomial on a normalised grid.

    The coordinates run over [-1, 1] on the grid passed in, so coefficients
    fitted on a full field can be evaluated on a crop by supplying that crop's
    own coordinates -- which is what keeps a stored aberration surface
    registered to the hologram after cropping.
    """
    y = torch.linspace(-1, 1, height, device=device, dtype=dtype).view(-1, 1).expand(height, width)
    x = torch.linspace(-1, 1, width, device=device, dtype=dtype).view(1, -1).expand(height, width)
    columns = [(x ** i) * (y ** j) for i in range(order + 1) for j in range(order + 1 - i)]
    return torch.stack(columns, dim=-1).reshape(-1, len(columns))


def fit_polynomial_surface(
    values: torch.Tensor, order: int, ridge: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Least-squares polynomial fit to a 2-D field.

    Returns (coefficients, fitted_surface, r_squared). Solved through the normal
    equations with a small ridge because the high-order monomials are strongly
    collinear on a dense grid and a plain ``lstsq`` loses accuracy above order 3.
    """
    height, width = values.shape
    design = polynomial_basis(height, width, order, values.device, torch.float64)
    flat = values.reshape(-1).to(torch.float64)

    gram = design.T @ design
    stabiliser = ridge * torch.diag(torch.diagonal(gram).clamp(min=1e-12))
    coefficients = torch.linalg.solve(gram + stabiliser, design.T @ flat)

    fitted = design @ coefficients
    residual = flat - fitted
    variance = float(flat.var())
    r_squared = 1.0 - float(residual.var()) / variance if variance > 0 else float("nan")
    return coefficients, fitted.reshape(height, width), r_squared


def detrend_polynomial(values: torch.Tensor, order: int) -> torch.Tensor:
    """Remove the best-fitting low-order surface, leaving the object structure."""
    _, fitted, _ = fit_polynomial_surface(values, order)
    return values.to(torch.float64) - fitted


def phase_skewness(values: torch.Tensor, detrend_order: int) -> float:
    """Skewness of a detrended phase field.

    Positive when sparse, optically denser objects sit on a flat background,
    which is what a field of cells in medium looks like.
    """
    if detrend_order < 2:
        raise ValueError(
            "detrend_order must be at least 2: the objective's curvature is a "
            "quadratic bowl whose own skewness exceeds the cells', so removing "
            "only a plane leaves the sign of the result set by the aberration "
            "rather than by the specimen."
        )
    residual = detrend_polynomial(values, detrend_order)
    centred = residual - residual.mean()
    deviation = centred.std().clamp(min=1e-12)
    return float((centred ** 3).mean() / deviation ** 3)


def resolve_conjugate(
    field: torch.Tensor,
    detrend_order: int,
    min_skewness: float = 0.0,
    unwrap: bool = True,
) -> tuple[torch.Tensor, list[bool], list[float]]:
    """Pick the physically correct one of the two conjugate sidebands.

    ``field`` is (B, 1, H, W) complex. Returns the corrected field, a per-sample
    flag saying whether it was conjugated, and the skewness the decision rested
    on. A sample whose |skewness| falls below ``min_skewness`` is left untouched
    and reported with its value, so an undecidable field is visible to the
    caller rather than silently assigned a sign.

    The phase is unwrapped before the test by default. Wrapping folds the
    surface into (-pi, pi] and destroys the asymmetry the rule reads: on this
    dataset the wrapped test is right on 6 of 13 fields, which is chance.
    """
    from .propagation import unwrap_phase_2d

    angle = torch.angle(field)
    if unwrap:
        angle = unwrap_phase_2d(angle)

    corrected, flipped, skewness = [], [], []
    for item in range(field.shape[0]):
        value = phase_skewness(angle[item, 0], detrend_order)
        flip = value < 0.0 and abs(value) >= min_skewness
        corrected.append(field[item].conj() if flip else field[item])
        flipped.append(bool(flip))
        skewness.append(value)
    return torch.stack(corrected), flipped, skewness
