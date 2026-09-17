"""Decide which propagation distance the forward-model arms should use.

Called by run_study.sh between calibration and the physics ablation. Prints one
line, ``<mode> <value>``, where mode is one of:

    fixed     a distance is available and trustworthy: either set by hand in
              config/base.yaml, or identified by scripts/calibrate_z.py with
              individual images agreeing on it. Held constant during training.

    learned   no identifiable distance, but the forward residual has a minimum
              somewhere. That minimum initialises an nn.Parameter which is then
              refined by gradient descent alongside the network weights. The
              converged value is logged every epoch and belongs in the paper.

    skip      nothing usable. The forward-model arms are left out rather than
              trained against a meaningless distance, because a wrong z makes
              the residual measure the error in z instead of the error in the
              reconstruction -- which is worse than not training the term.

A value set explicitly in the configuration always wins: if someone has been
given the real acquisition distance, no search should override it.

Only the in-line geometry genuinely constrains z. An off-axis hologram encodes
phase in its carrier fringes at any distance, so its residual is nearly flat and
its minimum means little. The same specimen was recorded at the same distance in
both, so the Gabor answer is preferred and then applies to both arms.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_ARMS = ("gabor", "off_axis")          # in-line first: it is the one that constrains z


def _calibration_path(config_path: str, calibration_path: str | None) -> Path:
    """Where calibrate_z.py writes, derived from the config rather than assumed.

    The default was the literal ``"runs/z_calibration.json"``, which silently
    diverges from ``paths.output_root`` -- and if it ever differed, this resolver
    returned "skip" and the caller printed "the term does not discriminate on
    any arm", which is a measurement claim about a file it never found.
    """
    if calibration_path is not None:
        return Path(calibration_path)
    from holoqpi.config import load_config

    return Path(load_config(config_path).paths.output_root) / "z_calibration.json"


def resolve(config_path: str = "config/base.yaml",
            calibration_path: str | None = None) -> tuple[str, str]:
    # EXCEPTIONS ARE NOT SWALLOWED HERE ANY MORE.
    #
    # A bare `except Exception: configured = None` made a typo'd path, a YAML
    # syntax error and a genuinely null distance_um indistinguishable. With
    # base.yaml setting 33.77, the correct answer is "fixed 33.770000"; a broken
    # config downgraded that to a calibration lookup or to "skip", and the
    # caller then reported "No usable propagation distance" as though it were a
    # result. A config that cannot be read is a hard error.
    from holoqpi.config import load_config

    configured = load_config(config_path).loss.forward_model.distance_um

    if configured is not None:
        return "fixed", f"{float(configured):.6f}"

    path = _calibration_path(config_path, calibration_path)
    if not path.is_file():
        return "skip", ""
    # A corrupt calibration file is reported, not treated as an absent one.
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError(
            f"{path} exists but could not be parsed ({exc}). Re-run "
            f"scripts/calibrate_z.py rather than treating this as 'no calibration'."
        ) from exc

    # A term whose residual FALLS when the phase is degraded would drive the
    # reconstruction away from the truth. Refuse outright for any arm where that
    # is the case; the decision is per modality, because on this data the
    # approximate operator discriminates correctly in-line and anti-correlates
    # off-axis.
    #
    # NOTE ON WHAT `forward_model_usable` MEANS. calibrate_z.py sets it to
    # `verdict == "usable"`, so it is also False for `marginal` and
    # `uninformative` -- not only for the anti-discriminative case the paragraph
    # above describes. The filter is conservative in the right direction; the
    # wording used to claim it tested only the sign.
    usable = [
        arm for arm in _ARMS if (payload.get(arm) or {}).get("forward_model_usable")
    ]
    if not usable:
        return "skip", ""

    for arm in usable:
        entry = payload.get(arm) or {}
        recommended = entry.get("recommended_z_um")
        if entry.get("identifiable") and recommended is not None:
            return "fixed", f"{float(recommended):.6f}"

    # Nothing identifiable. Fall back to the best forward minimum as a starting
    # point for gradient refinement, provided it is not sitting at zero (where
    # the in-line forward model is degenerate and carries no phase at all).
    for arm in usable:
        entry = payload.get(arm) or {}
        start = entry.get("z_forward_um")
        if start is not None and abs(float(start)) > 1e-6:
            return "learned", f"{float(start):.6f}"

    return "skip", ""


def usable_modalities(config_path: str = "config/base.yaml",
                      calibration_path: str | None = None) -> list[str]:
    """Arms whose forward-model verdict is `usable`.

    An absent calibration returns an empty list -- that is a legitimate "not
    measured yet". A corrupt one raises, because silently returning the same
    empty list would make an unreadable file look like a measured negative.
    """
    path = _calibration_path(config_path, calibration_path)
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError(
            f"{path} exists but could not be parsed ({exc}). Re-run "
            f"scripts/calibrate_z.py."
        ) from exc
    return [arm for arm in _ARMS if (payload.get(arm) or {}).get("forward_model_usable")]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--modalities", action="store_true",
                        help="print the arms the forward-model term may be trained on")
    # Both paths are overridable now. They were hardcoded in the signatures with
    # no CLI at all, so paths.output_root was bypassed entirely.
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--calibration", default=None,
                        help="default: <paths.output_root>/z_calibration.json")
    arguments = parser.parse_args()
    if arguments.modalities:
        print(" ".join(usable_modalities(arguments.config, arguments.calibration)))
    else:
        mode, value = resolve(arguments.config, arguments.calibration)
        print(mode, value)
