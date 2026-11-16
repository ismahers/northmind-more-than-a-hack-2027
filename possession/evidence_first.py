"""Four-view evidence fusion before pulse detection; frozen v1 fusion is reused."""
from dataclasses import dataclass, fields

import numpy as np

from possession.scale_selective.fusion import (
    fuse_scale_selective, short_pulse_mask, team_logit_from_probabilities,
)


@dataclass(frozen=True)
class Evidence:
    local: np.ndarray
    multi: np.ndarray
    teacher: np.ndarray
    alternate: np.ndarray


def aligned_evidence(primary_p, alternate_p, local, multi, swap=False):
    """Undo team swaps in conditional logits, before any nonlinear pulse gate.

    Reflecting XY needs no polarity change. Average teacher *logits*, not
    probabilities: this isolates the order of the pulse nonlinearity.
    """
    sign = -1 if swap else 1
    return Evidence(*(sign * np.asarray(a, dtype=np.float32) for a in (
        local, multi, team_logit_from_probabilities(primary_p),
        team_logit_from_probabilities(alternate_p))))


def mean_evidence(views):
    if len(views) != 4:
        raise ValueError("Exactly four aligned views required")
    shape = views[0].local.shape
    if len(shape) != 1:
        raise ValueError("One-dimensional evidence required")
    values = []
    for field in fields(Evidence):
        arrays = [np.asarray(getattr(view, field.name)) for view in views]
        if any(a.shape != shape or not np.isfinite(a).all() for a in arrays):
            raise ValueError("Invalid or misaligned view evidence")
        values.append(np.mean(arrays, axis=0, dtype=np.float64).astype(np.float32))
    return Evidence(*values)


def fuse_evidence(evidence, observed, run, params, config):
    return fuse_scale_selective(
        evidence.local, evidence.multi, observed, run,
        z_teacher=evidence.teacher, z_alt_teacher=evidence.alternate,
        params=params, obs_window=config["obs_window"],
        pulse_min_frames=config["pulse_min_frames"],
    )


def pulse_mask(evidence, observed, run, params, config):
    """Diagnostic only; the prediction uses the unchanged registered fusion."""
    weight = params["alt_seed_weight"]
    teacher = (1 - weight) * evidence.teacher + weight * evidence.alternate
    blend = .5 * params["teacher_blend"]
    sharp = (1 - blend) * evidence.local + blend * teacher
    return short_pulse_mask(sharp, evidence.multi, observed, run,
        min_frames=config["pulse_min_frames"], max_frames=params["pulse_max_frames"],
        min_conf=params["pulse_min_conf"])
