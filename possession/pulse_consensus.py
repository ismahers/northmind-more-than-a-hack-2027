"""Boundary agreement gate for the difference between the two frozen fusions."""
import numpy as np
from possession.evidence_first import mean_evidence, pulse_mask


def sharp(e, params):
    w = params["alt_seed_weight"]
    teacher = (1-w)*e.teacher + w*e.alternate
    blend = .5*params["teacher_blend"]
    return (1-blend)*e.local + blend*teacher


def spans(signal, run):
    # Zero is its own state, preserving H/A equivariance at exact ties.
    signs = np.sign(signal)
    edges = np.r_[0, np.flatnonzero((signs[1:] != signs[:-1]) |
                                  (run[1:] != run[:-1]))+1, len(signal)]
    return [(int(a), int(b), signs[a]) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def boundary_gate(views, observed, run, params, config):
    combined = mean_evidence(views)  # Also validates complete finite ensemble.
    run = np.asarray(run)
    if run.shape != combined.local.shape or np.asarray(observed).shape != run.shape:
        raise ValueError("Misaligned timeline")
    mask = pulse_mask(combined, observed, run, params, config)
    view_spans = [spans(sharp(e, params), run) for e in views]
    tolerance = config["pulse_min_frames"]  # Existing three-frame resolution.
    weight = np.zeros(len(run), np.float32)
    for a, b, sign in spans(sharp(combined, params), run):
        if sign == 0 or not mask[a:b].all():
            continue
        support = sum(any(s == sign and run[left] == run[a]
                          and abs(left-a) <= tolerance and abs(right-b) <= tolerance
                          and (min(b, right)-max(a, left))/max(b-a, right-left) >= .5
                          for left, right, s in rows) for rows in view_spans)
        if support >= 3:
            weight[a:b] = 1
    return weight


def candidates(old_z, new_z, views, observed, run, params, config):
    weight = boundary_gate(views, observed, run, params, config)
    return {"half_residual": .5*(old_z+new_z),
            "boundary_consensus": old_z + weight*(new_z-old_z)}, weight
