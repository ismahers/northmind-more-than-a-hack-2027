"""Independent binary live/dead prior and exact offline Viterbi decoding.

No file access or home/away labels. Callers must supply only permitted external
training views when fitting; unknown alive labels never bridge observations.
"""

from collections.abc import Mapping
import math

import numpy as np


def _positive_fps(fps):
    if isinstance(fps, (bool, np.bool_)) or not np.isscalar(fps):
        raise ValueError("fps must be positive and finite")
    try:
        value = float(fps)
    except (TypeError, ValueError) as exc:
        raise ValueError("fps must be positive and finite") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("fps must be positive and finite")
    return value


def _clock(time, run, n, fps):
    t = np.asarray(time, dtype=float)
    r = np.asarray(run)
    if t.shape != (n,) or r.shape != (n,):
        raise ValueError("time and run must have shape (T,)")
    if not np.all(np.isfinite(t)):
        raise ValueError("time must be finite")
    if r.dtype.kind not in "iuf" or not np.all(np.isfinite(r)):
        raise ValueError("run must contain finite integer identifiers")
    if np.any(r != np.floor(r)):
        raise ValueError("run identifiers must be integers")
    same_run = r[1:] == r[:-1]
    delta = np.diff(t)
    if np.any(same_run & (delta <= 0)):
        raise ValueError("time must increase strictly inside each contiguous run")
    # One representational ULP permits exact nominal boundary gaps after
    # subtraction; it does not introduce an empirical timing tolerance.
    connected = same_run & (delta <= np.nextafter(1.5 / fps, np.inf))
    return connected


def fit_binary_prior(training_views, fps=10.0):
    """Fit D/L transitions using adjacent known external training labels only.

    Views expose ``alive``, ``time`` and ``run`` attributes (or mapping keys).
    Alive codes are -1 unknown, 0 dead, 1 live. Counts start at one in each
    cell (Laplace smoothing). Run changes, unknowns, view boundaries and gaps
    greater than 1.5/fps break adjacency. No known adjacent pair is an error;
    pseudocounts alone are insufficient to fit a prior. Inputs are untouched.
    """
    fps = _positive_fps(fps)
    counts = np.ones((2, 2), dtype=np.int64)
    observed = 0
    for view in training_views:
        def field(name):
            return view[name] if isinstance(view, Mapping) else getattr(view, name)

        alive = np.asarray(field("alive"))
        if alive.ndim != 1 or alive.dtype.kind not in "biuf":
            raise ValueError("alive must be a one-dimensional numeric array")
        if not np.all(np.isin(alive, [-1, 0, 1])):
            raise ValueError("alive must contain only -1, 0 or 1")
        connected = _clock(field("time"), field("run"), len(alive), fps)
        valid = connected & (alive[:-1] >= 0) & (alive[1:] >= 0)
        previous = alive[:-1][valid].astype(np.int64)
        following = alive[1:][valid].astype(np.int64)
        np.add.at(counts, (previous, following), 1)
        observed += int(np.count_nonzero(valid))
    if observed == 0:
        raise ValueError("at least one known adjacent training pair is required")
    return {
        "state_order": ("D", "L"),
        "transition_counts": counts,
        "observed_transition_count": observed,
        "transition_probabilities": counts / counts.sum(axis=1, keepdims=True),
    }


def decode_live_probability(
    alive_probability, time, run, prior, fps=10.0, strength=0.5,
):
    """Return the maximum-score binary path as a boolean live mask.

    Emissions are log(1-p), log(p), with p clipped to [1e-7, 1-1e-7]. Each
    segment starts uniformly and transitions add 0.5*log(P). Strength is fixed
    at 0.5 for this preregistered recipe. Clock/run breaks reset the whole DP;
    clocks may restart on a new run. The algorithm uses O(T*2) time/storage.
    Exact predecessor ties prefer staying; terminal ties prefer dead. This
    is an offline path decoder, so future evidence within a segment can affect
    an earlier state. No home/away state or labels are accepted or consulted.
    """
    fps = _positive_fps(fps)
    if isinstance(strength, (bool, np.bool_)) or strength != 0.5:
        raise ValueError("strength is fixed at 0.5")
    p = np.asarray(alive_probability, dtype=float)
    if p.ndim != 1 or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("alive_probability must be finite with shape (T,) in [0,1]")
    connected = _clock(time, run, len(p), fps)
    if not isinstance(prior, Mapping) or "transition_probabilities" not in prior:
        raise ValueError("prior must contain transition_probabilities")
    transition = np.asarray(prior["transition_probabilities"], dtype=float)
    if transition.shape != (2, 2) or not np.all(np.isfinite(transition)):
        raise ValueError("transition probabilities must be finite with shape (2,2)")
    if np.any(transition <= 0) or not np.allclose(
        transition.sum(axis=1), 1.0, rtol=0, atol=1e-12,
    ):
        raise ValueError("transition probabilities must be positive with unit row sums")
    if "state_order" in prior and tuple(prior["state_order"]) != ("D", "L"):
        raise ValueError("prior state order must be D,L")
    result = np.empty(len(p), dtype=bool)
    if not len(p):
        return result
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    emission = np.column_stack((np.log1p(-clipped), np.log(clipped)))
    log_transition = 0.5 * np.log(transition)
    parent = np.empty((len(p), 2), dtype=np.int8)
    starts = np.r_[0, np.flatnonzero(~connected) + 1]
    stops = np.r_[starts[1:], len(p)]
    for start, stop in zip(starts, stops):
        score = emission[start] - math.log(2)
        for i in range(start + 1, stop):
            following_score = np.empty(2)
            for state in (0, 1):
                stay = score[state] + log_transition[state, state]
                other = score[1 - state] + log_transition[1 - state, state]
                predecessor = state if stay >= other else 1 - state
                parent[i, state] = predecessor
                following_score[state] = max(stay, other) + emission[i, state]
            score = following_score
        state = int(score[1] > score[0])
        for i in range(stop - 1, start - 1, -1):
            result[i] = bool(state)
            if i > start:
                state = int(parent[i, state])
    return result
