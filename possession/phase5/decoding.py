"""H/A-only local potentials. D transitions and duration tables stay frozen."""

import numpy as np
from numba import njit

from possession.learning.temporal import TemporalDecoder
from possession.night.decoders import change_evidence, dynamic_viterbi


@njit(cache=True)
def boundary_hsmm(emission, jumps, log_duration, log_survival, tail, tail_mass, boundary, ha_bonus):
    """Same unbounded-tail HSMM recurrence; HA bonus indexed by destination frame.

    A segment beginning at u receives bonus[u]. It never receives bonus[u-1].
    The original probability-variation boundary term is preserved separately.
    """
    n, cap = len(emission), log_duration.shape[1]
    prefix = np.zeros((n + 1, 3))
    for t in range(n):
        prefix[t + 1] = prefix[t] + emission[t]
    values = np.full((n + 1, 3), -np.inf)
    entry = np.full((n + 1, 3), -np.inf)
    entry[0] = -np.log(3.)
    from_state = np.full((n + 1, 3), -1, dtype=np.int8)
    start = np.zeros((n + 1, 3), dtype=np.int32)
    tail_best, tail_start = np.full(3, -np.inf), np.zeros(3, dtype=np.int32)
    for t in range(1, n + 1):
        for state in range(3):
            score, winning_start = -np.inf, 0
            for duration in range(1, min(cap - 1, t) + 1):
                u = t - duration
                duration_score = log_survival[state, duration] if t == n else log_duration[state, duration]
                value = entry[u, state] + prefix[t, state] - prefix[u, state] + duration_score
                if value > score:
                    score, winning_start = value, u
            u = t - cap
            if u >= 0:
                candidate = entry[u, state] - prefix[u, state] - u * tail[state, 0]
                if candidate > tail_best[state]:
                    tail_best[state], tail_start[state] = candidate, u
                exit_score = 0. if t == n else tail[state, 1]
                candidate = prefix[t, state] + (t - cap) * tail[state, 0] + tail_mass[state] + exit_score + tail_best[state]
                if candidate > score:
                    score, winning_start = candidate, tail_start[state]
            values[t, state], start[t, state] = score, winning_start
        for state in range(3):
            score, before = -np.inf, 0
            for old in range(3):
                bonus = ha_bonus[t] if t < n and old < 2 and state < 2 and old != state else 0.
                candidate = values[t, old] + jumps[old, state] + bonus
                if candidate > score:
                    score, before = candidate, old
            entry[t, state] = score + boundary[t - 1]
            from_state[t, state] = before
    state = np.argmax(values[n])
    result = np.empty(n, dtype=np.int8)
    t = n
    while t > 0:
        u = start[t, state]
        result[u:t] = state
        state, t = from_state[u, state], u
    return result


class AdaptiveDecoder:
    def __init__(self, specification, priors, view, boundary=None, strength=0., threshold=.5):
        self.specification, self.priors = specification, priors
        self.original = TemporalDecoder(specification, priors)
        self.boundary, self.strength, self.threshold = boundary, strength, threshold
        self.indices = {}
        for run in np.unique(view.run):
            ix = np.flatnonzero(view.run == run)
            self.indices[(float(view.time[ix[0]]), float(view.time[ix[-1]]), len(ix))] = ix

    def decode(self, p, times):
        if self.strength == 0:
            return self.original.decode(p, times)
        if self.boundary is None:
            ha, ah = change_evidence(p, 3)
            evidence = np.maximum(ha, ah)
        else:
            ix = self.indices[(float(times[0]), float(times[-1]), len(times))]
            alive = p[:, :2].sum(-1)
            # Inference gate uses predicted alive, never the true D mask.
            evidence = np.clip((self.boundary[ix] - self.threshold) / (1 - self.threshold), 0, 1) * alive
        evidence[0] = 0
        s, prior = self.specification, self.priors
        scale = s["strength"]
        # The unit is the original mean H/A transition cost. No hard minimum spell.
        unit = max(1., -.5 * float(prior["hmm_log_transition"][0, 1] + prior["hmm_log_transition"][1, 0]) * scale)
        bonus = np.ascontiguousarray(self.strength * unit * evidence, dtype=np.float64)
        emission = np.log(np.clip(p.astype(np.float64), 1e-7, 1))
        if s["kind"] == "hmm":
            return dynamic_viterbi(emission, prior["hmm_log_transition"] * scale, bonus, bonus)
        if s["kind"] != "hsmm":
            raise ValueError("Frozen phase5 reference must be HMM or HSMM")
        boundary = np.zeros(len(p), np.float64)
        boundary[:-1] = s.get("boundary_strength", 0.) * np.log(.01 + np.abs(p[1:] - p[:-1]).sum(-1) * .5)
        return boundary_hsmm(emission, prior["jump_log_transition"] * scale,
                             prior["log_duration"] * scale, prior["log_survival"] * scale,
                             prior["tail"] * scale, prior["tail_mass"] * scale, boundary, bonus)
