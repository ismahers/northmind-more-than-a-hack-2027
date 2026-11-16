"""Joint H/A/D conditional-potential decoders, fit on training labels only.

HSMM uses O(T*S*Dcap) time and O(T*S) backpointers. A running maximum
implements an unbounded geometric tail exactly, without a hard spell-length cap.
The final segment is right-censored (survival rather than exit probability).
"""

import numpy as np
from numba import njit

from .decoding import validate_probabilities


def label_runs(labels, run):
    for identity in np.unique(run):
        positions = np.flatnonzero(run == identity)
        a, b = positions[0], positions[-1] + 1
        changes = np.r_[a, np.flatnonzero(labels[a + 1:b] != labels[a:b - 1]) + a + 1, b]
        yield [(int(x), int(y), int(labels[x])) for x, y in zip(changes[:-1], changes[1:])]


def fit_priors(training_views, cap=300):
    counts = np.ones((3, 3), dtype=np.float64)
    durations = [[], [], []]
    for view in training_views:
        labels = np.where(view.alive == 0, 2, np.where(view.alive == 1, view.team, -1))
        for spans in label_runs(labels, view.run):
            for a, b, state in spans:
                if state >= 0:
                    counts[state, state] += max(0, b - a - 1)
            for previous, current in zip(spans[:-1], spans[1:]):
                if previous[2] >= 0 and current[2] >= 0:
                    counts[previous[2], current[2]] += 1
            # Exclude both edges and unknown-adjacent spans from duration estimation.
            for i, (a, b, state) in enumerate(spans):
                if state >= 0 and 0 < i < len(spans) - 1 and spans[i - 1][2] >= 0 and spans[i + 1][2] >= 0:
                    durations[state].append(b - a)
    transition = np.log(counts / counts.sum(-1, keepdims=True))
    jumps = counts.copy()
    np.fill_diagonal(jumps, 0)
    jump_log = np.log(np.maximum(jumps / jumps.sum(-1, keepdims=True), 1e-300))
    np.fill_diagonal(jump_log, -np.inf)
    pooled = durations[0] + durations[1]
    tables, survival, tails, mass = [], [], [], []
    for values in (pooled, pooled, durations[2]):
        lengths = np.asarray(values if values else [10], dtype=np.int64)
        global_hazard = np.clip(len(lengths) / lengths.sum(), 1e-4, 0.5)
        risk = np.array([(lengths >= d).sum() for d in range(1, cap)], dtype=float)
        exits = np.array([(lengths == d).sum() for d in range(1, cap)], dtype=float)
        hazard = np.clip((exits + 20 * global_hazard) / (risk + 20), 1e-6, 1 - 1e-6)
        log_survival = np.r_[0., np.cumsum(np.log1p(-hazard))]
        table = np.r_[-np.inf, log_survival[:-1] + np.log(hazard)]
        # cap includes the first tail duration. Its survival mass is retained.
        tail_risk = np.maximum(0, lengths - cap + 1).sum()
        tail_exits = (lengths >= cap).sum()
        tail_hazard = np.clip((tail_exits + 20 * global_hazard) / (tail_risk + 20), 1e-6, 1 - 1e-6)
        tables.append(table)
        survival.append(np.r_[-np.inf, log_survival[:-1]])
        tails.append([np.log1p(-tail_hazard), np.log(tail_hazard)])
        mass.append(log_survival[-1])
    return {"hmm_log_transition": transition, "jump_log_transition": jump_log,
            "log_duration": np.asarray(tables), "log_survival": np.asarray(survival),
            "tail": np.asarray(tails), "tail_mass": np.asarray(mass), "cap": cap,
            "duration_counts": [len(x) for x in durations]}


@njit(cache=True)
def hmm_viterbi(emission, transition):
    n = len(emission)
    back = np.zeros((n, 3), dtype=np.int8)
    previous = emission[0] - np.log(3.)
    for t in range(1, n):
        current = np.empty(3)
        for state in range(3):
            score, winner = -np.inf, 0
            for before in range(3):
                value = previous[before] + transition[before, state]
                if value > score:
                    score, winner = value, before
            current[state] = score + emission[t, state]
            back[t, state] = winner
        previous = current
    state = np.argmax(previous)
    result = np.empty(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        result[t] = state
        state = back[t, state]
    return result


@njit(cache=True)
def hsmm_viterbi(emission, jumps, log_duration, log_survival, tail, tail_mass, boundary, censor_end=True):
    n, cap = len(emission), log_duration.shape[1]
    prefix = np.zeros((n + 1, 3))
    for t in range(n):
        prefix[t + 1] = prefix[t] + emission[t]
    values = np.full((n + 1, 3), -np.inf)
    entry = np.full((n + 1, 3), -np.inf)
    entry[0] = -np.log(3.)
    from_state = np.full((n + 1, 3), -1, dtype=np.int8)
    start = np.zeros((n + 1, 3), dtype=np.int32)
    tail_best = np.full(3, -np.inf)
    tail_start = np.zeros(3, dtype=np.int32)
    for t in range(1, n + 1):
        for state in range(3):
            score, winning_start = -np.inf, 0
            for duration in range(1, min(cap - 1, t) + 1):
                u = t - duration
                duration_score = log_survival[state, duration] if censor_end and t == n else log_duration[state, duration]
                value = entry[u, state] + prefix[t, state] - prefix[u, state] + duration_score
                if value > score:
                    score, winning_start = value, u
            u = t - cap
            if u >= 0:
                candidate = entry[u, state] - prefix[u, state] - u * tail[state, 0]
                if candidate > tail_best[state]:
                    tail_best[state], tail_start[state] = candidate, u
                exit_score = 0. if censor_end and t == n else tail[state, 1]
                candidate = (prefix[t, state] + (t - cap) * tail[state, 0]
                             + tail_mass[state] + exit_score + tail_best[state])
                if candidate > score:
                    score, winning_start = candidate, tail_start[state]
            values[t, state], start[t, state] = score, winning_start
        for state in range(3):
            score, before = -np.inf, 0
            for old in range(3):
                candidate = values[t, old] + jumps[old, state]
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


class TemporalDecoder:
    def __init__(self, specification, priors):
        self.specification, self.priors = specification, priors

    def decode(self, probabilities, timestamps):
        p = validate_probabilities(probabilities)
        if not len(p) or len(p) != len(timestamps) or np.any(np.diff(timestamps) <= 0):
            raise ValueError("Decoder requires one nonempty monotone run")
        kind = self.specification["kind"]
        if kind == "argmax":
            return p.argmax(-1).astype(np.int8)
        if kind == "smooth":
            radius = self.specification["frames"] // 2
            cumulative = np.vstack((np.zeros((1, 3)), np.cumsum(p.astype(np.float64), axis=0)))
            lo = np.maximum(0, np.arange(len(p)) - radius)
            hi = np.minimum(len(p), np.arange(len(p)) + radius + 1)
            return (cumulative[hi] - cumulative[lo]).argmax(-1).astype(np.int8)
        strength = self.specification["strength"]
        emission = np.log(np.clip(p.astype(np.float64), 1e-7, 1))
        if kind == "hmm":
            return hmm_viterbi(emission, self.priors["hmm_log_transition"] * strength)
        if kind != "hsmm":
            raise ValueError("Unknown decoder")
        boundary = np.zeros(len(p), dtype=np.float64)
        # Available at inference, derived solely from output probabilities; not a learned head.
        boundary[:-1] = self.specification.get("boundary_strength", 0.) * np.log(
            0.01 + np.abs(p[1:] - p[:-1]).sum(-1) * 0.5)
        return hsmm_viterbi(emission, self.priors["jump_log_transition"] * strength,
            self.priors["log_duration"] * strength, self.priors["log_survival"] * strength,
            self.priors["tail"] * strength, self.priors["tail_mass"] * strength, boundary)


def decoder_grid(protocol):
    grid = protocol["decoder_grid"]
    result = [{"kind": "argmax"}]
    result += [{"kind": "smooth", "frames": w} for w in grid["smooth_frames"]]
    result += [{"kind": "hmm", "strength": s} for s in grid["hmm_strength"]]
    result += [{"kind": "hsmm", "strength": s, "boundary_strength": b}
               for s in grid["hsmm_strength"] for b in grid["hsmm_boundary_strength"]]
    return result
