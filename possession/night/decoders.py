"""Small, explicit temporal ablations on frozen probabilities; no target features."""

import numpy as np
from numba import njit

from possession.learning.decoding import validate_probabilities
from possession.learning.temporal import TemporalDecoder, hmm_viterbi


@njit(cache=True)
def dynamic_viterbi(emission, transition, ha_bonus, ah_bonus):
    n = len(emission)
    back = np.zeros((n, 3), dtype=np.int8)
    previous = emission[0] - np.log(3.)
    for t in range(1, n):
        current = np.empty(3)
        for state in range(3):
            score, winner = -np.inf, 0
            for old in range(3):
                bonus = ha_bonus[t] if old == 0 and state == 1 else ah_bonus[t] if old == 1 and state == 0 else 0.
                value = previous[old] + transition[old, state] + bonus
                if value > score:
                    score, winner = value, old
            current[state] = score + emission[t, state]
            back[t, state] = winner
        previous = current
    state = np.argmax(previous)
    path = np.empty(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        path[t], state = state, back[t, state]
    return path


def change_evidence(p, window):
    p = p.astype(np.float64)
    alive = np.maximum(1e-7, p[:, :2].sum(-1))
    h = p[:, 0] / alive
    prefix = np.r_[0., np.cumsum(h)]
    alive_prefix = np.r_[0., np.cumsum(alive)]
    t = np.arange(len(p))
    lo, hi = np.maximum(0, t - window), np.minimum(len(p), t + window)
    left = (prefix[t] - prefix[lo]) / np.maximum(1, t - lo)
    right = (prefix[hi] - prefix[t]) / np.maximum(1, hi - t)
    gate = np.minimum((alive_prefix[t] - alive_prefix[lo]) / np.maximum(1, t - lo),
                      (alive_prefix[hi] - alive_prefix[t]) / np.maximum(1, hi - t))
    ha = np.maximum(0., left - right) * (left > .5) * (right < .5) * gate
    ah = np.maximum(0., right - left) * (left < .5) * (right > .5) * gate
    ha[0], ah[0] = 0., 0.
    return ha, ah


def scaled_hmm_duration(priors, factor):
    transition = np.exp(priors["hmm_log_transition"]).copy()
    for state in (0, 1):
        hazard = 1 - transition[state, state]
        new_hazard = np.clip(hazard * factor, 1e-6, 1 - 1e-6)
        transition[state] *= new_hazard / hazard
        transition[state, state] = 1 - new_hazard
    result = priors["hmm_log_transition"].copy()
    result[:2] = np.log(transition[:2])
    return result


def scaled_hsmm_duration(priors, factor):
    result = {key: np.array(value, copy=True) for key, value in priors.items()}
    for state in (0, 1):
        hazard = np.clip(np.exp(priors["log_duration"][state, 1:] - priors["log_survival"][state, 1:]) * factor, 1e-6, 1 - 1e-6)
        log_s = np.r_[0., np.cumsum(np.log1p(-hazard))]
        result["log_duration"][state, 1:] = log_s[:-1] + np.log(hazard)
        result["log_survival"][state, 1:] = log_s[:-1]
        result["tail_mass"][state] = log_s[-1]
        tail_hazard = np.clip(np.exp(priors["tail"][state, 1]) * factor, 1e-6, 1 - 1e-6)
        result["tail"][state] = [np.log1p(-tail_hazard), np.log(tail_hazard)]
    return result


def refine_boundaries(path, p, radius, minimum):
    bounds = np.r_[0, np.flatnonzero(path[1:] != path[:-1]) + 1, len(path)]
    original = bounds.copy()
    states = path[bounds[:-1]]
    prefix = np.vstack([np.zeros((1, 3)), np.cumsum(np.log(np.clip(p.astype(np.float64), 1e-7, 1)), axis=0)])
    for j in range(1, len(bounds) - 1):
        lo = max(original[j] - radius, bounds[j - 1] + minimum)
        hi = min(original[j] + radius, bounds[j + 1] - minimum)
        if lo <= hi:
            positions = np.arange(lo, hi + 1)
            scores = prefix[positions, states[j - 1]] - prefix[positions, states[j]]
            bounds[j] = positions[np.argmax(scores)]
    result = path.copy()
    for a, b, state in zip(bounds[:-1], bounds[1:], states):
        result[a:b] = state
    return result


@njit(cache=True)
def confidence_hysteresis(path, p, threshold, required, decay):
    result = path.copy()
    state, accumulated, live_before = 0, 0., False
    for t in range(len(p)):
        if path[t] == 2:
            live_before, accumulated = False, 0.
            continue
        if not live_before:
            state, accumulated, live_before = path[t], 0., True
        other = 1 - state
        confidence = p[t, other] / max(1e-7, p[t, 0] + p[t, 1])
        if confidence >= threshold:
            q = min(1 - 1e-7, confidence)
            accumulated = decay * accumulated + np.log(q / max(1e-7, 1 - q))
        elif confidence <= 1 - threshold:
            accumulated = 0.
        else:
            accumulated *= decay
        if accumulated >= required:
            state, accumulated = other, 0.
        result[t] = state
    return result


class NightDecoder:
    def __init__(self, specification, bases, priors, base_cache=None):
        self.specification, self.bases, self.priors = specification, bases, priors
        self.cache = {} if base_cache is None else base_cache

    def base(self, family, p, timestamps):
        key = (family, float(timestamps[0]), float(timestamps[-1]), len(p))
        if key not in self.cache:
            self.cache[key] = TemporalDecoder(self.bases[family], self.priors).decode(p, timestamps)
        return self.cache[key]

    def decode(self, probabilities, timestamps):
        p = validate_probabilities(probabilities)
        if len(p) == 0 or len(p) != len(timestamps) or np.any(np.diff(timestamps) <= 0):
            raise ValueError("Decode one nonempty increasing run")
        s, family = self.specification, self.specification["family"]
        if family in ("frozen", "hmm_reference", "hsmm_reference"):
            return self.base(family.replace("_reference", ""), p, timestamps)
        if family == "refine":
            return refine_boundaries(self.base("frozen", p, timestamps), p, s["radius"], s["minimum"])
        if family == "hysteresis":
            return confidence_hysteresis(self.base("frozen", p, timestamps), p, s["threshold"], s["required"], s["decay"])
        if family == "hsmm_duration":
            return TemporalDecoder(self.bases["hsmm"], scaled_hsmm_duration(self.priors, s["factor"])).decode(p, timestamps)
        transition = self.priors["hmm_log_transition"].copy()
        strength = self.bases["hmm"]["strength"]
        ha, ah = np.zeros(len(p)), np.zeros(len(p))
        if family == "hmm_duration":
            transition = scaled_hmm_duration(self.priors, s["factor"])
        elif family == "hmm_asymmetric":
            transition[0, 1] *= s["ratio"] * np.exp(s["bias"])
            transition[1, 0] *= s["ratio"] * np.exp(-s["bias"])
        elif family == "hmm_evidence":
            transition[0, 1] *= s["ratio"]
            transition[1, 0] *= s["ratio"]
            ha, ah = change_evidence(p, s["window"])
            ha, ah = ha * s["bonus"], ah * s["bonus"]
        else:
            raise ValueError("Unknown decoder ablation")
        return dynamic_viterbi(np.log(np.clip(p.astype(float), 1e-7, 1)), transition * strength, ha, ah)


def candidate_grid(protocol):
    g = protocol["decoder_grid"]
    result = [{"family": "frozen"}, {"family": "hmm_reference"}, {"family": "hsmm_reference"}]
    result += [{"family": "hmm_asymmetric", "ratio": r, "bias": b}
               for r in g["asymmetric_team_penalty_ratios"] for b in g["direction_bias"]]
    result += [{"family": "hmm_evidence", "ratio": r, "bonus": b, "window": w}
               for r in g["evidence_team_penalty_ratios"] for b in g["evidence_strength"] for w in g["evidence_window_frames"]]
    result += [{"family": family, "factor": f} for family in ("hmm_duration", "hsmm_duration") for f in g["duration_hazard_factors"]]
    result += [{"family": "refine", "radius": r, "minimum": m} for r in g["refine_radius_frames"] for m in g["refine_min_segment_frames"]]
    result += [{"family": "hysteresis", "threshold": t, "required": e, "decay": g["hysteresis_decay"]}
               for t in g["hysteresis_threshold"] for e in g["hysteresis_evidence"]]
    return [{"id": str(i).zfill(3), **specification} for i, specification in enumerate(result)]
