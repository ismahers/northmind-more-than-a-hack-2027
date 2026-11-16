"""Only training GT determines weights; inference reuses the unchanged head."""
import numpy as np
from possession.external_data.audit import joint_labels, segments


def native_durations(labels, runs, fps, censored_unit_weight=False):
    if fps <= 0:
        raise ValueError("Positive native frequency required")
    out = np.zeros(len(labels), np.float32)
    for a, b in runs:
        edges = np.r_[a, np.flatnonzero(labels[a+1:b] != labels[a:b-1])+a+1, b]
        for lo, hi in zip(edges[:-1], edges[1:]):
            if 0 <= labels[lo] < 2:
                # Infinite sentinel only in loss-target preparation, never an input:
                # a censored spell gets clipped raw weight one.
                out[lo:hi] = np.inf if censored_unit_weight and (lo==a or hi==b) else (hi-lo)/fps
    return out


def train_durations(match, view, allowed_train_keys):
    if match.key not in allowed_train_keys:
        raise ValueError("Duration weighting is restricted to own training matches")
    durations = native_durations(joint_labels(match), segments(match), match.metadata["native_fps"],True)
    return durations[view.native_index]


def raw_weights(duration, labels, config, cap):
    valid = (labels >= 0) & (labels < 2)
    if np.any(valid & (duration <= 0)):
        raise ValueError("Supervised live anchors require native spell duration")
    out = np.zeros(len(labels), np.float32)
    out[valid] = np.clip((config["reference_seconds"]/np.maximum(duration[valid],
                        config["minimum_seconds"]))**config["power"], 1., cap)
    return out


def normalize_train(raw, labels):
    selected = [w[(y >= 0) & (y < 2)] for w, y in zip(raw, labels)]
    if not selected or sum(len(x) for x in selected) == 0:
        raise ValueError("No supervised training anchors")
    mean = float(np.concatenate(selected).mean())
    return [w/mean for w in raw], mean
