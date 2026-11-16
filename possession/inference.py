"""Infer possession labels from geometric and tactical evidence, without ground truth."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from games import GAME, GAMES, half_video_blocks
from starter.baseline import smooth
from possession.tactical_tagging import get_quick_restart_mask


@dataclass(frozen=True)
class Config:
    # Engineering hypotheses fixed before scoring, not estimated from public GT.
    half_window_frames: int = 50  # Keep the published reference's time scale.
    proximity_scale_m: float = 3.0
    team_margin_scale_m: float = 1.5
    outside_margin_m: float = 0.2
    outside_min_frames: int = 5
    stationary_window_frames: int = 75  # Three seconds at 25 fps.
    stationary_min_support: float = 0.8
    stationary_max_range_m: float = 0.75
    unattended_min_distance_m: float = 1.5
    tagging_prior_weight: float = 0.25


def window_sum(values, half_width):
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    index = np.arange(len(values))
    return (cumulative[np.minimum(len(values), index + half_width + 1)]
            - cumulative[np.maximum(0, index - half_width)])


def minimum_runs(mask, minimum):
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    result = np.zeros(len(mask), dtype=bool)
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop - start >= minimum:
            result[start:stop] = True
    return result


def team_strength(distances, config, team_prior=None):
    """Signed team evidence before temporal aggregation; no GT or file access."""
    nearest_distance = distances.min(axis=1)
    observed = np.isfinite(nearest_distance)
    strength = np.zeros(len(distances))
    strength[observed] = (
        np.exp(-nearest_distance[observed] / config.proximity_scale_m)
        * np.tanh((distances[observed, 1] - distances[observed, 0])
                  / config.team_margin_scale_m))
    if team_prior is not None:
        strength += config.tagging_prior_weight * team_prior * observed
    return strength


def block_dead_evidence(distances, ball, config, quick_restart_mask=None):
    """Independent geometric stopping cues for one half; return an in-memory mask.

    If `quick_restart_mask` is provided, suppress 'ball outside pitch' dead-ball
    evidence during quick restarts (throw-ins, corners, free kicks, goal kicks),
    since tracking often loses the ball there but play continues.
    """
    length, width = GAMES[GAME]["pitch"]
    nearest_distance = distances.min(axis=1)
    observed = np.isfinite(nearest_distance)
    visible = np.isfinite(ball).all(axis=1)
    margin = config.outside_margin_m
    outside = visible & ((ball[:, 0] < -margin) | (ball[:, 0] > length + margin)
                         | (ball[:, 1] < -margin) | (ball[:, 1] > width + margin))
    outside = minimum_runs(outside, config.outside_min_frames)

    # Suppress 'outside' dead-ball evidence during quick restarts
    if quick_restart_mask is not None:
        outside = outside & ~quick_restart_mask

    window = config.stationary_window_frames
    minimum = int(np.ceil(window * config.stationary_min_support))
    rolling = pd.DataFrame(ball).rolling(window, center=True, min_periods=minimum)
    extent = rolling.max() - rolling.min()
    diameter = np.hypot(extent[0].to_numpy(), extent[1].to_numpy())
    unattended = pd.Series(np.where(observed, nearest_distance, np.nan)).rolling(
        window, center=True, min_periods=minimum).min().to_numpy()
    stationary = visible & (diameter <= config.stationary_max_range_m) & (
        unattended >= config.unattended_min_distance_m)
    return outside | stationary


def predict(evidence, config=None, team_prior=None):
    """Return weighted-vote and weighted-vote-plus-stoppages predictions.

    Soft team evidence = proximity * tanh(distance margin / scale). D fallback
    remains the baseline's missing-observation heuristic. The second candidate
    adds observed out-of-pitch runs and stationary, unattended-ball windows.
    Geometry is noisy, so these are hypotheses, not authoritative referee calls.
    """
    config = config if config is not None else Config()
    n_frames = GAMES[GAME]["n_video_frames"]
    if (evidence.nearest_codes.shape != (n_frames,)
            or evidence.team_distance_m.shape != (n_frames, 2)
            or evidence.ball_xy_m.shape != (n_frames, 2)):
        raise ValueError("Evidence must cover the tactical timeline")
    if team_prior is not None and (
            team_prior.shape != (n_frames,) or not np.isfinite(team_prior).all()
            or np.any(np.abs(team_prior) > 1)):
        raise ValueError("Team prior must be finite, bounded and on the tactical timeline")

    # Quick restart mask: suppress 'ball outside' dead-ball evidence during set pieces
    quick_restart_mask = get_quick_restart_mask(window_frames=50)

    weighted = np.full(n_frames, "X", dtype="<U1")
    dead_evidence = np.zeros(n_frames, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        block = slice(first, last + 1)
        distances = evidence.team_distance_m[block]
        ball = evidence.ball_xy_m[block]
        # Tactical evidence only chooses a team where tracking supplies evidence.
        prior = None if team_prior is None else team_prior[block]
        strength = team_strength(distances, config, prior)
        score = window_sum(strength, config.half_window_frames)
        support = window_sum(np.abs(strength), config.half_window_frames)
        labels = smooth(evidence.nearest_codes[block], config.half_window_frames)
        # At zero evidence / exact ties retain the reference fallback. The
        # tolerance only handles floating point cancellation, not confidence.
        has_evidence = (support > 1e-10) & (np.abs(score) > 1e-10)
        labels[has_evidence] = np.where(score[has_evidence] > 0, "H", "A")
        weighted[block] = labels

        dead_evidence[block] = block_dead_evidence(
            distances, ball, config, quick_restart_mask[block]
        )

    geometry = weighted.copy()
    geometry[dead_evidence] = "D"
    return {"northmind_weighted": weighted, "northmind_geometry": geometry}, {
        "extra_dead_ball_frames": int(np.count_nonzero(dead_evidence & (weighted != "D"))),
    }
