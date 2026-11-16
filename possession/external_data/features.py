"""Deterministic pitch scaling only: no fitted statistics and no target-derived inputs."""

from dataclasses import dataclass

import numpy as np

from .audit import segments

FEATURE_VERSION = "geometry-v1"
NODE_DIM = 6
GLOBAL_DIM = 7


@dataclass
class Features:
    nodes: np.ndarray
    node_mask: np.ndarray
    global_features: np.ndarray
    native_index: np.ndarray
    run: np.ndarray
    time: np.ndarray
    alive: np.ndarray
    team: np.ndarray
    confidence: np.ndarray


def nearest_indices(sorted_times, requested):
    right = np.searchsorted(sorted_times, requested).clip(0, len(sorted_times) - 1)
    left = (right - 1).clip(0)
    return np.where(abs(sorted_times[left] - requested) <= abs(sorted_times[right] - requested), left, right)


def build_features(match, fps=10.0):
    if fps <= 0 or fps > match.metadata["native_fps"]:
        raise ValueError("Model FPS must be positive and no greater than native FPS")
    indices, run_ids, clocks = [], [], []
    for run_id, (a, b) in enumerate(segments(match)):
        # Grid is anchored independently per contiguous period, avoiding clock/halftime joins.
        duration = match.time[b - 1] - match.time[a]
        grid = match.time[a] + np.arange(int(np.floor(duration * fps + 1e-5)) + 1) / fps
        selected = a + nearest_indices(match.time[a:b], grid)
        indices.extend(selected.tolist())
        run_ids.extend([run_id] * len(grid))
        clocks.extend(grid.tolist())
    ix = np.asarray(indices, dtype=np.int64)
    run_ids = np.asarray(run_ids, dtype=np.int32)
    clocks = np.asarray(clocks, dtype=np.float64)
    scale = np.asarray(match.metadata["pitch_m"], dtype=np.float32) / 2
    mask = match.player_available[ix]
    xy = np.nan_to_num(match.players[ix] / scale, nan=0, posinf=0, neginf=0)
    ball_available = match.ball_available[ix]
    ball_xy = np.nan_to_num(match.ball[ix, :2] / scale, nan=0, posinf=0, neginf=0)
    relative = (xy - ball_xy[:, None, None, :]) * ball_available[:, None, None, None]
    nodes = np.concatenate((xy, relative, match.player_detected[ix, ..., None],
                            match.player_detection_known[ix, ..., None]), axis=-1)
    nodes *= mask[..., None]
    # Use native observations for age, rather than accidentally discarding a detection at 25Hz.
    ages = np.zeros(len(match.frame), dtype=np.float32)
    for a, b in segments(match):
        last = None
        for i in range(a, b):
            if match.ball_available[i] and (not match.ball_detection_known[i] or match.ball_detected[i]):
                last = match.time[i]
            ages[i] = 1 if last is None else min(10, match.time[i] - last) / 10
    global_features = np.column_stack((ball_xy,
        np.nan_to_num(match.ball[ix, 2], nan=0, posinf=0, neginf=0) / 10,
        ball_available, match.ball_detected[ix], match.ball_detection_known[ix], ages[ix]))
    # These are the only arrays consumed by the model. Native labels/provenance/IDs are excluded.
    return Features(nodes.astype(np.float32), mask, global_features.astype(np.float32), ix,
                    run_ids, clocks, match.alive[ix], match.team[ix], match.confidence[ix])
