"""Causal, label-free control memory from metric tracking geometry only.

Team zero has sign +1 and team one sign -1. Missing observations can retain
previously established memory, but never establish or refresh control.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ControlMemoryConfig:
    fps: float = 25.0
    speed_lag_s: float = 0.12
    max_distance_m: float = 1.8
    max_speed_m_s: float = 5.5
    min_rival_margin_m: float = 0.75
    confirmation_s: float = 0.20
    memory_horizon_s: float = 2.0
    max_gap_frames: float = 1.5

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")


@dataclass(frozen=True)
class ControlMemoryResult:
    memory_signal: np.ndarray
    supported: np.ndarray
    speed_m_s: np.ndarray
    candidate_team: np.ndarray
    confirmed_control: np.ndarray
    memory_age_s: np.ndarray
    segment_start: np.ndarray


def temporal_control_memory(
    distances, ball_xy, time, run, live_mask,
    config: ControlMemoryConfig = ControlMemoryConfig(),
) -> ControlMemoryResult:
    """Extract bounded memory without future frames, labels, or file access.

    ``distances`` and ``ball_xy`` have shape (T, 2), in metres; ``time`` is
    seconds. ``run`` contains finite integer segment identifiers. Time must be
    strictly increasing inside each contiguous run, but may restart in a new
    run. ``live_mask`` is strictly boolean (false includes D/X).

    A new live segment begins after a run boundary, gap > 1.5/fps, or non-live
    sample. Speed uses linear interpolation at exactly t-0.12s within the
    current contiguous finite-coordinate history; no zero speed is imputed.
    Confirmation requires the same candidate throughout 0.20 elapsed seconds.
    Once confirmed, each further qualifying observation refreshes memory;
    absent confirmation it decays linearly to zero over two seconds.

    Metadata signs use +1/-1 for teams, zero for no candidate/control. Age is
    NaN when there is no non-expired memory; supported is abs(signal) > 0.
    Inputs are never modified. All thresholds can be configured explicitly.
    """
    if not isinstance(config, ControlMemoryConfig):
        raise TypeError("config must be ControlMemoryConfig")
    d = np.asarray(distances, dtype=float)
    xy = np.asarray(ball_xy, dtype=float)
    t = np.asarray(time, dtype=float)
    r = np.asarray(run)
    live = np.asarray(live_mask)
    if t.ndim != 1:
        raise ValueError("time must be one-dimensional")
    n = len(t)
    if d.shape != (n, 2) or xy.shape != (n, 2):
        raise ValueError("distances and ball_xy must have shape (T, 2)")
    if r.shape != (n,) or live.shape != (n,):
        raise ValueError("run and live_mask must have shape (T,)")
    if live.dtype.kind != "b":
        raise ValueError("live_mask must be boolean")
    if r.dtype.kind not in "iuf" or not np.all(np.isfinite(r)):
        raise ValueError("run must contain finite integer identifiers")
    if np.any(r != np.floor(r)):
        raise ValueError("run identifiers must be integers")
    if not np.all(np.isfinite(t)):
        raise ValueError("time must be finite")
    if n > 1 and np.any((r[1:] == r[:-1]) & (np.diff(t) <= 0)):
        raise ValueError("time must increase within each contiguous run")
    if np.any(d < 0):
        raise ValueError("distances must be nonnegative or missing")

    signal = np.zeros(n, dtype=float)
    speed = np.full(n, np.nan)
    candidate = np.zeros(n, dtype=np.int8)
    confirmed = np.zeros(n, dtype=np.int8)
    age = np.full(n, np.nan)
    starts = np.zeros(n, dtype=bool)
    finite_xy = np.all(np.isfinite(xy), axis=1)
    finite_d = np.all(np.isfinite(d), axis=1)
    pending_team = owner = 0
    pending_since = last_confirmed = math.nan
    history_start = left = 0
    epsilon = 1e-9
    max_gap_s = config.max_gap_frames / config.fps

    for i in range(n):
        boundary = (
            i == 0 or r[i] != r[i - 1] or not live[i - 1]
            or t[i] - t[i - 1] > max_gap_s + epsilon
        )
        if boundary or not live[i]:
            pending_team = owner = 0
            pending_since = last_confirmed = math.nan
            history_start = left = i
        starts[i] = live[i] and boundary
        if not live[i]:
            continue

        # A missing coordinate breaks speed history, but not established memory.
        if not finite_xy[i]:
            history_start = left = i + 1
        elif t[i] - t[history_start] + epsilon >= config.speed_lag_s:
            target = t[i] - config.speed_lag_s
            while left + 1 < i and t[left + 1] <= target:
                left += 1
            right = left + 1
            if right <= i and t[left] <= target + epsilon:
                weight = (target - t[left]) / (t[right] - t[left])
                previous_xy = xy[left] + weight * (xy[right] - xy[left])
                speed[i] = np.linalg.norm(xy[i] - previous_xy) / config.speed_lag_s

        if finite_d[i] and np.isfinite(speed[i]):
            nearest = int(np.argmin(d[i]))
            rival = 1 - nearest
            if (
                d[i, nearest] <= config.max_distance_m
                and speed[i] <= config.max_speed_m_s
                and d[i, rival] - d[i, nearest] >= config.min_rival_margin_m
            ):
                candidate[i] = 1 if nearest == 0 else -1

        if candidate[i] == 0:
            pending_team = 0
            pending_since = math.nan
        else:
            if candidate[i] != pending_team:
                pending_team = int(candidate[i])
                pending_since = t[i]
            if t[i] - pending_since + epsilon >= config.confirmation_s:
                confirmed[i] = pending_team
                owner = pending_team
                last_confirmed = t[i]

        if owner:
            elapsed = t[i] - last_confirmed
            strength = max(0.0, 1.0 - elapsed / config.memory_horizon_s)
            if strength > 0:
                signal[i] = owner * strength
                age[i] = elapsed
            else:
                owner = 0

    return ControlMemoryResult(
        signal, np.abs(signal) > 0, speed, candidate, confirmed, age, starts
    )
