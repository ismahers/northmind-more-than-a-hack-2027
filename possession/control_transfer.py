"""Pure XY descriptions for fixed candidate slots; no action/identity inference.

Callers supply native samples, including the trailing history. No filesystem,
loader, labels, classification, interpolation or changing-nearest logic lives here.
"""

from dataclasses import dataclass
from math import hypot, isfinite
from typing import Mapping


@dataclass(frozen=True)
class NativeSample:
    time: float
    run: str
    ball: tuple[float, float] | None
    players: Mapping[str, tuple[float, float] | None]
    ball_observed: bool = True
    observed_slots: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Contract:
    native_dt: float = 0.04
    trailing_seconds: float = 0.12
    endpoint_tolerance: float = 0.02
    maximum_gap_frames: float = 1.5
    ball_jump_speed: float = 45.0
    player_jump_speed: float = 12.0

    def __post_init__(self):
        values = (self.native_dt, self.trailing_seconds, self.endpoint_tolerance,
                  self.maximum_gap_frames, self.ball_jump_speed, self.player_jump_speed)
        if not all(isfinite(v) and v > 0 for v in values):
            raise ValueError("Contract constants must be finite and positive")
        if self.endpoint_tolerance > self.native_dt / 2 + 1e-12:
            raise ValueError("Endpoint tolerance exceeds half a native frame")


def _valid_xy(xy):
    return xy is not None and len(xy) == 2 and all(isfinite(v) for v in xy)


def _distance(a, b):
    return hypot(a[0] - b[0], a[1] - b[1])


def nearest_index(times, target, tolerance):
    """Nearest timestamp, earlier on ties rounded to 1e-12 seconds.

    Boundary comparisons allow 1e-12 seconds of floating point roundoff only;
    timestamps themselves are retained unchanged.
    """
    if not times or not isfinite(target) or tolerance < 0:
        return None
    index = min(range(len(times)), key=lambda j: (round(abs(times[j] - target), 12), times[j], j))
    return index if abs(times[index] - target) <= tolerance + 1e-12 else None


def _point_reasons(sample, slot):
    reasons = []
    if not sample.ball_observed or not _valid_xy(sample.ball):
        reasons.append("ball_unobserved_or_nonfinite")
    if slot not in sample.observed_slots or not _valid_xy(sample.players.get(slot)):
        reasons.append("slot_unobserved_or_nonfinite")
    return reasons


def _edge_reasons(previous, current, slot, contract):
    reasons = []
    dt = current.time - previous.time
    if previous.run != current.run:
        reasons.append("run_change")
    if dt <= 0:
        reasons.append("nonincreasing_time")
    elif dt > contract.native_dt * contract.maximum_gap_frames + 1e-12:
        reasons.append("native_gap")
    if dt > 0:
        if (_valid_xy(previous.ball) and _valid_xy(current.ball)
                and _distance(previous.ball, current.ball) / dt > contract.ball_jump_speed + 1e-9):
            reasons.append("ball_jump")
        a, b = previous.players.get(slot), current.players.get(slot)
        if _valid_xy(a) and _valid_xy(b) and _distance(a, b) / dt > contract.player_jump_speed + 1e-9:
            reasons.append("slot_jump")
    return reasons


def measure(samples, candidates, start=-0.6, end=1.5, contract=Contract()):
    """Describe native rows in [start,end]; candidate values are immutable slot keys.

    Both d and u are omitted when the trailing support is invalid. A break marks
    every omitted row so downstream drawing cannot join through absent support.
    Support is checked at ALL native samples, even those omitted by later scans.
    Monotonicity is a structural requirement: unsorted input is never sorted into
    apparent continuity. The run is an opaque segment key, not a person identity.
    """
    if not isfinite(start) or not isfinite(end) or end < start:
        raise ValueError("Invalid window")
    if not candidates or len(set(candidates.values())) != len(candidates):
        raise ValueError("Provide distinct fixed candidate slots")
    samples = tuple(samples)
    times = [s.time for s in samples]
    if not all(isfinite(t) for t in times):
        raise ValueError("Native timestamps must be finite")
    time_order_valid = all(b > a for a, b in zip(times, times[1:]))
    start_index = nearest_index(times, start, contract.endpoint_tolerance)
    end_index = nearest_index(times, end, contract.endpoint_tolerance)
    rows = []
    for index, sample in enumerate(samples):
        left = times[start_index] if start_index is not None else start
        right = times[end_index] if end_index is not None else end
        if not left - 1e-12 <= sample.time <= right + 1e-12:
            continue
        row = {"time": sample.time, "run": sample.run, "candidates": {}}
        target = sample.time - contract.trailing_seconds
        previous_index = nearest_index(times[:index], target, contract.endpoint_tolerance)
        for name, slot in candidates.items():
            reasons = []
            if not time_order_valid:
                reasons.append("nonincreasing_time")
            if previous_index is None:
                reasons.append("trailing_endpoint_missing")
            else:
                for j in range(previous_index, index + 1):
                    reasons.extend(_point_reasons(samples[j], slot))
                    if j > previous_index:
                        reasons.extend(_edge_reasons(samples[j - 1], samples[j], slot, contract))
            reasons = tuple(sorted(set(reasons)))
            d = u = None
            if not reasons:
                earlier = samples[previous_index]
                elapsed = sample.time - earlier.time
                ball_v = tuple((sample.ball[k] - earlier.ball[k]) / elapsed for k in (0, 1))
                player_v = tuple((sample.players[slot][k] - earlier.players[slot][k]) / elapsed for k in (0, 1))
                d = _distance(sample.ball, sample.players[slot])
                u = _distance(ball_v, player_v)
            row["candidates"][name] = {"slot": slot, "d": d, "u": u,
                                       "break": bool(reasons), "reasons": reasons}
        rows.append(row)
    # Endpoint existence plus all intervening native rows: scan output alone is
    # never sufficient evidence of complete window support.
    boundary_ok = start_index is not None and end_index is not None
    supported = bool(rows) and boundary_ok and all(
        not value["break"] for row in rows for value in row["candidates"].values())
    return {"rows": rows, "whole_window_supported": supported, "window": (start, end)}


def scan_nearest(measurement, targets, contract=Contract()):
    """Select rows nearest scan targets without interpolating or hiding support.

    Missing scan rows remain explicit. Whole-window support is inherited from the
    native measurement and cannot become true by dropping bad native rows.
    """
    times = [row["time"] for row in measurement["rows"]]
    rows = []
    for target in targets:
        j = nearest_index(times, target, contract.endpoint_tolerance)
        rows.append({"target": target, "sample": None if j is None else measurement["rows"][j]})
    return {"rows": rows, "whole_window_supported": measurement["whole_window_supported"]
            and all(row["sample"] is not None for row in rows)}


def descriptive_changes(measurement, center=0.0):
    """Native-sample means and differences, only with complete window support.

    No scores, rankings, learned weights, semantic thresholds or significance.
    """
    if not measurement["whole_window_supported"]:
        return None
    if measurement["window"][0] > center - 0.6 or measurement["window"][1] < center + 1.5:
        return None
    intervals = {"pre": (-0.6, -0.2), "post": (0.2, 0.6), "late": (1.0, 1.5)}
    output = {}
    for name in measurement["rows"][0]["candidates"]:
        result = {}
        for metric in ("d", "u"):
            means = {}
            for label, (left, right) in intervals.items():
                values = [row["candidates"][name][metric] for row in measurement["rows"]
                          if center + left - 1e-12 <= row["time"] <= center + right + 1e-12]
                if not values or any(value is None for value in values):
                    return None
                means[label] = sum(values) / len(values)
            result[metric] = {**means, "post_minus_pre": means["post"] - means["pre"],
                              "late_minus_pre": means["late"] - means["pre"]}
        output[name] = result
    return output
