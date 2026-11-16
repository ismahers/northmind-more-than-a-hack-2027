"""Array-only, label-free geometric diagnostic; returns aggregate JSON values only.

Unknown detection is admissible geometry, never certified visibility. Slots are
fixed at each selected center and are not claims about person identity.
"""

from dataclasses import replace
from math import ceil, floor

import numpy as np

from possession.control_transfer import (
    Contract, NativeSample, descriptive_changes, measure, nearest_index, scan_nearest,
)

WIDTHS = np.array([1., 2., 2., 1., 1., 2., 2.])
SCANS = [round(-.6 + j * .1, 10) for j in range(22)]
CONTRACT = Contract()


def _validated(arrays):
    # Explicit allowlist access: never enumerate arbitrary loader/label keys.
    if arrays["metadata"].get("native_fps") != 25:
        raise ValueError("Geometry requires native 25 Hz metadata")
    names = ("time", "period", "frame", "players", "ball", "player_available",
             "ball_available", "player_detected", "player_detection_known",
             "ball_detected", "ball_detection_known")
    a = {key: np.asarray(arrays[key]) for key in names}
    n = len(a["time"])
    p = a["players"]
    if p.ndim != 4 or p.shape[0] != n or p.shape[1] != 2 or p.shape[3] != 2:
        raise ValueError("Invalid player geometry shape")
    shapes = {"time": (n,), "period": (n,), "frame": (n,), "ball": (n, 3),
              "player_available": p.shape[:3], "player_detected": p.shape[:3],
              "player_detection_known": p.shape[:3], "ball_available": (n,),
              "ball_detected": (n,), "ball_detection_known": (n,)}
    if any(a[key].shape != shape for key, shape in shapes.items()):
        raise ValueError("Invalid geometry/flag shape")
    if not np.isfinite(a["time"]).all() or not np.isfinite(a["frame"]).all():
        raise ValueError("Nonfinite native clock")
    for key in names[5:]:
        if not np.isin(a[key], [False, True]).all():
            raise ValueError("Invalid geometry flag")
        a[key] = a[key].astype(bool, copy=False)
    same = a["period"][1:] == a["period"][:-1]
    if np.any(same & ((np.diff(a["time"]) <= 0) | (np.diff(a["frame"]) <= 0))):
        raise ValueError("Native clock must increase within a period")
    a["player_ok"] = (np.isfinite(p).all(axis=-1) & a["player_available"]
                      & (~a["player_detection_known"] | a["player_detected"]))
    a["ball_ok"] = (np.isfinite(a["ball"][:, :2]).all(axis=-1) & a["ball_available"]
                    & (~a["ball_detection_known"] | a["ball_detected"]))
    return a


def select_centers(a):
    """Select before any geometric support checks; return run/bounds/grid/index."""
    times = a["time"]
    if not len(times):
        return []
    cuts = np.flatnonzero((a["period"][1:] != a["period"][:-1])
                         | (np.diff(times) > .06 + 1e-12)) + 1
    bounds = np.r_[0, cuts, len(times)]
    selected = []
    for run, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        first = ceil((float(times[lo]) + .6 - 1e-12) / 10)
        last = floor((float(times[hi - 1]) - 1.5 + 1e-12) / 10)
        local = times[lo:hi]
        for k in range(first, last + 1):
            grid = float(k * 10)
            insertion = int(np.searchsorted(local, grid))
            nearby = [j for j in (insertion - 1, insertion) if 0 <= j < len(local)]
            j = min(nearby, key=lambda j: (round(abs(float(local[j]) - grid), 12), float(local[j])))
            if abs(float(local[j]) - grid) > .02 + 1e-12:
                j = None
            selected.append((run, int(lo), int(hi), grid, None if j is None else int(lo + j)))
    if len(selected) > 100:
        selected = [selected[floor(k * (len(selected) - 1) / 99)] for k in range(100)]
    return selected


def _candidates(a, index, team):
    if index is None or not a["ball_ok"][index]:
        return None
    ball, players = a["ball"][index, :2], a["players"][index]
    own = np.flatnonzero(a["player_ok"][index, team]).tolist()
    other = np.flatnonzero(a["player_ok"][index, 1 - team]).tolist()
    if len(own) < 2 or not other:
        return None
    def closest(slots, positions, target):
        return min(slots, key=lambda j: (float(np.linalg.norm(positions[j] - target)), j))
    actor = closest(own, players[team], ball)
    mate = closest([j for j in own if j != actor], players[team], players[team, actor])
    rival = closest(other, players[1 - team], ball)
    return {"A": (team, actor), "C": (team, mate), "R": (1 - team, rival)}


def _samples(a, center, candidates):
    run, lo, hi, grid, index = center
    # Selection/cap use nominal grid; measurement uses the selected native t.
    grid = float(a["time"][index])
    left = np.searchsorted(a["time"][lo:hi], grid - .6 - .12 - .04 - 1e-12) + lo
    right = np.searchsorted(a["time"][lo:hi], grid + 1.5 + .04 + 1e-12, side="right") + lo
    keys = {name: f"{team}:{slot}" for name, (team, slot) in candidates.items()}
    samples = []
    for j in range(left, right):
        players = {keys[name]: tuple(a["players"][j, team, slot])
                   for name, (team, slot) in candidates.items()}
        observed = frozenset(keys[name] for name, (team, slot) in candidates.items()
                             if a["player_ok"][j, team, slot])
        samples.append(NativeSample(float(a["time"][j] - grid), str(run),
                                    tuple(a["ball"][j, :2]), players,
                                    bool(a["ball_ok"][j]), observed))
    return samples, keys, int(left)


def _velocity(samples, index, slot=None):
    times = [sample.time for sample in samples]
    previous = nearest_index(times[:index], times[index] - .12, .02)
    if previous is None:
        raise ValueError("Supported window lacks initial trailing velocity")
    first = samples[previous].ball if slot is None else samples[previous].players[slot]
    last = samples[index].ball if slot is None else samples[index].players[slot]
    return (np.asarray(last) - first) / (times[index] - times[previous])


def _window(a, center, candidates):
    _, lo, hi, _, index = center
    effective = float(a["time"][index])
    if (effective - .6 < float(a["time"][lo]) - 1e-12
            or effective + 1.5 > float(a["time"][hi - 1]) + 1e-12):
        return None
    samples, keys, first = _samples(a, center, candidates)
    observed = measure(samples, keys)
    scan = scan_nearest(observed, SCANS)
    changes = descriptive_changes(observed)
    if not scan["whole_window_supported"] or changes is None:
        return None
    middle = center[4] - first
    anchor = samples[middle]
    ball_v = _velocity(samples, middle)
    velocities = {name: _velocity(samples, middle, slot) for name, slot in keys.items()}
    uniform = []
    for sample in samples:
        elapsed = sample.time - anchor.time
        if elapsed < -1e-12:
            uniform.append(sample)
        else:
            uniform.append(replace(sample, ball=tuple(np.asarray(anchor.ball) + elapsed * ball_v),
                                   players={slot: tuple(np.asarray(anchor.players[slot]) + elapsed * velocities[name])
                                            for name, slot in keys.items()}))
    # Observed guards already passed on the exact common native support. The
    # hypothetical continuation must not be censored for its own geometry.
    hypothetical = measure(uniform, keys, contract=replace(CONTRACT, ball_jump_speed=1e100,
                                                         player_jump_speed=1e100))
    baseline = descriptive_changes(hypothetical)
    if baseline is None:
        raise ValueError("Uniform reference lost common support")
    proximity = {name: float(np.linalg.norm(np.asarray(anchor.ball) - anchor.players[slot]))
                 for name, slot in keys.items()}
    initial = np.array([proximity["A"], proximity["C"], np.linalg.norm(ball_v),
                        np.linalg.norm(velocities["A"]), np.linalg.norm(velocities["C"]),
                        np.linalg.norm(ball_v - velocities["A"]), np.linalg.norm(ball_v - velocities["C"])])
    values = {}
    for name in keys:
        for metric in ("d", "u"):
            for label, value in changes[name][metric].items():
                values[f"observed.{name}.{metric}.{label}"] = value
                if label.endswith("minus_pre"):
                    values[f"residual.{name}.{metric}.{label}"] = value - baseline[name][metric][label]
    # Count the exact trailing/measurement support, excluding spare padding.
    native_times = [sample.time for sample in samples]
    start = nearest_index(native_times, observed["rows"][0]["time"] - .12, .02)
    end = nearest_index(native_times, observed["rows"][-1]["time"], .02)
    indices = np.arange(first + start, first + end + 1)
    flags = {"ball": [len(indices), int(a["ball_detection_known"][indices].sum()),
                      int((a["ball_detection_known"][indices] & a["ball_detected"][indices]).sum())],
             "players": [0, 0, 0]}
    joint_known = a["ball_detection_known"][indices].copy()
    joint_detected = a["ball_detected"][indices].copy()
    for team, slot in candidates.values():
        known = a["player_detection_known"][indices, team, slot]
        detected = a["player_detected"][indices, team, slot]
        flags["players"] = [x + y for x, y in zip(flags["players"],
                                                [len(indices), int(known.sum()), int((known & detected).sum())])]
        joint_known &= known
        joint_detected &= detected
    flags["joint"] = [len(indices), int(joint_known.sum()),
                      int((joint_known & joint_detected).sum())]
    return {"order": (center[0], center[3]), "initial": initial, "values": values,
            "proximity": proximity, "flags": flags}


def distribution(values):
    values = list(values)
    if not values:
        return {"count": 0, "q25": None, "median": None, "q75": None}
    quartiles = np.quantile(values, [.25, .5, .75], method="linear")
    return dict(zip(("count", "q25", "median", "q75"), [len(values), *map(float, quartiles)]))


def _pairs(windows):
    remaining = set(range(len(windows)))
    bins = [tuple(np.floor(w["initial"] / WIDTHS).astype(int)) for w in windows]
    pairs = []
    for i, window in enumerate(windows):
        if i not in remaining:
            continue
        remaining.remove(i)
        candidates = [j for j in remaining if bins[j] == bins[i]
                      and windows[j]["order"] != window["order"]]
        if candidates:
            j = min(candidates, key=lambda j: (float(np.linalg.norm(
                (window["initial"] - windows[j]["initial"]) / WIDTHS)), windows[j]["order"]))
            remaining.remove(j)
            pairs.append((window, windows[j]))
    return pairs


def _aggregate(windows, coverage, pairs):
    metrics = sorted(windows[0]["values"]) if windows else []
    changes = {key: distribution(w["values"][key] for w in windows) for key in metrics}
    contrasts = {}
    for key in metrics:
        if not key.endswith("minus_pre"):
            continue
        differences = [later["values"][key] - earlier["values"][key] for earlier, later in pairs]
        contrasts[key] = {"signed": distribution(differences),
                          "absolute": distribution(abs(v) for v in differences)}
    detection = {}
    for kind in ("ball", "players", "joint"):
        total, known, detected = [sum(w["flags"][kind][i] for w in windows) for i in range(3)]
        detection[kind] = {"sample_uses": total, "known": known, "known_detected": detected,
                           "known_fraction": known / total if total else None,
                           "detected_fraction_of_known": detected / known if known else None}
    return {"coverage": coverage,
            "discard_reasons": {"candidate_unavailable": coverage["candidate_sets"] - coverage["candidate_valid"],
                                "window_support_failed": coverage["candidate_valid"] - coverage["supported"]},
            "detection_on_supported_sample_uses": detection,
            "initial_proximity": {name: distribution(w["proximity"][name] for w in windows)
                                  for name in ("A", "C", "R")},
            "changes": changes,
            "matching": {"pairs": len(pairs), "paired_windows": 2 * len(pairs),
                         "unmatched_windows": len(windows) - 2 * len(pairs),
                         "paired_fraction": 2 * len(pairs) / len(windows) if windows else None,
                         "contrasts": contrasts}}


def analyse_geometry(arrays, check=lambda: None):
    """Describe one match, without I/O, labels, timestamps or slot records in output."""
    check()
    a = _validated(arrays)
    centers = select_centers(a)
    teams, all_windows, all_pairs = {}, [], []
    totals = {"selected_centers": len(centers), "candidate_sets": 2 * len(centers),
              "candidate_valid": 0, "supported": 0, "discarded": 0}
    for team in (0, 1):
        windows = []
        coverage = {"selected_centers": len(centers), "candidate_sets": len(centers),
                    "candidate_valid": 0, "supported": 0, "discarded": 0}
        for center in centers:
            check()
            candidates = _candidates(a, center[4], team)
            if candidates is None:
                continue
            coverage["candidate_valid"] += 1
            window = _window(a, center, candidates)
            if window is not None:
                windows.append(window)
        coverage["supported"] = len(windows)
        coverage["discarded"] = len(centers) - len(windows)
        pairs = _pairs(windows)
        teams[f"team{team}"] = _aggregate(windows, coverage, pairs)
        all_windows.extend(windows)
        all_pairs.extend(pairs)
        for key in ("candidate_valid", "supported", "discarded"):
            totals[key] += coverage[key]
    return {"global": _aggregate(all_windows, totals, all_pairs), "teams": teams,
            "semantic_claim": False, "model_promoted": False}
