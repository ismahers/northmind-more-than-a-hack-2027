"""JSON-serializable diagnostics, including unknown and censored intervals."""

import numpy as np


def segments(match):
    """Clock-contiguous runs; fail closed on duplicate/backward clocks or period re-entry."""
    n = len(match.frame)
    cuts = [0]
    completed = set()
    for i in range(1, n):
        if match.period[i] != match.period[i - 1]:
            completed.add(int(match.period[i - 1]))
            if int(match.period[i]) in completed:
                raise ValueError("Noncontiguous period records")
            cuts.append(i)
        else:
            dt = match.time[i] - match.time[i - 1]
            if dt <= 0 or match.frame[i] <= match.frame[i - 1]:
                raise ValueError("Nonmonotone or duplicated clock/frame")
            if dt > 1.5 / match.metadata["native_fps"]:
                cuts.append(i)
    return list(zip(cuts, cuts[1:] + [n]))


def joint_labels(match):
    labels = np.full(len(match.frame), -1, dtype=np.int8)
    state_valid = match.confidence[:, 0] > 0
    labels[(match.alive == 0) & state_valid] = 2
    valid = (match.alive == 1) & state_valid & (match.team >= 0) & (match.confidence[:, 1] > 0)
    labels[valid] = match.team[valid]
    return labels


def audit(match):
    match.validate()
    labels = joint_labels(match)
    runs, temporal_error = [], None
    try:
        runs = segments(match)
    except ValueError as exc:
        temporal_error = str(exc)
    counts, durations, transitions, censored = [], [], 0, 0
    for start, stop in runs:
        values = labels[start:stop]
        bounds = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            code = int(values[a])
            if code in (0, 1):
                counts.append(code)
                durations.append(float(match.time[start + b - 1] - match.time[start + a]
                                       + 1 / match.metadata["native_fps"]))
                censored += int(a == 0 or b == len(values) or values[a - 1] < 0
                                or (b < len(values) and values[b] < 0))
        transitions += int(np.count_nonzero((values[1:] != values[:-1])
                                          & (values[1:] >= 0) & (values[:-1] >= 0)))
    pitch = np.asarray(match.metadata["pitch_m"])
    bad_players = np.any(np.abs(match.players) > pitch / 2 + 5, axis=-1) & match.player_available
    dts = np.diff(match.time)[match.period[1:] == match.period[:-1]]
    def value_counts(values):
        names, amounts = np.unique(values, return_counts=True)
        return {str(name): int(amount) for name, amount in zip(names, amounts)}
    return {
        "schema_version": 1, "key": match.key, "sources": match.metadata["sources"],
        "native_fps": match.metadata["native_fps"], "frames": len(labels),
        "pitch_m": match.metadata["pitch_m"], "team_ids": match.metadata.get("team_ids"),
        "roster_sizes": (match.player_ids != "").sum(-1).tolist(),
        "players_with_coordinates_per_team": match.player_available.any(0).sum(-1).tolist(),
        "periods": [{"period": int(period), "frames": int((match.period == period).sum()),
                     "state_coverage": float((match.alive[match.period == period] >= 0).mean()),
                     "joint_coverage": float((labels[match.period == period] >= 0).mean())}
                    for period in np.unique(match.period)],
        "duration_s": sum(float(match.time[b - 1] - match.time[a] + 1 / match.metadata["native_fps"])
                          for a, b in runs),
        "ball_available_fraction": float(match.ball_available.mean()),
        "ball_detected_fraction": float(match.ball_detected.mean()),
        "ball_detection_known_fraction": float(match.ball_detection_known.mean()),
        "players_available_mean_per_team": match.player_available.sum(-1).mean(0).tolist(),
        "players_detected_mean_per_team": match.player_detected.sum(-1).mean(0).tolist(),
        "state_coverage": float(((match.alive >= 0) & (match.confidence[:, 0] > 0)).mean()),
        "team_coverage": float(((match.team >= 0) & (match.confidence[:, 1] > 0)).mean()),
        "joint_coverage": float((labels >= 0).mean()),
        "label_counts": {name: int((labels == code).sum()) for code, name in enumerate(("H", "A", "D"))},
        "unknown_joint_frames": int((labels < 0).sum()),
        "conditional_team_counts": {name: int((match.team == code).sum()) for code, name in enumerate(("H", "A"))},
        "possessions_including_censored": len(counts), "censored_possessions": censored,
        "possession_durations_s": durations, "transitions": transitions,
        "temporal_error": temporal_error, "contiguous_runs": len(runs),
        "within_period_dt_min_s": float(dts.min()) if len(dts) else None,
        "within_period_dt_max_s": float(dts.max()) if len(dts) else None,
        "irregular_native_intervals": int((abs(dts - 1 / match.metadata["native_fps"]) > 0.005).sum()),
        "large_geometry_player_records": int(bad_players.sum()),
        "large_geometry_ball_records": int(((np.abs(match.ball[:, :2]) > pitch / 2 + 10).any(-1)
                                             & match.ball_available).sum()),
        "negative_ball_height_records": int((match.ball[:, 2] < -0.5).sum()),
        "invalid_player_coordinate_records": int((~match.player_available
                                                   & np.isfinite(match.players).any(-1)).sum()),
        "native_status_values": sorted(set(match.native_status.tolist())),
        "native_status_counts": value_counts(match.native_status),
        "native_possession_counts": value_counts(match.native_possession),
        "active_native_possession_counts": value_counts(match.native_possession[match.alive == 1]),
        "adapter_issues": match.metadata.get("issues", {}),
        "label_policy": match.metadata.get("label_policy", "synthetic"),
    }
