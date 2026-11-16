"""Label-free control cues. Missing derivatives are NaN, never zero evidence.

Track indices must remain aligned within the input; continuity is only a proxy
for identity. No inferred possession labels or contact detections are produced.
"""
import numpy as np


def velocity(xy, available, time, run, lag, max_speed, max_gap=.15):
    """Backward differences over real time, with every intervening sample valid."""
    n = len(time)
    if lag < 1 or max_speed <= 0 or xy.shape[0] != n or available.shape != xy.shape[:-1]:
        raise ValueError("Invalid motion contract")
    out = np.full(xy.shape, np.nan, dtype=np.float64)
    valid = available & np.isfinite(xy).all(-1)
    if n <= lag:
        return out, np.zeros(available.shape, bool)
    dt = np.diff(time)
    # Timestamps must be regularly sampled within a run, including intermediate steps.
    step = (dt > 0) & (dt <= max_gap) & (run[1:] == run[:-1])
    one_dt = np.where(step, dt, 1).reshape((-1,) + (1,) * (xy.ndim - 2))
    local_speed = np.linalg.norm(np.diff(xy, axis=0), axis=-1) / one_dt
    edge = (valid[1:] & valid[:-1] & np.isfinite(local_speed)
            & (local_speed <= max_speed) & step.reshape((-1,) + (1,) * (valid.ndim - 1)))
    keep = valid[lag:].copy()
    for offset in range(lag):
        keep &= edge[offset:n-lag+offset]
    elapsed = (time[lag:] - time[:-lag]).reshape((-1,) + (1,) * (xy.ndim - 1))
    value = (xy[lag:] - xy[:-lag]) / np.where(elapsed > 0, elapsed, 1)
    out[lag:] = np.where(keep[..., None], value, np.nan)
    return out, np.isfinite(out).all(-1)


def extract(ball, players, ball_available, player_available, time, run, fps, spec, lag_s):
    """Metres/seconds in, continuous descriptors + explicit support masks out."""
    lag = max(1, int(round(lag_s * fps)))
    vb, vb_ok = velocity(ball, ball_available, time, run, lag, spec["max_ball_speed_mps"], 1.5/fps)
    vp, vp_ok = velocity(players, player_available, time, run, lag, spec["max_player_speed_mps"], 1.5/fps)
    observed = (ball_available & np.isfinite(ball).all(-1))[:, None, None]
    present = player_available & np.isfinite(players).all(-1)
    distance = np.where(observed & present, np.linalg.norm(players - ball[:, None, None], axis=-1), np.inf)
    flat = distance.reshape(len(time), -1)
    nearest_index = flat.argmin(-1)
    nearest = flat.min(-1)
    relative_all = np.linalg.norm(vp - vb[:, None, None], axis=-1).reshape(len(time), -1)
    relative = relative_all[np.arange(len(time)), nearest_index]
    relative[~np.isfinite(nearest)] = np.nan
    speed = np.linalg.norm(vb, axis=-1)
    # Signed change in SPEED, not magnitude of vector acceleration.
    speed_xy = np.column_stack((speed, np.zeros(len(time))))
    derivative, accel_ok = velocity(speed_xy, vb_ok, time, run, lag, np.inf, 1.5/fps)
    acceleration = derivative[:, 0]
    decel = accel_ok & (acceleration < -spec["deceleration_epsilon_mps2"])
    # Completed deceleration episodes only: both ends observed in the same run.
    episodes = []
    start = None
    for j in range(len(time)):
        same = j > 0 and run[j] == run[j-1] and accel_ok[j] and accel_ok[j-1]
        if not same:
            start = None
        elif decel[j] and not decel[j-1]:
            start = j
        elif not decel[j] and decel[j-1] and start is not None:
            episodes.append((start, j, float(time[j] - time[start])))
            start = None
    return {"distance": nearest, "relative_speed": relative, "ball_speed": speed,
            "acceleration": acceleration, "ball_velocity_valid": vb_ok,
            "relative_valid": np.isfinite(relative), "acceleration_valid": accel_ok,
            "decelerating": decel, "episodes": episodes}


def distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {"count": int(len(values)), "quantiles_10_50_90":
            np.quantile(values, [.1, .5, .9]).tolist() if len(values) else None}


def summarize(cues, spec):
    """Aggregates only. Near/far are geometric groups, NOT control labels."""
    d = cues["distance"]
    near = np.isfinite(d) & (d <= spec["near_m"])
    far = np.isfinite(d) & (d >= spec["far_m"])
    def fraction(num, denom):
        return float(np.count_nonzero(num) / np.count_nonzero(denom)) if np.any(denom) else None
    row = {"frames": len(d), "geometry_coverage": float(np.isfinite(d).mean()),
           "ball_velocity_coverage": float(cues["ball_velocity_valid"].mean()),
           "near_frames": int(near.sum()),
           "near_relative_velocity_coverage": fraction(near & cues["relative_valid"], near),
           "relative_speed_near_mps": distribution(cues["relative_speed"][near]),
           "relative_speed_far_mps": distribution(cues["relative_speed"][far]),
           "completed_decelerations_s": distribution([e[2] for e in cues["episodes"]])}
    for name, mask in (("near", near), ("far", far)):
        support = mask & cues["acceleration_valid"]
        row[name + "_deceleration_fraction"] = fraction(support & cues["decelerating"], support)
    row["coverage_gate"] = (row["ball_velocity_coverage"] >= spec["min_ball_velocity_coverage"]
        and row["near_relative_velocity_coverage"] is not None
        and row["near_relative_velocity_coverage"] >= spec["min_near_relative_velocity_coverage"])
    return row
