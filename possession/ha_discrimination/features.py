"""Label-free metric geometry, masked causal motion; no player identities."""
import numpy as np

NAMES = ["home_d1", "away_d1", "home_d3mean", "away_d3mean", "d1_away_minus_home", "d3_away_minus_home",
         "home_density3", "away_density3", "home_density5", "away_density5", "ball_available",
         "home_available_fraction", "away_available_fraction", "ball_vx", "ball_vy", "ball_speed",
         "velocity_valid", "proximity_change_half_second", "change_valid", "ball_age"]


def lag_difference(values, valid, run, time, lag):
    result = np.zeros_like(values, dtype=np.float32); keep = np.zeros(len(values), bool)
    # Require every intervening observation and a continuous 10Hz clock, not just endpoints.
    for j in range(lag, len(values)):
        a = j-lag
        if run[a] == run[j] and valid[a:j+1].all() and np.all(np.diff(time[a:j+1]) <= .10001) and time[j] > time[a]:
            result[j] = values[j]-values[a]; keep[j] = True
    return result, keep


def geometry(view, pitch):
    scale = np.asarray(pitch, np.float32)/2
    ball = view.global_features[:, :2]*scale
    available = view.global_features[:, 3] > .5
    present = view.node_mask
    players = view.nodes[..., :2]*scale
    d = np.linalg.norm(players-ball[:, None, None, :], axis=-1)
    d = np.where(present & available[:, None, None], d, np.inf)
    ordered = np.sort(d, axis=-1)
    nearest = ordered[..., 0]
    top = ordered[..., :3]
    counts = np.isfinite(top).sum(-1)
    d3 = np.where(counts > 0, np.where(np.isfinite(top), top, 0).sum(-1)/np.maximum(1, counts), np.inf)
    both = np.isfinite(nearest).all(-1)
    delta = np.where(both, np.where(np.isfinite(nearest[:, 1]), nearest[:, 1], 0)-np.where(np.isfinite(nearest[:, 0]), nearest[:, 0], 0), 0)
    delta3 = np.where(both, np.where(np.isfinite(d3[:, 1]), d3[:, 1], 0)-np.where(np.isfinite(d3[:, 0]), d3[:, 0], 0), 0)
    displacement, velocity_valid = lag_difference(ball, available, view.run, view.time, 1)
    velocity = displacement/.1
    change, change_valid = lag_difference(delta[:, None], both, view.run, view.time, 5)
    density3, density5 = (d < 3).sum(-1)/11, (d < 5).sum(-1)/11
    x = np.column_stack((np.where(np.isfinite(nearest), np.minimum(nearest, 50)/25, 0),
                         np.where(np.isfinite(d3), np.minimum(d3, 50)/25, 0),
                         np.clip(delta/25, -2, 2), np.clip(delta3/25, -2, 2), density3, density5,
                         available, present.sum(-1)/11,
                         np.clip(velocity/30, -2, 2), np.clip(np.linalg.norm(velocity, axis=1)/30, 0, 2),
                         velocity_valid, np.clip(change[:, 0]/25, -2, 2), change_valid, view.global_features[:, 6])).astype("f4")
    if x.shape != (len(view.time), 20) or not np.isfinite(x).all():
        raise ValueError("Invalid geometry features")
    return x, {"nearest": nearest, "delta": delta, "available": available, "both": both,
               "velocity_valid": velocity_valid, "speed": np.linalg.norm(velocity, axis=1)}
