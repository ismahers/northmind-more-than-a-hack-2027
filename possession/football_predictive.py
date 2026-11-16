"""Small geometry-only team predictor, independent of neural training emissions."""
import numpy as np
import torch
from possession.football_control import extract
from possession.external_data.audit import segments
from possession.external_data.features import nearest_indices

NAMES = ["proximity", "negative_distance", "density"] + [
    f"{name}_{lag}" for lag in (.12, .24)
    for name in ("comovement", "motion_support", "signed_speed_change", "deceleration_age")]


def geometry_features(ball, players, ba, pa, time, run, fps, sample, spec):
    """No labels, IDs, neural outputs or model parameters enter feature generation."""
    observed = ba & np.isfinite(ball).all(-1)
    distance = np.linalg.norm(players-ball[:, None, None], axis=-1)
    valid = pa & np.isfinite(players).all(-1) & observed[:, None, None]
    distance = np.where(valid, distance, np.inf)
    nearest = distance.min(-1)
    active = np.isfinite(nearest).all(-1)
    prox = np.exp(-nearest/3.)
    per_team = [prox, np.where(np.isfinite(nearest), -np.minimum(nearest, 20)/20, 0),
                (distance <= 3.).sum(-1)/11]
    for lag in spec["lag_seconds"]:
        fields = [[] for _ in range(4)]
        for side in (0, 1):
            cues = extract(ball, players[:, side:side+1], observed, pa[:, side:side+1],
                           time, run, fps, spec, lag)
            relative = cues["relative_speed"]
            supported = cues["relative_valid"]
            accel = cues["acceleration"]
            acc_ok = cues["acceleration_valid"]
            age = np.zeros(len(time))
            for j in range(1, len(time)):
                if run[j] == run[j-1] and cues["decelerating"][j] and cues["decelerating"][j-1]:
                    age[j] = min(2., age[j-1] + time[j]-time[j-1])
            fields[0].append(prox[:, side]*np.exp(-np.nan_to_num(relative, nan=100)/5)*supported)
            fields[1].append(prox[:, side]*supported)
            fields[2].append(prox[:, side]*np.clip(np.nan_to_num(accel)/20, -1, 1)*acc_ok)
            fields[3].append(prox[:, side]*age/2*acc_ok)
        per_team.extend(np.column_stack(values) for values in fields)
    per_team = np.stack(per_team, -1)
    x = (per_team[:, 0]-per_team[:, 1])[sample]
    active = active[sample]
    x[~active] = 0
    if x.shape[1] != len(NAMES) or not np.isfinite(x).all():
        raise ValueError("Geometry feature contract failed")
    return x.astype(np.float64), active


def features_for(match, view, missing, spec):
    """Expand each 10Hz corruption to native geometry BEFORE differentiating."""
    native_run = np.empty(len(match.time), np.int32)
    native_missing = np.ones(len(match.time), bool)
    for run, (a, b) in enumerate(segments(match)):
        ix = np.flatnonzero(view.run == run)
        if not len(ix):
            raise ValueError("Missing model run")
        mapping = ix[nearest_indices(view.time[ix], match.time[a:b])]
        native_missing[a:b] = missing[mapping]
        native_run[a:b] = run
    return geometry_features(match.ball[:, :2], match.players,
        match.ball_available & ~native_missing, match.player_available,
        match.time, native_run, match.metadata["native_fps"], view.native_index, spec)


def ablate(x, arm):
    out = np.array(x, copy=True)
    if arm == "proximity":
        out[:, 3:] = 0
    elif arm != "football":
        raise ValueError("Unknown geometry arm")
    return out


def fit(x, y, weight, arm, spec, check=lambda: None):
    """Zero-centred RMS scaling from own train only preserves H/A equivariance."""
    x = ablate(x, arm)
    weight = weight / weight.sum()
    scale = np.maximum(np.sqrt(np.sum(weight[:, None]*x*x, axis=0)), .05)
    inputs = torch.tensor(x/scale, dtype=torch.float64)
    targets = torch.tensor(y, dtype=torch.float64)
    weights = torch.tensor(weight, dtype=torch.float64)
    theta = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=spec["learning_rate"])
    history = []
    for step in range(spec["steps"]):
        check()
        optimizer.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(inputs@theta, targets,
                    reduction="none")*weights).sum() + .5*spec["l2"]*theta.square().sum()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite geometry loss")
        loss.backward()
        optimizer.step()
        if step in (0, spec["steps"]-1):
            history.append(float(loss.detach()))
    return {"arm": arm, "theta": theta.detach().numpy().tolist(), "scale": scale.tolist(),
            "feature_names": NAMES, "steps": spec["steps"], "train_loss_first_last": history,
            "training_samples": len(y), "neural_training_predictions_used": False}


def predict(model, x):
    logits = ablate(x, model["arm"])/np.array(model["scale"])@np.array(model["theta"])
    return 1/(1+np.exp(-np.clip(logits, -30, 30)))


def combine(base, conditional, active, spec):
    """Fixed convex blend of conditional team only; unchanged alive emissions."""
    out = np.array(base, dtype=np.float64, copy=True)
    alive = out[:, :2].sum(-1)
    home = np.divide(out[:, 0], alive, out=np.full(len(base), .5), where=alive > 1e-12)
    mixed = (1-spec["geometry_weight"])*home+spec["geometry_weight"]*conditional
    out[active, 0] = alive[active]*mixed[active]
    out[active, 1] = alive[active]*(1-mixed[active])
    return out
