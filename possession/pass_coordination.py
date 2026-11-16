"""Fixed, label-free ball-turn and prior-receiver kinematics. No learning."""
import numpy as np
from possession.football_control import velocity
from possession.external_data.features import nearest_indices

ARMS = ("frozen_oof", "turn_proximity_space", "prior_receiver_coordination",
        "stationary_receiver", "permuted_prior_velocities")


def receivers(ball, positions, prior, outgoing, spec, mode="prior"):
    """Same observed receiver set for all controls; no semantic pass/contact input."""
    motion = np.minimum(np.linalg.norm(prior, axis=-1)/spec["receiver_speed_scale_mps"], 1)
    u = prior.copy()
    if mode == "stationary":
        u[:] = 0
    elif mode == "permuted":
        u = np.roll(u, 1, axis=0)
        motion = np.minimum(np.linalg.norm(u, axis=-1)/spec["receiver_speed_scale_mps"], 1)
    elif mode != "prior":
        raise ValueError("Unknown receiver control")
    relative = outgoing-u
    square = np.sum(relative*relative, axis=-1)
    lo, hi = spec["intercept_horizon_s"]
    tau = np.divide(np.sum((positions-ball)*relative, axis=-1), square,
                    out=np.full(len(positions), lo), where=square > 1e-12)
    tau = np.clip(tau, lo, hi)
    miss = positions+tau[:, None]*u-ball-tau[:, None]*outgoing
    score = motion*np.exp(-np.sum(miss*miss, axis=-1)/(2*spec["intercept_miss_scale_m"]**2))
    return float(score.max())


def candidates(m, run, spec, check=lambda: None):
    lag = round(spec["ball_velocity_interval_s"]*spec["native_hz"])
    vb, ok = velocity(m.ball, m.ball_available, m.time, run, lag,
                      spec["max_ball_speed_mps"], spec["max_contiguous_gap_frames"]/spec["native_hz"])
    result = []
    for r in np.unique(run):
        check(); ix = np.flatnonzero(run == r); a, b = ix[0], ix[-1]+1
        grid = m.time[a]+np.arange(int(np.floor((m.time[b-1]-m.time[a])*spec["scan_hz"]))+1)/spec["scan_hz"]
        scan = a+nearest_indices(m.time[a:b], grid)
        eligible = []
        for i in scan:
            if i+lag >= b or not (ok[i] and ok[i+lag]):
                continue
            incoming, outgoing = vb[i], vb[i+lag]
            speeds = np.linalg.norm([incoming, outgoing], axis=1)
            if speeds.min() < spec["candidate_min_ball_speed_mps"]:
                continue
            cosine = float(np.clip(incoming@outgoing/np.prod(speeds), -1, 1))
            if cosine > np.cos(np.deg2rad(spec["candidate_min_turn_degrees"])):
                continue
            distances = np.where(m.player_available[i], np.linalg.norm(m.players[i]-m.ball[i], axis=-1), np.inf)
            if distances.min() <= spec["candidate_max_nearest_actor_m"]:
                eligible.append((int(i), cosine))
        # Nonmaximum suppression on eligible centres, independent of all labels/OOF.
        accepted = []
        for i, cosine in sorted(eligible, key=lambda item: (item[1], item[0])):
            if all(abs(m.time[i]-m.time[j]) > spec["local_maximum_radius_s"]+1e-8 for j, _ in accepted):
                accepted.append((i, cosine))
        result.extend((i, cosine, vb[i+lag].copy()) for i, cosine in accepted)
    result.sort(key=lambda row: row[0])
    if len(result) > spec["max_events_per_match"]:
        chosen = np.round(np.linspace(0, len(result)-1, spec["max_events_per_match"])).astype(int)
        result = [result[i] for i in chosen]
    return result


def player_motion(m, run, i, window, spec):
    a, b = [i+round(t*spec["native_hz"]) for t in window]
    shape = m.players.shape[1:]
    if a < 0 or b >= len(m.time) or run[a] != run[b] or run[a] != run[i]:
        return np.full(shape, np.nan), np.zeros(shape[:-1], bool)
    v, valid = velocity(m.players[a:b+1], m.player_available[a:b+1], m.time[a:b+1], run[a:b+1],
                        b-a, spec["max_player_speed_mps"], spec["max_contiguous_gap_frames"]/spec["native_hz"])
    return v[-1], valid[-1]


def event_features(m, run, i, cosine, outgoing, spec):
    prior, valid = player_motion(m, run, i, spec["receiver_prior_interval_s"], spec)
    post, post_valid = player_motion(m, run, i, spec["receiver_post_control_interval_s"], spec)
    distances = np.where(m.player_available[i], np.linalg.norm(m.players[i]-m.ball[i], axis=-1), np.inf)
    actors = distances.argmin(-1); nearest = distances.min(-1)
    g = np.zeros(2); c = np.full((4, 2), np.nan); counts = np.zeros(2, int)
    for side in (0, 1):
        if not np.isfinite(nearest).all():
            continue
        if nearest[side] <= spec["candidate_max_nearest_actor_m"]:
            margin = (nearest[1-side]-nearest[side])/spec["opponent_margin_scale_m"]
            g[side] = (1-cosine)/2*np.exp(-nearest[side]/spec["actor_distance_scale_m"])/(1+np.exp(-np.clip(margin, -100, 100)))
        keep = valid[side] & m.player_available[i, side] & (distances[side] <= spec["receiver_max_current_distance_m"])
        keep[actors[side]] = False
        counts[side] = int(keep.sum())
        if not keep.any():
            continue
        pos, u = m.players[i, side, keep], prior[side, keep]
        for j, mode in enumerate(("prior", "stationary", "permuted")):
            c[j, side] = receivers(m.ball[i], pos, u, outgoing, spec, mode)
        if post_valid[side, keep].all():
            c[3, side] = receivers(m.ball[i], pos, post[side, keep], outgoing, spec)
    return g, c, counts


def evaluate(data, spec):
    truth = data["truth"]
    live = truth < 2
    support = live & np.isfinite(data["c"][:, :3]).all(axis=(1, 2)) & np.isfinite(data["base_home"])
    y = truth[support]; home = np.clip(data["base_home"][support], spec["probability_clip"], 1-spec["probability_clip"])
    g, c = data["g"][support], data["c"][support]
    logits = np.log(home/(1-home)); base = (logits < 0).astype(int)
    metrics = {}
    evidence = [np.zeros_like(g), g, g*c[:, 0], g*c[:, 1], g*c[:, 2]]
    for name, e in zip(ARMS, evidence):
        pred = ((logits+spec["conditional_logit_weight"]*(e[:, 0]-e[:, 1])) < 0).astype(int)
        confusion = [[int(((y == t) & (pred == p)).sum()) for p in (0, 1)] for t in (0, 1)]
        recalls = [confusion[t][t]/sum(confusion[t]) if sum(confusion[t]) else None for t in (0, 1)]
        metrics[name] = {"balanced_accuracy": float(np.mean(recalls)) if None not in recalls else None,
                         "confusion": confusion, "corrected_oof_errors": int(((base != y) & (pred == y)).sum()),
                         "new_oof_errors": int(((base == y) & (pred != y)).sum())}
    coverage = {"geometry_candidates": len(truth), "live_candidates": int(live.sum()),
                "dead_or_unknown_candidates": int((~live).sum()), "supported_live": int(support.sum()),
                "native_home": int((y == 0).sum()), "native_away": int((y == 1).sum()),
                "oof_wrong": int((base != y).sum()),
                "support_fraction": float(support.sum()/max(1, live.sum())),
                "single_receiver_side_events": int(((data["counts"] == 1).any(-1) & support).sum()),
                "post_control_supported": int((np.isfinite(data["c"][:, 3]).all(-1) & support).sum())}
    gate = (coverage["supported_live"] >= spec["minimum_supported_live_events_per_match"]
            and min(coverage["native_home"], coverage["native_away"]) >= spec["minimum_events_per_team_per_match"]
            and coverage["oof_wrong"] >= spec["minimum_oof_wrong_events_per_match"]
            and coverage["support_fraction"] >= spec["minimum_matched_support_fraction"])
    # Secondary post-exit control is reported only on its narrower, explicitly counted support.
    post_mask = np.isfinite(c[:, 3]).all(-1)
    post_accuracy = None
    if post_mask.any() and all((y[post_mask] == t).any() for t in (0, 1)):
        e = g[post_mask]*c[post_mask, 3]
        pred = ((logits[post_mask]+e[:, 0]-e[:, 1]) < 0).astype(int)
        post_accuracy = float(np.mean([(pred[y[post_mask] == t] == t).mean() for t in (0, 1)]))
    return {"coverage": coverage, "coverage_gate": bool(gate), "arms": metrics,
            "secondary_post_control_balanced_accuracy": post_accuracy}


def summarize(matches, spec):
    covered = all(v["coverage_gate"] for v in matches.values())
    gains = {}
    full = "prior_receiver_coordination"
    for arm in ARMS:
        values = [v["arms"][full]["balanced_accuracy"]-v["arms"][arm]["balanced_accuracy"]
                  for v in matches.values() if v["arms"][full]["balanced_accuracy"] is not None
                  and v["arms"][arm]["balanced_accuracy"] is not None]
        gains[arm] = float(np.mean(values)) if len(values) == len(matches) else None
    corrected = sum(v["arms"][full]["corrected_oof_errors"] for v in matches.values())
    harmed = sum(v["arms"][full]["new_oof_errors"] for v in matches.values())
    gain_gate = (covered and gains[ARMS[0]] >= spec["minimum_mean_gain_vs_frozen_oof"]
                 and all(v["arms"][full]["balanced_accuracy"] >= v["arms"][ARMS[0]]["balanced_accuracy"] for v in matches.values())
                 and all(gains[a] >= spec["minimum_mean_gain_vs_each_geometry_control"] for a in (ARMS[1], ARMS[3], ARMS[4])))
    harm_gate = corrected >= spec["minimum_total_corrected_oof_errors"] and corrected >= harmed*spec["minimum_corrections_to_new_errors_ratio"]
    return {"matches": matches, "mean_gain": gains, "total_corrections": corrected, "total_new_errors": harmed,
            "coverage_gate": covered, "gain_gate": bool(gain_gate), "harm_gate": bool(harm_gate),
            "status": "inconclusive_coverage" if not covered else "prioritize_interval_probe" if gain_gate and harm_gate else "close_fixed_formulation",
            "training_performed": False, "validation_accessed": False, "test_accessed": False,
            "target_accessed": False, "promoted": False,
            "interpretation": "Event-level exploratory diagnosis only; no pass/rebound semantic labels, interval gain or independent confirmation."}
