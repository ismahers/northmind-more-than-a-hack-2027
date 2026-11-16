"""Globally decode team control while keeping geometric stopping evidence separate."""

from dataclasses import dataclass

import numpy as np

from games import GAME, GAMES, half_video_blocks
from possession.inference import Config, block_dead_evidence, team_strength
from possession.tactical_tagging import get_quick_restart_mask, load_restart_and_box_prior
from possession.st_tcn_target import load_cached_st_tcn_target


@dataclass(frozen=True)
class SequenceConfig:
    # Fixed engineering choices before scoring. No public-GT parameter search.
    switch_cost: float = 35.0  # Base cost (used if adaptive disabled)
    max_bridge_frames: int = 125  # Five seconds of unsupported D at most.
    # Adaptive switch cost parameters (all nonnegative)
    switch_cost_min: float = 14.0   # Cost when evidence is STRONG (|strength| >= 0.25)
    switch_cost_max: float = 55.0   # Cost when evidence is MODERATE
    switch_cost_zero: float = 80.0  # Cost when NO evidence (strength = 0)
    # Strength smoothing: temporal window (frames) to average evidence.
    strength_smooth_frames: int = 5
    # Extra tactical prior weight for sequence decoder (on top of inference weight 0.25).
    extra_prior_weight: float = 0.08
    # Weight for set-piece restarts, box entries, and goals prior.
    restart_prior_weight: float = 0.20
    # Weights for 4-way equivariant ST+TCN + Scale-Selective neural team signal
    neural_weight_observed: float = 0.25
    neural_weight_occluded: float = 0.45
    # Ball speed threshold (m/s) above which straight ball flight damps pass-by proximity
    pass_speed_threshold_ms: float = 7.5

    def __post_init__(self):
        for name, val in (
            ("switch_cost", self.switch_cost),
            ("switch_cost_min", self.switch_cost_min),
            ("switch_cost_max", self.switch_cost_max),
            ("switch_cost_zero", self.switch_cost_zero),
            ("extra_prior_weight", self.extra_prior_weight),
            ("restart_prior_weight", self.restart_prior_weight),
            ("neural_weight_observed", self.neural_weight_observed),
            ("neural_weight_occluded", self.neural_weight_occluded),
            ("pass_speed_threshold_ms", self.pass_speed_threshold_ms),
        ):
            if (isinstance(val, bool) or not isinstance(val, (int, float))
                    or not np.isfinite(val) or val < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if (isinstance(self.max_bridge_frames, bool)
                or not isinstance(self.max_bridge_frames, int) or self.max_bridge_frames < 0):
            raise ValueError("max_bridge_frames must be a nonnegative integer")
        if not (self.switch_cost_min <= self.switch_cost_max <= self.switch_cost_zero):
            raise ValueError("Expected switch_cost_min <= switch_cost_max <= switch_cost_zero")
        if (isinstance(self.strength_smooth_frames, bool)
                or not isinstance(self.strength_smooth_frames, int) or self.strength_smooth_frames < 0):
            raise ValueError("strength_smooth_frames must be a nonnegative integer")


def runs(mask):
    """Half-open spans of true values, including spans touching either edge."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def _smooth_strength(strength, window_frames):
    """Temporal smoothing of evidence strength using a centered moving average."""
    if window_frames <= 1 or len(strength) == 0:
        return np.asarray(strength, dtype=float)
    k = window_frames // 2
    eff_win = 2 * k + 1
    kernel = np.ones(eff_win, dtype=float) / eff_win
    padded = np.pad(np.asarray(strength, dtype=float), k, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed


def _ball_kinematics_weights(ball_xy, distances, pass_speed_threshold_ms=7.5, fps=25.0):
    """Compute per-frame kinematic scaling in [0.25, 1.25] from ball trajectory.

    - High-speed straight flight (pass/clearance in flight > pass_speed_threshold_ms):
      damps proximity evidence so the ball flying past an opponent does not trigger
      an uncontrolled turnover (CHALLENGE.md rule).
    - Low-speed close control (< 1.8m and <= 5.5m/s) or direction inflection with
      close player (< 2.0m): boosts control evidence.
    """
    ball = np.asarray(ball_xy, dtype=float)
    dist = np.asarray(distances, dtype=float)
    n = len(ball)
    weights = np.ones(n, dtype=float)
    if n < 3:
        return weights

    visible = np.isfinite(ball).all(axis=1)
    nearest_d = dist.min(axis=1)
    has_player = np.isfinite(nearest_d)

    # Estimate instantaneous velocity and direction change over +/- 1 frame
    vel_prev = np.zeros((n, 2), dtype=float)
    vel_next = np.zeros((n, 2), dtype=float)
    valid_prev = np.zeros(n, dtype=bool)
    valid_next = np.zeros(n, dtype=bool)

    valid_step = visible[1:] & visible[:-1]
    step_v = np.where(valid_step[:, None], (ball[1:] - ball[:-1]) * fps, 0.0)

    vel_prev[1:] = step_v
    valid_prev[1:] = valid_step
    vel_next[:-1] = step_v
    valid_next[:-1] = valid_step

    speed_prev = np.linalg.norm(vel_prev, axis=1)
    speed_next = np.linalg.norm(vel_next, axis=1)
    speed = np.where(
        valid_prev & valid_next,
        0.5 * (speed_prev + speed_next),
        np.where(valid_prev, speed_prev, speed_next),
    )
    has_speed = valid_prev | valid_next

    # Cosine of direction change where both incoming and outgoing velocities exist
    both_moving = valid_prev & valid_next & (speed_prev > 1.5) & (speed_next > 1.5)
    dot = np.sum(vel_prev * vel_next, axis=1)
    denom = np.maximum(speed_prev * speed_next, 1e-6)
    cos_angle = np.where(both_moving, np.clip(dot / denom, -1.0, 1.0), 1.0)

    # Straight fast flight: speed > threshold and not deflecting/receiving sharply
    straight_flight = has_speed & (speed > pass_speed_threshold_ms) & (cos_angle > 0.65)
    excess_speed = np.maximum(0.0, speed - pass_speed_threshold_ms)
    flight_damp = np.clip(np.exp(-excess_speed / 5.5), 0.25, 1.0)
    weights = np.where(straight_flight, flight_damp, weights)

    # True control point: close player (< 1.8m) with controlled speed (<= 5.5 m/s)
    # or sharp inflection (cos_angle < 0.35) near a player (< 2.0m)
    close_controlled = has_player & (nearest_d <= 1.8) & has_speed & (speed <= 5.5)
    close_inflection = has_player & (nearest_d <= 2.0) & both_moving & (cos_angle < 0.35)
    weights = np.where(close_controlled | close_inflection, 1.20, weights)

    return weights


def _adaptive_switch_costs(strength, config):
    """Compute per-frame switch costs from evidence strength."""
    abs_strength = np.abs(strength)
    costs = np.where(
        abs_strength >= 0.25,
        config.switch_cost_min,
        config.switch_cost_zero - (abs_strength / 0.25) * (config.switch_cost_zero - config.switch_cost_min),
    )
    return np.clip(costs, config.switch_cost_min, config.switch_cost_zero)


def decode_control(strength, switch_cost, fallback="H"):
    """Exact two-state dynamic program on one uninterrupted live span.

    Maximize sum(+strength for H, -strength for A) - cost * team changes.
    Missing evidence contributes zero, so it does not invent a handover.
    Backtracking locates the whole sequence jointly, without a confirmation lag.
    Ties prefer staying; final ties use the supplied reference team.

    switch_cost can be a scalar (constant cost) or 1D array (per-frame adaptive cost).
    """
    values = np.asarray(strength, dtype=float)
    costs = np.asarray(switch_cost, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or fallback not in ("H", "A"):
        raise ValueError("Invalid control evidence")
    if costs.ndim == 0:
        costs = np.full(len(values), float(costs))
    if costs.ndim != 1 or costs.shape != values.shape or not np.isfinite(costs).all() or np.any(costs < 0):
        raise ValueError("Invalid switch_cost array")
    if not len(values):
        return np.array([], dtype="<U1")
    predecessor = np.empty((len(values), 2), dtype=np.uint8)
    home, away = values[0], -values[0]
    for frame in range(1, len(values)):
        cost = costs[frame]
        home_from_away = away - cost > home
        away_from_home = home - cost > away
        predecessor[frame, 0] = 1 if home_from_away else 0
        predecessor[frame, 1] = 0 if away_from_home else 1
        home, away = (
            (away - cost if home_from_away else home) + values[frame],
            (home - cost if away_from_home else away) - values[frame],
        )
    state = (0 if home > away else 1) if home != away else (0 if fallback == "H" else 1)
    output = np.empty(len(values), dtype="<U1")
    for frame in range(len(values) - 1, -1, -1):
        output[frame] = "H" if state == 0 else "A"
        if frame:
            state = int(predecessor[frame, state])
    return output


def _filter_oscillations(labels, max_oscillation_window=48, min_oscillations=3,
                         min_possession_frames=20):
    """Filter each uninterrupted live span; D/X are barriers to all evidence."""
    values = np.asarray(labels)
    if values.ndim != 1:
        return values
    result = values.copy()
    for start, stop in runs(np.isin(values, ("H", "A"))):
        result[start:stop] = _filter_live_oscillations(
            values[start:stop], max_oscillation_window, min_oscillations,
            min_possession_frames)
    return result


def _filter_live_oscillations(values, max_window, min_transitions, min_frames):
    result = values.copy()
    transitions = np.flatnonzero(values[1:] != values[:-1]) + 1
    i = 0
    while i < len(transitions):
        j = i + 1
        while j < len(transitions) and transitions[j] - transitions[i] <= max_window:
            j += 1
        if j - i >= min_transitions:
            # Only the interior of the cluster may change. In particular, the
            # last cluster never consumes an arbitrarily long trailing spell.
            start, stop = int(transitions[i]), int(transitions[j - 1])
            before, after = values[start - 1], values[stop]
            interior = values[start:stop]
            home, away = np.count_nonzero(interior == "H"), np.count_nonzero(interior == "A")
            if before == after:
                result[start:stop] = before
            elif home != away:
                result[start:stop] = "H" if home > away else "A"
            # Ties preserve the input, maintaining H/A exchange symmetry.
            i = j
        else:
            i += 1
    # Decide all bounded blips from one snapshot: no cascading merge and no
    # fictitious run spanning a D/X separator.
    snapshot = result.copy()
    bounds = np.r_[0, np.flatnonzero(snapshot[1:] != snapshot[:-1]) + 1, len(snapshot)]
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if (0 < start < stop < len(snapshot) and stop - start <= min_frames
                and snapshot[start - 1] == snapshot[stop] != snapshot[start]):
            result[start:stop] = snapshot[start - 1]
    return result


def _align_transitions_to_geometry(seq_labels, geom_labels, window_frames=20):
    """Align sequence model transitions to nearest same-direction geometry transitions.

    Never overwrites D or X frames and requires matching transition polarity (H->A vs A->H).
    """
    seq = np.asarray(seq_labels)
    geom = np.asarray(geom_labels)
    if seq.shape != geom.shape:
        return seq

    result = seq.copy()
    n = len(seq)

    seq_trans = []
    geom_trans = []
    for i in range(1, n):
        if seq[i] in ("H", "A") and seq[i - 1] in ("H", "A") and seq[i] != seq[i - 1]:
            seq_trans.append(i)
        if geom[i] in ("H", "A") and geom[i - 1] in ("H", "A") and geom[i] != geom[i - 1]:
            geom_trans.append(i)

    if not seq_trans or not geom_trans:
        return result

    geom_trans = np.array(geom_trans)
    for s_idx in seq_trans:
        before_team = result[s_idx - 1]
        after_team = result[s_idx]
        if before_team not in ("H", "A") or after_team not in ("H", "A") or before_team == after_team:
            continue

        # Only consider geometry transitions with the SAME direction (before_team -> after_team)
        same_dir = [
            g for g in geom_trans
            if abs(int(g) - s_idx) <= window_frames
            and geom[g - 1] == before_team
            and geom[g] == after_team
        ]
        if not same_dir:
            continue
        g_idx = int(min(same_dir, key=lambda g: abs(int(g) - s_idx)))
        if g_idx == s_idx:
            continue

        run_start = s_idx
        while run_start > 0 and result[run_start - 1] == before_team:
            run_start -= 1
        run_end = s_idx
        while run_end < n and result[run_end] == after_team:
            run_end += 1

        if g_idx > s_idx:
            if g_idx < run_end and np.isin(result[s_idx:g_idx], ("H", "A")).all():
                result[s_idx:g_idx] = before_team
        else:
            if g_idx > run_start and np.isin(result[g_idx:s_idx], ("H", "A")).all():
                result[g_idx:s_idx] = after_team

    return result


def bridge_unsupported_stops(labels, protected_dead, max_frames):
    """Fill only bounded D runs between the same team, absent geometric evidence.

    Called separately per half. No merging across an edge, X or a protected D.
    Decisions use the original labels, so one bridge cannot trigger another.
    """
    values = np.asarray(labels)
    protected = np.asarray(protected_dead, dtype=bool)
    if values.ndim != 1 or protected.shape != values.shape or max_frames < 0:
        raise ValueError("Invalid stopping mask or bridge limit")
    result = values.copy()
    for start, stop in runs(values == "D"):
        if (stop - start <= max_frames and start > 0 and stop < len(values)
                and not protected[start:stop].any()
                and values[start - 1] in ("H", "A")
                and values[start - 1] == values[stop]):
            result[start:stop] = values[start - 1]
    return result


def predict_sequence(evidence, reference_labels, team_prior, config=None, sequence_config=None):
    """Two fixed ablations: sequence control; same control plus conservative D bridging."""
    config = config if config is not None else Config()
    sequence_config = sequence_config if sequence_config is not None else SequenceConfig()
    n_frames = GAMES[GAME]["n_video_frames"]
    reference = np.asarray(reference_labels)
    if (reference.shape != (n_frames,) or not np.isin(reference, ("H", "A", "D", "X")).all()
            or evidence.team_distance_m.shape != (n_frames, 2)
            or evidence.ball_xy_m.shape != (n_frames, 2)
            or team_prior.shape != (n_frames,) or not np.isfinite(team_prior).all()
            or np.any(np.abs(team_prior) > 1)):
        raise ValueError("Sequence inputs must match the tactical timeline")

    quick_restart_mask = get_quick_restart_mask(window_frames=50)
    restart_box_prior = load_restart_and_box_prior(n_frames=n_frames, post_restart_frames=50)
    neural_cache = load_cached_st_tcn_target(n_frames=n_frames)

    plain = np.full(n_frames, "X", dtype="<U1")
    bridged = plain.copy()
    bridged_frames = 0
    geometry_only = np.full(n_frames, "X", dtype="<U1")

    for _, first, last in half_video_blocks(GAME):
        block = slice(first, last + 1)
        values = reference[block]
        if np.any(values == "X"):
            raise ValueError("Reference must supply H/A/D inside each half")
        distances = evidence.team_distance_m[block]
        ball_xy = evidence.ball_xy_m[block]
        observed = np.isfinite(distances.min(axis=1))

        # 1. Base geometric + tactical strength weighted by ball kinematics (pass flight vs control)
        kin_weights = _ball_kinematics_weights(
            ball_xy, distances, pass_speed_threshold_ms=sequence_config.pass_speed_threshold_ms
        )
        raw_geom = team_strength(distances, config, None) * kin_weights
        strength = np.clip(raw_geom + config.tagging_prior_weight * team_prior[block] * observed, -1, 1)

        if sequence_config.extra_prior_weight > 0:
            strength = np.clip(strength + sequence_config.extra_prior_weight * team_prior[block], -1, 1)

        if sequence_config.restart_prior_weight > 0:
            strength = np.clip(
                strength + sequence_config.restart_prior_weight * restart_box_prior[block], -1, 1
            )

        # 2. Fuse 4-way Equivariant Set Transformer + TCN + Scale-Selective neural team signal
        if neural_cache is not None:
            neural_sig = neural_cache["team_signal"][block]
            w_neural = np.where(
                observed,
                sequence_config.neural_weight_observed,
                sequence_config.neural_weight_occluded,
            )
            strength = np.clip((1.0 - 0.5 * w_neural) * strength + w_neural * neural_sig, -1, 1)

        strength = _smooth_strength(strength, sequence_config.strength_smooth_frames)

        geom_strength = np.clip(raw_geom, -1, 1)
        geom_strength = _smooth_strength(geom_strength, sequence_config.strength_smooth_frames)

        protected = block_dead_evidence(
            distances, ball_xy, config, quick_restart_mask[block]
        )
        merged = bridge_unsupported_stops(values, protected, sequence_config.max_bridge_frames)

        geom_decoded = values.copy()
        for start, stop in runs(values != "D"):
            seg_geom = geom_strength[start:stop]
            geom_costs = _adaptive_switch_costs(seg_geom, sequence_config)
            geom_decoded[start:stop] = decode_control(
                seg_geom, geom_costs, fallback=values[start])
        geom_decoded = _filter_oscillations(geom_decoded)
        if np.any(geom_decoded[protected] != "D"):
            geom_decoded[protected] = "D"
        geometry_only[block] = geom_decoded

        for is_bridged, (labels, target) in enumerate(((values, plain), (merged, bridged))):
            decoded = labels.copy()
            for start, stop in runs(labels != "D"):
                seg_strength = strength[start:stop]
                adaptive_costs = _adaptive_switch_costs(seg_strength, sequence_config)
                decoded[start:stop] = decode_control(
                    seg_strength, adaptive_costs, fallback=labels[start])
            decoded = _filter_oscillations(decoded)
            if is_bridged:
                # Also bridge unsupported D runs whose flanking teams agree after DP decoding
                decoded = bridge_unsupported_stops(decoded, protected, sequence_config.max_bridge_frames)
                decoded = _filter_oscillations(decoded)
            if np.any(decoded[protected] != "D"):
                raise ValueError("Sequence decoding removed independent dead-ball evidence")
            target[block] = decoded

        bridged_frames += int(np.count_nonzero((bridged[block] != "D") & (values == "D")))

    plain = _align_transitions_to_geometry(plain, geometry_only, window_frames=20)
    bridged = _align_transitions_to_geometry(bridged, geometry_only, window_frames=20)

    if not np.array_equal(plain == "D", reference == "D"):
        raise ValueError("Control-only decoding changed dead-ball decisions")
    changed_dead = (bridged == "D") != (reference == "D")
    if np.any(bridged[changed_dead] == "D"):
        raise ValueError("Bridging may remove unsupported D only")
    return {"northmind_sequence": plain, "northmind_sequence_bridged": bridged}, {
        "bridged_dead_frames": bridged_frames,
        "control_only_preserves_dead_mask": True,
        "protected_geometry_preserved": True,
        "uses_equivariant_st_tcn": neural_cache is not None,
        "neural_cache_fingerprint": None if neural_cache is None else neural_cache["fingerprint"],
    }
