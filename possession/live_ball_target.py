"""Pure target live/dead-mask ablation; no loaders, files or ground truth.

The original full target H/A evidence and global geometry alignment are kept.
The returned candidate is experimental and has not passed every external gate.
"""
import numpy as np


def recover_cached_live(match,view,cache,prior,n_frames):
    """Recover 10Hz anchors only when their native round trip is EXACT.

    The target cache stores float32 alive probabilities; external validation
    used float64 averages. This recovers the stored values, without claiming
    that the two precisions or the full target recipe were jointly validated.
    Only clocks/indexes are read from match/view, never their supervision.
    """
    from possession.st_tcn_target import _native_signals
    from possession.live_ball import decode_live_probability
    alive = np.asarray(cache["alive_prob"])
    if (alive.shape!=(n_frames,) or alive.dtype!=np.float32 or not np.isfinite(alive).all()
            or np.any((alive<0)|(alive>1))):
        raise ValueError("Invalid native alive cache")
    anchors = np.asarray(match.frame)[np.asarray(view.native_index)]
    if np.any((anchors<0)|(anchors>=n_frames)):
        raise ValueError("Anchor outside native timeline")
    alive10 = alive[anchors]
    zeros = np.zeros(len(view.time))
    roundtrip = _native_signals(match,view,zeros,alive10,n_frames)[1]
    if not np.array_equal(roundtrip,alive):
        raise ValueError("Alive anchor round trip failed")
    live10 = decode_live_probability(alive10,view.time,view.run,prior,fps=10.,strength=.5)
    live25 = _native_signals(match,view,zeros,live10.astype(np.float32),n_frames)[1]>.5
    return live25


def _decode_span(values,strength,geom_strength,protected,config):
    from possession.sequence_decoding import (runs,decode_control,_adaptive_switch_costs,
        _filter_oscillations)
    alignment,decoded = values.copy(),values.copy()
    for start,stop in runs(values!="D"):
        alignment[start:stop] = decode_control(geom_strength[start:stop],
            _adaptive_switch_costs(geom_strength[start:stop],config),fallback=values[start])
        decoded[start:stop] = decode_control(strength[start:stop],
            _adaptive_switch_costs(strength[start:stop],config),fallback=values[start])
    alignment = _filter_oscillations(alignment)
    alignment[protected] = "D"
    decoded = _filter_oscillations(decoded)
    if np.any(decoded[protected]!="D"):
        raise ValueError("Independent geometric dead evidence changed")
    return decoded,alignment


def decode_target_candidate(evidence,reference_labels,team_prior,quick_restart_mask,
        restart_box_prior,neural_cache,learned_live,blocks,config=None,sequence_config=None):
    """Return baseline and candidate with the EXACT current plain H/A decoder.

    Both use tactical/restart priors, kinematic geometry, the same neural team
    signal, target smoothing/costs/filter and GLOBAL transition alignment.
    Candidate D = protected geometric D OR not learned-live. Newly live samples
    previously D fall back to sign of unsmoothed fused strength, exact tie H.
    X outside half blocks stays X. No memory or D bridging is introduced.
    """
    from possession.inference import Config,team_strength,block_dead_evidence
    from possession.sequence_decoding import (SequenceConfig,_ball_kinematics_weights,
        _smooth_strength,_align_transitions_to_geometry)
    config = Config() if config is None else config
    seq = SequenceConfig() if sequence_config is None else sequence_config
    reference = np.asarray(reference_labels)
    n = len(reference)
    prior,restarts = np.asarray(team_prior,float),np.asarray(restart_box_prior,float)
    quick,live = np.asarray(quick_restart_mask),np.asarray(learned_live)
    signal = np.asarray(neural_cache["team_signal"],float)
    if (reference.shape!=(n,) or not np.isin(reference,("H","A","D","X")).all()
            or evidence.team_distance_m.shape!=(n,2) or evidence.ball_xy_m.shape!=(n,2)
            or prior.shape!=(n,) or restarts.shape!=(n,) or signal.shape!=(n,)
            or quick.shape!=(n,) or quick.dtype!=np.bool_ or live.shape!=(n,) or live.dtype!=np.bool_
            or any(not np.isfinite(x).all() or np.any(abs(x)>1) for x in (prior,restarts,signal))):
        raise ValueError("Invalid target ablation arrays")
    baseline,candidate,geom_old,geom_new = [np.full(n,"X",dtype="<U1") for _ in range(4)]
    covered = np.zeros(n,bool)
    for _,first,last in blocks:
        if (not isinstance(first,(int,np.integer)) or not isinstance(last,(int,np.integer))
                or first<0 or last<first or last>=n or covered[first:last+1].any()):
            raise ValueError("Invalid half boundaries")
        covered[first:last+1] = True
        block = slice(first,last+1)
        values = reference[block]
        if np.any(values=="X"):
            raise ValueError("Unexpected X inside target half")
        distances,ball = evidence.team_distance_m[block],evidence.ball_xy_m[block]
        observed = np.isfinite(distances.min(axis=1))
        kin = _ball_kinematics_weights(ball,distances,pass_speed_threshold_ms=seq.pass_speed_threshold_ms)
        raw_geom = team_strength(distances,config,None)*kin
        strength = np.clip(raw_geom+config.tagging_prior_weight*prior[block]*observed,-1,1)
        if seq.extra_prior_weight>0:
            strength = np.clip(strength+seq.extra_prior_weight*prior[block],-1,1)
        if seq.restart_prior_weight>0:
            strength = np.clip(strength+seq.restart_prior_weight*restarts[block],-1,1)
        weight = np.where(observed,seq.neural_weight_observed,seq.neural_weight_occluded)
        strength = np.clip((1-.5*weight)*strength+weight*signal[block],-1,1)
        unsmoothed = strength.copy()
        strength = _smooth_strength(strength,seq.strength_smooth_frames)
        geom_strength = _smooth_strength(np.clip(raw_geom,-1,1),seq.strength_smooth_frames)
        protected = block_dead_evidence(distances,ball,config,quick[block])
        new_values = values.copy()
        new_dead = protected | ~live[block]
        newly_live = ~new_dead & (values=="D")
        new_values[newly_live] = np.where(unsmoothed[newly_live]>=0,"H","A")
        new_values[new_dead] = "D"
        baseline[block],geom_old[block] = _decode_span(values,strength,geom_strength,protected,seq)
        candidate[block],geom_new[block] = _decode_span(new_values,strength,geom_strength,protected,seq)
    if np.any(reference[~covered]!="X"):
        raise ValueError("Reference outside half must be X")
    baseline = _align_transitions_to_geometry(baseline,geom_old,window_frames=20)
    candidate = _align_transitions_to_geometry(candidate,geom_new,window_frames=20)
    if not np.array_equal(baseline=="D",reference=="D"):
        raise ValueError("Baseline stopping mask changed")
    if np.any(baseline[~covered]!="X") or np.any(candidate[~covered]!="X"):
        raise ValueError("Half separator changed")
    return baseline,candidate
