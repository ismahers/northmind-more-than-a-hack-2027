"""Antisymmetric availability- and duration-scale-gated temporal logit fusion."""
import numpy as np

from possession.temporal_probe.windows import emissions


def team_logit_from_probabilities(probabilities):
    """Extract conditional log(P(H|live) / P(A|live)) from [P(H), P(A), P(D)]."""
    p = np.asarray(probabilities, dtype=np.float64)
    alive = np.clip(p[:, 0] + p[:, 1], 1e-7, 1.0)
    h = np.clip(p[:, 0] / alive, 1e-6, 1.0 - 1e-6)
    return np.log(h / (1.0 - h)).astype(np.float32)


def observation_gate(ball_available, run, window=9):
    """Per-run local ball observation density in [0, 1]."""
    obs = np.asarray(ball_available, dtype=np.float32) > 0.5
    run_arr = np.asarray(run)
    if window <= 1 or len(obs) == 0:
        return obs.astype(np.float32)
    half = window // 2
    kernel = np.ones(2 * half + 1, dtype=np.float32) / float(2 * half + 1)
    gate = np.zeros(len(obs), dtype=np.float32)
    for r in np.unique(run_arr):
        ix = np.flatnonzero(run_arr == r)
        if not len(ix):
            continue
        padded = np.pad(obs[ix].astype(np.float32), half, mode="edge")
        gate[ix] = np.convolve(padded, kernel, mode="valid")
    return np.clip(gate, 0.0, 1.0)


def short_pulse_mask(z_sharp, z_multi, ball_available, run,
                     min_frames=3, max_frames=30, min_conf=0.20):
    """Identify observed short sign-consistent runs (0.3s-3.0s at 10Hz) of z_sharp.

    Strictly invariant to H/A label negation: depends only on sign changes of
    z_sharp and |z_sharp|.
    """
    z_s = np.asarray(z_sharp, dtype=np.float32)
    z_m = np.asarray(z_multi, dtype=np.float32)
    obs = np.asarray(ball_available, dtype=np.float32) > 0.5
    run_arr = np.asarray(run)
    mask = np.zeros(len(z_s), dtype=bool)
    if len(z_s) == 0:
        return mask

    signs = np.where(z_s >= 0.0, 1, -1)
    # Break runs at period boundaries or sign flips
    breaks = np.r_[0, np.flatnonzero((signs[1:] != signs[:-1]) | (run_arr[1:] != run_arr[:-1])) + 1, len(z_s)]
    for start, stop in zip(breaks[:-1], breaks[1:]):
        duration = stop - start
        if min_frames <= duration <= max_frames:
            seg_obs = obs[start:stop]
            if seg_obs.mean() >= 0.5:
                conf = float(np.mean(np.abs(z_s[start:stop])))
                # Also check if multiscale damped or opposed this short run
                multi_damped = float(np.mean(signs[start] * z_m[start:stop])) < conf
                if conf >= min_conf and multi_damped:
                    mask[start:stop] = True
    return mask


def fuse_scale_selective(z_local, z_multi, ball_available, run,
                         z_teacher=None, z_alt_teacher=None, params=None,
                         obs_window=9, pulse_min_frames=3):
    """Fuse local, multiscale, and teacher conditional team logits.

    - During ball occlusions (low observation density), routes to multiscale + teacher
      to preserve long-range trajectory continuity.
    - During observed play, routes short sign-consistent pulses (<= pulse_max_frames)
      to boosted local logits (recovering <3s spells thatHMM/multiscale smooth out),
      while blending multiscale + local on longer possessions (>=3s) to prevent
      mid-possession fragmentation and suppress 1-2 frame noise spikes.
    """
    p = params or {}
    alpha_long = float(p.get("alpha_long", 0.20))
    pulse_max_frames = int(p.get("pulse_max_frames", 30))
    pulse_min_conf = float(p.get("pulse_min_conf", 0.20))
    pulse_boost = float(p.get("pulse_boost", 1.35))
    teacher_blend = float(p.get("teacher_blend", 0.20))
    alt_seed_weight = float(p.get("alt_seed_weight", 0.0))

    z_loc = np.asarray(z_local, dtype=np.float32)
    z_mul = np.asarray(z_multi, dtype=np.float32)
    z_teach = z_loc if z_teacher is None else np.asarray(z_teacher, dtype=np.float32)
    if z_alt_teacher is not None and alt_seed_weight > 0.0:
        z_alt = np.asarray(z_alt_teacher, dtype=np.float32)
        z_teach = (1.0 - alt_seed_weight) * z_teach + alt_seed_weight * z_alt

    # Sharp local signal (optionally stabilized slightly by teacher)
    z_sharp = (1.0 - 0.5 * teacher_blend) * z_loc + (0.5 * teacher_blend) * z_teach
    # Smooth occlusion signal (multiscale + teacher)
    z_occ = (1.0 - teacher_blend) * z_mul + teacher_blend * z_teach
    # Long observed possession signal
    z_long = alpha_long * z_sharp + (1.0 - alpha_long) * z_occ

    w_obs = observation_gate(ball_available, run, window=obs_window)
    pulses = short_pulse_mask(
        z_sharp, z_mul, ball_available, run,
        min_frames=pulse_min_frames,
        max_frames=pulse_max_frames,
        min_conf=pulse_min_conf,
    )

    z_obs = np.where(pulses, pulse_boost * z_sharp, z_long)
    # Gate smoothly by observation density (high w_obs -> z_obs; occluded -> z_occ)
    obs_weight = np.where(w_obs >= 0.90, 1.0, w_obs ** 2).astype(np.float32)
    return (obs_weight * z_obs + (1.0 - obs_weight) * z_occ).astype(np.float32)


def fused_emissions(z_fused, base_probabilities, alt_probabilities=None, alt_alive_weight=0.0):
    """Combine conditional team logit with frozen (or multi-seed) alive probabilities."""
    base_p = np.asarray(base_probabilities, dtype=np.float32)
    if alt_probabilities is not None and alt_alive_weight > 0.0:
        alt_p = np.asarray(alt_probabilities, dtype=np.float32)
        blended_p = (1.0 - alt_alive_weight) * base_p + alt_alive_weight * alt_p
        return emissions(z_fused, blended_p)
    return emissions(z_fused, base_p)

