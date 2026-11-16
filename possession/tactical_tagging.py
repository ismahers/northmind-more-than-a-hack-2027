"""Convert automatic tactical tagging to bounded team evidence in memory."""

import numpy as np

from games import source_frames
from starter.load_smart_tagging import check_time_base, load


# Team names/aliases and code vocabulary are documented in DATA.md. These
# interpret attacking phases as evidence for the tagged side and defensive
# phases as evidence for its opponent. They do not define possession boundaries.
ATTACKING = {
    "BUILD UP", "PROGRESSION", "BALL IN FINAL THIRD", "BALL IN THE BOX",
    "CREATING CHANCES", "LONG BALL", "ATTACKING TRANSITION",
}
DEFENDING = {
    "DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD",
    "DEFENDING IN ATTACKING THIRD", "DEFENSIVE TRANSITION",
}
BOX_AND_GOALS = {"PLAYERS IN THE BOX", "GOALS"}
TEAM_SIGN = {"barcelona": 1, "atletic club": -1, "athletic club": -1}

# Set piece types that indicate a quick restart (ball back in play fast)
QUICK_RESTART_TYPES = {
    "throw-in", "corner kick", "free kick", "goal kick",
    "kick off (start)", "kick off (after goal)"
}


def load_team_prior():
    events = load()
    if not check_time_base(events)["agrees"]:
        raise ValueError("Automatic tagging does not pass the documented clock check")
    n_frames = source_frames()
    votes = np.zeros(n_frames + 1)
    counts = np.zeros(n_frames + 1)
    used = 0
    for event in events:
        code = (event["code"] or "").strip().upper()
        side = TEAM_SIGN.get((event["tags"].get("Team") or "").strip().casefold())
        if side is None or code not in ATTACKING | DEFENDING:
            continue
        start = max(0, min(n_frames, event["start_frame"]))
        stop = max(0, min(n_frames, event["end_frame"]))
        if start >= stop:
            continue
        sign = side if code in ATTACKING else -side
        votes[start] += sign
        votes[stop] -= sign
        counts[start] += 1
        counts[stop] -= 1
        used += 1
    votes = np.cumsum(votes[:-1])
    counts = np.cumsum(counts[:-1])
    # Overlapping tags cannot accumulate unlimited influence. Conflicting
    # phases cancel rather than selecting whichever interval was read last.
    prior = np.divide(votes, counts, out=np.zeros(n_frames), where=counts > 0)
    return prior, {"events_loaded": len(events), "events_used": used,
                   "clock_verified": True, "covered_frames": int(np.count_nonzero(counts))}


def get_quick_restart_mask(window_frames: int = 50) -> np.ndarray:
    """
    Return a boolean mask of frames that are within `window_frames` of a
    quick restart (throw-in, corner, free kick, goal kick, kick off).
    These frames should not be marked as dead ball due to 'ball outside pitch'.
    """
    events = load()
    if not check_time_base(events)["agrees"]:
        raise ValueError("Automatic tagging does not pass the documented clock check")
    n_frames = source_frames()
    mask = np.zeros(n_frames, dtype=bool)
    for event in events:
        code = (event["code"] or "").strip().upper()
        if code != "SET PIECES":
            continue
        tipo = (event["tags"].get("Type") or "").strip().lower()
        if tipo not in QUICK_RESTART_TYPES:
            continue
        start = max(0, min(n_frames, event["start_frame"]))
        stop = max(0, min(n_frames, event["end_frame"]))
        # Extend window around the set piece
        start = max(0, start - window_frames)
        stop = min(n_frames, stop + window_frames)
        mask[start:stop] = True
    return mask


def load_restart_and_box_prior(n_frames: int | None = None, post_restart_frames: int = 50) -> np.ndarray:
    """Extract bounded team directional cues from SET PIECES, PLAYERS IN THE BOX, and GOALS.

    Used only inside sequence_decoding so historical reference hashes remain unchanged.
    """
    target_n = source_frames() if n_frames is None else int(n_frames)
    try:
        events = load()
        if not check_time_base(events)["agrees"]:
            return np.zeros(target_n, dtype=float)
    except Exception:
        return np.zeros(target_n, dtype=float)

    votes = np.zeros(target_n + 1, dtype=float)
    counts = np.zeros(target_n + 1, dtype=float)
    for event in events:
        code = (event["code"] or "").strip().upper()
        side = TEAM_SIGN.get((event["tags"].get("Team") or "").strip().casefold())
        if side is None:
            continue
        start = max(0, min(target_n, int(event["start_frame"])))
        stop = max(0, min(target_n, int(event["end_frame"])))
        if start >= stop:
            continue
        if code == "SET PIECES":
            tipo = (event["tags"].get("Type") or "").strip().lower()
            if tipo in QUICK_RESTART_TYPES:
                # Restarting team puts the ball into play; extend slightly after end_frame
                r_end = min(target_n, stop + post_restart_frames)
                votes[start] += 1.5 * side
                votes[r_end] -= 1.5 * side
                counts[start] += 1.5
                counts[r_end] -= 1.5
        elif code in BOX_AND_GOALS:
            votes[start] += 1.0 * side
            votes[stop] -= 1.0 * side
            counts[start] += 1.0
            counts[stop] -= 1.0

    votes = np.cumsum(votes[:-1])
    counts = np.cumsum(counts[:-1])
    return np.divide(votes, counts, out=np.zeros(target_n, dtype=float), where=counts > 0)
