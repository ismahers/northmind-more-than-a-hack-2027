"""Extract geometric possession features from tactical ATD, in memory only."""

from dataclasses import dataclass
import hashlib
import pickle
from pathlib import Path

import numpy as np

from games import GAME, GAMES, require, source_frames
from starter.baseline import team_of_each_track


@dataclass
class Evidence:
    nearest_codes: np.ndarray
    team_distance_m: np.ndarray
    ball_xy_m: np.ndarray


def load_evidence(use_cache=True):
    """Stream the documented ATD format, enforcing exact frame order and slot count.

    All per-frame geometry stays in memory. Exceptions deliberately contain no
    input tokens. The command entry point additionally suppresses parser errors.
    """
    meta, raw = require(["atd_meta", "atd_raw"])
    teams = team_of_each_track(meta)
    scale = np.array(GAMES[GAME]["pitch"])
    n_frames = source_frames()
    n_slots = len(teams)
    teams_sha = hashlib.sha256(teams.tobytes()).hexdigest()

    cache_dir = Path("evaluation/reports")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "features_cache.pkl"

    if use_cache and cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if (cached.get("n_frames") == n_frames
                    and cached.get("nearest_codes") is not None
                    and len(cached["nearest_codes"]) == n_frames
                    and cached.get("teams_sha", teams_sha) == teams_sha):
                print(f"[cache] Loaded features from {cache_path}")
                return Evidence(
                    cached["nearest_codes"],
                    cached["team_distance_m"],
                    cached["ball_xy_m"],
                )
        except Exception:
            pass

    # --- First pass: count valid frames per slot to filter ghost tracks ---
    slot_valid_counts = np.zeros(n_slots, dtype=np.int32)
    MAX_SPEED_MS = 30.0  # Filter out impossibly fast tracks (ghost tracks)
    with open(raw, encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            positions_3d = np.fromstring(parts[1].replace(";", ","), sep=",")
            if positions_3d.size != n_slots * 3:
                continue
            positions_3d = positions_3d.reshape(n_slots, 3)
            pos_finite = np.isfinite(positions_3d[:, :2]).all(axis=1)
            speed_ok = positions_3d[:, 2] <= MAX_SPEED_MS
            valid = pos_finite & speed_ok
            slot_valid_counts += valid.astype(np.int32)

    MIN_VALID_FRAMES = 50  # ~2 seconds at 25 fps
    slot_mask = slot_valid_counts >= MIN_VALID_FRAMES
    if not slot_mask.all():
        print(f"[tracking] Filtered {int((~slot_mask).sum())} ghost tracks "
              f"(<{MIN_VALID_FRAMES} valid frames or speed >{MAX_SPEED_MS} m/s)")

    # --- Second pass: compute features with filtered slots ---
    teams_filtered = teams[slot_mask]
    nearest = np.full(n_frames, -1, dtype=np.int8)
    distances = np.full((n_frames, 2), np.inf)
    ball_xy = np.full((n_frames, 2), np.nan)
    count = 0
    with open(raw, encoding="utf-8") as handle:
        for frame, line in enumerate(handle):
            parts = line.strip().split(":")
            if len(parts) != 3 or frame >= n_frames or int(parts[0]) != frame + 1:
                raise ValueError("ATD frame count or order does not match tactical video")
            positions_3d = np.fromstring(parts[1].replace(";", ","), sep=",")
            ball = np.fromstring(parts[2], sep=",")
            if positions_3d.size != n_slots * 3 or ball.size != 3:
                raise ValueError("Unexpected ATD group dimensions")
            positions_3d = positions_3d.reshape(n_slots, 3)

            positions_3d = positions_3d[slot_mask]
            positions = positions_3d[:, :2]

            if np.isfinite(ball[:2]).all():
                ball_xy[frame] = ball[:2] * scale
                squared = np.sum(((positions - ball[:2]) * scale) ** 2, axis=1)
                squared[~np.isfinite(squared)] = np.inf
                if squared.size > 0:
                    best_slot = int(np.argmin(squared))
                    if np.isfinite(squared[best_slot]):
                        nearest[frame] = teams_filtered[best_slot]
                    for team in (0, 1):
                        group = squared[teams_filtered == team]
                        if group.size:
                            distances[frame, team] = np.sqrt(group.min())
            count += 1
    if count != n_frames:
        raise ValueError("ATD is incomplete for the tactical timeline")

    evidence = Evidence(nearest, distances, ball_xy)
    cache_data = {
        "nearest_codes": nearest,
        "team_distance_m": distances,
        "ball_xy_m": ball_xy,
        "n_frames": n_frames,
        "teams_hash": hash(teams.tobytes()),
        "teams_sha": teams_sha,
    }
    if use_cache:
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)
        print(f"[cache] Saved features to {cache_path}")
    return evidence
