"""Native JSONL adapter; null possession never supplies dead-ball supervision."""

from collections import Counter
import json

import numpy as np

from .schema import empty_match, finite_number, sources


def rows(path):
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def clock(value):
    parts = str(value).split(":")
    if len(parts) != 3:
        raise ValueError("Expected SkillCorner HH:MM:SS timestamp")
    h, m, s = map(float, parts)
    result = h * 3600 + m * 60 + s
    if not np.isfinite(result):
        raise ValueError("Invalid timestamp")
    return result


def load_skillcorner(metadata_path, tracking_path):
    with open(metadata_path, encoding="utf-8") as stream:
        meta = json.load(stream)
    team_ids = [meta["home_team"]["id"], meta["away_team"]["id"]]
    rosters = [sorted(str(p["id"]) for p in meta["players"] if p["team_id"] == team)
               for team in team_ids]
    if not all(rosters):
        raise ValueError("Both SkillCorner rosters are required")
    index = {pid: (side, i) for side, roster in enumerate(rosters) for i, pid in enumerate(roster)}
    frames, periods, stamps, times, issues = [], [], [], [], Counter()
    for row in rows(tracking_path):
        if row.get("period") not in (1, 2, 3, 4):
            issues["nonplaying_period_records"] += 1
            continue
        frames.append(int(row["frame"]))
        periods.append(int(row["period"]))
        stamps.append(str(row["timestamp"]))
        times.append(clock(row["timestamp"]))
    pitch = [finite_number(meta.get("pitch_length")), finite_number(meta.get("pitch_width"))]
    # Dimensions are explicitly required. Guessing 105x68 can silently distort features.
    if not np.isfinite(pitch).all():
        raise ValueError("SkillCorner metadata must supply pitch_length and pitch_width")
    match = empty_match("skillcorner", meta["id"], frames, periods, stamps, times,
                        rosters, pitch, 10.0, sources([metadata_path, tracking_path]))
    seen = set()
    i = 0
    for row in rows(tracking_path):
        if row.get("period") not in (1, 2, 3, 4):
            continue
        key = (row["period"], int(row["frame"]))
        if key in seen:
            raise ValueError("Duplicate SkillCorner period/frame")
        seen.add(key)
        frame_players = set()
        for player in row.get("player_data") or []:
            pid = str(player.get("player_id"))
            if pid not in index:
                issues["unknown_player_id"] += 1
                continue
            side, p = index[pid]
            if pid in frame_players:
                raise ValueError("Duplicate player within frame")
            frame_players.add(pid)
            match.players[i, side, p] = [finite_number(player.get(a)) for a in ("x", "y")]
            detected = player.get("is_detected")
            match.player_detection_known[i, side, p] = isinstance(detected, bool)
            match.player_detected[i, side, p] = detected is True
        ball = row.get("ball_data") or {}
        match.ball[i] = [finite_number(ball.get(a)) for a in ("x", "y", "z")]
        detected = ball.get("is_detected")
        match.ball_detection_known[i] = isinstance(detected, bool)
        match.ball_detected[i] = detected is True
        poss = row.get("possession") or {}
        group = poss.get("group")
        match.native_possession[i] = json.dumps(poss, sort_keys=True, separators=(",", ":"))
        if group in ("home team", "away team"):
            match.team[i] = int(group == "away team")
            match.confidence[i, 1] = 1
            match.provenance[i, 1] = "native:possession.group|conditional"
        elif group is not None:
            issues["unknown_possession_group"] += 1
        i += 1
    match.player_available[:] = np.isfinite(match.players).all(-1)
    match.ball_available[:] = np.isfinite(match.ball[:, :2]).all(-1)
    match.metadata.update(team_ids=team_ids, issues=dict(issues),
                          observation_semantics="available includes extrapolated; detected is native bool",
                          label_policy="group supervises conditional team only; no alive/dead labels inferred",
                          transformations=["native metric coordinates retained", "no events/phases used as features",
                                           "native JSON/JSONL retained by hash", "no null=>D conversion"])
    return match.validate()
