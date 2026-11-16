"""Streaming two-pass reader for the public Sportec position XML, retaining dead ball."""

from collections import Counter
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .schema import empty_match, finite_number, sources

PERIODS = {"firstHalf": 1, "secondHalf": 2, "firstHalfExtra": 3, "secondHalfExtra": 4}
IDSSE_IDS = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")


def tag(element):
    return element.tag.rsplit("}", 1)[-1]


def records(path):
    """Yield attributes only; clear children immediately, not entire per-player trees."""
    current = None
    stack = []
    for event, node in ET.iterparse(path, events=("start", "end")):
        name = tag(node)
        if event == "start":
            stack.append(node)
            if name == "FrameSet":
                current = dict(node.attrib)
            elif name == "Frame" and current is not None:
                yield current, dict(node.attrib)
        else:
            if name == "FrameSet":
                current = None
            node.clear()
            stack.pop()
            if stack:
                stack[-1].remove(node)


def load_idsse(metadata_path, positions_path):
    root = ET.parse(metadata_path).getroot()
    general = next(x for x in root.iter() if tag(x) == "General")
    environment = next(x for x in root.iter() if tag(x) == "Environment")
    match_id = general.attrib["MatchId"]
    pitch = [float(environment.attrib[x]) for x in ("PitchX", "PitchY")]
    team_ids = [general.attrib["HomeTeamId"], general.attrib["GuestTeamId"]]
    rosters = [[], []]
    for team in (x for x in root.iter() if tag(x) == "Team"):
        if team.get("TeamId") in team_ids:
            side = team_ids.index(team.get("TeamId"))
            rosters[side] = sorted({x.attrib["PersonId"] for x in team.iter() if tag(x) == "Player"})
    if not all(rosters):
        raise ValueError("Both team rosters are required")
    index = {pid: (side, i) for side, roster in enumerate(rosters) for i, pid in enumerate(roster)}
    clocks, issues = {}, Counter()
    for group, row in records(positions_path):
        if group.get("MatchId", match_id) != match_id:
            raise ValueError("Mismatched position and metadata match IDs")
        period = PERIODS.get(group.get("GameSection"))
        if period is None or not row.get("N", "").isdigit():
            issues["invalid_frame_or_period"] += 1
            continue
        if group.get("TeamId") != "BALL" and group.get("PersonId") not in index:
            continue
        key = period, int(row["N"])
        stamp = row.get("T", "")
        try:
            parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("Timezone required")
            seconds = parsed.timestamp()
        except ValueError:
            issues["invalid_timestamp"] += 1
            continue
        if key in clocks and abs(clocks[key][1] - seconds) > 0.001:
            issues["inconsistent_object_timestamps"] += 1
        if key not in clocks or group.get("TeamId") == "BALL":
            clocks[key] = stamp, seconds
    keys = sorted(clocks)
    match = empty_match("idsse", match_id, [k[1] for k in keys], [k[0] for k in keys],
                        [clocks[k][0] for k in keys], [clocks[k][1] for k in keys],
                        rosters, pitch, 25.0, sources([metadata_path, positions_path]))
    lookup = {key: i for i, key in enumerate(keys)}
    seen_ball = np.zeros(len(keys), dtype=bool)
    seen_player = np.zeros(match.player_available.shape, dtype=bool)
    for group, row in records(positions_path):
        period = PERIODS.get(group.get("GameSection"))
        if period is None or not row.get("N", "").isdigit():
            continue
        i = lookup.get((period, int(row["N"])))
        obj = "ball" if group.get("TeamId") == "BALL" else group.get("PersonId")
        if i is None or (obj != "ball" and obj not in index):
            continue
        duplicate = seen_ball[i] if obj == "ball" else seen_player[(i, *index[obj])]
        if duplicate:
            issues["duplicate_object_frames"] += 1
            raise ValueError("Duplicate object/frame: ambiguous source")
        if obj == "ball":
            seen_ball[i] = True
        else:
            seen_player[(i, *index[obj])] = True
        xy = [finite_number(row.get(axis)) for axis in ("X", "Y")]
        if obj == "ball":
            match.ball[i] = [*xy, finite_number(row.get("Z"))]
            status, poss = row.get("BallStatus", ""), row.get("BallPossession", "")
            match.native_status[i], match.native_possession[i] = status, poss
            if status in ("0", "1"):
                match.alive[i] = int(status)
                match.confidence[i, 0] = 1
                match.provenance[i, 0] = "native:BallStatus"
            else:
                issues["invalid_ball_status"] += 1
            if status == "1" and poss in ("1", "2"):
                match.team[i] = int(poss) - 1
                match.confidence[i, 1] = 1
                match.provenance[i, 1] = "native:BallPossession|active"
            elif status == "1":
                issues["invalid_active_possession"] += 1
        else:
            side, player = index[obj]
            if group.get("TeamId") != team_ids[side]:
                raise ValueError("Player assigned to an inconsistent team")
            match.players[i, side, player] = xy
    match.player_available[:] = np.isfinite(match.players).all(-1)
    match.ball_available[:] = np.isfinite(match.ball[:, :2]).all(-1)
    match.metadata.update(team_ids=team_ids, issues=dict(issues),
                          observation_semantics="finite observed-file record; visual detection unknown",
                          label_policy="status 0=>D; status 1 & possession 1/2=>H/A; else partial/unknown",
                          transformations=["native metric coordinates retained", "no event-derived labels",
                                           "all periods and dead-ball frames retained", "native XML retained by hash"])
    return match.validate()


def discover_idsse(directory):
    result = []
    for path in sorted(Path(directory).glob("*matchinformation*.xml")):
        match_id = path.stem.rsplit("_", 1)[-1]
        positions = list(Path(directory).glob("*positions*" + match_id + ".xml"))
        if len(positions) != 1:
            raise ValueError("Need exactly one position file per metadata file")
        result.append((path, positions[0]))
    if not result:
        raise ValueError("No IDSSE match files found")
    return result
