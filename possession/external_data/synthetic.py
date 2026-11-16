"""Generated fixtures only: no real names, match assets, or competition ground truth."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def write_idsse_fixture(directory, match_id="DFL-MAT-J03WMX", frames_per_half=100, invalid=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = ET.Element("PutDataRequest")
    info = ET.SubElement(metadata, "MatchInformation")
    ET.SubElement(info, "General", MatchId=match_id, HomeTeamId="HOME", GuestTeamId="AWAY")
    ET.SubElement(info, "Environment", PitchX="105", PitchY="68")
    teams = ET.SubElement(info, "Teams")
    for side, pids in (("HOME", ("h1", "h2")), ("AWAY", ("a1",))):
        players = ET.SubElement(ET.SubElement(teams, "Team", TeamId=side), "Players")
        for pid in pids:
            ET.SubElement(players, "Player", PersonId=pid)
    root = ET.Element("PutDataRequest")
    positions = ET.SubElement(root, "Positions")
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for period, section in enumerate(("firstHalf", "secondHalf")):
        for team, pid in (("BALL", "ball"), ("HOME", "h1"), ("HOME", "h2"), ("AWAY", "a1")):
            group = ET.SubElement(positions, "FrameSet", GameSection=section, MatchId=match_id,
                                  TeamId=team, PersonId=pid)
            for i in range(frames_per_half):
                timestamp = origin + timedelta(seconds=period * 3600 + i / 25)
                row = {"N": str(10000 + period * 100000 + i), "T": timestamp.isoformat(),
                       "X": str((i % 10) / 10 + (5 if team == "HOME" else -5)), "Y": "2"}
                if team == "BALL":
                    phase = min(3, i * 4 // frames_per_half)
                    row.update(Z="0.1", BallStatus="0" if phase == 2 else "1",
                               BallPossession="1" if phase in (0, 2) else "2")
                    if i == 3:
                        row["X"] = "NaN"  # Tracking absent while supervision remains valid.
                    if invalid and i == 4:
                        row["BallStatus"], row["BallPossession"] = "9", "3"
                ET.SubElement(group, "Frame", row)
    meta_path = directory / ("DFL_02_01_matchinformation_" + match_id + ".xml")
    pos_path = directory / ("DFL_04_03_positions_raw_observed_" + match_id + ".xml")
    ET.ElementTree(metadata).write(meta_path, encoding="utf-8", xml_declaration=True)
    ET.ElementTree(root).write(pos_path, encoding="utf-8", xml_declaration=True)
    return meta_path, pos_path


def write_skillcorner_fixture(directory, n=20):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    meta = {"id": 42, "home_team": {"id": 10}, "away_team": {"id": 20},
            "pitch_length": 105, "pitch_width": 68,
            "players": [{"id": 1, "team_id": 10}, {"id": 2, "team_id": 20}]}
    meta_path, track_path = directory / "42_match.json", directory / "42_tracking_extrapolated.jsonl"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with track_path.open("w", encoding="utf-8") as stream:
        for i in range(n):
            row = {"frame": i + 10, "period": 1, "timestamp": "00:00:" + format(i / 10, "06.3f"),
                   "player_data": [{"player_id": 1, "x": 3, "y": 2, "is_detected": i % 2 == 0},
                                   {"player_id": 2, "x": -3, "y": -2, "is_detected": True}],
                   "ball_data": {"x": i / 10, "y": 0, "z": 0.1, "is_detected": i % 2 == 0},
                   "possession": {"group": "home team", "player_id": 1} if i % 3 else None}
            stream.write(json.dumps(row) + "\n")
    return meta_path, track_path
