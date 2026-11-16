"""Versioned native-rate tracking contract. Unknown labels are -1, never D."""

from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sources(paths):
    result = []
    for path in paths:
        path = Path(path)
        record = {"name": path.name, "sha256": file_hash(path)}
        receipt_path = path.with_name(path.name + ".receipt.json")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("sha256") != record["sha256"]:
                raise ValueError("Native source does not match its download receipt")
            record["download_receipt"] = receipt
        result.append(record)
    return result


@dataclass
class Match:
    metadata: dict
    frame: np.ndarray
    period: np.ndarray
    timestamp: np.ndarray             # Exact provider timestamp string.
    time: np.ndarray                  # Seconds; monotone within each period.
    player_ids: np.ndarray            # [2, roster capacity], empty = padding.
    players: np.ndarray               # [T, 2, roster capacity, 2], metres, centred.
    player_available: np.ndarray      # Finite coordinate record, including extrapolation.
    player_detected: np.ndarray
    player_detection_known: np.ndarray
    ball: np.ndarray                  # [T, 3], metres; z may be NaN independently.
    ball_available: np.ndarray        # XY availability, not ball state.
    ball_detected: np.ndarray
    ball_detection_known: np.ndarray
    alive: np.ndarray                 # -1 unknown, 0 dead, 1 alive.
    team: np.ndarray                  # -1 unknown, 0 home, 1 away; conditional on alive.
    confidence: np.ndarray            # [T, 2]: state, team; 0 means no supervision.
    provenance: np.ndarray            # [T, 2]: state, team.
    native_status: np.ndarray
    native_possession: np.ndarray

    @property
    def key(self):
        return self.metadata["provider"] + ":" + self.metadata["match_id"]

    def validate(self):
        n = len(self.frame)
        if not n or self.players.ndim != 4 or self.players.shape[:2] != (n, 2):
            raise ValueError("Empty or malformed tracking")
        p = self.players.shape[2]
        shapes = {"player_ids": (2, p), "players": (n, 2, p, 2),
                  "player_available": (n, 2, p), "player_detected": (n, 2, p),
                  "player_detection_known": (n, 2, p), "ball": (n, 3),
                  "confidence": (n, 2), "provenance": (n, 2)}
        for field in fields(self):
            if field.name == "metadata":
                continue
            value = getattr(self, field.name)
            if value.shape != shapes.get(field.name, (n,)) or value.dtype.hasobject:
                raise ValueError("Invalid array contract: " + field.name)
        if not np.isin(self.alive, [-1, 0, 1]).all() or not np.isin(self.team, [-1, 0, 1]).all():
            raise ValueError("Invalid supervision")
        if not np.isfinite(self.time).all() or not np.isfinite(self.confidence).all():
            raise ValueError("Nonfinite clock/confidence")
        if np.any((self.confidence < 0) | (self.confidence > 1)):
            raise ValueError("Confidence outside [0,1]")
        if np.any(self.confidence[:, 0][self.alive < 0]) or np.any(self.confidence[:, 1][self.team < 0]):
            raise ValueError("Unknown targets cannot carry confidence")
        if np.any((self.alive == 0) & (self.team >= 0)):
            raise ValueError("Dead frames cannot supervise conditional team")
        if not np.array_equal(self.player_available, np.isfinite(self.players).all(-1)):
            raise ValueError("Player availability mismatch")
        if not np.array_equal(self.ball_available, np.isfinite(self.ball[:, :2]).all(-1)):
            raise ValueError("Ball availability mismatch")
        if self.metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported schema version")
        if any(not np.isfinite(x) or x <= 0 for x in self.metadata["pitch_m"]):
            raise ValueError("Invalid pitch dimensions")
        return self

    def save(self, path):
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "metadata"}
        # Fixed-width construction buffers are deliberately generous. Persist only the
        # necessary Unicode width, preserving every string and reducing loaded RAM.
        for name, value in arrays.items():
            if value.dtype.kind == "U":
                width = max(1, int(np.char.str_len(value).max(initial=0)))
                arrays[name] = value.astype("U" + str(width))
        # Unicode scalars/arrays: loading never needs pickle.
        np.savez_compressed(path, metadata=np.array(json.dumps(self.metadata)), **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            result = cls(metadata=json.loads(str(data["metadata"])),
                         **{f.name: data[f.name] for f in fields(cls) if f.name != "metadata"})
        return result.validate()

    @staticmethod
    def read_metadata(path):
        with np.load(path, allow_pickle=False) as data:
            return json.loads(str(data["metadata"]))


def empty_match(provider, match_id, frames, periods, timestamps, times, rosters, pitch, fps, raw_sources):
    if any(len(roster) != len(set(roster)) for roster in rosters) or set(rosters[0]) & set(rosters[1]):
        raise ValueError("Player identities must be unique within and across teams")
    n = len(frames)
    p = max(1, *(len(x) for x in rosters))
    ids = np.full((2, p), "", dtype="U64")
    for side, roster in enumerate(rosters):
        ids[side, :len(roster)] = roster
    return Match(
        {"schema_version": SCHEMA_VERSION, "provider": provider, "match_id": str(match_id),
         "pitch_m": list(pitch), "native_fps": fps, "sources": raw_sources,
         "coordinate_system": "metres; centre origin; provider fixed axes; no half rotation",
         "issues": {}},
        np.asarray(frames, dtype=np.int64), np.asarray(periods, dtype=np.int16),
        np.asarray(timestamps, dtype="U64"), np.asarray(times, dtype=np.float64), ids,
        np.full((n, 2, p, 2), np.nan, dtype=np.float32),
        np.zeros((n, 2, p), dtype=bool), np.zeros((n, 2, p), dtype=bool),
        np.zeros((n, 2, p), dtype=bool), np.full((n, 3), np.nan, dtype=np.float32),
        np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), np.zeros(n, dtype=bool),
        np.full(n, -1, dtype=np.int8), np.full(n, -1, dtype=np.int8),
        np.zeros((n, 2), dtype=np.float32), np.full((n, 2), "unknown", dtype="U64"),
        np.full(n, "", dtype="U32"), np.full(n, "", dtype="U256"))


def finite_number(value):
    try:
        number = float(value)
        return number if np.isfinite(number) else np.nan
    except (TypeError, ValueError):
        return np.nan
