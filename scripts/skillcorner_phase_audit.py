"""Public phase interval audit and one deterministic tracking alignment probe."""
import csv
import hashlib
import io
import json
from pathlib import Path
import ssl
import time
from urllib.request import Request, urlopen
import zipfile

import certifi
import numpy as np

from scripts.skillcorner_audit import ROOT, DEST as INVENTORY, read, write, sha

DEST = ROOT / "evaluation/reports/external/skillcorner-phase-audit-v1"
PROTOCOL = ROOT / "configs/skillcorner_phase_audit_v1.json"


def seconds(value):
    parts = str(value).split(":")
    return sum(float(v)*60**i for i, v in enumerate(reversed(parts)))


def phase_labels(rows, teams):
    """Conservative half-open intervals; caller audits endpoint conventions."""
    arrays, summary = {}, {"rows": len(rows), "invalid_rows": 0, "unknown_team_rows": 0,
        "duration_exclusive_abs_error_max": 0., "duration_inclusive_abs_error_max": 0., "periods": {}}
    for period in sorted({int(r["period"]) for r in rows}):
        group = [r for r in rows if int(r["period"]) == period]
        n = max(int(r["frame_end"]) for r in group)+1
        labels, hits = np.full(n, -1, np.int8), np.zeros(n, np.int16)
        offsets = []
        for r in group:
            a, b = int(r["frame_start"]), int(r["frame_end"])
            if a < 0 or b <= a:
                summary["invalid_rows"] += 1
                continue
            team_id = int(r["team_in_possession_id"])
            if team_id not in teams:
                summary["unknown_team_rows"] += 1
                continue
            label = teams.index(team_id)
            current = labels[a:b]
            different = (hits[a:b] > 0) & (current != label)
            current[hits[a:b] == 0] = label
            current[different] = -2
            hits[a:b] += 1
            duration = float(r["duration"])
            summary["duration_exclusive_abs_error_max"] = max(summary["duration_exclusive_abs_error_max"], abs((b-a)/10-duration))
            summary["duration_inclusive_abs_error_max"] = max(summary["duration_inclusive_abs_error_max"], abs((b-a+1)/10-duration))
            offsets.extend((seconds(r["time_start"])-a/10, seconds(r["time_end"])-b/10))
        valid = labels >= 0
        arrays[f"labels_{period}"] = labels
        summary["periods"][str(period)] = {"first_frame": min(int(r["frame_start"]) for r in group),
            "last_frame_exclusive": max(int(r["frame_end"]) for r in group),
            "covered_frames": int(valid.sum()), "conflict_frames": int((labels == -2).sum()),
            "overlap_frames": int((hits > 1).sum()), "uncovered_frames_in_span": int((hits == 0)[min(int(r["frame_start"]) for r in group):].sum()),
            "clock_offset_min_seconds": min(offsets) if offsets else None,
            "clock_offset_max_seconds": max(offsets) if offsets else None}
    return arrays, summary


def run():
    DEST.mkdir(parents=True, exist_ok=True)
    p = read(PROTOCOL)
    inventory = read(INVENTORY/"summary.json")
    for name, digest in read(INVENTORY/"result-seal.json")["outputs"].items():
        if sha((INVENTORY/name).read_bytes()) != digest:
            raise ValueError("Inventory seal mismatch")
    receipts = read(INVENTORY/"inputs-manifest.json")
    for name, receipt in receipts.items():
        if sha((INVENTORY/"inputs"/name).read_bytes()) != receipt["sha256"]:
            raise ValueError("Public inventory input changed")
    start = time.monotonic()
    summaries, sample_arrays = {}, None
    for key in inventory["revisions"]["current"]["matches"]:
        folder = INVENTORY/f"inputs/current/{key}"
        meta = read(folder/"metadata.json")
        text = (folder/"phases.csv").read_text(encoding="utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(text)))
        teams = [int(meta[t]["id"]) for t in ("home_team", "away_team")]
        arrays, summary = phase_labels(rows, teams)
        out = DEST/f"labels/{key}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **arrays)
        summaries[key] = summary
        if int(key) == p["sample_tracking_match"]:
            sample_arrays = arrays
    if sample_arrays is None:
        raise ValueError("Predeclared tracking sample absent")
    match = p["sample_tracking_match"]
    url = f"https://media.githubusercontent.com/media/SkillCorner/opendata/{inventory['current_revision']}/data/matches/{match}/{match}_tracking_extrapolated.jsonl"
    path = DEST/"inputs"/f"{match}-tracking.jsonl"
    receipt_path = path.with_suffix(".receipt.json")
    if path.exists():
        receipt = read(receipt_path)
        if receipt["url"] != url or sha(path.read_bytes()) != receipt["sha256"]:
            raise ValueError("Tracking input changed")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        digest, size = hashlib.sha256(), 0
        context = ssl.create_default_context(cafile=certifi.where())
        with urlopen(Request(url, headers={"User-Agent": "Northmind-public-audit"}), timeout=45, context=context) as source, path.with_suffix(".part").open("wb") as stream:
            while chunk := source.read(1024*1024):
                if time.monotonic()-start > p["max_seconds"] or size+len(chunk) > p["max_tracking_bytes"]:
                    raise TimeoutError("Tracking audit budget")
                stream.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        path.with_suffix(".part").replace(path)
        receipt = {"url": url, "bytes": size, "sha256": digest.hexdigest()}
        write(receipt_path, receipt)
    alignment = dict(rows=0, playing_rows=0, duplicate_period_frames=0, nonmonotone_frames=0,
        phase_covered=0, phase_unknown=0, phase_conflict=0, group_known_covered=0,
        group_agrees_covered=0, phase_covered_absent_tracking=0, clock_residual_max_seconds=0.)
    seen, last, offsets = set(), {}, {}
    # Phase frame-clock offset must be constant before using it as alignment evidence.
    for period, row in summaries[str(match)]["periods"].items():
        if abs(row["clock_offset_max_seconds"]-row["clock_offset_min_seconds"]) > .011:
            raise ValueError("Nonconstant phase frame-clock offset")
        offsets[int(period)] = row["clock_offset_min_seconds"]
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if time.monotonic()-start > p["max_seconds"]:
                raise TimeoutError("Tracking parse budget")
            row = json.loads(line)
            alignment["rows"] += 1
            period, frame = row.get("period"), int(row["frame"])
            if period not in (1, 2, 3, 4):
                continue
            alignment["playing_rows"] += 1
            identity = (period, frame)
            alignment["duplicate_period_frames"] += int(identity in seen)
            alignment["nonmonotone_frames"] += int(frame <= last.get(period, -1))
            seen.add(identity)
            last[period] = frame
            labels = sample_arrays.get(f"labels_{period}", ())
            label = labels[frame] if 0 <= frame < len(labels) else -1
            alignment["phase_covered"] += int(label >= 0)
            alignment["phase_unknown"] += int(label == -1)
            alignment["phase_conflict"] += int(label == -2)
            group = (row.get("possession") or {}).get("group")
            if label >= 0 and group in ("home team", "away team"):
                alignment["group_known_covered"] += 1
                alignment["group_agrees_covered"] += int(label == int(group == "away team"))
            if period in offsets:
                alignment["clock_residual_max_seconds"] = max(alignment["clock_residual_max_seconds"], abs(seconds(row["timestamp"])-frame/10-offsets[period]))
    alignment["phase_covered_absent_tracking"] = sum(1 for name, labels in sample_arrays.items()
        for frame in np.flatnonzero(labels >= 0) if (int(name.split("_")[1]), int(frame)) not in seen)
    report = {"protocol": p, "revision": inventory["current_revision"], "matches": summaries,
        "tracking_sample": alignment, "tracking_receipt": receipt,
        "new_independent_matches": inventory["comparison"]["current_ids_not_in_config"],
        "full_HAD_validated": False, "D_labels_created": 0,
        "label_kind": "weak positive alive and conditional team inside conflict-free phase intervals; all other frames unknown",
        "training_run": False, "target_accessed": False, "seconds": time.monotonic()-start}
    write(DEST/"summary.json", report)
    outputs = [DEST/"summary.json", *sorted((DEST/"labels").glob("*.npz"))]
    write(DEST/"result-seal.json", {"outputs": {x.relative_to(DEST).as_posix(): sha(x.read_bytes()) for x in outputs},
        "sources": {x.relative_to(ROOT).as_posix(): sha(x.read_bytes()) for x in (Path(__file__), PROTOCOL, ROOT/"scripts/skillcorner_audit.py")},
        "inventory_summary_sha256": sha((INVENTORY/"summary.json").read_bytes())})
    return 0


def export():
    # Package only derived labels, receipts and audit metadata, never raw tracking.
    names = []
    for folder in (INVENTORY, DEST):
        seal = read(folder/"result-seal.json")
        for name, digest in seal["outputs"].items():
            if sha((folder/name).read_bytes()) != digest:
                raise ValueError("Sealed audit output changed")
            names.append(folder/name)
        for name, digest in seal["sources"].items():
            if sha((ROOT/name).read_bytes()) != digest:
                raise ValueError("Audit source changed")
        names.append(folder/"result-seal.json")
    if sha((INVENTORY/"summary.json").read_bytes()) != read(DEST/"result-seal.json")["inventory_summary_sha256"]:
        raise ValueError("Parent inventory changed")
    out = ROOT/"evaluation/reports/handoff/northmind-skillcorner-audit-2026-10-29.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {f.relative_to(ROOT).as_posix(): sha(f.read_bytes()) for f in names}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in names:
            z.write(f, f.relative_to(ROOT).as_posix())
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(out) as z:
        if z.testzip() is not None or any(sha(z.read(n)) != h for n, h in manifest.items()):
            raise ValueError("Archive failed verification")
    public = ROOT/"docs/public-results/2026-10-29"
    write(public/"skillcorner-inventory-v1.json", read(INVENTORY/"summary.json"))
    write(public/"skillcorner-phases-v1.json", read(DEST/"summary.json"))
    write(public/"skillcorner-audit-package.json", {"archive": out.name, "sha256": sha(out.read_bytes()),
        "bytes": out.stat().st_size, "files": len(manifest), "verified": True,
        "raw_tracking_included": False, "scope": "public audit metadata, receipts, derived weak labels; no model or target outputs"})
    return 0
