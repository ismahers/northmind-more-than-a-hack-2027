"""Reviewed train-geometry-only feasibility study; invoke via scripts.run_local."""
import hashlib
import json
from pathlib import Path
import platform
import time
import unittest
import numpy as np

from possession.football_control import extract, summarize
from possession.external_data.schema import file_hash

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "evaluation/reports/external/football-control-v1"
PUBLIC = ROOT / "docs/public-results/2026-10-29"
CONFIG = ROOT / "configs/football_control_v1.json"
KEYS = ["idsse:DFL-MAT-" + x for x in ("J03WOH", "J03WOY", "J03WPY")]
SOURCES = ["configs/football_control_v1.json", "possession/football_control.py",
           "scripts/football_control.py", "scripts/run_local.py", "tests/test_football_control.py",
           "configs/external_sources.json", "possession/external_data/schema.py"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sources():
    # Stable across checkout newline policies; inputs use exact binary hashes.
    return {n: hashlib.sha256((ROOT/n).read_text(encoding="utf-8-sig").replace("\r\n", "\n").encode()).hexdigest()
            for n in SOURCES}


def tests():
    suite = unittest.defaultTestLoader.discover(str(ROOT/"tests"), pattern="test_football_control.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    write(DEST/"synthetic-status.json", {"tests": result.testsRun,
          "success": result.wasSuccessful(), "failures": len(result.failures), "errors": len(result.errors),
          "sources": sources()})
    return 0 if result.wasSuccessful() and result.testsRun > 0 else 1


def run():
    started = time.perf_counter()
    spec = read(CONFIG)
    if spec["matches"] != KEYS or any(spec[k] for k in
        ("labels_access", "validation_access", "test_access", "target_access", "fit_model", "parameter_search")):
        raise ValueError("Train geometry only")
    test_status = read(DEST/"synthetic-status.json")
    if not test_status["success"] or test_status["sources"] != sources():
        raise ValueError("Synthetic tests required first")
    # Read manifest metadata only; NEVER hash/open non-allowed match files.
    manifest = read(ROOT/"evaluation/reports/external/folds/idsse-only.json")
    input_records = {}
    for key in KEYS:
        item = manifest["artifacts"][key]
        path = Path(item["path"]).resolve()
        if not path.is_relative_to((ROOT/"data/external/processed").resolve()) or path.suffix != ".npz":
            raise ValueError("Unexpected public input path")
        if file_hash(path) != item["sha256"]:
            raise ValueError("Public training input changed")
        input_records[key] = {"path": path.relative_to(ROOT).as_posix(), "sha256": item["sha256"]}
    stamp = {"protocol": spec, "sources": sources(), "inputs": input_records,
             "environment": {"python": platform.python_version(), "numpy": np.__version__}}
    registration = DEST/"registration.json"
    if registration.exists() and read(registration) != stamp:
        raise ValueError("Immutable study changed")
    write(registration, stamp)  # BEFORE loading geometry or calculating results.
    rows = {}
    for key, item in input_records.items():
        if time.perf_counter()-started > spec["max_seconds"]:
            raise TimeoutError("Feasibility budget exhausted")
        with np.load(ROOT/item["path"], allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"]))
            if meta["provider"] + ":" + meta["match_id"] != key:
                raise ValueError("Training identity mismatch")
            fps = float(meta["native_fps"])
            ball, players = z["ball"][:, :2], z["players"]
            t, period, frame = z["time"], z["period"], z["frame"]
            ba, pa = z["ball_available"], z["player_available"]
            # Deliberately never access alive/team/confidence/provenance/native labels.
        if not np.isfinite(t).all():
            raise ValueError("Invalid geometry time")
        dt = np.diff(t)
        if np.any((period[1:] == period[:-1]) & ((dt <= 0) | (np.diff(frame) <= 0))):
            raise ValueError("Nonmonotone time")
        cuts = (period[1:] != period[:-1]) | (dt > 1.5/fps)
        run_id = np.r_[0, np.cumsum(cuts)]
        rows[key] = {}
        for lag in spec["lag_seconds"]:
            if time.perf_counter()-started > spec["max_seconds"]:
                raise TimeoutError("Feasibility budget exhausted")
            cue = extract(ball, players, ba, pa, t, run_id, fps, spec, lag)
            rows[key][str(lag)] = summarize(cue, spec)
        if file_hash(ROOT/item["path"]) != item["sha256"]:
            raise ValueError("Input changed during study")
    if sources() != stamp["sources"]:
        raise ValueError("Source changed during study")
    if time.perf_counter()-started > spec["max_seconds"]:
        raise TimeoutError("Feasibility budget exhausted")
    report = {"version": spec["version"], "registration_sha256": file_hash(registration),
              "protocol": spec, "sources": stamp["sources"], "inputs": input_records,
              "environment": stamp["environment"], "synthetic": read(DEST/"synthetic-status.json"),
              "geometry_only": True, "labels_read": False, "target_accessed": False,
              "validation_accessed": False, "test_accessed": False, "model_trained": False,
              "seconds": time.perf_counter()-started, "matches": rows,
              "all_coverage_gates": all(r["coverage_gate"] for m in rows.values() for r in m.values()),
              "interpretation": "Coverage gates only; near/far are NOT control labels. No predictive improvement measured."}
    write(DEST/"public-summary.json", report)
    write(DEST/"result-seal.json", {"summary_sha256": file_hash(DEST/"public-summary.json"),
          "registration_sha256": file_hash(registration)})
    return verify()


def verify():
    report = read(DEST/"public-summary.json")
    stamp = read(DEST/"registration.json")
    seal = read(DEST/"result-seal.json")
    if (seal["summary_sha256"] != file_hash(DEST/"public-summary.json")
            or seal["registration_sha256"] != file_hash(DEST/"registration.json")
            or report["registration_sha256"] != file_hash(DEST/"registration.json")
            or sources() != stamp["sources"] or report["sources"] != stamp["sources"]
            or report["inputs"] != stamp["inputs"] or report["protocol"] != stamp["protocol"]
            or set(report["matches"]) != set(KEYS) or not report["synthetic"]["success"]
            or any(report[k] for k in ("labels_read", "target_accessed", "validation_accessed", "test_accessed", "model_trained"))):
        raise ValueError("Feasibility provenance failed")
    for item in stamp["inputs"].values():
        if file_hash(ROOT/item["path"]) != item["sha256"]:
            raise ValueError("Input hash mismatch")
    return 0


def export():
    verify()
    # Reviewed writer exports ONLY the fixed aggregate object, never NPZs or logs.
    report = read(DEST/"public-summary.json")
    write(PUBLIC/"football-control-v1.json", report)
    write(PUBLIC/"football-control-v1-registration.json", read(DEST/"registration.json"))
    write(PUBLIC/"football-control-v1-seal.json", read(DEST/"result-seal.json"))
    if file_hash(PUBLIC/"football-control-v1.json") != file_hash(DEST/"public-summary.json"):
        raise ValueError("Export changed aggregate bytes")
    return 0
