"""Reviewed geometry-only I/O. Invoke exclusively through scripts.run_local.

No Match loader, manifest, supervision array or per-window persistence is used.
Exceptions remain in the launcher's private log; public exports are schema checked.
"""
import hashlib
import json
import math
import platform
import sys
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "evaluation/reports/external/control-transfer-geometry-v1"
PUBLIC = ROOT / "docs/public-results/2026-10-30"
CONFIG = "configs/control_transfer_geometry_v1.json"
PARENT = "docs/public-results/2026-10-29/football-control-v1-registration.json"
SOURCE_REVISION = "a715a38dfbaf5f58e431727c2b78d174101a703c"
MATCHES = ("J03WOH", "J03WOY", "J03WPY")
HASHES = (
    "34504202e4d148c77119ba1540e8608302833505614b5cb2a784387cac52402b",
    "556d6316ef3e90a59b30093aaab1c6ad416f54f0806b53554939d63a1ee857cf",
    "09aeb251d470461dbcafde5fa85767fd5a77d4e7a587c7b14ddf2f535c17b271",
)
ALLOWED_INPUTS = {"idsse:DFL-MAT-" + match: {
    "path": "data/external/processed/idsse_DFL-MAT-" + match + ".npz", "sha256": digest}
    for match, digest in zip(MATCHES, HASHES)}
ARRAY_KEYS = ("time", "period", "frame", "players", "ball", "player_available",
              "ball_available", "player_detected", "player_detection_known",
              "ball_detected", "ball_detection_known")
SOURCES = ("scripts/control_transfer_geometry.py", "possession/control_transfer_geometry.py",
           "possession/control_transfer.py", "tests/test_control_transfer.py",
           "tests/test_control_transfer_geometry.py", "tests/test_control_transfer_geometry_runner.py",
           CONFIG, "configs/external_sources.json", "scripts/run_local.py",
           "docs/experiments/CONTROL_TRANSFER_GEOMETRY_2026-10-30.md", PARENT)
FLAGS = {"geometry_only": True, "labels_read": False, "target_accessed": False,
         "validation_accessed": False, "test_accessed": False, "model_trained": False,
         "new_inference": False, "semantic_hypothesis_tested": False, "promoted": False}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def sources():
    return {name: hashlib.sha256((ROOT/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for name in SOURCES}


def file_hash(path, check=lambda: None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            check()
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def budget(started, seconds=600):
    def check():
        if time.perf_counter() - started > seconds:
            raise TimeoutError("Geometry budget exhausted")
    return check


def allowed_path(path, key):
    if key not in ALLOWED_INPUTS:
        raise ValueError("Match not allowed")
    actual = Path(path).resolve()
    expected = (ROOT/ALLOWED_INPUTS[key]["path"]).resolve()
    if not actual.is_relative_to(ROOT.resolve()) or actual != expected:
        raise ValueError("Input path not allowed")
    return actual


def load_geometry(path, key, check=lambda: None):
    """Read exactly metadata and the geometry/detection allowlist, never labels."""
    path = allowed_path(path, key)
    check()
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if (metadata.get("provider") != "idsse"
                or metadata.get("match_id") != key.split(":", 1)[1]
                or metadata.get("native_fps") != 25):
            raise ValueError("Geometry metadata mismatch")
        # Deliberately discard unrelated metadata; no provider strings/IDs enter results.
        arrays = {"metadata": {name: metadata[name] for name in ("provider", "match_id", "native_fps")}}
        for name in ARRAY_KEYS:
            check()
            arrays[name] = archive[name]
    check()
    n = len(arrays["time"])
    players = arrays["players"]
    if not n or players.ndim != 4 or players.shape[:2] != (n, 2) or players.shape[-1] != 2:
        raise ValueError("Invalid geometry shape")
    p = players.shape[2]
    shapes = {"players": (n, 2, p, 2), "ball": (n, 3),
              **{name: (n, 2, p) for name in ARRAY_KEYS if name.startswith("player_")}}
    for name in ARRAY_KEYS:
        value = arrays[name]
        if value.dtype.hasobject or value.shape != shapes.get(name, (n,)):
            raise ValueError("Invalid geometry array contract")
        if name.endswith(("available", "detected", "known")) and value.dtype.kind != "b":
            raise ValueError("Detection flag must be boolean")
    if not np.isfinite(arrays["time"]).all():
        raise ValueError("Invalid geometry clock")
    return arrays


def protocol():
    spec = read(ROOT/CONFIG)
    parent = read(ROOT/PARENT)
    if (parent.get("inputs") != ALLOWED_INPUTS
            or parent["protocol"]["matches"] != list(ALLOWED_INPUTS)
            or parent["protocol"]["role"] != "intersection of historical train folds 0 and 6"):
        raise ValueError("Historical public registration mismatch")
    source = read(ROOT/"configs/external_sources.json")["idsse"]
    if (source["revision"] != SOURCE_REVISION or source["license"] != "CC-BY-4.0"
            or source["repo"] != "pysport/idsse-data"
            or any(match not in source["matches"] for match in MATCHES)):
        raise ValueError("Public source mismatch")
    if (spec["matches"] != list(ALLOWED_INPUTS) or spec["budget"] != {"seconds": 600, "threads": 4}
            or spec["flags"] != FLAGS or spec["max_centers_per_match"] != 100
            or spec["source"] != {"provider": "idsse", "repo": "pysport/idsse-data",
                                   "revision": SOURCE_REVISION, "license": "CC-BY-4.0"}):
        raise ValueError("Geometry protocol mismatch")
    return spec


def check_inputs(inputs, check=lambda: None):
    if inputs != ALLOWED_INPUTS:
        raise ValueError("Input registration mismatch")
    for key, record in inputs.items():
        if file_hash(allowed_path(ROOT/record["path"], key), check) != record["sha256"]:
            raise ValueError("Input checksum mismatch")


def tests():
    started = time.perf_counter()
    before = sources()
    suite = unittest.TestSuite()
    for pattern in ("test_control_transfer.py", "test_control_transfer_geometry.py",
                    "test_control_transfer_geometry_runner.py"):
        suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/"tests"), pattern=pattern))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    success = (result.wasSuccessful() and result.testsRun >= 6 and not result.skipped
               and before == sources() and time.perf_counter()-started < 30)
    status = {"success": success, "sources": before,
          "synthetic_only": True, "dataset_accessed": False,
          "failure_test_ids": [test.id() for test, _ in result.failures],
          "error_test_ids": [test.id() for test, _ in result.errors],
          "counts": {"run": result.testsRun, "failures": len(result.failures),
                     "errors": len(result.errors), "skipped": len(result.skipped)},
          "environment": environment(), "seconds": time.perf_counter()-started}
    write(DEST/"synthetic-status.json", status)
    write(PUBLIC/"control-transfer-geometry-v1-synthetic.json", status)
    return 0 if success else 1


def environment():
    return {"python": sys.version.split()[0], "numpy": np.__version__, "system": platform.system()}


def exact(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("Unexpected public schema")


def count(value):
    if type(value) is not int or value < 0:
        raise ValueError("Invalid count")


def number(value, nullable=False):
    if nullable and value is None:
        return
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Invalid finite scalar")


def distribution(value):
    exact(value, ("count", "q25", "median", "q75"))
    count(value["count"])
    for key in ("q25", "median", "q75"):
        number(value[key], nullable=value["count"] == 0)
    if value["count"] == 0 and any(value[key] is not None for key in ("q25", "median", "q75")):
        raise ValueError("Empty distribution mismatch")
    if value["count"] and not value["q25"] <= value["median"] <= value["q75"]:
        raise ValueError("Invalid quartiles")


METRICS = {"observed."+a+"."+m+"."+s for a in "ACR" for m in ("d", "u")
           for s in ("pre", "post", "late", "post_minus_pre", "late_minus_pre")}
METRICS |= {"residual."+a+"."+m+"."+s for a in "ACR" for m in ("d", "u")
            for s in ("post_minus_pre", "late_minus_pre")}
CONTRASTS = {key for key in METRICS if key.endswith("minus_pre")}


def aggregate(value, team=False):
    exact(value, ("coverage", "discard_reasons", "detection_on_supported_sample_uses", "initial_proximity", "changes", "matching"))
    coverage = value["coverage"]
    exact(coverage, ("selected_centers", "candidate_sets", "candidate_valid", "supported", "discarded"))
    for n in coverage.values():
        count(n)
    if not (coverage["selected_centers"] <= 100
            and coverage["candidate_sets"] == coverage["selected_centers"] * (1 if team else 2)
            and coverage["supported"] <= coverage["candidate_valid"] <= coverage["candidate_sets"]
            and coverage["discarded"] == coverage["candidate_sets"]-coverage["supported"]):
        raise ValueError("Coverage counts mismatch")
    detection = value["detection_on_supported_sample_uses"]
    exact(value["discard_reasons"], ("candidate_unavailable", "window_support_failed"))
    for n in value["discard_reasons"].values():
        count(n)
    if (sum(value["discard_reasons"].values()) != coverage["discarded"]
            or value["discard_reasons"]["candidate_unavailable"] != coverage["candidate_sets"]-coverage["candidate_valid"]
            or value["discard_reasons"]["window_support_failed"] != coverage["candidate_valid"]-coverage["supported"]):
        raise ValueError("Discard reasons mismatch")
    exact(detection, ("ball", "players", "joint"))
    for item in detection.values():
        exact(item, ("sample_uses", "known", "known_detected", "known_fraction", "detected_fraction_of_known"))
        for k in ("sample_uses", "known", "known_detected"):
            count(item[k])
        if not item["known_detected"] <= item["known"] <= item["sample_uses"]:
            raise ValueError("Detection counts mismatch")
        for k, numerator, denominator in (("known_fraction", "known", "sample_uses"),
                                           ("detected_fraction_of_known", "known_detected", "known")):
            expected = item[numerator]/item[denominator] if item[denominator] else None
            if item[k] != expected:
                raise ValueError("Detection fraction mismatch")
    exact(value["initial_proximity"], "ACR")
    for d in value["initial_proximity"].values():
        distribution(d)
        if d["count"] != coverage["supported"]:
            raise ValueError("Initial support denominator mismatch")
    exact(value["changes"], METRICS if coverage["supported"] else ())
    for d in value["changes"].values():
        distribution(d)
        if d["count"] != coverage["supported"]:
            raise ValueError("Distribution denominator mismatch")
    matching = value["matching"]
    exact(matching, ("pairs", "paired_windows", "unmatched_windows", "paired_fraction", "contrasts"))
    for k in ("pairs", "paired_windows", "unmatched_windows"):
        count(matching[k])
    if (matching["paired_windows"] != matching["pairs"]*2
            or matching["paired_windows"]+matching["unmatched_windows"] != coverage["supported"]
            or matching["paired_fraction"] != (matching["paired_windows"]/coverage["supported"] if coverage["supported"] else None)):
        raise ValueError("Matching counts mismatch")
    exact(matching["contrasts"], CONTRASTS if coverage["supported"] else ())
    for item in matching["contrasts"].values():
        exact(item, ("signed", "absolute"))
        for d in item.values():
            distribution(d)
            if d["count"] != matching["pairs"]:
                raise ValueError("Pair denominator mismatch")


def validate_result(result):
    exact(result, ("global", "teams", "semantic_claim", "model_promoted"))
    if result["semantic_claim"] is not False or result["model_promoted"] is not False:
        raise ValueError("Forbidden interpretation")
    exact(result["teams"], ("team0", "team1"))
    aggregate(result["global"])
    for team in result["teams"].values():
        aggregate(team, team=True)
    for key in ("candidate_sets", "candidate_valid", "supported", "discarded"):
        if sum(t["coverage"][key] for t in result["teams"].values()) != result["global"]["coverage"][key]:
            raise ValueError("Team coverage mismatch")


def run():
    from possession.control_transfer_geometry import analyse_geometry
    started = time.perf_counter()
    check = budget(started)
    spec = protocol()
    before = sources()
    tested = read(DEST/"synthetic-status.json")
    if tested.get("success") is not True or tested.get("sources") != before:
        raise ValueError("Current successful synthetic tests required")
    if any((DEST/name).exists() for name in ("summary.json", "result-seal.json")):
        raise ValueError("Completed run is immutable")
    check_inputs(ALLOWED_INPUTS, check)
    stamp = {"id": "control-transfer-geometry-v1", "protocol": spec, "sources": before,
             "inputs": ALLOWED_INPUTS, "environment": environment(), "flags": FLAGS,
             "role": "three reused public train matches; exploratory geometry only"}
    marker = DEST/"registration.json"
    if marker.exists():
        if read(marker) != stamp:
            raise ValueError("Registration is immutable")
    else:
        write(marker, stamp, exclusive=True)  # BEFORE loading arrays or computation.
    results = {}
    for key, record in ALLOWED_INPUTS.items():
        check()
        results[key] = analyse_geometry(load_geometry(ROOT/record["path"], key, check), check=check)
        validate_result(results[key])
        if results[key]["global"]["coverage"]["supported"] == 0:
            raise ValueError("Zero geometry support; stop without expansion")
    check_inputs(ALLOWED_INPUTS, check)
    if before != sources():
        raise ValueError("Sources changed")
    check()
    report = {"id": stamp["id"], "flags": FLAGS, "role": stamp["role"],
              "sources": before, "inputs": ALLOWED_INPUTS, "matches": results,
              "environment": environment(), "seconds": time.perf_counter()-started,
              "registration_sha256": file_hash(marker, check)}
    write(DEST/"summary.json", report, exclusive=True)
    write(DEST/"result-seal.json", {"registration_sha256": file_hash(marker, check),
          "summary_sha256": file_hash(DEST/"summary.json", check)}, exclusive=True)
    return verify()


def verify():
    check = budget(time.perf_counter())
    spec = protocol()
    stamp, seal, report = (read(DEST/name) for name in ("registration.json", "result-seal.json", "summary.json"))
    exact(stamp, ("id", "protocol", "sources", "inputs", "environment", "flags", "role"))
    exact(seal, ("registration_sha256", "summary_sha256"))
    exact(report, ("id", "flags", "role", "sources", "inputs", "matches", "environment", "seconds", "registration_sha256"))
    if (stamp["protocol"] != spec or stamp["sources"] != sources()
            or stamp["flags"] != FLAGS or report["flags"] != FLAGS
            or stamp["inputs"] != ALLOWED_INPUTS or report["inputs"] != ALLOWED_INPUTS
            or stamp["id"] != "control-transfer-geometry-v1" or report["id"] != stamp["id"]
            or stamp["role"] != "three reused public train matches; exploratory geometry only"
            or report["role"] != stamp["role"] or report["sources"] != stamp["sources"]
            or report["environment"] != stamp["environment"]
            or seal["registration_sha256"] != file_hash(DEST/"registration.json", check)
            or report["registration_sha256"] != seal["registration_sha256"]
            or seal["summary_sha256"] != file_hash(DEST/"summary.json", check)):
        raise ValueError("Geometry seal mismatch")
    number(report["seconds"])
    if not 0 <= report["seconds"] <= 600:
        raise ValueError("Geometry budget mismatch")
    exact(report["environment"], ("python", "numpy", "system"))
    if any(not isinstance(x, str) or len(x)>100 for x in report["environment"].values()):
        raise ValueError("Environment schema mismatch")
    exact(report["matches"], ALLOWED_INPUTS)
    for result in report["matches"].values():
        validate_result(result)
        if result["global"]["coverage"]["supported"] == 0:
            raise ValueError("Zero support")
    check_inputs(stamp["inputs"], check)
    if sources() != stamp["sources"]:
        raise ValueError("Sources changed during verification")
    verification = {"id": stamp["id"], "verified": True, "flags": FLAGS,
                    "registration_sha256": seal["registration_sha256"],
                    "summary_sha256": seal["summary_sha256"],
                    "sources_and_inputs_unchanged": True}
    # Only explicitly validated aggregate objects cross into versioned files.
    write(PUBLIC/"control-transfer-geometry-v1-registration.json", stamp)
    write(PUBLIC/"control-transfer-geometry-v1-summary.json", report)
    write(PUBLIC/"control-transfer-geometry-v1-verification.json", verification)
    return 0
