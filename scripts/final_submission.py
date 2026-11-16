"""Inference-only final delivery. Run only through scripts.run_local.

No ground truth, evaluator, score comparison or parameter search is used.
All prediction hashes and input provenance remain in the ignored local folder.
"""
import csv
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evaluation/reports/final-delivery"
PUBLIC_STATUS = ROOT / "docs/public-results/2026-11-16/final-delivery-status.json"
EXPECTED_ROWS = 199319


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def serialize_submission(labels, expected_rows=EXPECTED_ROWS):
    """Strict UTF-8/LF serialization; never normalize invalid labels silently."""
    if len(labels) != expected_rows:
        raise ValueError("Invalid prediction count")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("frame", "label"))
    for frame, label in enumerate(labels):
        if not isinstance(label, str) or label not in {"H", "A", "D", "X"}:
            raise ValueError("Invalid prediction label")
        writer.writerow((frame, label))
    return stream.getvalue().encode("utf-8")


def _write_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_bytes((json.dumps(record, sort_keys=True, indent=2) + "\n").encode("utf-8"))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_status(status):
    # Explicit whitelist: no input paths, prediction hashes, counts or labels.
    allowed = ("phase", "validated", "reproducible", "candidate_exists",
               "fallback_exists", "exception_type", "environment")
    safe = {key: status[key] for key in allowed if key in status}
    _write_json(OUTPUT / "status.json", safe)
    _write_json(PUBLIC_STATUS, safe)


def _guard_existing(path, content):
    """Preserve an existing differing artifact and refuse to replace it."""
    if path.exists() and path.read_bytes() != content:
        digest = sha256_file(path)
        backup = path.with_name(path.stem + ".previous-" + digest + path.suffix)
        if not backup.exists():
            with backup.open("xb") as handle, path.open("rb") as original:
                for block in iter(lambda: original.read(1024 * 1024), b""):
                    handle.write(block)
        elif sha256_file(backup) != digest:
            raise ValueError("Invalid preserved prediction")
        raise FileExistsError("Existing prediction differs; preserved without replacement")


def _validate_bytes(content, expected_rows=EXPECTED_ROWS):
    from evaluation import possession_metrics as pm
    temporary = OUTPUT / ("validation-" + uuid.uuid4().hex + ".csv")
    try:
        temporary.write_bytes(content)
        if pm.validate_submission(pm.read_label_csv(temporary), expected_rows):
            raise ValueError("Invalid submission contract")
    finally:
        temporary.unlink(missing_ok=True)


def _generate(config, sequence_config, set_phase):
    import numpy as np
    from possession.inference import predict
    from possession.sequence_decoding import predict_sequence
    from possession.tactical_tagging import load_team_prior
    from possession.tracking_features import load_evidence

    set_phase("geometry")
    # Each generation independently streams ATD; never trust the historical
    # feature cache, whose original contract did not hash the raw input.
    evidence = load_evidence(use_cache=False)
    prior, _ = load_team_prior()
    tagged, _ = predict(evidence, config, team_prior=prior)
    fallback = tagged["northmind_geometry"]
    set_phase("decode")
    decoded, checks = predict_sequence(evidence, fallback, prior, config, sequence_config)
    if (not checks["uses_equivariant_st_tcn"] or not checks["neural_cache_fingerprint"]
            or not checks["control_only_preserves_dead_mask"]
            or not np.array_equal(decoded["northmind_sequence"] == "D", fallback == "D")):
        raise ValueError("Required neural provenance or stopping invariant missing")
    return {"Northmind.csv": serialize_submission(decoded["northmind_sequence"]),
            "Northmind-fallback.csv": serialize_submission(fallback)}, checks


def _provenance():
    from games import asset_paths
    inputs = {name: sha256_file(asset_paths()[name])
              for name in ("atd_meta", "atd_raw", "smart_tagging")}
    source_paths = ("scripts/final_submission.py", "games.py", "starter/baseline.py",
                    "starter/load_smart_tagging.py", "possession/inference.py",
                    "possession/tracking_features.py", "possession/tactical_tagging.py",
                    "possession/sequence_decoding.py", "possession/st_tcn_target.py",
                    "possession/target_domain.py")
    return inputs, {name: sha256_file(ROOT / name) for name in source_paths}


def main():
    status = {"phase": "restore", "validated": False, "reproducible": False,
              "candidate_exists": False, "fallback_exists": False,
              "environment": {"python": platform.python_version(), "platform": sys.platform}}

    def set_phase(phase):
        status["phase"] = phase
        _publish_status(status)

    try:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        set_phase("restore")
        from games import GAME, GAMES
        from possession.inference import Config
        from possession.sequence_decoding import SequenceConfig
        from possession.st_tcn_target import compute_and_cache_st_tcn_target, load_cached_st_tcn_target
        if GAME != "BAR-ATH" or GAMES[GAME]["n_video_frames"] != EXPECTED_ROWS:
            raise ValueError("Unexpected target contract")
        config, sequence_config = Config(), SequenceConfig()
        set_phase("neural")
        if compute_and_cache_st_tcn_target() != 0:
            raise RuntimeError("Neural computation failed")
        if load_cached_st_tcn_target(EXPECTED_ROWS) is None:
            raise ValueError("Required neural cache unavailable")
        inputs_before, sources_before = _provenance()
        first, checks = _generate(config, sequence_config, set_phase)
        second, repeated_checks = _generate(config, sequence_config, set_phase)
        set_phase("validate")
        inputs_after, sources_after = _provenance()
        if (inputs_before != inputs_after
                or sources_before != sources_after or first != second
                or checks["neural_cache_fingerprint"] != repeated_checks["neural_cache_fingerprint"]):
            raise ValueError("Delivery regeneration was not identical")
        for content in first.values():
            _validate_bytes(content)
        # Preflight both names before committing either artifact.
        for name, content in first.items():
            _guard_existing(OUTPUT / name, content)
        for name, content in first.items():
            path = OUTPUT / name
            temporary = OUTPUT / (name + "." + uuid.uuid4().hex + ".tmp")
            try:
                temporary.write_bytes(content)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        _write_json(OUTPUT / "manifest.json", {
            "candidate": "northmind_sequence", "fallback": "northmind_tagged_geometry",
            "selection": "provisional fixed rule preserving D; no target score selection",
            "uses_gt": False, "evaluated": False, "rows": EXPECTED_ROWS,
            "config": asdict(config), "sequence_config": asdict(sequence_config),
            "inputs_sha256": inputs_after, "sources_sha256": sources_after,
            "neural_cache_fingerprint": checks["neural_cache_fingerprint"],
            "outputs_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in first.items()},
            "independent_generations": 2, "reproducible": True,
            "environment": status["environment"],
        })
        status.update(phase="complete", validated=True, reproducible=True,
                      candidate_exists=True, fallback_exists=True)
        _publish_status(status)
        return 0
    except BaseException as error:
        # Never expose messages, tracebacks, paths or input/output values.
        status["exception_type"] = type(error).__name__
        status["candidate_exists"] = (OUTPUT / "Northmind.csv").is_file()
        status["fallback_exists"] = (OUTPUT / "Northmind-fallback.csv").is_file()
        try:
            _publish_status(status)
        except BaseException:
            pass
        return 1


def test():
    """Synthetic-only entry point, suitable for the reviewed launcher."""
    import unittest
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_final_submission.py")
    return 0 if unittest.TextTestRunner().run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
