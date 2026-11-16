"""Independent v3 output audit; handles Windows paths without rewriting seals.

Only metadata, public aggregates and byte hashes are inspected. No inference,
per-case content, raw data, logs or target files are loaded by this audit.
"""
import hashlib
import json
from pathlib import Path

from possession.external_data.schema import file_hash
from possession.long_diagnosis_run import ROOT, DEST, validate_protocol
from possession.long_diagnosis import paired_uncertainty
from possession.night.common import atomic_json

TASKS = ("audit-publish-long-diagnosis", "test-long-diagnosis-audit")


def normalized_outputs(outputs):
    normalized = {name.replace("\\", "/"): digest for name, digest in outputs.items()}
    if len(normalized) != len(outputs) or any(".." in name.split("/") or name.startswith("/") for name in normalized):
        raise ValueError("Unsafe or duplicate result paths")
    return normalized


def matches(path, expected):
    if file_hash(path) == expected:
        return True
    if path != ROOT / "scripts/run_local.py":
        return False
    lines = path.read_bytes().splitlines(keepends=True)
    kept = [line for line in lines if not any(line.lstrip().startswith(('"'+name+'":').encode()) for name in TASKS)]
    return len(lines)-len(kept) == 2 and hashlib.sha256(b"".join(kept)).hexdigest() == expected


def audit():
    registration = DEST / "registration.json"
    record = json.loads(registration.read_text())
    validate_protocol(record["protocol"])
    # Verify every recorded source/dependency. No source or model exception is
    # added here: only our two launcher declarations may be removed byte-exactly.
    for name, digest in record["sources"].items():
        if not matches(ROOT / name, digest):
            raise ValueError("Changed v3 source")
    for name, digest in record["dependencies"].items():
        if not matches(Path(name), digest):
            raise ValueError("Changed v3 dependency")
    seal = json.loads((DEST / "result-seal.json").read_text())
    fingerprint = file_hash(registration)
    if seal["registration_sha256"] != fingerprint:
        raise ValueError("Registration seal mismatch")
    outputs = normalized_outputs(seal["outputs"])
    expected = {"public-summary.json"}
    for fold in (0, 6):
        for scenario in record["protocol"]["scenarios"]:
            expected.update(f"{kind}/fold-{fold}/{scenario}.{ext}" for kind, ext in
                (("cache", "npz"), ("cache", "json"), ("cases", "json"), ("cases", "npz")))
    if set(outputs) != expected:
        raise ValueError("Incomplete outputs")
    for name, digest in outputs.items():
        if file_hash(DEST / name) != digest:
            raise ValueError("Output changed")
    result = json.loads((DEST / "public-summary.json").read_text())
    if (any(result[k] for k in ("test_accessed", "target_accessed", "training", "new_fusions", "automatic_promotion"))
            or result["registration_sha256"] != fingerprint
            or {(r["fold"], r["scenario"]) for r in result["rows"]} !=
               {(f, s) for f in (0, 6) for s in record["protocol"]["scenarios"]}
            or len(result["rows"]) != 6
            or not all(r["historical_metrics_parity"] for r in result["rows"])
            or result["paired_uncertainty"] != paired_uncertainty(result["rows"])):
        raise ValueError("Invalid result scope/accounting")
    atomic_json(DEST / "public-audit-v1.json", {"version": "long-diagnosis-output-audit-v1",
        "verified": True, "registration_sha256": fingerprint, "result_seal_sha256": file_hash(DEST / "result-seal.json"),
        "output_files": len(outputs), "rows": len(result["rows"]),
        "audit_sources": {name: file_hash(ROOT / name) for name in
            ("scripts/audit_long_diagnosis.py", "scripts/public_handoff_portable.py", "tests/test_long_diagnosis_audit.py")},
        "path_normalization": "backslash to slash, without rewriting any original file",
        "original_verifier": "FAILED on Windows path separators; original preserved",
        "test_accessed": False, "target_accessed": False})
    return 0


def main():
    audit()
    from scripts.public_handoff_portable import export
    return export()
