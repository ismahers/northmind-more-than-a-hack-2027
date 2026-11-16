"""Compare restricted files locally; export booleans only, never their hashes."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.digest()


def main():
    output = Path("evaluation/reports/final-delivery")
    record = {"different_source_root": True, "candidate_byte_identical": False,
              "fallback_byte_identical": False, "verified": False}
    try:
        for label, name in (("candidate", "Northmind.csv"), ("fallback", "Northmind-fallback.csv")):
            record[label + "_byte_identical"] = digest(ROOT / output / name) == digest(ROOT / "_pub" / output / name)
        record["verified"] = record["candidate_byte_identical"] and record["fallback_byte_identical"]
    except Exception as error:
        record["exception_type"] = type(error).__name__
    destination = ROOT / "docs/public-results/2026-11-16/submission-replay-status.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return 0 if record["verified"] else 1
