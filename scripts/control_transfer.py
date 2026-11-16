"""Reviewed synthetic-only runner. Never imports loaders or reads datasets/logs."""

import hashlib
import json
import platform
import sys
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "possession/control_transfer.py", "tests/test_control_transfer.py",
    "scripts/control_transfer.py", "configs/control_transfer_v1.json",
    "scripts/run_local.py", "docs/experiments/CONTROL_TRANSFER_PROTOCOL_2026-10-30.md",
)


def _source_hashes():
    # All registered sources are text. Git checkouts may use CRLF or LF;
    # canonical LF hashes identify the same source on either platform.
    return {name: hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for name in SOURCES}


def main():
    """Run exactly the synthetic suite, seal sources, export only safe counts."""
    started = time.perf_counter()
    before = _source_hashes()
    suite = unittest.defaultTestLoader.discover(
        str(ROOT / "tests"), pattern="test_control_transfer.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    stable = before == _source_hashes()
    success = result.wasSuccessful() and result.testsRun >= 6 and not result.skipped and stable
    report = {
        "id": "control-transfer-v1-synthetic",
        "synthetic_only": True,
        "dataset_accessed": False,
        "real_hypothesis_tested": False,
        "model_promoted": False,
        "success": success,
        "source_integrity_unchanged": stable,
        "source_hash_policy": "SHA-256 of text source bytes with CRLF normalized to LF",
        "python_version": sys.version.split()[0],
        "platform": platform.system(),
        "elapsed_seconds": time.perf_counter() - started,
        "tests": {"run": result.testsRun, "failures": len(result.failures),
                  "errors": len(result.errors), "skipped": len(result.skipped)},
        "fixture_count": 6,
        "native_hz": 25,
        "scan_hz": 10,
        "trailing_seconds": 0.12,
        "source_sha256": before,
    }
    # This object contains only constants, counts and source hashes. No exception
    # text, fixture curves, input values or paths from an external dataset enter it.
    local = ROOT / "evaluation/reports/control-transfer-v1"
    local.mkdir(parents=True, exist_ok=True)
    (local / "synthetic-status.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if success:
        public = ROOT / "docs/public-results/2026-10-30"
        public.mkdir(parents=True, exist_ok=True)
        (public / "control-transfer-v1-synthetic.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if success else 1


def tests():
    return main()


if __name__ == "__main__":
    raise SystemExit(main())
