"""Reviewed control-memory entry points; use scripts.run_local only.

Tests are synthetic; validation uses public external IDSSE data exclusively.
No target data, target GT, video, or restricted log is read here.
"""
from pathlib import Path
import hashlib
import json
import platform
import unittest


ROOT = Path(__file__).resolve().parents[1]


def tests():
    suite = unittest.TestSuite()
    for name in ("test_control_memory.py", "test_control_memory_validation.py"):
        suite.addTests(unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern=name))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sources = {}
    for name in ("possession/control_memory.py", "possession/control_memory_validation.py",
                 "tests/test_control_memory.py", "tests/test_control_memory_validation.py",
                 "scripts/control_memory.py", "configs/control_memory_validation_v1.json"):
        path = ROOT / name
        if path.is_file():
            content = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").encode("utf-8")
            sources[name] = hashlib.sha256(content).hexdigest()
    record = {"synthetic_only": True, "tests": result.testsRun,
              "success": result.testsRun > 0 and result.wasSuccessful(),
              "failures": len(result.failures), "errors": len(result.errors),
              "skipped": len(result.skipped), "python": platform.python_version(),
              "sources_sha256_lf": sources}
    destination = ROOT / "docs/public-results/2026-11-17/control-memory-tests.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0 if record["success"] else 1


def run():
    from possession.control_memory_validation import run as execute
    return execute()


def verify():
    from possession.control_memory_validation import verify as execute
    return execute()


def export():
    from possession.control_memory_validation import export as execute
    return execute()
