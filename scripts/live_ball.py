"""Reviewed external live-ball pilot; invoke through scripts.run_local only."""
import hashlib
import json
from pathlib import Path
import platform
import unittest

ROOT = Path(__file__).resolve().parents[1]
TEST_SOURCES = ("possession/live_ball.py", "possession/live_ball_validation.py",
                "tests/test_live_ball.py", "tests/test_live_ball_validation.py",
                "scripts/live_ball.py", "configs/live_ball_validation_v1.json")


def tests():
    suite = unittest.TestSuite()
    for name in ("test_live_ball.py", "test_live_ball_validation.py"):
        suite.addTests(unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern=name))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    hashes = {name: hashlib.sha256((ROOT / name).read_text(
        encoding="utf-8-sig").replace("\r\n", "\n").encode("utf-8")).hexdigest()
        for name in TEST_SOURCES}
    status = {"synthetic_only": True, "tests": result.testsRun,
              "success": result.testsRun > 0 and result.wasSuccessful(),
              "errors": len(result.errors), "failures": len(result.failures),
              "skipped": len(result.skipped), "python": platform.python_version(),
              "sources_sha256_lf": hashes}
    destination = ROOT / "docs/public-results/2026-11-17/live-ball-tests.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(status, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0 if status["success"] else 1


def run():
    from possession.live_ball_validation import run as execute
    return execute()


def verify():
    from possession.live_ball_validation import verify as execute
    return execute()


def export():
    from possession.live_ball_validation import export as execute
    return execute()
