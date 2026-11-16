"""Fixed public-validation commands; only invoked by the reviewed launcher."""


def freeze_main():
    from possession.long_diagnosis_run import freeze
    return freeze()


def main():
    from possession.long_diagnosis_run import run
    return run()


def verify_main():
    from possession.long_diagnosis_run import verify
    status = verify()
    if status:
        return status
    from scripts.public_handoff import export
    return export()


def test_main():
    """Synthetic tests only; publish test IDs/status, never exception messages."""
    import json
    from pathlib import Path
    import unittest
    root = Path(__file__).resolve().parents[1]
    suite = unittest.defaultTestLoader.discover(str(root / "tests"), pattern="test_long_diagnosis.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    path = root / "evaluation/reports/external/long-diagnosis-v3/synthetic-test-status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scope": "synthetic only; no data read", "tests": result.testsRun,
        "passed": result.wasSuccessful(), "failures": [t.id() for t, _ in result.failures],
        "errors": [t.id() for t, _ in result.errors]}, indent=2)+"\n")
    return 0 if result.wasSuccessful() else 1
