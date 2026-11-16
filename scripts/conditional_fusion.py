"""Reviewed entries, output captured from creation by scripts.run_local."""
from possession.conditional_fusion_run import run, verify, replay, export


def tests():
    import unittest
    from possession.conditional_fusion_run import write, PUBLIC, ROOT
    failures = []
    class Result(unittest.TextTestResult):
        def addFailure(self, test, err):
            failures.append({"test": test.id(), "kind": "assertion"})
            super().addFailure(test, err)
        def addError(self, test, err):
            failures.append({"test": test.id(), "kind": err[0].__name__, "synthetic_stage": getattr(test, "stage", "none")})
            super().addError(test, err)
    suite = unittest.defaultTestLoader.discover(str(ROOT/"tests"), pattern="test_conditional_fusion.py")
    result = unittest.TextTestRunner(verbosity=2, resultclass=Result).run(suite)
    write(PUBLIC/"conditional-fusion-test-status.json", {"tests": result.testsRun,
          "success": result.wasSuccessful(), "synthetic_failures": failures})
    return 0 if result.wasSuccessful() and result.testsRun else 1
