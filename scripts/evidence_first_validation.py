"""Reviewed public-only CLI; run exclusively via scripts.run_local."""
from possession.evidence_first_validation import run, verify


def main():
    return run()


def verify_main():
    return verify()
