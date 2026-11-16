"""Public validation CLI, output captured by run_local from child creation."""
from possession.pulse_consensus_validation import run, verify

def main():
    return run()

def verify_main():
    return verify()
