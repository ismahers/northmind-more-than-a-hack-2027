"""Reviewed public-data HA diagnostics/pilot; stdout/stderr via run_local."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("diagnose", "pilot", "report"), default="diagnose")
    args = parser.parse_args()
    if args.stage == "diagnose":
        from possession.ha_discrimination.diagnosis import run
    else:
        from possession.ha_discrimination.pilot import run
    return run() if args.stage == "diagnose" else run(args.stage)
