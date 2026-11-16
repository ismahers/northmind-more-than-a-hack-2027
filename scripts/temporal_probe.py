"""Reviewed entry point for bounded public frozen-embedding ablations."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("bootstrap", "select", "test", "report", "all"), default="all")
    args = parser.parse_args()
    if args.stage == "bootstrap":
        from possession.temporal_probe.common import exclusive_run
        from possession.temporal_probe.teacher import bootstrap
        with exclusive_run():
            bootstrap()
    elif args.stage == "report":
        from possession.temporal_probe.common import exclusive_run
        from possession.temporal_probe.report import report
        with exclusive_run():
            report()
    else:
        from possession.temporal_probe.experiment import run
        run(args.stage)
    return 0
