"""Reviewed public-only local-head continuation; use scripts.run_local."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("audit", "select", "test", "report", "all"), default="all")
    args = parser.parse_args()
    from possession.local_head_budget.experiment import run
    return run(args.stage)
