"""Run only reviewed public IDSSE experiments through scripts.run_local."""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilots", "validation", "test", "all"), default="all")
    args = parser.parse_args()
    from possession.phase5.experiment import run
    run(args.stage)
    return 0
