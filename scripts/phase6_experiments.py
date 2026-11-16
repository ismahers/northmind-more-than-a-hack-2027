"""Reviewed public IDSSE pilots; no target-match assets opened here."""

import argparse


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=("pilots","test","all"),default="pilots")
    args=parser.parse_args()
    from possession.phase6.experiment import run
    run(args.stage)
    return 0
