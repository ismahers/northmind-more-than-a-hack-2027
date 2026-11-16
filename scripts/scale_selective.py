"""CLI entry point for scale-selective temporal fusion on public IDSSE pilots."""
import argparse

from possession.scale_selective.experiment import run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "select", "test", "report"), default="all")
    args = parser.parse_args(argv)
    return run(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())

