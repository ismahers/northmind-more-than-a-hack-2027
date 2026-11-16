"""Reviewed public-only CLI; run through scripts.run_local, never expose traces."""
import argparse
from possession.ball_attention.experiment import run
from possession.ball_attention.common import DEST
from possession.night.common import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",choices=("equivalence","pilot","report"),default="pilot")
    args = parser.parse_args()
    try:
        result = run(args.stage)
        atomic_json(DEST/"status.json",{"stage":args.stage,"ok":result==0})
        return result
    except Exception as exc:
        # Exception text/trace stays captured. Only type is a safe public status.
        atomic_json(DEST/"status.json",{"stage":args.stage,"ok":False,"error_type":type(exc).__name__})
        raise


if __name__=="__main__":
    raise SystemExit(main())
