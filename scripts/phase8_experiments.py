"""Complete serial orchestration of the public-only Phase8 study."""
import argparse
from possession.phase8.training import run_primary
from possession.phase8.evaluation import primary_selection,mixed_training,tests


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=("all","train","select","mixed","test","transfer"),default="all")
    args=parser.parse_args()
    if args.stage in ("train","all"):run_primary()
    if args.stage in ("select","all"):primary_selection()
    if args.stage in ("mixed","all"):mixed_training()
    if args.stage in ("test","all"):tests()
    if args.stage in ("transfer","all"):
        from possession.phase8.transfer import run
        run()
    return 0
