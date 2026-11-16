"""Run resumable public-only Phase7 pilots and frozen confirmation."""
import argparse
from possession.phase7.experiment import run
from possession.phase7.public import DEST

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=("pilots","test","transfer","all"),default="all")
    args=parser.parse_args()
    if args.stage=="all":
        run("pilots")
        from scripts.phase7_freeze import main as freeze
        freeze()
        run("test");run("transfer")
    else:
        if args.stage in ("test","transfer") and not (DEST/"independent_architecture_selection.json").exists():
            raise ValueError("Freeze own-fold architecture choices before new confirmation")
        run(args.stage)
    return 0
