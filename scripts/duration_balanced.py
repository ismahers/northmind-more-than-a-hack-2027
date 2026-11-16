"""No bootstrap entry point: existing public parents are mandatory."""
import argparse
from possession.duration_balanced.common import audit


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",choices=("isma","phase8"),default="isma")
    parser.add_argument("--stage",choices=("audit","select","test","report","all"),default="audit")
    args=parser.parse_args()
    if args.stage=="audit":
        audit(args.source)
        return 0
    if args.stage=="report":
        from possession.duration_balanced.report import report
        return report(args.source)
    from possession.duration_balanced.experiment import run
    result=run(args.source,args.stage)
    if result==0 and args.stage=="all":
        from possession.duration_balanced.report import report
        result=report(args.source)
    return result
