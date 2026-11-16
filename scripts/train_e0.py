"""Train/evaluate only public external artifacts recorded in the frozen LOMO manifest."""

import argparse
import json
from pathlib import Path

from possession.external_data.schema import file_hash
from possession.learning.protocol import Fold
from possession.learning.training import TrainConfig, evaluate_checkpoint, train
from scripts.external_data import EXTERNAL, REPORTS, ROOT


def manifest(name):
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("Invalid manifest name")
    record = json.loads((REPORTS / "folds" / (name + ".json")).read_text(encoding="utf-8"))
    paths = []
    for item in record["artifacts"].values():
        path = Path(item["path"]).resolve()
        if not path.is_relative_to(EXTERNAL.resolve()) or file_hash(path) != item["sha256"]:
            raise ValueError("Only unchanged external artifacts in data/external are accepted")
        paths.append(path)
    return record, paths


def parser_base():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--run", default="e0", help="Unique experiment name, no path separators")
    parser.add_argument("--manifest", default="idsse-only", help="Frozen manifest name")
    return parser


def run_directory(args):
    if not args.run or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.run):
        raise ValueError("Invalid run name")
    return REPORTS / "experiments" / args.run / ("fold-" + str(args.fold))


def train_main():
    parser = parser_base()
    parser.add_argument("--config", default="configs/e0.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    record, paths = manifest(args.manifest)
    if not 0 <= args.fold < len(record["folds"]):
        raise ValueError("Unknown fold")
    specification = record["folds"][args.fold]
    fold = Fold(specification["name"], *(tuple(specification[k]) for k in ("train", "validation", "test")))
    config = TrainConfig.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8")))
    output = run_directory(args)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Experiment exists; use a fresh run name or resume explicitly")
    train(paths, fold, config, output, resume=output / "last.pt" if args.resume else None)
    return 0


def evaluate_main():
    parser = parser_base()
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--allow-partial", action="store_true", help="Diagnostic only: unknown labels censor spells")
    args = parser.parse_args()
    _, paths = manifest(args.manifest)
    output = run_directory(args)
    evaluate_checkpoint(paths, output / "best.pt", args.split, output / args.split, args.allow_partial)
    return 0
