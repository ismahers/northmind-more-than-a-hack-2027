"""Run isolated public-external night experiments through scripts.run_local."""

import argparse
import json
from pathlib import Path
import time

import torch

from possession.night.common import DEST, OLD, register, Budget, BudgetStop, atomic_json, progress
from scripts.train_e0 import manifest


def train_stage(protocol, budget):
    from possession.night.cache import PublicStore
    from possession.night.training import profile, continue_trajectory
    data, _ = manifest("idsse-only")
    store = PublicStore(data["artifacts"])
    profile(store, data, protocol)
    for i in protocol["training_folds"]:
        for condition in protocol["training_conditions"]:
            budget.check()
            continue_trajectory(store, data, i, condition, protocol, budget)
    # All four trajectories complete before any new held-out model evaluation.
    from possession.night.training_evaluation import evaluate_extensions
    evaluate_extensions(store, data, protocol, budget)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("train", "decode", "reuse", "combine", "report", "verify"), required=True)
    args = parser.parse_args()
    protocol = register()
    torch.set_num_threads(protocol["cpu_threads"])
    DEST.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (DEST / "execution.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget, status = Budget(protocol), "FAILED"
        try:
            if args.stage == "train":
                train_stage(protocol, budget)
            elif args.stage == "decode":
                from possession.night.decoding_study import run_decoders
                run_decoders(protocol, budget)
            elif args.stage == "reuse":
                from possession.night.reuse import run_reuse
                run_reuse(protocol, budget)
            elif args.stage == "combine":
                from possession.night.combination import run_combination
                run_combination(protocol, budget)
            elif args.stage == "report":
                from scripts.night_report import report
                report(protocol)
            else:
                from scripts.night_report import verify
                verify(protocol)
            status = "OK"
            progress(args.stage + "_complete")
        except BudgetStop:
            status = "BUDGET_STOP"
            progress("budget_stop", interrupted_stage=args.stage, resumable=True)
        finally:
            budget.close(args.stage, status)
    return 0
