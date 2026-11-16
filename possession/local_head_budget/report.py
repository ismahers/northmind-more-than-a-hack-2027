"""Aggregate public metrics and verify the immutable continuation and seal."""
import csv

import numpy as np
import torch

from possession.external_data.schema import file_hash
from possession.night.common import atomic_json
from possession.temporal_probe.common import read_json
from .common import ROOT, DEST, parent_checkpoint, cap_path
from .experiment import assert_sealed
from .training import state_equal


def verify(record):
    spec = record["protocol"]; digest = file_hash(DEST / "registration.json")
    checks = {}
    for name, expected in record["sources"].items():
        checks["source:"+name] = file_hash(ROOT / name) == expected
    for name, expected in record["references"].items():
        checks["parent:"+name] = file_hash(ROOT / name) == expected
    assert_sealed(spec, digest)
    predictions = list((DEST / "predictions").glob("**/test/*.npz"))
    checks["both_folds_sealed_before_test"] = bool(predictions) and (DEST / "selection-seal.json").stat().st_mtime_ns < min(p.stat().st_mtime_ns for p in predictions)
    for fold in spec["pilots"]:
        selected = read_json(DEST / f"selections/fold-{fold}.json")
        checks[f"fold{fold}/no_test_selection"] = not selected["test_used"]
        checks[f"fold{fold}/exact_resume_proof"] = read_json(DEST / f"training/fold-{fold}/resume-proof.json")["exact"]
        parent = torch.load(parent_checkpoint(fold), map_location="cpu", weights_only=False)
        parent_generator = torch.Generator(); parent_generator.set_state(parent["loader_rng"])
        complete = read_json(DEST / f"training/fold-{fold}/complete.json")
        steps = complete["steps_per_epoch"]
        windows = complete["training_windows"]
        previous = parent
        for cap in spec["budgets"]:
            state = torch.load(cap_path(fold, cap), map_location="cpu", weights_only=False)
            first_path = DEST.with_name("local-head-budget-phase8-v1") / f"training/fold-{fold}/cap-{cap}.pt"
            if cap > 8 and first_path.is_file():
                first = torch.load(first_path, map_location="cpu", weights_only=False)
                checks[f"fold{fold}/{cap}/identical_with_explicit_determinism"] = all(
                    state_equal(state[k], first[k]) for k in ("model", "optimizer", "loader_rng", "torch_rng", "numpy_rng", "python_rng"))
            checks[f"fold{fold}/{cap}/checkpoint_frozen"] = file_hash(cap_path(fold, cap)) == selected["checkpoint_hashes"][str(cap)]
            checks[f"fold{fold}/{cap}/exact_epochs"] = len(state["history"]) == cap
            checks[f"fold{fold}/{cap}/same_history_prefix"] = state["history"][:len(previous["history"])] == previous["history"]
            checks[f"fold{fold}/{cap}/correct_updates"] = all(r["steps"] == steps for r in state["history"])
            checks[f"fold{fold}/{cap}/adamw_steps"] = all(int(s["step"]) == cap*steps for s in state["optimizer"]["state"].values())
            checks[f"fold{fold}/{cap}/finite"] = all(bool(torch.isfinite(value).all()) for value in state["model"].values())
            for _ in range(cap-len(previous["history"])):
                torch.randperm(windows, generator=parent_generator)
            checks[f"fold{fold}/{cap}/expected_sampler_continuation"] = torch.equal(parent_generator.get_state(), state["loader_rng"])
            previous = state
    rows = read_json(DEST / "all_rows.json")
    checks["24_test_rows"] = len(rows) == 24
    all_predictions = list((DEST / "predictions").glob("**/*.npz"))
    checks["36_prediction_files"] = len(all_predictions) == 36
    for path in all_predictions:
        with np.load(path, allow_pickle=False) as z:
            p = z["probabilities"]
            checks["normalized:"+str(path.relative_to(DEST))] = bool(np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1, atol=2e-6))
            first_path = DEST.with_name("local-head-budget-phase8-v1") / path.relative_to(DEST)
            if first_path.is_file():
                with np.load(first_path, allow_pickle=False) as first:
                    checks["identical_explicit_determinism:"+str(path.relative_to(DEST))] = np.array_equal(p, first["probabilities"])
    checks["deterministic_algorithms_enabled"] = torch.are_deterministic_algorithms_enabled()
    checks["registered_cpu_determinism"] = record["environment"]["deterministic_algorithms"] and record["environment"]["device"] == "cpu"
    result = {"passed": all(checks.values()), "checks": checks, "count": len(checks)}
    atomic_json(DEST / "verification.json", result)
    return result


def compact(metrics):
    fields = ("spellF1", "macroF1", "long_recall", "short_recall", "short_recovered", "short_gt",
              "false_possessions_per_minute", "false_transitions_per_minute", "fragmented_gt", "fused_predictions", "conditional_BCE", "transitions")
    return {k: metrics[k] for k in fields if k in metrics}


def report(record):
    verification = verify(record)
    if not verification["passed"]:
        return 1
    spec = record["protocol"]
    selections = {str(fold): read_json(DEST / f"selections/fold-{fold}.json") for fold in spec["pilots"]}
    rows = read_json(DEST / "all_rows.json")
    folds = {}
    for fold in spec["pilots"]:
        selection = selections[str(fold)]
        folds[str(fold)] = {"teacher_seed": record["teachers"][str(fold)]["seed"],
                            "head_seed": spec["seed"], "winner": selection["winner"],
                            "unconstrained_best": selection["unconstrained_best"], "gates": selection["gates"],
                            "validation_objectives": selection["objectives"],
                            "validation": {cap: {scenario: compact(m) for scenario, m in group.items()} for cap, group in selection["trials"].items()},
                            "test": {str(cap): {r["scenario"]: compact(r["metrics"]) for r in rows if r["fold"] == fold and r["cap"] == cap}
                                     for cap in spec["budgets"]+["teacher"]}}
    training = {str(fold): read_json(DEST / f"training/fold-{fold}/complete.json") for fold in spec["pilots"]}
    costs = {"new_trajectory_epochs": 16, "reused_parent_epochs": 16, "extra_exact_resume_proof_epochs": 2,
             "training_seconds": sum(row["seconds"] for value in training.values() for row in value["history"][8:]),
             "resume_proof_seconds": sum(read_json(DEST / f"training/fold-{fold}/resume-proof.json")["seconds"] for fold in spec["pilots"]),
             "registered_sessions": read_json(DEST / "compute.json"),
             "new_teacher_training": 0, "new_embedding_exports": 0}
    summary = {"folds": folds, "decision": read_json(DEST / "decision.json"), "cost": costs,
               "training_loss_by_fold": {fold: [{"epoch": row["epoch"], "loss": row["loss"]} for row in value["history"]] for fold, value in training.items()},
               "verification_passed": True, "verification_count": verification["count"]}
    if (DEST / "D_diagnosis.json").exists():
        summary["D_diagnosis"] = read_json(DEST / "D_diagnosis.json")
    atomic_json(DEST / "summary.json", summary)
    with (DEST / "results.csv").open("w", newline="") as stream:
        fields = ["fold", "cap", "scenario", "spellF1", "macroF1", "short_recall", "short_recovered", "short_gt", "long_recall", "false_transitions_per_minute", "false_possessions_per_minute", "fragmented_gt", "fused_predictions"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore"); writer.writeheader()
        for row in rows:
            writer.writerow({**row, **row["metrics"]})
    lines = ["# Local H/A head — budgets8/12/16", "", "Frozen Phase8 teachers; same original BCE and HMM. Selection uses own validation only.", "",
             "| Fold | Cap | Validation robust | Validation short | Selected |", "|---|---:|---:|---:|---|"]
    for fold, value in folds.items():
        for cap in spec["budgets"]:
            m = value["validation"][str(cap)]["clean"]
            lines.append(f"| {fold} | {cap} | {value['validation_objectives'][str(cap)]:.5f} | {m['short_recovered']}/{m['short_gt']} | {value['winner'] == cap} |")
    lines += ["", "| Fold | Cap | Test clean | Short recovered | Long recall | FP HA/min | Occluded5 | Occluded15 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for fold, value in folds.items():
        for cap in spec["budgets"]:
            group = value["test"][str(cap)]; m = group["clean"]
            lines.append(f"| {fold} | {cap} | {m['spellF1']:.5f} | {m['short_recovered']}/{m['short_gt']} | {m['long_recall']:.4f} | {m['false_transitions_per_minute']:.4f} | {group['missing40_5s']['spellF1']:.5f} | {group['missing40_15s']['spellF1']:.5f} |")
    lines += ["", "Common validation-approved extended budgets: "+str(summary["decision"]["common_budget_passes_both_validation_gates"]),
              "Cost and D diagnostics: summary.json. Test comparisons never change the frozen selections."]
    (DEST / "REPORT.md").write_text("\n".join(lines)+"\n")
    return 0
