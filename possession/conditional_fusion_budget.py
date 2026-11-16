"""Reviewed metadata-only public compute audit, before any OOF training."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from possession.external_data.schema import file_hash
from possession.football_predictive_run import SPLITS

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/"evaluation/reports/external/conditional-fusion-budget-v1"
PUBLIC = ROOT/"docs/public-results/2026-10-29"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


def units():
    result = []
    for fold, split in SPLITS.items():
        train = sorted(split["train"])
        for index in (0, 1):
            heldout = train[2*index:2*index+2]
            fit = [key for key in train if key not in heldout]
            if set(fit) & set(heldout) or len(fit) != 2 or len(heldout) != 2:
                raise ValueError("Invalid crossfit plan")
            result.append({"outer_fold": int(fold), "inner_fold": index,
                           "train": fit, "heldout": heldout,
                           "excluded": [split[k] for k in ("calibration", "validation", "test")]})
    return result


def eligible(training_keys, selection_keys, heldout, outer_excluded):
    used = set(training_keys) | set(selection_keys)
    return not bool(used & (set(heldout) | set(outer_excluded)))


def forecast(epoch_seconds, inference_seconds, spec):
    # Eight teachers, 20 epochs each, with two rather than four train matches.
    teacher = len(units())*len(spec["teacher_seeds"])*sum(spec["teacher_epochs"])*.5
    def estimate(per_epoch):
        return spec["forecast_safety_factor"]*(teacher*per_epoch+inference_seconds
            +spec["forecast_heads_embeddings_seconds"]+spec["forecast_gate_validation_seconds"])
    return {"teacher_model_count": 8, "head_model_count": 8,
        "teacher_epoch_equivalents_four_match": teacher,
        "median_estimate_seconds": estimate(float(np.median(epoch_seconds))),
        "p90_estimate_seconds": estimate(float(np.quantile(epoch_seconds, .9))),
        "note": "Projection, not runtime guarantee: linear match-count scaling, historical inference proxy and fixed overhead reserves."}


def run():
    spec = read(ROOT/"configs/conditional_fusion_v1.json")
    public_risk = read(PUBLIC/"sequence-risk-v1.json")
    provenance_path = ROOT/"evaluation/reports/external/sequence-risk-v1/registration.json"
    if file_hash(provenance_path) != public_risk["registration_sha256"]:
        raise ValueError("Published provenance mismatch")
    provenance = read(provenance_path)
    times, checkpoints, reuse = [], {}, []
    for fold in spec["folds"]:
        for seed in spec["teacher_seeds"]:
            relative = f"evaluation/reports/external/temporal-teachers-v1/training/fold-{fold}/seed-{seed}/binary/last.pt"
            path = ROOT/relative
            if file_hash(path) != provenance["dependencies"][relative]:
                raise ValueError("Public teacher checkpoint changed")
            state = torch.load(path, map_location="cpu", weights_only=False)
            split = state["split"]
            if split != SPLITS[str(fold)]:
                raise ValueError("Unexpected historical split")
            history_seconds = [float(x["seconds"]) for x in state["history"]]
            times.extend(history_seconds)
            checkpoints[relative] = {"sha256": file_hash(path), "train_count": len(split["train"]),
                                     "recorded_epochs": len(history_seconds),
                                     "epoch_median_s": float(np.median(history_seconds))}
            for unit in units():
                if unit["outer_fold"] == fold:
                    # Existing seed/recipe choice used own validation; exclude it too.
                    reuse.append(eligible(split["train"], [split["validation"]], unit["heldout"], unit["excluded"]))
    prediction = forecast(times, public_risk["seconds"], spec)
    report = {"protocol": spec, "units": units(), "checkpoint_metadata": checkpoints,
              "reusable_existing_teacher_count": int(sum(reuse)), "forecast": prediction,
              "fits_budget_p90": prediction["p90_estimate_seconds"] <= spec["max_compute_seconds"],
              "training_started": False, "raw_data_accessed": False, "validation_scored": False,
              "test_accessed": False, "target_accessed": False,
              "sources": {n: hashlib.sha256((ROOT/n).read_text(encoding="utf-8-sig").replace("\r\n", "\n").encode()).hexdigest()
                          for n in ("possession/conditional_fusion_budget.py", "configs/conditional_fusion_v1.json")}}
    write(DEST/"public-summary.json", report)
    # Fixed reviewed metadata-only export; never emits weights, labels or traces.
    write(PUBLIC/"conditional-fusion-budget-v1.json", report)
    return 0
