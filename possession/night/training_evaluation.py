"""Freeze extension choices using only their own validation, then evaluate public tests."""

import json
import time

import numpy as np
import torch

from possession.learning.corruption import hide_ball, stable_seed
from possession.learning.metrics import score_match
from possession.learning.study import save_probabilities
from possession.learning.temporal import TemporalDecoder
from possession.learning.training import TrainConfig, new_model
from scripts.phase3 import CachedRunDecoder, metric_row
from .common import DEST, OLD, atomic_json, objective, robust_eligible, progress
from .training import old_directory, infer


def evaluate_extensions(store, data, protocol, budget):
    cases = [c for c in json.loads((OLD / "protocol.json").read_text())["corruptions"]
             if c["name"] in protocol["primary_scenarios"]]
    for i in protocol["training_folds"]:
        target = DEST / f"rows/training-fold-{i}.json"
        if target.exists():
            continue
        previous = json.loads((OLD / f"selections/fold-{i}.json").read_text())
        candidates = []
        for condition in protocol["training_conditions"]:
            training = json.loads((DEST / f"training/{condition}/fold-{i}/training.json").read_text())
            for cap in training["caps"]:
                scores = cap["validation_scores"][cap["frozen_family"]]
                candidates.append({"condition": condition, "cap": cap["cap"], "family": cap["frozen_family"],
                    "selected_epoch": cap["selected_epoch"], "objective": objective(scores, protocol), "scores": scores,
                    "eligible": robust_eligible(scores, previous["winner"]["scores"], protocol)})
        eligible = [row for row in candidates if row["eligible"]]
        if not eligible:
            raise ValueError("Original cap12 reference must remain eligible")
        winner = max(eligible, key=lambda row: (row["objective"], -row["cap"]))
        selection = {"outer_test_used": False, "candidates": candidates, "winner": winner,
            "fixed_decoders": previous["conditions"], "reference_phase3": previous["winner"]}
        selection_path = DEST / f"training_selections/fold-{i}.json"
        if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
            raise ValueError("Frozen training selection changed")
        atomic_json(selection_path, selection)
        # Nothing below this point can modify the selection.
        test_key = data["folds"][i]["test"][0]
        match, view = store.match(test_key), store.view(test_key)
        rows = []
        for condition in protocol["training_conditions"]:
            training = json.loads((DEST / f"training/{condition}/fold-{i}/training.json").read_text())
            with np.load(old_directory(i, condition) / "priors.npz") as z:
                priors = dict(z)
            metric_cache, model_cache = {}, {}
            old_epoch = json.loads((old_directory(i, condition) / "training.json").read_text())["chosen"]["selected_epoch"]
            for cap in training["caps"]:
                epoch = cap["selected_epoch"]
                families = list(dict.fromkeys(["argmax", "hmm", "hsmm", cap["frozen_family"]]))
                for case in cases:
                    budget.check()
                    name = case["name"]
                    hidden = None
                    probability_path = DEST / f"training_test_predictions/fold-{i}/{condition}/epoch-{epoch}/{name}.npz"
                    if epoch == old_epoch:
                        source = OLD / f"test/seed-2027/fold-{i}/{condition}/{name}/probabilities.npz"
                        with np.load(source) as z:
                            p = z["probabilities"]
                            np.testing.assert_array_equal(z["time"], view.time)
                        probability_origin = str(source)
                        fraction = 0. if name == "clean" else None
                    elif probability_path.exists():
                        with np.load(probability_path) as z:
                            p = z["probabilities"]
                        probability_origin = str(probability_path)
                        fraction = 0. if name == "clean" else None
                    else:
                        if epoch not in model_cache:
                            state = torch.load(cap["checkpoint"], map_location="cpu", weights_only=False)
                            model = new_model(TrainConfig.from_dict(state["config"]))
                            model.load_state_dict(state["model"])
                            model_cache = {epoch: model}
                        corrupted, hidden = hide_ball(view, case["fraction"], case["block_seconds"], stable_seed(test_key, 701))
                        start = time.perf_counter()
                        p = infer(model_cache[epoch], corrupted, protocol)
                        inference_s = time.perf_counter() - start
                        save_probabilities(probability_path, view, p)
                        atomic_json(probability_path.with_suffix(".timing.json"), {"inference_s": inference_s})
                        probability_origin, fraction = str(probability_path), float(hidden.mean())
                    if fraction is None:
                        # Same deterministic mask as phase3; no labels used.
                        from possession.learning.corruption import block_mask
                        fraction = float(block_mask(view.run, case["fraction"], round(case["block_seconds"] * 10), stable_seed(test_key, 701)).mean())
                    for family in families:
                        key = (epoch, name, family)
                        if key not in metric_cache:
                            start = time.perf_counter()
                            if epoch == old_epoch:
                                metrics = json.loads((OLD / f"test/seed-2027/fold-{i}/{condition}/{name}/{family}.json").read_text())
                                origin = "reused_phase3_metrics"
                            else:
                                specification = previous["conditions"][condition]["by_family"][family]["specification"]
                                metrics = score_match(match, view, p, CachedRunDecoder(TemporalDecoder(specification, priors)))
                                origin = "measured_new_checkpoint"
                            metric_cache[key] = (metrics, origin, time.perf_counter() - start)
                        metrics, origin, elapsed = metric_cache[key]
                        chosen = condition == winner["condition"] and cap["cap"] == winner["cap"] and family == winner["family"]
                        row = metric_row(metrics, i, condition, name, family, protocol["seed"], cap["cap"], epoch, fraction, chosen)
                        row.update(experiment="training_duration", origin=origin, probability_path=probability_origin,
                            evaluation_s=elapsed, epochs_executed=cap["epochs_executed"],
                            reused_early_stopped_result=cap.get("reused_early_stopped_result", False))
                        rows.append(row)
                        atomic_json(DEST / f"training_metrics/fold-{i}/{condition}/cap-{cap['cap']}/{name}-{family}.json", metrics)
                    progress("training_test", fold=i, condition=condition, cap=cap["cap"], scenario=name)
        atomic_json(target, rows)
