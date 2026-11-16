"""Bounded, resumable public IDSSE experiment. Selection precedes all test reads."""

from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from possession.external_data.schema import file_hash
from possession.learning.corruption import hide_ball, stable_seed
from possession.learning.data import Windows, collate
from possession.learning.metrics import SCORER_PATH, score_match
from possession.learning.model import masked_losses
from possession.learning.study import ScoreContext
from possession.learning.temporal import TemporalDecoder
from possession.learning.training import forward, seed_everything, atomic_checkpoint, memory_usage
from possession.night.cache import PublicStore
from possession.night.common import atomic_json, OLD, DEST as NIGHT
from possession.night.decoding_study import selected_source, load_predictions
from possession.night.decoders import change_evidence
from possession.night.training import old_directory
from scripts.external_data import ROOT, REPORTS
from scripts.train_e0 import manifest
from scripts.phase3 import metric_row, extra_diagnostics
from .model import BoundaryE0, BoundaryWindows, boundary_collate, boundary_targets, boundary_loss
from .decoding import AdaptiveDecoder
from .metrics import diagnostics

DEST = REPORTS / "phase5-v1"
CONFIG = ROOT / "configs/phase5_protocol.json"


def register():
    protocol = json.loads(CONFIG.read_text())
    sources = list((ROOT / "possession/phase5").glob("*.py")) + [ROOT / "scripts/phase5_experiments.py"]
    fingerprint = {"protocol": protocol, "source": {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
                   "official_scorer": file_hash(SCORER_PATH)}
    path = DEST / "registration.json"
    if path.exists():
        if json.loads(path.read_text()) != fingerprint:
            raise ValueError("Existing phase5 run is immutable; use a new version for changed code/protocol")
    else:
        atomic_json(path, fingerprint)
    return protocol


def progress(stage, **fields):
    atomic_json(DEST / "progress.json", {"stage": stage, **fields})


class Budget:
    def __init__(self, seconds):
        self.path = DEST / "compute.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"seconds": 0., "sessions": []}
        self.start, self.limit = time.perf_counter(), seconds

    def check(self):
        if time.perf_counter() - self.start + self.data["seconds"] >= self.limit:
            raise TimeoutError("Budget reached; resume from durable checkpoints")

    def close(self, stage, status):
        elapsed = time.perf_counter() - self.start
        self.data["seconds"] += elapsed
        self.data["sessions"].append({"stage": stage, "status": status, "seconds": elapsed})
        atomic_json(self.path, self.data)


def get_store(data):
    # This cache already exists for all seven matches. No old result is written.
    for key in data["artifacts"]:
        if not (NIGHT / "feature_cache" / key.replace(":", "_") / "manifest.json").exists():
            raise ValueError("Expected reusable public feature cache is absent")
    return PublicStore(data["artifacts"])


def source(i):
    previous, condition, bases, priors = selected_source(i)
    cap = previous["conditions"][condition]["cap"]
    return {"condition": condition, "cap": cap,
            "checkpoint": old_directory(i, condition) / f"cap-{cap}.pt",
            "specification": bases["frozen"], "priors": priors}


def original_validation(i, src, protocol):
    return {name: load_predictions(OLD / f"validation_predictions/fold-{i}/{src['condition']}/{name}.npz")[1]
            for name in protocol["scenarios"]}


def scenarios(view, key, protocol):
    result = {"clean": (view, np.zeros(len(view.time), bool))}
    for name, length in (("missing40_5s", 5.), ("missing40_15s", 15.)):
        result[name] = hide_ball(view, .4, length, stable_seed(key, 701))
    return result


def predict(model, view):
    dataset = Windows({"public": view}, 256, model.context)
    p, q = np.zeros((len(view.time), 3), np.float32), np.zeros(len(view.time), np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), 8):
            items = [dataset[i] for i in range(start, min(start + 8, len(dataset)))]
            out = forward(model, collate(items))
            pp, qq = out["probabilities"].numpy(), out["boundary_logit"].sigmoid().numpy()
            for item, pred, boundary in zip(items, pp, qq):
                left, a, b = item["left"], item["a"], item["b"]
                p[a:b], q[a:b] = pred[a - left:b - left], boundary[a - left:b - left]
    return p, q


def save_prediction(path, view, p, q):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, probabilities=p, boundary=q, time=view.time, run=view.run, native_index=view.native_index)


def load_prediction(path):
    with np.load(path, allow_pickle=False) as z:
        return z["probabilities"], z["boundary"]


def make_model(src, seed):
    saved = torch.load(src["checkpoint"], map_location="cpu", weights_only=False)
    seed_everything(seed)
    model = BoundaryE0(**{k: saved["config"][k] for k in ("spatial_dim", "channels", "dilations", "dropout")})
    missing, unexpected = model.load_state_dict(saved["model"], strict=False)
    if set(missing) != {"boundary_head.weight", "boundary_head.bias"} or unexpected:
        raise ValueError("Unexpected warm-start architecture")
    return model


def evaluate_cases(context, predictions, src, protocol, strength=0., threshold=.5, learned=False):
    values = {}
    for name in protocol["scenarios"]:
        p, q = predictions[name]
        decoder = AdaptiveDecoder(src["specification"], src["priors"], context.view,
                                  q if learned else None, strength, threshold)
        hard = context.decoded(p, decoder)
        metrics = context.pm.spell_scores(context.gt, hard, context.runs)
        row = {"spellF1": metrics["mean_f1"]}
        if name == "clean":
            diag = diagnostics(context, hard)
            row.update(long_recall=diag["duration_groups"]["ge10"]["iou"]["0.5"]["recall"],
                       fragmented_gt=diag["fragmented_gt"], fused_predictions=diag["fused_predictions"])
        values[name] = row
    score = sum(w * values[name]["spellF1"] for w, name in zip(protocol["score_weights"], protocol["scenarios"]))
    return {"metrics": values, "objective": score}


def eligible(trial, reference, protocol):
    return all(trial["metrics"][name]["spellF1"] >= reference["metrics"][name]["spellF1"] + protocol["robustness_floor"]
               for name in protocol["scenarios"][1:]) and (
                   trial["metrics"]["clean"]["long_recall"] >= reference["metrics"]["clean"]["long_recall"] + protocol["long_spell_recall_floor"])


def train_arm(store, data, i, kind, src, protocol, budget):
    folder = DEST / "training" / f"fold-{i}" / kind
    complete = folder / "training.json"
    if complete.exists():
        return json.loads(complete.read_text())
    folder.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    fold = data["folds"][i]
    keys = [k for k in fold["train"] if k.startswith("idsse:")]
    assert not set(keys) & set(fold["test"] + fold["validation"])
    val_key = fold["validation"][0]
    views = {key: store.view(key) for key in keys}
    validation = store.view(val_key)
    targets = {key: boundary_targets(store.match(key), view, protocol["boundary_tolerance_s"])
               for key, view in views.items()}
    positives = sum(float((target * mask).sum()) for target, mask in targets.values())
    total = sum(int(mask.sum()) for _, mask in targets.values())
    if positives <= 0:
        raise ValueError("No true H/A boundary supervision")
    pos_weight = min(protocol["positive_weight_cap"], (total - positives) / positives)
    context = ScoreContext(store.match(val_key), validation)
    cases = scenarios(validation, val_key, protocol)
    val_targets = {val_key: boundary_targets(store.match(val_key), validation, protocol["boundary_tolerance_s"])}
    val_loader = DataLoader(BoundaryWindows({val_key: validation}, val_targets, 256, 15),
                            batch_size=4, collate_fn=boundary_collate)
    model = make_model(src, protocol["seed"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"], weight_decay=.0001)
    generator = torch.Generator().manual_seed(protocol["seed"])
    history, best_score, best_epoch, stale = [], -np.inf, 0, 0
    last_path = folder / "last.json"
    if last_path.exists():
        checkpoint = torch.load(folder / json.loads(last_path.read_text())["checkpoint"], weights_only=False, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"])
        generator.set_state(checkpoint["loader_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        history, best_score, best_epoch, stale = (checkpoint[k] for k in ("history", "best_score", "best_epoch", "stale"))
    while len(history) < protocol["max_epochs"]:
        if kind != "control" and len(history) >= protocol["minimum_epochs"] and stale >= protocol["patience"]:
            break
        budget.check()
        epoch = len(history) + 1
        progress("training", fold=i, loss=kind, epoch=epoch)
        epoch_start = time.perf_counter()
        length = protocol["training_block_seconds"][(epoch - 1) % 3]
        degraded = {key: hide_ball(view, protocol["training_missing_fraction"], length,
                                   stable_seed(key, protocol["seed"] + epoch * 1000))[0] for key, view in views.items()}
        loader = DataLoader(BoundaryWindows(degraded, targets, 256, model.context), batch_size=protocol["batch_size"],
                            shuffle=True, generator=generator, collate_fn=boundary_collate)
        train_stats = epoch_pass(model, loader, optimizer, pos_weight, kind, protocol)
        validation_stats = epoch_pass(model, val_loader, None, pos_weight, kind, protocol)
        predictions = {name: predict(model, value[0]) for name, value in cases.items()}
        scores = evaluate_cases(context, predictions, src, protocol)
        if scores["objective"] > best_score + .0001:
            best_score, best_epoch, stale = scores["objective"], epoch, 0
        else:
            stale += 1
        history.append({"epoch": epoch, "training": train_stats, "validation_loss": validation_stats,
                        "selection": scores, "seconds": time.perf_counter() - epoch_start})
        # Save complete RNG/optimizer state after validation for exact continuation.
        atomic_checkpoint(folder / f"epoch-{epoch:02d}.pt", {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
            "loader_rng": generator.get_state(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
            "history": history, "best_score": best_score, "best_epoch": best_epoch, "stale": stale,
            "source_checkpoint": str(src["checkpoint"]), "source_sha256": file_hash(src["checkpoint"]),
            "kind": kind, "protocol": protocol, "train_keys": keys, "validation_key": val_key})
        for name, (p, q) in predictions.items():
            save_prediction(folder / f"validation/epoch-{epoch:02d}/{name}.npz", validation, p, q)
        atomic_json(last_path, {"checkpoint": f"epoch-{epoch:02d}.pt"})
    result = {"fold": i, "kind": kind, "best_epoch": best_epoch, "best_score": best_score, "history": history,
              "positive_weight": pos_weight, "positive_band_frames": positives, "valid_frames": total,
              "source_checkpoint": str(src["checkpoint"]), "source_sha256": file_hash(src["checkpoint"]),
              "train_keys": keys, "validation_key": val_key, "seconds": time.perf_counter() - begin,
              "memory": memory_usage(torch.device("cpu")), "early_stopped": len(history) < protocol["max_epochs"]}
    atomic_json(complete, result)
    return result


def epoch_pass(model, loader, optimizer, pos_weight, kind, protocol):
    model.train(optimizer is not None)
    base_totals, boundary_total, count, gradients = np.zeros(4), 0., 0, []
    for batch in loader:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            output = forward(model, batch)
            base, stats = masked_losses(output, batch["alive"], batch["team"], batch["confidence"], batch["core"])
            aux = boundary_loss(output["boundary_logit"], batch["boundary_target"],
                                batch["boundary_valid"] & batch["core"], pos_weight,
                                "bce" if kind == "control" else kind, protocol["focal_gamma"])
            loss = base + (0. if kind == "control" else protocol["boundary_loss_weight"]) * aux
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite phase5 loss")
            if optimizer is not None:
                loss.backward()
                gradients.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)))
                optimizer.step()
        base_totals += stats.numpy()
        boundary_total += float(aux.detach())
        count += 1
    state_loss, team_loss = base_totals[0] / max(1, base_totals[1]), base_totals[2] / max(1, base_totals[3])
    return {"state_loss": state_loss, "team_loss": team_loss, "boundary_loss_batch_mean": boundary_total / max(1, count),
            "base_loss": state_loss + team_loss, "gradient_norm_mean": float(np.mean(gradients)) if gradients else None,
            "gradient_norm_max": max(gradients) if gradients else None, "batches": count}


def stored_predictions(i, kind, epoch, protocol):
    return {name: load_prediction(DEST / f"training/fold-{i}/{kind}/validation/epoch-{epoch:02d}/{name}.npz")
            for name in protocol["scenarios"]}


def proxy(context, probabilities):
    result = np.zeros(len(probabilities), np.float32)
    for selected, _, _ in context.parts:
        ha, ah = change_evidence(probabilities[selected], 3)
        result[selected] = np.maximum(ha, ah)
    return result


def threshold_selection(context, predictions, protocol, learned):
    trials = []
    for threshold in protocol["head_thresholds"]:
        values = []
        for name in protocol["scenarios"]:
            p, q = predictions[name]
            signal = q * p[:, :2].sum(-1) if learned else proxy(context, p)
            hard = context.decoded(p, TemporalDecoder({"kind": "argmax"}, {}))
            values.append(diagnostics(context, hard, signal, threshold)["head_HA"]["0.5"]["f1"])
        trials.append({"threshold": threshold, "objective": sum(w * x for w, x in zip(protocol["score_weights"], values))})
    return max(trials, key=lambda row: row["objective"])["threshold"]


def select_fold(store, data, i, protocol, budget):
    path = DEST / f"selections/fold-{i}.json"
    if path.exists():
        return json.loads(path.read_text())
    src = source(i)
    summaries = {kind: train_arm(store, data, i, kind, src, protocol, budget) for kind in protocol["losses"]}
    val_key = data["folds"][i]["validation"][0]
    context = ScoreContext(store.match(val_key), store.view(val_key))
    baseline = {name: (p, None) for name, p in original_validation(i, src, protocol).items()}
    A = evaluate_cases(context, baseline, src, protocol)
    boundary_trials = []
    for kind in ("bce", "focal"):
        epoch = summaries[kind]["best_epoch"]
        predictions = stored_predictions(i, kind, epoch, protocol)
        trial = evaluate_cases(context, predictions, src, protocol)
        trial.update(kind=kind, epoch=epoch, eligible=eligible(trial, A, protocol))
        boundary_trials.append(trial)
    admissible = [r for r in boundary_trials if r["eligible"]] or boundary_trials
    C = max(admissible, key=lambda row: row["objective"])
    predictions = stored_predictions(i, C["kind"], C["epoch"], protocol)
    grids = {}
    for system, inputs, learned in (("B", baseline, False), ("D", predictions, True)):
        trials = []
        for strength in protocol["adaptive_strengths"]:
            for threshold in (protocol["head_thresholds"] if learned and strength else [.5]):
                budget.check()
                trial = evaluate_cases(context, inputs, src, protocol, strength, threshold, learned)
                trial.update(strength=strength, threshold=threshold, eligible=eligible(trial, A, protocol))
                trials.append(trial)
        grids[system] = {"trials": trials, "winner": max([r for r in trials if r["eligible"]] or trials,
                                                           key=lambda row: row["objective"])}
    candidates = {"A": {**A, "eligible": True}, "B": grids["B"]["winner"],
                  "C": C, "D": grids["D"]["winner"]}
    winner = max((name for name, row in candidates.items() if row["eligible"]), key=lambda name: candidates[name]["objective"])
    result = {"fold": i, "outer_test_used": False, "baseline_condition": src["condition"],
              "baseline_cap": src["cap"], "frozen_decoder": src["specification"],
              "boundary_kind": C["kind"], "boundary_epoch": C["epoch"], "boundary_trials": boundary_trials,
              "control_best_epoch": summaries["control"]["best_epoch"], "control_matched_epoch": C["epoch"],
              "grids": grids, "candidates": candidates, "winner": winner,
              "head_threshold": threshold_selection(context, predictions, protocol, True),
              "proxy_threshold": threshold_selection(context, baseline, protocol, False)}
    atomic_json(path, result)
    return result


def load_arm_model(i, kind, epoch, src, protocol):
    model = make_model(src, protocol["seed"])
    saved = torch.load(DEST / f"training/fold-{i}/{kind}/epoch-{epoch:02d}.pt", weights_only=False, map_location="cpu")
    model.load_state_dict(saved["model"])
    return model


def test_fold(store, data, i, protocol, budget):
    output = DEST / f"rows/fold-{i}.json"
    if output.exists():
        return json.loads(output.read_text())
    # All seven selection files must exist before any new outer evaluation.
    selections = [json.loads((DEST / f"selections/fold-{j}.json").read_text()) for j in range(7)]
    selection, src = selections[i], source(i)
    key = data["folds"][i]["test"][0]
    view = store.view(key)
    context = ScoreContext(store.match(key), view)
    models = {"boundary": load_arm_model(i, selection["boundary_kind"], selection["boundary_epoch"], src, protocol),
              "control_matched": load_arm_model(i, "control", selection["control_matched_epoch"], src, protocol),
              "control_best": load_arm_model(i, "control", selection["control_best_epoch"], src, protocol)}
    rows = []
    for name, (degraded, missing) in scenarios(view, key, protocol).items():
        budget.check()
        progress("test", fold=i, scenario=name)
        _, baseline = load_predictions(OLD / f"test/seed-2027/fold-{i}/{src['condition']}/{name}/probabilities.npz")
        predictions = {"original": (baseline, None)}
        for kind, model in models.items():
            path = DEST / f"predictions/fold-{i}/{kind}/{name}.npz"
            if path.exists():
                predictions[kind] = load_prediction(path)
            else:
                predictions[kind] = predict(model, degraded)
                save_prediction(path, view, *predictions[kind])
        for system, kind in (("A", "original"), ("B", "original"), ("C", "boundary"), ("D", "boundary"),
                             ("control_matched", "control_matched"), ("control_best", "control_best"),
                             ("raw_A", "original"), ("raw_C", "boundary")):
            start = time.perf_counter()
            p, q = predictions[kind]
            if system.startswith("raw"):
                decoder = TemporalDecoder({"kind": "argmax"}, {})
            else:
                trial = selection["grids"][system]["winner"] if system in ("B", "D") else {"strength": 0., "threshold": .5}
                decoder = AdaptiveDecoder(src["specification"], src["priors"], view,
                                          q if system == "D" else None, trial["strength"], trial["threshold"])
            hard = context.decoded(p, decoder)
            learned = kind == "boundary"
            signal = q * p[:, :2].sum(-1) if learned else proxy(context, p)
            threshold = selection["head_threshold"] if learned else selection["proxy_threshold"]
            detail = diagnostics(context, hard, signal, threshold)
            full = score_match(store.match(key), view, p, decoder)
            row = metric_row(full, i, src["condition"], name, system, protocol["seed"], src["cap"],
                             selection["boundary_epoch"] if learned else 0, float(missing.mean()), system == selection["winner"])
            row.update(system=system, match=key, duration_groups=detail["duration_groups"],
                       decoded_HA=detail["decoded_HA"], head_HA=detail["head_HA"], head_threshold=threshold,
                       evaluation_s=time.perf_counter() - start,
                       independent=extra_diagnostics(store.match(key), view, p, decoder, missing))
            rows.append(row)
            atomic_json(DEST / f"metrics/fold-{i}/{name}/{system}.json", {"row": row, "official": full})
        atomic_json(DEST / f"rows/fold-{i}-partial.json", rows)
    atomic_json(output, rows)
    return rows


def run(stage):
    protocol = register()
    torch.set_num_threads(protocol["cpu_threads"])
    data, _ = manifest("idsse-only")
    store = get_store(data)
    budget = Budget(protocol["budget_seconds"])
    status = "failed"
    try:
        if stage in ("pilots", "validation", "all"):
            order = protocol["pilot_folds"] if stage == "pilots" else protocol["pilot_folds"] + [i for i in range(7) if i not in protocol["pilot_folds"]]
            for i in order:
                select_fold(store, data, i, protocol, budget)
        if stage in ("test", "all"):
            rows = []
            for i in range(7):
                rows.extend(test_fold(store, data, i, protocol, budget))
            atomic_json(DEST / "all_rows.json", rows)
        status = "complete"
        progress(stage, status=status)
    finally:
        budget.close(stage, status)
