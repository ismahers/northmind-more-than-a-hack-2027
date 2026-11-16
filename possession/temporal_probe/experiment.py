"""Train/select all pilots before evaluating new test exports; bounded public-only run."""
import numpy as np
import torch

from possession.external_data.schema import file_hash
from possession.interval_probe.data import load
from possession.interval_probe.decoding import ProposalDecoder
from possession.interval_probe.proposals import generate, calibration_targets, bonuses
from possession.learning.study import ScoreContext
from possession.learning.temporal import fit_priors
from possession.night.common import atomic_json
from possession.phase5.metrics import diagnostics
from possession.phase6.candidates import fit_logistic, logit
from possession.phase6.decoding import DirectionalDecoder

from .common import DEST, Store, Budget, register, read_json, exclusive_run
from .data import get_exports
from .selection import choose, promotion
from .training import train, load_head
from .windows import predict, emissions


def priors(store, teacher):
    return fit_priors([store.harmonized(key) for key in teacher["split"]["train"]])


def pack(model, data, view, config):
    team, distance, quality = predict(model, data, config["core_frames"], config["batch_size"])
    p = emissions(team, data["probabilities"])
    if not np.isfinite(p).all() or not np.allclose(p.sum(1), 1, atol=2e-6):
        raise ValueError("Invalid reheaded probability distribution")
    box, raw = np.empty((0, 3), np.int64), np.empty(0, float)
    if model.intervals:
        if not np.isfinite(distance).all() or not np.isfinite(quality).all():
            raise ValueError("Invalid localization output")
        alive = p[:, :2].sum(1)
        h = 1/(1+np.exp(-np.clip(team, -30, 30)))
        score = np.column_stack((h, 1-h)) * np.sqrt(1/(1+np.exp(-np.clip(quality, -30, 30)))) * alive[:, None]
        box, raw = generate(view, distance, score, config["proposal"])
    return {"p": p, "boxes": box, "raw": raw}


def metrics(context, p, prior, pack_value=None, calibrator=None, strength=0.):
    if strength:
        reward, _ = bonuses(calibrator, pack_value["raw"], strength)
        decoder = ProposalDecoder(context.view, prior, pack_value["boxes"], reward)
    else:
        decoder = DirectionalDecoder(context.view, prior, "hmm", .5)
    hard = context.decoded(p, decoder)
    detail = diagnostics(context, hard)
    groups = detail["duration_groups"]
    short_gt = sum(groups[name]["gt"] for name in ("lt1", "1to3"))
    short_matched = sum(groups[name]["iou"]["0.5"]["matched_gt"] for name in ("lt1", "1to3"))
    return {"spellF1": float(context.pm.spell_scores(context.gt, hard, context.runs)["mean_f1"]),
            "long_recall": groups["ge10"]["iou"]["0.5"]["recall"],
            "short_recall": short_matched/max(1, short_gt), "duration_groups": groups,
            "fragmented_gt": detail["fragmented_gt"], "fused_predictions": detail["fused_predictions"]}


def select(store, fold, teacher, config, spec, budget, registration_hash):
    dest = DEST / f"selections/fold-{fold}.json"
    if dest.is_file():
        return
    train(store, fold, teacher, config, spec, budget, registration_hash)
    parts = teacher["split"]
    key = parts["validation"]
    view = store.harmonized(key)
    context = ScoreContext(store.match(key), view)
    prior = priors(store, teacher)
    validation = {name: load(path) for name, path in get_exports(store, key, fold, teacher, spec, budget).items()}
    baseline = {name: metrics(context, value["probabilities"], prior) for name, value in validation.items()}
    ckey = parts["calibration"]
    cview = store.harmonized(ckey)
    cc = ScoreContext(store.match(ckey), cview)
    calibration = {name: load(path) for name, path in get_exports(store, ckey, fold, teacher, spec, budget).items()}
    trials, calibrators = [], {}
    for arm in config["arms"]:
        model = load_head(fold, teacher, arm, config)
        fitted = None
        if model.intervals:
            packs = [pack(model, value, cview, config) for value in calibration.values()]
            x = np.concatenate([logit(value["raw"]) for value in packs])[:, None]
            y = np.concatenate([calibration_targets(cc, value["boxes"]) for value in packs])
            fitted = fit_logistic(x, y) if len(y) else {"mean": [0.], "std": [1.],
                        "weights": [float(logit(.01)), 0.], "candidate_prevalence": .01, "n": 0, "positives": 0}
            calibrators[arm] = fitted
        val_packs = {name: pack(model, value, view, config) for name, value in validation.items()}
        strengths = config["bonus_strengths"] if model.intervals else [0.]
        for strength in strengths:
            budget.check()
            measured = {name: metrics(context, value["p"], prior, value, fitted, strength)
                        for name, value in val_packs.items()}
            trials.append({"arm": arm, "strength": strength, "metrics": measured})
    winner = choose(trials, baseline, config)
    # Each promotion is local to this fold. Never pool pilot validations to set another fold's recipe.
    gates = {arm: promotion(choose([t for t in trials if t["arm"] == arm], baseline, config)["metrics"], baseline, config)
             for arm in config["arms"]}
    checkpoints = {arm: file_hash(DEST / f"training/fold-{fold}/{arm}/last.pt") for arm in config["arms"]}
    atomic_json(dest, {"teacher": teacher, "baseline": baseline, "trials": trials,
                      "calibrators": calibrators, "winner": winner, "own_fold_promotion": gates,
                      "checkpoints": checkpoints, "registration_sha256": registration_hash, "test_used": False})


def freeze(config, registration_hash):
    selections = {str(i): file_hash(DEST / f"selections/fold-{i}.json") for i in config["pilots"]}
    record = {"selections": selections, "registration_sha256": registration_hash, "test_used": False}
    path = DEST / "selection-seal.json"
    if path.is_file() and read_json(path) != record:
        raise ValueError("Pilot selections changed after sealing")
    if not path.is_file():
        atomic_json(path, record)


def test(store, fold, teacher, config, spec, budget, registration_hash):
    freeze(config, registration_hash)
    path = DEST / f"rows/fold-{fold}.json"
    if path.is_file():
        return read_json(path)
    chosen = read_json(DEST / f"selections/fold-{fold}.json")
    key = teacher["split"]["test"]
    context = ScoreContext(store.match(key), store.harmonized(key))
    prior = priors(store, teacher)
    cases = {name: load(p) for name, p in get_exports(store, key, fold, teacher, spec, budget).items()}
    rows = []
    for scenario, value in cases.items():
        rows.append({"fold": fold, "arm": "teacher", "scenario": scenario, "system": "hmm", "strength": 0.,
                     "metrics": metrics(context, value["probabilities"], prior)})
    for arm in config["arms"]:
        model = load_head(fold, teacher, arm, config)
        fitted = chosen["calibrators"].get(arm)
        # No test-based choice: best strength for this arm is frozen using own validation.
        best = choose([t for t in chosen["trials"] if t["arm"] == arm], chosen["baseline"], config)
        strength = best["strength"] if best["arm"] == arm else 0.
        for scenario, value in cases.items():
            budget.check()
            current = pack(model, value, context.view, config)
            for system, bonus in (("hmm", 0.), ("proposal", strength)) if model.intervals else (("hmm", 0.),):
                rows.append({"fold": fold, "arm": arm, "scenario": scenario, "system": system, "strength": bonus,
                             "metrics": metrics(context, current["p"], prior, current, fitted, bonus)})
    atomic_json(path, rows)
    return rows


def run(stage="all"):
    with exclusive_run():
        registration, records = register()
        config, spec = registration["protocol"], registration["teacher_protocol"]
        torch.set_num_threads(config["cpu_threads"])
        digest = file_hash(DEST / "registration.json")
        store, budget, status = Store(records), Budget(config["max_compute_seconds"]), "failed"
        try:
            if stage in ("all", "select"):
                for fold in config["pilots"]:
                    select(store, fold, registration["teachers"][str(fold)], config, spec, budget, digest)
                freeze(config, digest)
            if stage in ("all", "test"):
                rows = []
                for fold in config["pilots"]:
                    rows.extend(test(store, fold, registration["teachers"][str(fold)], config, spec, budget, digest))
                atomic_json(DEST / "all_rows.json", rows)
            status = "complete"
        finally:
            budget.close(stage, status)
