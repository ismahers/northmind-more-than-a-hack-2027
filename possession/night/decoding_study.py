"""Inner-match decoder search on cached emissions; new metrics never choose parameters."""

import hashlib
import json
from types import SimpleNamespace
import time

import numpy as np

from possession.external_data.schema import file_hash
from possession.learning.metrics import score_match
from possession.learning.study import ScoreContext
from scripts.phase3 import CachedRunDecoder, metric_row
from scripts.train_e0 import manifest
from .cache import PublicStore
from .common import DEST, OLD, atomic_json, objective, robust_eligible, progress, source_hashes
from .decoders import NightDecoder, candidate_grid
from .training import old_directory


def load_predictions(path):
    with np.load(path, allow_pickle=False) as archive:
        return SimpleNamespace(time=archive["time"], run=archive["run"], native_index=archive["native_index"]), archive["probabilities"]


def quick_scores(context, p, decoder, diagnostics=False):
    hard = context.decoded(p, decoder)
    pm = context.pm
    score = pm.spell_scores(context.gt, hard, context.runs)
    result = {"spellF1": score["mean_f1"],
              "macroF1": pm.macro_f1(pm.scored_confusion(pm.confusion(context.gt, hard)))}
    if diagnostics:
        truth = pm.possession_spells(context.gt, context.runs)
        prediction = pm.possession_spells(hard, context.runs)
        truth_degree, pred_degree = np.zeros(len(truth), int), np.zeros(len(prediction), int)
        for _, a, b in pm.spell_overlaps(truth, prediction):
            truth_degree[a] += 1
            pred_degree[b] += 1
        result.update(fragmented_gt=int((truth_degree > 1).sum()), fused_predictions=int((pred_degree > 1).sum()),
                      gt_spells=len(truth), pred_spells=len(prediction))
    return result


def guard_reasons(metrics, baseline, protocol):
    scores = {case: row["spellF1"] for case, row in metrics.items()}
    reference = {case: row["spellF1"] for case, row in baseline.items()}
    reasons = []
    if not robust_eligible(scores, reference, protocol):
        reasons.append("occlusion_score_below_reference_minus_0.02")
    actual, previous = metrics["clean"], baseline["clean"]
    settings = protocol["decoder_guard"]
    if actual["fused_predictions"] > previous["fused_predictions"] + settings["fusion_increase"]:
        reasons.append("more_validation_fusions")
    allowance = max(settings["fragmentation_absolute_allowance"],
                    int(np.ceil(settings["fragmentation_fraction_increase"] * previous["fragmented_gt"])))
    if actual["fragmented_gt"] > previous["fragmented_gt"] + allowance:
        reasons.append("validation_fragmentation_allowance_exceeded")
    return reasons


def selected_source(i):
    previous = json.loads((OLD / f"selections/fold-{i}.json").read_text())
    condition = previous["winner"]["condition"]
    bases = {family: value["specification"] for family, value in previous["conditions"][condition]["by_family"].items()}
    bases["frozen"] = previous["winner"]["specification"]
    with np.load(old_directory(i, condition) / "priors.npz", allow_pickle=False) as archive:
        priors = dict(archive)
    return previous, condition, bases, priors


def select_fold(store, data, i, protocol, budget):
    previous, condition, bases, priors = selected_source(i)
    source_paths = {case: OLD / f"validation_predictions/fold-{i}/{condition}/{case}.npz"
                    for case in protocol["primary_scenarios"]}
    fingerprint = {"protocol": protocol, "source": source_hashes(["possession/night/decoders.py",
        "possession/night/decoding_study.py"]), "condition": condition,
        "validation_probabilities": {case: file_hash(path) for case, path in source_paths.items()}}
    target = DEST / f"decoder_selections/fold-{i}.json"
    if target.exists():
        result = json.loads(target.read_text())
        if result["fingerprint"] != fingerprint:
            raise ValueError("Frozen decoder selection differs from current inputs")
        return result
    search_manifest = DEST / f"decoder_trials/fold-{i}/fingerprint.json"
    if search_manifest.exists() and json.loads(search_manifest.read_text()) != fingerprint:
        raise ValueError("Partial search provenance changed; keep it and register a new run")
    atomic_json(search_manifest, fingerprint)
    started = time.perf_counter()
    validation_key = data["folds"][i]["validation"][0]
    match = store.match(validation_key)
    validation = {case: load_predictions(path) for case, path in source_paths.items()}
    context = ScoreContext(match, validation["clean"][0])
    for view, p in validation.values():
        np.testing.assert_array_equal(view.time, context.view.time)
    caches, trials, baseline = {case: {} for case in validation}, [], None
    for specification in candidate_grid(protocol):
        budget.check()
        trial_path = DEST / f"decoder_trials/fold-{i}/{specification['id']}.json"
        if trial_path.exists():
            trial = json.loads(trial_path.read_text())
            if trial["specification"] != specification:
                raise ValueError("Partial trial grid differs")
        else:
            start = time.perf_counter()
            metrics = {case: quick_scores(context, p, NightDecoder(specification, bases, priors, caches[case]), case == "clean")
                       for case, (_, p) in validation.items()}
            if baseline is None:
                baseline = metrics
            scores = {case: value["spellF1"] for case, value in metrics.items()}
            reasons = guard_reasons(metrics, baseline, protocol)
            trial = {"specification": specification, "metrics": metrics, "objective": objective(scores, protocol),
                "eligible": not reasons, "rejection_reasons": reasons, "validation_s": time.perf_counter() - start}
            atomic_json(trial_path, trial)
        if baseline is None:
            baseline = trial["metrics"]
        trials.append(trial)
        progress("decoder_validation", fold=i, candidate=specification["id"], family=specification["family"])
    for case in protocol["primary_scenarios"]:
        if baseline[case]["spellF1"] != previous["winner"]["scores"][case]:
            raise ValueError("Frozen decoder no longer reproduces original validation score")
    finalists = {}
    for trial in trials:
        family = trial["specification"]["family"]
        if trial["eligible"] and (family not in finalists or trial["objective"] > finalists[family]["objective"]):
            finalists[family] = trial
    winner = max(finalists.values(), key=lambda trial: trial["objective"])
    result = {"fingerprint": fingerprint, "outer_test_used": False, "condition": condition,
        "winner": winner, "finalists": finalists, "trials": trials,
        "unconstrained_best": max(trials, key=lambda trial: trial["objective"]),
        "validation_search_s": time.perf_counter() - started,
        "discarded_families": sorted({t["specification"]["family"] for t in trials} - finalists.keys())}
    atomic_json(target, result)
    return result


def test_fold(store, data, i, selection, protocol, budget):
    previous, condition, bases, priors = selected_source(i)
    test_key = data["folds"][i]["test"][0]
    match = store.match(test_key)
    rows = []
    previous_rows = json.loads((OLD / f"rows/fold-{i}.json").read_text())
    all_cases = json.loads((OLD / "protocol.json").read_text())["corruptions"]
    for case in all_cases:
        name = case["name"]
        source = OLD / f"test/seed-2027/fold-{i}/{condition}/{name}/probabilities.npz"
        view, p = load_predictions(source)
        context, cache = ScoreContext(match, view), {}
        fraction = next(row["actual_hidden_fraction"] for row in previous_rows if row["condition"] == condition and row["scenario"] == name)
        finalists = selection["finalists"] if name in protocol["primary_scenarios"] else {
            selection["winner"]["specification"]["family"]: selection["winner"]}
        metrics_cache = {}
        for family, trial in finalists.items():
            budget.check()
            specification = trial["specification"]
            path = DEST / f"decoder_metrics/fold-{i}/{name}/{specification['id']}.json"
            if path.exists():
                record = json.loads(path.read_text())
                metrics, timing, origin = record["metrics"], record["elapsed_s"], record["origin"]
            else:
                started = time.perf_counter()
                decoder = CachedRunDecoder(NightDecoder(specification, bases, priors, cache))
                hard = context.decoded(p, decoder)
                signature = hashlib.sha256(hard.tobytes()).hexdigest()
                if family == "frozen":
                    metrics = json.loads((source.parent / (previous["winner"]["family"] + ".json")).read_text())
                    origin = "reused_phase3_metrics"
                elif signature in metrics_cache:
                    metrics = metrics_cache[signature]
                    origin = "identical_decoded_path_reused"
                else:
                    metrics = score_match(match, view, p, decoder)
                    origin = "measured_decoder_on_cached_emissions"
                metrics_cache[signature] = metrics
                timing = time.perf_counter() - started
                atomic_json(path, {"metrics": metrics, "elapsed_s": timing, "origin": origin, "specification": specification})
            training = previous["conditions"][condition]
            selected = specification["id"] == selection["winner"]["specification"]["id"]
            row = metric_row(metrics, i, condition, name, family, 2027, training["cap"], training["selected_epoch"], fraction, selected)
            row.update(experiment="cached_decoder", decoder_id=specification["id"], probability_path=str(source),
                validation_objective=trial["objective"], evaluation_s=timing, origin=origin)
            rows.append(row)
        progress("decoder_test", fold=i, scenario=name)
    atomic_json(DEST / f"rows/decoder-fold-{i}.json", rows)


def run_decoders(protocol, budget):
    data, _ = manifest("idsse-only")
    store = PublicStore(data["artifacts"])
    for i in protocol["decoder_folds"]:
        target = DEST / f"rows/decoder-fold-{i}.json"
        selection = select_fold(store, data, i, protocol, budget)
        if target.exists():
            continue
        test_fold(store, data, i, selection, protocol, budget)
