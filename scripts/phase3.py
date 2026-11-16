"""Reviewed CPU study on public external data only. Every decision precedes outer test access."""

from dataclasses import asdict
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from possession.external_data.features import build_features, nearest_indices
from possession.external_data.audit import segments
from possession.external_data.schema import Match, file_hash
from possession.learning.corruption import hide_ball, stable_seed
from possession.learning.decoding import ArgmaxDecoder
from possession.learning.metrics import score_match, official_scorer
from possession.learning.protocol import Fold
from possession.learning.study import (ScoreContext, fit_trajectory, load_study_model,
    save_json, save_probabilities)
from possession.learning.temporal import TemporalDecoder, decoder_grid
from possession.learning.training import TrainConfig, train, evaluate_checkpoint, predict
from scripts.external_data import ROOT, REPORTS
from scripts.train_e0 import manifest

DEST = REPORTS / "phase3"
PROTOCOL_PATH = ROOT / "configs/phase3_protocol.json"


class CachedRunDecoder:
    """Reuse the same decoded paths for official metrics and oracle diagnostics."""
    def __init__(self, decoder):
        self.decoder, self.cache = decoder, {}

    def decode(self, probabilities, timestamps):
        key = (float(timestamps[0]), float(timestamps[-1]), len(timestamps))
        if key not in self.cache:
            self.cache[key] = self.decoder.decode(probabilities, timestamps)
        return self.cache[key]


def environment():
    protocol = json.loads(PROTOCOL_PATH.read_text())
    frozen = DEST / "protocol.json"
    if frozen.exists() and json.loads(frozen.read_text()) != protocol:
        raise ValueError("Protocol changed after study registration")
    save_json(frozen, protocol)
    torch.set_num_threads(4)
    a, _ = manifest("idsse-only")
    return protocol, a


def directory(fold, condition, seed=2027):
    return DEST / "training" / condition / ("seed-" + str(seed)) / ("fold-" + str(fold))


def progress(stage, fold, **extra):
    save_json(DEST / "progress.json", {"stage": stage, "fold": fold, **extra})


def baseline(protocol, data, folds):
    paths = [Path(row["path"]) for row in data["artifacts"].values()]
    for i in folds:
        output = DEST / "baseline" / ("fold-" + str(i))
        if (output / "test/metrics.json").exists():
            continue
        progress("baseline", i)
        f = data["folds"][i]
        fold = Fold(f["name"], *(tuple(f[k]) for k in ("train", "validation", "test")))
        config = TrainConfig.from_dict(json.loads((ROOT / "configs/e0_idsse_cpu_first.json").read_text()))
        train(paths, fold, config, output)
        evaluate_checkpoint(paths, output / "best.pt", "validation", output / "validation")
        evaluate_checkpoint(paths, output / "best.pt", "test", output / "test")
    progress("baseline_complete", None)


def durations(protocol, data, folds):
    for i in folds:
        progress("duration", i)
        fit_trajectory(data["artifacts"], data["folds"][i], protocol, directory(i, "A_clean"),
                       max_epochs=max(protocol["epoch_caps"]))
    progress("duration_complete", None)


def conditions(protocol, data, folds):
    combined, _ = manifest("idsse-skillcorner-phase3")
    for i in folds:
        chosen = json.loads((directory(i, "A_clean") / "training.json").read_text())["chosen"]
        for condition in ("A_blocks", "B_clean", "B_blocks"):
            progress("conditions", i, condition=condition, epoch_cap=chosen["budget"])
            source = combined if condition.startswith("B") else data
            fit_trajectory(source["artifacts"], source["folds"][i], protocol, directory(i, condition),
                           condition=condition, max_epochs=chosen["budget"])
    progress("conditions_complete", None)


def prediction_set(model, match, protocol, cases, seed=701):
    view = build_features(match)
    result = {}
    for case in cases:
        corrupted, mask = hide_ball(view, case["fraction"], case["block_seconds"], stable_seed(match.key, seed))
        p = predict(model, corrupted, 256, torch.device("cpu"))
        result[case["name"]] = (p, float(mask.mean()), mask)
    return view, result


def extra_diagnostics(match, view, probabilities, decoder, missing=None):
    ctx = ScoreContext(match, view)
    hard = ctx.decoded(probabilities, decoder)
    pm = official_scorer()
    # Exact 25Hz GT state/team; no resampled GT in oracle diagnostics.
    conditional_team = np.where(probabilities[:, 0] >= probabilities[:, 1], 0, 1)
    native_team = np.full(len(ctx.gt), 3, dtype=np.int8)
    native_alive = np.zeros(len(ctx.gt), dtype=float)
    hidden = np.zeros(len(ctx.gt), dtype=bool)
    for selected, mapping, output in ctx.parts:
        native_team[output] = conditional_team[selected][mapping]
        native_alive[output] = 1 - probabilities[selected, 2][mapping]
        if missing is not None:
            hidden[output] = missing[selected][mapping]
    known = ctx.gt != 3
    true_dead = np.where(ctx.gt == 2, 2, native_team).astype(np.int8)
    true_dead[~known] = 3
    true_team = hard.copy()
    valid_team = (ctx.gt < 2) & (hard != 2)
    true_team[valid_team] = ctx.gt[valid_team]
    result = {}
    for name, labels in (("true_D_predicted_conditional_team", true_dead), ("true_team_where_defined_predicted_D", true_team)):
        metric = pm.evaluate(ctx.gt, labels, bootstrap_n=0)
        result[name] = {key: metric[key] for key in ("spell_f1_mean", "macro_f1", "alive_dead_f1")}
    truth, pred = pm.possession_spells(ctx.gt, ctx.runs), pm.possession_spells(hard, ctx.runs)
    used_truth, used_pred, starts, ends = set(), set(), [], []
    for overlap, a, b in pm.spell_overlaps(truth, pred):
        if overlap < .5:
            break
        if a not in used_truth and b not in used_pred:
            used_truth.add(a)
            used_pred.add(b)
            starts.append((pred[b][0] - truth[a][0]) / 25)
            ends.append((pred[b][1] - truth[a][1]) / 25)
    truth_alive, predicted_alive = ctx.gt < 2, native_alive >= .5
    tp = int((truth_alive & predicted_alive & known).sum())
    fp = int((~truth_alive & predicted_alive & known).sum())
    fn = int((truth_alive & ~predicted_alive & known).sum())
    strata = {}
    for name, mask in (("hidden", hidden & known), ("observed", ~hidden & known)):
        strata[name] = official_scorer().macro_f1(official_scorer().scored_confusion(
            official_scorer().confusion(ctx.gt[mask], hard[mask]))) if np.any(mask) else None
    return {"oracles_analysis_only": result,
        "oracle_note": "No true team exists during GT D; true-team oracle leaves those false-live errors unchanged",
        "conditional_team_accuracy_gt_alive": float((native_team[truth_alive] == ctx.gt[truth_alive]).mean()),
        "head_alive_F1_at_0_5": 2 * tp / max(1, 2 * tp + fp + fn),
        "macroF1_hidden_frames": strata["hidden"], "macroF1_observed_frames": strata["observed"],
        "matched_start_MAE_s": float(np.mean(np.abs(starts))) if starts else None,
        "matched_end_MAE_s": float(np.mean(np.abs(ends))) if ends else None,
        "matched_endpoint_spells": len(starts)}


def metric_row(metrics, fold, condition, case, decoder, seed, cap, selected_epoch, missing=0., selected=False):
    diagnostics = metrics["diagnostics"]
    boundary = metrics["transitions"][25] if 25 in metrics["transitions"] else metrics["transitions"]["25"]
    result = {"fold": fold, "condition": condition, "scenario": case, "decoder": decoder,
        "seed": seed, "epoch_cap": cap, "selected_epoch": selected_epoch, "validation_selected_pipeline": selected,
        "spellF1": metrics["spell_f1_mean"], "macroF1": metrics["macro_f1"],
        "alive_dead_F1": metrics["alive_dead_f1"], "team_accuracy_true_alive": metrics["team_acc_gt_alive"],
        "gt_spells": metrics["n_gt_spells"], "pred_spells": metrics["n_pred_spells"],
        "fragmented_gt": diagnostics["gt_spells_overlapped_by_multiple_same_team_predictions"],
        "fused_predictions": diagnostics["predicted_spells_overlapping_multiple_same_team_gt"],
        "false_D_interruptions": diagnostics["false_internal_D_interruptions"],
        "false_D_frames": diagnostics["false_dead_frames_on_true_alive"],
        "false_team_changes": diagnostics["incorrect_team_changes_at_1s"],
        "boundary_precision_1s": boundary["precision"], "boundary_recall_1s": boundary["recall"],
        "boundary_MAE_s": boundary["mean_abs_latency"] / 25 if boundary["mean_abs_latency"] is not None else None,
        "actual_hidden_fraction": missing}
    result.update({"f1_iou_" + str(k): v["f1"] for k, v in metrics["spell_f1"].items()})
    return result


def validation_selection(protocol, data, i):
    match = Match.load(data["artifacts"][data["folds"][i]["validation"][0]]["path"])
    validation_cases = [case for case in protocol["corruptions"] if case["name"] in ("clean", "missing40_5s", "missing40_15s")]
    results = {}
    for condition in protocol["conditions"]:
        training = json.loads((directory(i, condition) / "training.json").read_text())
        cap = training["chosen"]["budget"]
        model, state = load_study_model(directory(i, condition), cap)
        view, scenarios = prediction_set(model, match, protocol, validation_cases)
        ctx = ScoreContext(match, view)
        with np.load(directory(i, condition) / "priors.npz", allow_pickle=False) as archive:
            priors = {key: archive[key] for key in archive.files}
        trials, best = [], {}
        for specification in decoder_grid(protocol):
            start = time.perf_counter()
            decoder = TemporalDecoder(specification, priors)
            scores = {name: ctx.spell(values[0], decoder) for name, values in scenarios.items()}
            objective = .5 * scores["clean"] + .25 * scores["missing40_5s"] + .25 * scores["missing40_15s"]
            trial = {"specification": specification, "objective": objective, "scores": scores,
                     "three_validation_decodes_and_scoring_s": time.perf_counter() - start}
            trials.append(trial)
            kind = specification["kind"]
            if kind not in best or objective > best[kind]["objective"]:
                best[kind] = trial
        results[condition] = {"cap": cap, "selected_epoch": state["epoch"], "by_family": best, "trials": trials}
        for name, (p, _, _) in scenarios.items():
            save_probabilities(DEST / "validation_predictions" / ("fold-" + str(i)) / condition / (name + ".npz"), view, p)
        progress("validation_decoders", i, condition=condition)
    winner = max(((condition, family, row) for condition, result in results.items()
                  for family, row in result["by_family"].items()), key=lambda x: x[2]["objective"])
    selection = {"outer_test_used": False, "conditions": results,
                 "winner": {"condition": winner[0], "family": winner[1], **winner[2]}}
    # Persist complete validation choices BEFORE any test match is opened.
    save_json(DEST / "selections" / ("fold-" + str(i) + ".json"), selection)
    return selection


def test_condition(protocol, data, i, condition, selection, seed=2027, fixed_cap=None, families=None):
    train_dir = directory(i, condition, seed)
    cap = selection["conditions"][condition]["cap"] if fixed_cap is None else fixed_cap
    model, state = load_study_model(train_dir, cap)
    key = data["folds"][i]["test"][0]
    match = Match.load(data["artifacts"][key]["path"])
    view, scenarios = prediction_set(model, match, protocol, protocol["corruptions"])
    with np.load(train_dir / "priors.npz", allow_pickle=False) as archive:
        priors = {name: archive[name] for name in archive.files}
    rows = []
    for name, (p, fraction, missing) in scenarios.items():
        output = DEST / "test" / ("seed-" + str(seed)) / ("fold-" + str(i)) / condition / name
        save_probabilities(output / "probabilities.npz", view, p)
        # Native 25Hz probabilities retain input frame/period identities for future decoders.
        native_p = np.zeros((len(match.frame), 3), dtype=np.float32)
        for run, (a, b) in enumerate(segments(match)):
            ix = np.flatnonzero(view.run == run)
            native_p[a:b] = p[ix[nearest_indices(view.time[ix], match.time[a:b])]]
        np.savez_compressed(output / "native_probabilities.npz", probabilities=native_p,
                            frame=match.frame, period=match.period, time=match.time)
        for family, chosen in selection["conditions"][condition]["by_family"].items():
            if families is not None and family not in families:
                continue
            decoder = CachedRunDecoder(TemporalDecoder(chosen["specification"], priors))
            started = time.perf_counter()
            metrics = score_match(match, view, p, decoder)
            diagnostic = extra_diagnostics(match, view, p, decoder, missing)
            metrics.update(diagnostic)
            metrics["decode_and_diagnostics_s"] = time.perf_counter() - started
            save_json(output / (family + ".json"), metrics)
            winner = selection["winner"]
            row = metric_row(metrics, i, condition, name, family, seed, cap, state["epoch"], fraction,
                             condition == winner["condition"] and family == winner["family"])
            row.update({k: v for k, v in diagnostic.items() if k != "oracle_note"})
            rows.append(row)
        progress("outer_test", i, condition=condition, scenario=name, seed=seed)
    return rows


def decode(protocol, data, folds):
    for i in folds:
        target = DEST / "rows" / ("fold-" + str(i) + ".json")
        if target.exists():
            continue
        selection = validation_selection(protocol, data, i)
        rows = []
        for condition in protocol["conditions"]:
            rows += test_condition(protocol, data, i, condition, selection)
        # Prespecified cap comparisons; choice already frozen using validation.
        match = Match.load(data["artifacts"][data["folds"][i]["test"][0]]["path"])
        view = build_features(match)
        duration = json.loads((directory(i, "A_clean") / "training.json").read_text())
        for candidate in duration["caps"]:
            model, state = load_study_model(directory(i, "A_clean"), candidate["budget"])
            p = predict(model, view, 256, torch.device("cpu"))
            save_probabilities(DEST / "durations" / ("fold-" + str(i)) /
                               (str(candidate["budget"]) + "-probabilities.npz"), view, p)
            metrics = score_match(match, view, p)
            save_json(DEST / "durations" / ("fold-" + str(i)) / (str(candidate["budget"]) + ".json"), metrics)
            rows.append(metric_row(metrics, i, "duration_" + str(candidate["budget"]), "clean", "argmax", 2027,
                                   candidate["budget"], state["epoch"]))
        save_json(target, rows)
    progress("decoding_complete", None)


def cache_duration_probabilities(protocol, data, folds):
    """Backfill optional cached emissions without changing any saved selection or score."""
    for i in folds:
        duration = json.loads((directory(i, "A_clean") / "training.json").read_text())
        missing = [c for c in duration["caps"] if not (DEST / "durations" / ("fold-" + str(i)) /
                   (str(c["budget"]) + "-probabilities.npz")).exists()]
        if not missing:
            continue
        match = Match.load(data["artifacts"][data["folds"][i]["test"][0]]["path"])
        view = build_features(match)
        for candidate in missing:
            model, _ = load_study_model(directory(i, "A_clean"), candidate["budget"])
            p = predict(model, view, 256, torch.device("cpu"))
            save_probabilities(DEST / "durations" / ("fold-" + str(i)) /
                               (str(candidate["budget"]) + "-probabilities.npz"), view, p)


def additional_seeds(protocol, data, folds):
    combined, _ = manifest("idsse-skillcorner-phase3")
    for i in folds:
        selection = json.loads((DEST / "selections" / ("fold-" + str(i) + ".json")).read_text())
        condition, family = selection["winner"]["condition"], selection["winner"]["family"]
        cap = selection["conditions"][condition]["cap"]
        source = combined if condition.startswith("B") else data
        for seed in protocol["seeds"][1:]:
            target = DEST / "rows" / ("seed-" + str(seed) + "-fold-" + str(i) + ".json")
            if target.exists():
                continue
            progress("seed_variability", i, seed=seed, condition=condition)
            fit_trajectory(source["artifacts"], source["folds"][i], protocol, directory(i, condition, seed),
                           condition, seed, cap)
            rows = test_condition(protocol, data, i, condition, selection, seed, cap, [family])
            save_json(target, rows)
    progress("seeds_complete", None)


def report(protocol, data):
    rows = []
    for path in sorted((DEST / "rows").glob("*.json")):
        rows.extend(json.loads(path.read_text()))
    for i in range(7):
        path = DEST / "baseline" / ("fold-" + str(i)) / "test/metrics.json"
        if path.exists():
            metrics = next(iter(json.loads(path.read_text()).values()))
            rows.append(metric_row(metrics, i, "E0_2epochs", "clean", "argmax", 2027, 2,
                                   metrics["experiment"]["selected_epoch"]))
    rows.sort(key=lambda row: (row["fold"], row["seed"], row["condition"], row["scenario"], row["decoder"]))
    save_json(DEST / "all_rows.json", rows)
    columns = list(dict.fromkeys(k for row in rows for k in row if k != "oracles_analysis_only"))
    with (DEST / "all_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (DEST / "per_fold").mkdir(exist_ok=True)
    for i in range(7):
        with (DEST / "per_fold" / ("fold-" + str(i) + ".csv")).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(row for row in rows if row["fold"] == i)
    grouped = {}
    for row in rows:
        if row["seed"] == 2027:
            key = (row["condition"], row["scenario"], row["decoder"])
            grouped.setdefault(key, []).append(row)
    summary = []
    for (condition, scenario, decoder), values in sorted(grouped.items()):
        entry = {"condition": condition, "scenario": scenario, "decoder": decoder, "folds": len(values),
            "spellF1_mean": float(np.mean([r["spellF1"] for r in values])),
            "spellF1_sample_SD": float(np.std([r["spellF1"] for r in values], ddof=1)) if len(values) > 1 else None,
            "macroF1_mean": float(np.mean([r["macroF1"] for r in values]))}
        for field in ("macroF1", "alive_dead_F1", "team_accuracy_true_alive", "gt_spells", "pred_spells",
                      "fragmented_gt", "fused_predictions", "false_D_interruptions", "false_team_changes",
                      "boundary_MAE_s", "matched_start_MAE_s", "matched_end_MAE_s"):
            numbers = [r[field] for r in values if r.get(field) is not None]
            entry[field + "_mean"] = float(np.mean(numbers)) if numbers else None
            entry[field + "_sample_SD"] = float(np.std(numbers, ddof=1)) if len(numbers) > 1 else None
        summary.append(entry)
    selected = [row for row in rows if row["validation_selected_pipeline"]]
    selected_summary = []
    for scenario in [case["name"] for case in protocol["corruptions"]]:
        values = [row for row in selected if row["scenario"] == scenario]
        seed_means = [np.mean([r["spellF1"] for r in values if r["seed"] == seed])
                      for seed in protocol["seeds"] if any(r["seed"] == seed for r in values)]
        selected_summary.append({"scenario": scenario, "rows": len(values), "seed_means": seed_means,
            "mean_spellF1": float(np.mean([r["spellF1"] for r in values])) if values else None,
            "seed_2027_match_sample_SD": float(np.std([r["spellF1"] for r in values if r["seed"] == 2027], ddof=1))
                if sum(r["seed"] == 2027 for r in values) > 1 else None,
            "SD_across_seed_means": float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else None})
    paired = []
    rng = np.random.default_rng(2027)
    for control, experiment in (("A_clean", "A_blocks"), ("A_clean", "B_clean"), ("A_blocks", "B_blocks")):
        for scenario in [case["name"] for case in protocol["corruptions"]]:
            a = {r["fold"]: r["spellF1"] for r in rows if r["condition"] == control and r["decoder"] == "argmax"
                 and r["scenario"] == scenario and r["seed"] == 2027}
            b = {r["fold"]: r["spellF1"] for r in rows if r["condition"] == experiment and r["decoder"] == "argmax"
                 and r["scenario"] == scenario and r["seed"] == 2027}
            if len(a) == len(b) == 7:
                diff = np.array([b[i] - a[i] for i in range(7)])
                boot = rng.choice(diff, (5000, 7), replace=True).mean(-1)
                paired.append({"control": control, "experiment": experiment, "scenario": scenario,
                    "mean_delta_spellF1": float(diff.mean()), "paired_match_95CI": np.percentile(boot, [2.5, 97.5]).tolist()})
    decoder_effects = []
    decoder_rng = np.random.default_rng(2028)
    for condition in protocol["conditions"]:
        for scenario in [case["name"] for case in protocol["corruptions"]]:
            for control, experiment in (("argmax", "smooth"), ("argmax", "hmm"), ("hmm", "hsmm")):
                a = {r["fold"]: r["spellF1"] for r in rows if r["condition"] == condition and r["decoder"] == control
                     and r["scenario"] == scenario and r["seed"] == 2027}
                b = {r["fold"]: r["spellF1"] for r in rows if r["condition"] == condition and r["decoder"] == experiment
                     and r["scenario"] == scenario and r["seed"] == 2027}
                if len(a) == len(b) == 7:
                    diff = np.array([b[i] - a[i] for i in range(7)])
                    boot = decoder_rng.choice(diff, (5000, 7), replace=True).mean(-1)
                    decoder_effects.append({"condition": condition, "control": control, "experiment": experiment,
                        "scenario": scenario, "mean_delta_spellF1": float(diff.mean()),
                        "paired_match_95CI": np.percentile(boot, [2.5, 97.5]).tolist()})
    schedules, selections, fold_table = [], [], []
    for i in range(7):
        training_path = directory(i, "A_clean") / "training.json"
        if training_path.exists():
            training = json.loads(training_path.read_text())
            schedules.append({"fold": i, "chosen": training["chosen"], "caps": training["caps"],
                              "epochs_executed": len(training["history"]), "total_s": training["total_s"]})
        selection_path = DEST / "selections" / ("fold-" + str(i) + ".json")
        if selection_path.exists():
            selection = json.loads(selection_path.read_text())
            selections.append({"fold": i, **selection["winner"],
                "cap": selection["conditions"][selection["winner"]["condition"]]["cap"]})
        fold_row = {"fold": i, "test_match": data["folds"][i]["test"][0]}
        for row in rows:
            if row["fold"] != i or row["seed"] != 2027:
                continue
            if row["condition"] == "E0_2epochs":
                fold_row["E0_2epochs_spellF1"] = row["spellF1"]
            if row["condition"] == "A_clean" and row["scenario"] == "clean" and row["decoder"] == "argmax":
                fold_row["trained_argmax_spellF1"] = row["spellF1"]
            if row["validation_selected_pipeline"] and row["scenario"] in ("clean", "missing40_5s", "missing40_15s"):
                fold_row[row["scenario"] + "_selected_spellF1"] = row["spellF1"]
                if row["scenario"] == "clean":
                    fold_row.update(condition=row["condition"], decoder=row["decoder"], macroF1=row["macroF1"],
                        gt_spells=row["gt_spells"], pred_spells=row["pred_spells"],
                        fragmented_gt=row["fragmented_gt"], fused_predictions=row["fused_predictions"],
                        false_D_interruptions=row["false_D_interruptions"],
                        boundary_start_MAE_s=row.get("matched_start_MAE_s"),
                        boundary_end_MAE_s=row.get("matched_end_MAE_s"))
        fold_table.append(fold_row)
    save_json(DEST / "fold_table.json", fold_table)
    document = {"source": "real public external data", "rows": len(rows),
        "configurations": summary, "validation_selected_pipeline": selected_summary, "paired_effects_argmax": paired,
        "paired_effects_decoders": decoder_effects,
        "training_schedules": schedules, "selections": selections,
        "warning": "Outer scores describe prespecified comparisons; per-fold choices use only own inner validation"}
    save_json(DEST / "summary.json", document)
    from scripts.phase3_report import write_report
    write_report(DEST, rows, document, fold_table, protocol)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("baseline", "duration", "conditions", "decode", "seeds", "report", "cache"), required=True)
    parser.add_argument("--fold", type=int, action="append")
    args = parser.parse_args()
    protocol, data = environment()
    folds = args.fold if args.fold is not None else range(7)
    if any(i not in range(7) for i in folds):
        raise ValueError("Invalid fold")
    functions = {"baseline": baseline, "duration": durations, "conditions": conditions,
                 "decode": decode, "seeds": additional_seeds, "cache": cache_duration_probabilities}
    if args.stage == "report":
        report(protocol, data)
    else:
        functions[args.stage](protocol, data, folds)
    return 0
