"""Aggregate only public external audits/results, never competition data or logs."""

import argparse
import csv
import json

import numpy as np

from possession.external_data.idsse import IDSSE_IDS
from scripts.external_data import REPORTS, write_json


def audit_summary_main():
    rows = []
    for mid in IDSSE_IDS:
        path = REPORTS / "audits" / ("idsse_DFL-MAT-" + mid + ".json")
        record = json.loads(path.read_text(encoding="utf-8"))
        durations = record["possession_durations_s"]
        rows.append({"match": mid, "frames": record["frames"], "duration_min": record["duration_s"] / 60,
            "pitch_m": record["pitch_m"], "roster_sizes": record["roster_sizes"],
            "players_with_coordinates": record["players_with_coordinates_per_team"],
            "players_available_mean": record["players_available_mean_per_team"],
            "ball_available": record["ball_available_fraction"], "joint_coverage": record["joint_coverage"],
            "unknown_frames": record["unknown_joint_frames"], "labels": record["label_counts"],
            "status": record["native_status_counts"], "possession": record["native_possession_counts"],
            "active_possession": record["active_native_possession_counts"],
            "possessions": record["possessions_including_censored"], "censored": record["censored_possessions"],
            "possession_duration_median_s": float(np.median(durations)) if durations else None,
            "possession_duration_max_s": max(durations) if durations else None,
            "transitions": record["transitions"], "runs": record["contiguous_runs"],
            "irregular_intervals": record["irregular_native_intervals"],
            "dt_min": record["within_period_dt_min_s"], "dt_max": record["within_period_dt_max_s"],
            "temporal_error": record["temporal_error"], "issues": record["adapter_issues"],
            "large_geometry_players": record["large_geometry_player_records"],
            "large_geometry_ball": record["large_geometry_ball_records"],
            "invalid_partial_player_coordinates": record["invalid_player_coordinate_records"],
            "negative_ball_heights": record["negative_ball_height_records"]})
    ready = all(r["joint_coverage"] == 1 and not r["temporal_error"] and not r["issues"] for r in rows)
    skillcorner = []
    for path in sorted((REPORTS / "audits").glob("skillcorner_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        skillcorner.append({name: record[name] for name in ("key", "frames", "duration_s", "state_coverage",
            "team_coverage", "joint_coverage", "ball_available_fraction", "ball_detected_fraction",
            "adapter_issues", "temporal_error", "contiguous_runs")})
    write_json(REPORTS / "idsse_audit_summary.json", {"source": "public IDSSE only", "matches": rows,
        "total_frames": sum(r["frames"] for r in rows), "complete_labels_and_clocks": ready,
        "skillcorner_auxiliary": skillcorner,
        "note": "Geometry exceptions require review; finite XY is not independently annotated visual visibility"})
    return 0


def benchmark_summary_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="e0-idsse-cpu-pilot")
    args = parser.parse_args()
    if not args.run or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.run):
        raise ValueError("Invalid run")
    rows = []
    for i in range(7):
        directory = REPORTS / "experiments" / args.run / ("fold-" + str(i))
        path = directory / "test" / "metrics.json"
        row = {"fold": i, "status": "not_run"}
        if path.exists():
            values = json.loads(path.read_text(encoding="utf-8"))
            if len(values) != 1:
                raise ValueError("Expected one held-out IDSSE match per fold")
            key, metrics = next(iter(values.items()))
            latency = metrics["transitions"]["25"]["mean_abs_latency"]
            row.update(status="evaluated", match=key, spellF1=metrics["spell_f1_mean"],
                macroF1=metrics["macro_f1"], gt_spells=metrics["n_gt_spells"], pred_spells=metrics["n_pred_spells"],
                partial_labels=metrics["evaluation_contract"]["partial_annotation_diagnostic"],
                bounded_pilot=metrics["experiment"]["config"].get("max_train_batches") is not None,
                selected_epoch=metrics["experiment"]["selected_epoch"],
                false_D=metrics["diagnostics"]["false_internal_D_interruptions"],
                false_team_changes=metrics["diagnostics"]["incorrect_team_changes_at_1s"],
                fragmentation=metrics["diagnostics"]["gt_spells_overlapped_by_multiple_same_team_predictions"],
                fusion=metrics["diagnostics"]["predicted_spells_overlapping_multiple_same_team_gt"],
                boundary_precision_1s=metrics["transitions"]["25"]["precision"],
                boundary_recall_1s=metrics["transitions"]["25"]["recall"],
                boundary_mean_absolute_error_s=latency / 25 if latency is not None else None,
                ball_coordinates_available_frames=metrics["ball_availability"]["strata"]["coordinates_available"]["frames"],
                ball_coordinates_missing_frames=metrics["ball_availability"]["strata"]["coordinates_missing"]["frames"],
                probability_mean_HAD=metrics["probability_diagnostics"]["mean_HAD"])
            row.update({"f1_iou_" + str(threshold): score["f1"] for threshold, score in metrics["spell_f1"].items()})
        rows.append(row)
    target = REPORTS / "benchmarks" / args.run
    write_json(target.with_suffix(".json"), {"source": "public IDSSE only", "folds": rows,
        "note": "Unrun folds are explicit. A bounded CPU pilot is not a seven-fold benchmark."})
    with target.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        names = list(dict.fromkeys(name for row in rows for name in row))
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    return 0
