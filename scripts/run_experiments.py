"""Generate and evaluate Northmind candidates using real tactical tracking.

python -m scripts.run_local evaluate-tracking
The launcher captures all worker output locally and displays only OK/FAILED.
Raw assets and extracted per-frame features are never printed or exported.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evaluation" / "reports" / "northmind-tracking-v1"


def run(with_tagging=False, sequence=False):
    import numpy as np

    from evaluation import possession_metrics as pm
    from possession.inference import Config, predict
    from possession.tracking_features import load_evidence
    from starter.baseline import VARIANTS, smooth, write_submission

    config = Config()
    if sequence and not with_tagging:
        raise ValueError("Sequence experiment requires the fixed tracking-plus-tagging reference")
    output = (ROOT / "evaluation" / "reports" / "sequence-model-v2" if sequence else
              ROOT / "evaluation" / "reports" / "northmind-tracking-tagging-v2"
              if with_tagging else OUTPUT)
    print("Loading tactical tracking; restricted input contents will not be displayed.", flush=True)
    evidence = load_evidence()
    reference = ROOT / "evaluation" / "results" / "baseline_raw.csv"
    expected = pm.read_label_csv(reference)
    reason = pm.validate_submission(expected, len(evidence.nearest_codes))
    if reason:
        raise ValueError("Invalid public reference CSV")
    raw = smooth(evidence.nearest_codes, 0)
    mismatches = int(np.count_nonzero(pm.label_codes(raw) != pm.label_codes(expected)))
    if mismatches:
        raise ValueError("Feature loader does not reproduce reference raw predictions")
    print("Feature loader: 0 differing raw baseline labels; exact frame order verified.", flush=True)
    candidates, aggregate = predict(evidence, config)
    if with_tagging:
        from possession.tactical_tagging import load_team_prior
        prior, tagging_aggregate = load_team_prior()
        tagged, tagged_aggregate = predict(evidence, config, team_prior=prior)
        for name in ("northmind_weighted", "northmind_geometry"):
            if not np.array_equal(candidates[name] == "D", tagged[name] == "D"):
                raise ValueError("Team-only tagging must not change dead-ball decisions")
        candidates["northmind_tagged"] = tagged["northmind_weighted"]
        candidates["northmind_tagged_geometry"] = tagged["northmind_geometry"]
        aggregate["tagging"] = tagging_aggregate
        aggregate["tagged_extra_dead_ball_frames"] = tagged_aggregate["extra_dead_ball_frames"]
        aggregate["tagging_preserves_dead_ball_mask"] = True
    sequence_config = None
    if sequence:
        from possession.sequence_decoding import SequenceConfig, predict_sequence
        sequence_config = SequenceConfig()
        decoded, sequence_checks = predict_sequence(
            evidence, candidates["northmind_tagged_geometry"], prior, config, sequence_config)
        if not sequence_checks["uses_equivariant_st_tcn"]:
            raise ValueError("Sequence v2 requires compute-target-st-tcn first")
        candidates.update(decoded)
        aggregate["sequence"] = sequence_checks
    submissions = output / "submissions"
    submissions.mkdir(parents=True, exist_ok=True)
    manifest = {"config": asdict(config), "source": "tactical ATD",
                "tagging_requires_tracking": True,
                "uses_gt_for_inference": False, "raw_baseline_label_differences": mismatches,
                "aggregate": aggregate, "predictions": {}}
    if sequence:
        manifest["sequence_config"] = asdict(sequence_config)
        manifest["parameter_search"] = False
    for name, labels in candidates.items():
        path = submissions / (name + ".csv")
        write_submission(str(path), labels)
        table = pm.read_label_csv(path)
        if pm.validate_submission(table, len(evidence.nearest_codes)):
            raise ValueError("Generated prediction does not satisfy the submission contract")
        manifest["predictions"][name] = {
            "sha256_lf": hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "label_counts": {label: int(np.count_nonzero(labels == label)) for label in pm.LABELS},
        }
        print("{}: {} rows, submission format valid.".format(name, len(labels)), flush=True)
    if sequence:
        previous_path = ROOT / "evaluation" / "reports" / "northmind-tracking-tagging-v2" / "experiment.json"
        if previous_path.is_file():
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
            if previous["config"] != manifest["config"]:
                raise ValueError("Previous reference uses different geometric parameters")
            for name in ("northmind_weighted", "northmind_geometry", "northmind_tagged", "northmind_tagged_geometry"):
                if previous["predictions"][name]["sha256_lf"] != manifest["predictions"][name]["sha256_lf"]:
                    raise ValueError("The sequence experiment changed an existing reference prediction")
            aggregate["previous_reference_hashes_preserved"] = True
    # Only these four named public files are used for baseline comparison.
    for name, _ in VARIANTS:
        shutil.copyfile(ROOT / "evaluation" / "results" / (name + ".csv"),
                        submissions / (name + ".csv"))
    (output / "experiment.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Public labels are used for evaluation only after the predictions exist.
    subprocess.run([
        sys.executable, str(ROOT / "evaluation" / "run_evaluation.py"),
        "--gt", str(ROOT / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"),
        "--results", str(submissions), "--out", str(output),
        "--bootstrap", "1000", "--no-detail",
    ], check=True)
    if sequence:
        # Compare locally AFTER inference and official evaluation. This report
        # is never printed or used for parameter tuning / automatic promotion.
        baseline = json.loads((output / "reports" / "baseline_smooth50.json").read_text())
        reference_result = json.loads((output / "reports" / "northmind_tagged_geometry.json").read_text())
        comparison = {"public_window_only": True, "automatic_model_selection": False,
                      "reference": "northmind_tagged_geometry", "candidates": {}}
        for name in ("northmind_sequence", "northmind_sequence_bridged"):
            result = json.loads((output / "reports" / (name + ".json")).read_text())
            comparison["candidates"][name] = {
                "spell_f1_mean": result["spell_f1_mean"], "macro_f1": result["macro_f1"],
                "spell_delta_vs_reference": result["spell_f1_mean"] - reference_result["spell_f1_mean"],
                "spell_delta_vs_smooth50": result["spell_f1_mean"] - baseline["spell_f1_mean"],
                "spell_bootstrap_5_95": [result["spell_bootstrap_low"], result["spell_bootstrap_high"]],
                "n_pred_spells": result["n_pred_spells"],
            }
        # Preserve the prior run and compare only after inference/evaluation.
        previous_folder = ROOT / "evaluation/reports/sequence-model-v1/reports"
        comparison["previous_sequence_run"] = {}
        for name in ("northmind_sequence", "northmind_sequence_bridged"):
            previous_report = previous_folder / (name + ".json")
            if previous_report.is_file():
                old = json.loads(previous_report.read_text(encoding="utf-8"))
                new = comparison["candidates"][name]
                comparison["previous_sequence_run"][name] = {
                    "spell_delta": new["spell_f1_mean"] - old["spell_f1_mean"],
                    "macro_delta": new["macro_f1"] - old["macro_f1"],
                    "diagnostic_only": True,
                }

        (output / "comparison.json").write_text(
            json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--with-tagging", action="store_true",
                        help="also evaluate bounded automatic tactical team evidence")
    parser.add_argument("--sequence", action="store_true",
                        help="evaluate fixed sequence decoding and conservative missing-stop bridging")
    args = parser.parse_args()
    try:
        run(with_tagging=args.with_tagging or args.sequence, sequence=args.sequence)
        return 0
    except Exception as error:
        print("Northmind run failed ({}); input contents suppressed.".format(
            type(error).__name__), file=sys.stderr)
        return 2


if __name__ == "__main__":
    from scripts.run_local import run_task
    sys.exit(run_task("evaluate-tracking", sys.argv[1:]))
