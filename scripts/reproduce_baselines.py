"""Regenerate official baselines without printing any restricted asset contents.

Run from the repository root: python -m scripts.run_local reproduce-baselines
The launcher captures comparisons locally and displays only OK/FAILED.
Original baseline CSVs stay intact.
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evaluation" / "reports" / "reproduced-baselines"


def reproduce():
    import numpy as np

    from evaluation import possession_metrics as pm
    from games import GAME, GAMES, require, source_frames
    from starter.baseline import (
        VARIANTS, nearest_player_labels, smooth, team_of_each_track, write_submission,
    )

    meta, raw = require(["atd_meta", "atd_raw"])
    codes = nearest_player_labels(raw, team_of_each_track(meta),
                                  source_frames(), GAMES[GAME]["pitch"])
    submissions = OUTPUT / "submissions"
    submissions.mkdir(parents=True, exist_ok=True)
    report = {"n_frames": len(codes), "baselines": {}}
    for name, window in VARIANTS:
        labels = smooth(codes, window)
        path = submissions / (name + ".csv")
        write_submission(str(path), labels)
        reference = ROOT / "evaluation" / "results" / path.name
        table = pm.read_label_csv(reference)
        reason = pm.validate_submission(table, len(labels))
        if reason:
            raise ValueError("Invalid public baseline")
        differences = int(np.count_nonzero(pm.label_codes(labels) != pm.label_codes(table)))
        # Canonicalize line endings: Windows and Unix may serialize the same
        # file differently while preserving every frame and label exactly.
        digest = lambda p: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        report["baselines"][name] = {
            "different_labels": differences,
            "canonical_csv_matches": digest(path) == digest(reference),
            "prediction_sha256_lf": digest(path),
        }
        print("{}: {} differing labels; normalized CSV identical: {}".format(
            name, differences, report["baselines"][name]["canonical_csv_matches"]), flush=True)
    report["all_match"] = all(
        item["different_labels"] == 0 and item["canonical_csv_matches"]
        for item in report["baselines"].values())
    (OUTPUT / "reproduction.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main():
    try:
        return 0 if reproduce()["all_match"] else 1
    except Exception as error:
        # Parser exceptions can contain a raw input token. Never print them.
        print("Baseline reproduction failed ({}); input contents suppressed.".format(
            type(error).__name__), file=sys.stderr)
        return 2


if __name__ == "__main__":
    from scripts.run_local import run_task
    sys.exit(run_task("reproduce-baselines", sys.argv[1:]))
