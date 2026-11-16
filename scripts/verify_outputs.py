"""Check real-run outputs without displaying any frame-level content.

After reproducing baselines and both experiments: python -m scripts.run_local verify-outputs
No restricted input files are opened by this verification command.
The launcher captures verification output locally and displays only OK/FAILED.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "evaluation" / "reports"


def digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def check(sequence=False):
    import numpy as np

    from evaluation import possession_metrics as pm
    from games import GAME, GAMES, half_video_blocks
    from starter.baseline import VARIANTS

    expected_rows = GAMES[GAME]["n_video_frames"]
    inside = np.zeros(expected_rows, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        inside[first:last + 1] = True
    for name, _ in VARIANTS:
        original = ROOT / "evaluation" / "results" / (name + ".csv")
        rebuilt = REPORTS / "reproduced-baselines" / "submissions" / original.name
        if digest(original) != digest(rebuilt):
            raise ValueError("Rebuilt baseline differs from the public original")

    for folder, names in (
        ("northmind-tracking-v1", ("northmind_weighted", "northmind_geometry")),
        ("northmind-tracking-tagging-v2", (
            "northmind_weighted", "northmind_geometry", "northmind_tagged", "northmind_tagged_geometry")),
    ):
        manifest = json.loads((REPORTS / folder / "experiment.json").read_text())
        labels = {}
        for name in names:
            path = REPORTS / folder / "submissions" / (name + ".csv")
            table = pm.read_label_csv(path)
            if pm.validate_submission(table, expected_rows):
                raise ValueError("Invalid generated submission")
            labels[name] = pm.label_codes(table)
            if np.any(labels[name][~inside] != pm.X):
                raise ValueError("Predictions outside public half boundaries must be X")
            if digest(path) != manifest["predictions"][name]["sha256_lf"]:
                raise ValueError("Prediction no longer matches its recorded hash")
        for base, geometry in (("northmind_weighted", "northmind_geometry"),
                               ("northmind_tagged", "northmind_tagged_geometry")):
            if base not in labels:
                continue
            changed = labels[base] != labels[geometry]
            if np.any(labels[geometry][changed] != pm.D):
                raise ValueError("Geometry variant may only add dead-ball evidence")
        if "northmind_tagged" in labels:
            for plain, tagged in (("northmind_weighted", "northmind_tagged"),
                                   ("northmind_geometry", "northmind_tagged_geometry")):
                if not np.array_equal(labels[plain] == pm.D, labels[tagged] == pm.D):
                    raise ValueError("Tagging changed the dead-ball mask")

    for name in ("northmind_weighted", "northmind_geometry"):
        left = REPORTS / "northmind-tracking-v1" / "submissions" / (name + ".csv")
        right = REPORTS / "northmind-tracking-tagging-v2" / "submissions" / (name + ".csv")
        if digest(left) != digest(right):
            raise ValueError("Independent tracking runs are not reproducible")
    if sequence:
        folder = REPORTS / "sequence-model-v2"
        manifest = json.loads((folder / "experiment.json").read_text())
        names = ("northmind_weighted", "northmind_geometry", "northmind_tagged",
                 "northmind_tagged_geometry", "northmind_sequence", "northmind_sequence_bridged")
        labels = {}
        for name in names:
            path = folder / "submissions" / (name + ".csv")
            table = pm.read_label_csv(path)
            if pm.validate_submission(table, expected_rows):
                raise ValueError("Invalid sequence experiment submission")
            labels[name] = pm.label_codes(table)
            if np.any(labels[name][~inside] != pm.X) or np.any(labels[name][inside] == pm.X):
                raise ValueError("Sequence experiment does not respect the public half boundaries")
            if digest(path) != manifest["predictions"][name]["sha256_lf"]:
                raise ValueError("Sequence submission does not match its recorded hash")
        for name in names[:4]:
            if digest(folder / "submissions" / (name + ".csv")) != digest(
                    REPORTS / "northmind-tracking-tagging-v2" / "submissions" / (name + ".csv")):
                raise ValueError("Sequence work changed an existing model")
        reference_dead = labels["northmind_tagged_geometry"] == pm.D
        if not np.array_equal(labels["northmind_sequence"] == pm.D, reference_dead):
            raise ValueError("Control-only decoding changed the dead mask")
        if np.any((labels["northmind_sequence_bridged"] == pm.D) & ~reference_dead):
            raise ValueError("Bridging introduced a new stopping decision")
        if not manifest["aggregate"]["sequence"]["protected_geometry_preserved"]:
            raise ValueError("Protected geometric stopping evidence was not verified")
        if not manifest["aggregate"]["sequence"]["uses_equivariant_st_tcn"]:
            raise ValueError("Sequence v2 must include the verified neural ensemble")
        from possession.st_tcn_target import _read_cache
        fingerprint = manifest["aggregate"]["sequence"]["neural_cache_fingerprint"]
        if _read_cache(fingerprint, expected_rows) is None:
            raise ValueError("Sequence output and neural cache provenance differ")

        comparison = json.loads((folder / "comparison.json").read_text())
        if comparison["automatic_model_selection"] or not comparison["public_window_only"]:
            raise ValueError("Public comparison must not select or tune the model")
    print("Checks passed: exact baseline reproduction, submission contracts, half boundaries,")
    print("recorded hashes, repeatable tracking outputs, geometry changes and team-only tagging.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sequence", action="store_true")
    args = parser.parse_args()
    try:
        check(sequence=args.sequence)
        return 0
    except Exception as error:
        print("Output verification failed ({}); file contents suppressed.".format(
            type(error).__name__), file=sys.stderr)
        return 1


if __name__ == "__main__":
    from scripts.run_local import run_task
    sys.exit(run_task("verify-outputs", sys.argv[1:]))
