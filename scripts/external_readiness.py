"""Real external-data leakage and temporal checks; labels are public IDSSE only."""

from types import SimpleNamespace

import numpy as np

from possession.external_data.audit import joint_labels, segments
from possession.external_data.features import build_features
from possession.external_data.schema import Match, file_hash
from possession.learning.data import Windows
from possession.learning.metrics import SCORER_PATH, score_match
from scripts.external_data import REPORTS, write_json
from scripts.train_e0 import manifest


def main():
    document, paths = manifest("idsse-only")
    rows = []
    for path in paths:
        match = Match.load(path)
        view = build_features(match, 10)
        original_alive, original_team = match.alive.copy(), match.team.copy()
        np.testing.assert_array_equal(view.alive, original_alive[view.native_index])
        np.testing.assert_array_equal(view.team, original_team[view.native_index])
        temporal_error = float(np.max(np.abs(view.time - match.time[view.native_index])))
        if temporal_error > 0.020001:
            raise ValueError("Unexpected 25Hz->10Hz timestamp misalignment")
        for run in np.unique(view.run):
            if not np.all(np.diff(view.native_index[view.run == run]) > 0):
                raise ValueError("Resampling reordered or duplicated native frames")
        coverage = np.zeros(len(view.time), dtype=int)
        for item in Windows({match.key: view}, 256, 15):
            coverage[item["a"]:item["b"]] += 1
            a, b = item["left"], item["left"] + len(item["nodes"])
            if len(set(view.run[a:b])) != 1:
                raise ValueError("Window spans a period/gap")
        np.testing.assert_array_equal(coverage, 1)
        labels = joint_labels(match)
        native_run = np.zeros(len(match.frame), dtype=np.int32)
        for run_id, (a, b) in enumerate(segments(match)):
            native_run[a:b] = run_id
        oracle_native = np.eye(3, dtype=np.float32)[labels]
        native_timeline = SimpleNamespace(time=match.time, run=native_run)
        sanity = score_match(match, native_timeline, oracle_native)
        if sanity["spell_f1_mean"] != 1 or sanity["macro_f1"] != 1:
            raise ValueError("Official evaluator/native timeline sanity failed")
        sampled = score_match(match, view, np.eye(3, dtype=np.float32)[labels[view.native_index]])
        # In-memory perturbation only. Never write mutated labels or the source arrays.
        match.alive[:] = -1
        match.team[:] = -1
        match.confidence[:] = 0
        match.native_status[:] = "x"
        match.native_possession[:] = "x"
        match.provenance[:] = "changed-targets"
        changed = build_features(match, 10)
        for name in ("nodes", "node_mask", "global_features", "time", "run", "native_index"):
            np.testing.assert_array_equal(getattr(view, name), getattr(changed, name))
        rows.append({"key": match.key, "model_frames_10hz": len(view.time),
            "max_native_sampling_offset_s": temporal_error, "windows_cover_once": True,
            "target_mutation_does_not_change_inputs": True, "native_identity_sanity_spellF1": sanity["spell_f1_mean"],
            "native_identity_sanity_macroF1": sanity["macro_f1"],
            "oracle_sampled_10hz_spellF1": sampled["spell_f1_mean"],
            "oracle_sampled_10hz_macroF1": sampled["macro_f1"],
            "note": "Oracles verify data plumbing/resampling only; these are not trained-model results"})
        del match, view, changed
    for fold in document["folds"]:
        groups = [set(fold[name]) for name in ("train", "validation", "test")]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError("Leaked match split")
    write_json(REPORTS / "readiness.json", {"source": "public IDSSE only", "passed": True,
        "scorer_sha256": file_hash(SCORER_PATH), "matches": rows,
        "folds": document["folds"], "normalization": "fixed pitch scaling; no fitted statistics",
        "training_test_access": "metadata and integrity hash only; no test arrays loaded by trainer",
        "events_as_features": False, "pretrained": False})
    return 0
