"""Use the official scorer unchanged on a 25Hz timeline; never open competition GT."""

from functools import lru_cache
import importlib.util
from pathlib import Path

import numpy as np

from possession.external_data.audit import joint_labels, segments
from possession.external_data.features import nearest_indices
from possession.external_data.schema import file_hash
from .decoding import ArgmaxDecoder, validate_probabilities

SCORER_PATH = Path(__file__).resolve().parents[2] / "evaluation" / "possession_metrics.py"


@lru_cache(maxsize=1)
def official_scorer():
    spec = importlib.util.spec_from_file_location("northmind_official_metrics", SCORER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_match(match, features, probabilities, decoder=None, require_complete=True):
    probabilities = validate_probabilities(probabilities)
    if len(probabilities) != len(features.time):
        raise ValueError("Predictions and model timeline differ")
    decoder = decoder or ArgmaxDecoder()
    pm = official_scorer()
    native_labels = joint_labels(match)
    gt_chunks, pred_chunks, available_chunks = [], [], []
    for run_id, (a, b) in enumerate(segments(match)):
        times = match.time[a] + np.arange(int(round((match.time[b - 1] - match.time[a]) * 25)) + 1) / 25
        native_ix = a + nearest_indices(match.time[a:b], times)
        model_ix = np.flatnonzero(features.run == run_id)
        if not len(model_ix):
            raise ValueError("Missing model run")
        # Decode each run independently, then map its hard labels to the canonical grid.
        decoded = decoder.decode(probabilities[model_ix], features.time[model_ix])
        pred = decoded[nearest_indices(features.time[model_ix], times)]
        labels = native_labels[native_ix].copy()
        if require_complete and np.any(labels < 0):
            raise ValueError("Incomplete H/A/D: audit partial labels before primary evaluation")
        labels[labels < 0] = 3
        # The scorer's boundary matcher relies on gaps exceeding its tolerances.
        # One X splits spells but can still permit boundary matches across halves.
        separator = np.full(max(pm.TAUS) + 1, 3, dtype=np.int8)
        gt_chunks.extend((labels, separator))
        pred_chunks.extend((pred, separator))
        available_chunks.extend((match.ball_available[native_ix], np.zeros(len(separator), dtype=bool)))
    gt, pred = np.concatenate(gt_chunks), np.concatenate(pred_chunks)
    available = np.concatenate(available_chunks)
    if not np.any(gt != 3):
        raise ValueError("No jointly labeled H/A/D frames to score")
    result = pm.evaluate(gt, pred, bootstrap_n=0)
    runs = pm.valid_runs(gt)
    truth, predicted = pm.possession_spells(gt, runs), pm.possession_spells(pred, runs)
    overlaps = pm.spell_overlaps(truth, predicted)
    truth_degree, pred_degree = np.zeros(len(truth), dtype=int), np.zeros(len(predicted), dtype=int)
    for _, i, j in overlaps:
        truth_degree[i] += 1
        pred_degree[j] += 1
    result["diagnostics"] = {
        "gt_spells_overlapped_by_multiple_same_team_predictions": int((truth_degree > 1).sum()),
        "predicted_spells_overlapping_multiple_same_team_gt": int((pred_degree > 1).sum()),
        "note": "overlap diagnostics, not causal error attribution; see official one-to-one scores",
    }
    false_dead_frames = int(((gt < 2) & (pred == 2)).sum())
    interruptions = 0
    for a, b, team in truth:
        for left, right, state in pm.segments(pred, [(a, b)]):
            if state == 2 and left > a and right < b and pred[left - 1] == team and pred[right] == team:
                interruptions += 1
    gt_switches = [x for x in pm.transitions(gt, runs) if x[1] < 2 and x[2] < 2]
    pred_switches = [x for x in pm.transitions(pred, runs) if x[1] < 2 and x[2] < 2]
    _, _, incorrect_switches = pm.match_transitions(gt_switches, pred_switches, 25, typed="pair")
    result["diagnostics"].update(false_dead_frames_on_true_alive=false_dead_frames,
        false_internal_D_interruptions=interruptions, incorrect_team_changes_at_1s=len(incorrect_switches),
        team_change_tolerance_frames=25)
    availability = {}
    for name, selected in (("coordinates_available", available), ("coordinates_missing", ~available)):
        selected = selected & (gt != 3)
        if np.any(selected):
            cm = pm.scored_confusion(pm.confusion(gt[selected], pred[selected]))
            availability[name] = {"frames": int(selected.sum()), "macro_f1": pm.macro_f1(cm),
                                  "accuracy": pm.accuracy(cm)}
        else:
            availability[name] = {"frames": 0, "macro_f1": None, "accuracy": None}
    result["ball_availability"] = {"strata": availability,
        "visual_detection_known_fraction": float(match.ball_detection_known.mean()),
        "semantics": "finite XY availability; visual visibility is not annotated by IDSSE",
        "note": "Frame metrics only; masking visibility intervals would artificially split possession spells"}
    flags = pm.matched_gt_flags(overlaps, len(truth), 0.5)
    result["per_possession"] = [{"start_canonical_frame": a, "stop_canonical_frame": b,
        "team": "HA"[team], "duration_s": (b - a) / 25, "matched_at_iou_0_5": bool(flags[i]),
        "same_team_prediction_overlaps": int(truth_degree[i]),
        "ball_coordinate_availability": float(available[a:b].mean()),
        "false_dead_frames": int((pred[a:b] == 2).sum()),
        "wrong_team_frames": int((pred[a:b] == 1 - team).sum())}
        for i, (a, b, team) in enumerate(truth)]
    result["evaluation_contract"] = {
        "match_key": match.key, "fps": 25, "scorer_sha256": file_hash(SCORER_PATH),
        "joint_native_coverage": float((native_labels >= 0).mean()),
        "partial_annotation_diagnostic": bool(np.any(native_labels < 0)),
        "warning": "Unknown intervals censor spells; partial scores cannot be compared with complete H/A/D scores",
    }
    return result
