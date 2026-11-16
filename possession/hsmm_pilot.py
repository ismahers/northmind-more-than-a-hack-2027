"""Frozen-decoder comparison and portable public replay, without raw inputs."""
from types import SimpleNamespace
import numpy as np

from possession.learning.study import ScoreContext
from possession.learning.metrics import official_scorer
from possession.learning.temporal import TemporalDecoder
from possession.phase5.metrics import diagnostics
from possession.local_head_budget.metrics import transition_counts
from possession.long_diagnosis import GlobalMatcher, matrix


def save_context(path, context):
    arrays = {"gt": context.gt, "time": context.view.time, "run": context.view.run,
              "parts": np.asarray(len(context.parts))}
    for i, (selected, mapping, output) in enumerate(context.parts):
        arrays[f"selected_{i}"] = selected
        arrays[f"mapping_{i}"] = mapping
        arrays[f"output_{i}"] = np.asarray([output.start, output.stop])
    np.savez_compressed(path, **arrays)


def load_context(path):
    # No paths, tracking geometry, teams or training labels are needed to replay.
    context = ScoreContext.__new__(ScoreContext)
    context.pm = official_scorer()
    with np.load(path, allow_pickle=False) as z:
        context.gt = z["gt"].copy()
        context.view = SimpleNamespace(time=z["time"].copy(), run=z["run"].copy())
        context.parts = [(z[f"selected_{i}"].copy(), z[f"mapping_{i}"].copy(),
            slice(*map(int, z[f"output_{i}"]))) for i in range(int(z["parts"]))]
    context.runs = context.pm.valid_runs(context.gt)
    return context


def score(context, hard):
    full = context.pm.evaluate(context.gt, hard, bootstrap_n=0)
    groups = diagnostics(context, hard)["duration_groups"]
    short_gt = sum(groups[g]["gt"] for g in ("lt1", "1to3"))
    short = sum(groups[g]["iou"]["0.5"]["matched_gt"] for g in ("lt1", "1to3"))
    return {"spellF1": float(context.pm.spell_scores(context.gt, hard, context.runs)["mean_f1"]),
        "macroF1": float(full["macro_f1"]), "short_recall": short/max(1, short_gt),
        "short_recovered": short, "short_gt": short_gt,
        "long_recall": groups["ge10"]["iou"]["0.5"]["recall"],
        "false_transitions_per_minute": transition_counts(context, hard)["direct_HA"]["false_per_minute"]}


def compare(context, prior, probabilities, protocol):
    specs = {"registered_multiseed": protocol["control_decoder"],
             "four_view_v2": protocol["control_decoder"], "four_view_hsmm": protocol["decoder"]}
    predictions, metrics = {}, {}
    for name, spec in specs.items():
        source = "four_view_v2" if name == "four_view_hsmm" else name
        hard = context.decoded(probabilities[source], TemporalDecoder(spec, prior))
        predictions[name] = hard
        metrics[name] = score(context, hard)
    matcher = GlobalMatcher(context.pm, context.gt)
    candidate = matcher.score(predictions["four_view_hsmm"], verify=True)
    changes = {}
    for ref in protocol["references"]:
        baseline = matcher.score(predictions[ref], verify=True)
        changes[ref] = {"long": matrix(baseline, candidate, matcher.long),
                        "short": matrix(baseline, candidate, matcher.short)}
    return metrics, predictions, changes
