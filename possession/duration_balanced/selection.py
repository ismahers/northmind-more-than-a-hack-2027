"""Predeclared gates, relative to original local classifier and frozen teacher."""
import math


def objective(metrics, spec):
    return sum(w*metrics[n]["spellF1"] for n,w in zip(spec["scenarios"], spec["score_weights"]))


def safeguards(metrics, baseline, spec):
    gate = spec["promotion"]
    clean, base = metrics["clean"], baseline["clean"]
    return {
        "occlusion": all(metrics[n]["spellF1"] >= baseline[n]["spellF1"]+gate["occlusion_floor_delta"] for n in spec["scenarios"][1:]),
        "long_recall": clean["long_recall"] >= base["long_recall"]+gate["long_recall_floor_delta"],
        "false_possessions": all(metrics[n]["false_possessions_per_minute"] <= baseline[n]["false_possessions_per_minute"]+gate["false_possessions_per_minute_increase_max"] for n in spec["scenarios"]),
        "fragmentation": all(metrics[n]["fragmented_gt"] <= baseline[n]["fragmented_gt"]+
            max(gate["fragmentation_increase_absolute_floor"],
                math.ceil(baseline[n]["fragmented_gt"]*gate["fragmentation_increase_relative_max"]))
            for n in spec["scenarios"])}


def promotion(metrics, original, teacher, spec):
    gate = spec["promotion"]
    checks = {"robust_gain": objective(metrics,spec) >= objective(original,spec)+gate["robust_gain_over_original"],
              "short_recall_gain": metrics["clean"]["short_recall"] >= original["clean"]["short_recall"]+gate["short_recall_gain_over_original"]}
    checks.update({"vs_original_"+k:v for k,v in safeguards(metrics, original, spec).items()})
    checks.update({"vs_teacher_"+k:v for k,v in safeguards(metrics, teacher, spec).items()})
    return {"passed":all(checks.values()), "checks":checks}


def select(trials, teacher, spec):
    original = trials["original"]
    gates = {a:promotion(m,original,teacher,spec) for a,m in trials.items() if a!="original"}
    allowed = ["original"]+[a for a,g in gates.items() if g["passed"]]
    winner = max(allowed, key=lambda a:objective(trials[a],spec))
    return {"winner":winner, "gates":gates, "test_used":False}
