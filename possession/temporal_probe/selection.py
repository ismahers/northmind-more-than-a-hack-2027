"""Pure own-fold selection and predeclared promotion gates; no test input."""


def objective(metrics, config):
    return sum(w * metrics[name]["spellF1"]
               for name, w in zip(config["scenarios"], config["score_weights"]))


def eligible(metrics, baseline, config):
    gate = config["promotion"]
    return (all(metrics[name]["spellF1"] >= baseline[name]["spellF1"] + gate["occlusion_floor_delta"]
                for name in config["scenarios"][1:])
            and metrics["clean"]["long_recall"] >= baseline["clean"]["long_recall"] + gate["long_recall_floor_delta"])


def choose(trials, baseline, config):
    allowed = [dict(arm="teacher", strength=0., metrics=baseline, objective=objective(baseline, config))]
    for trial in trials:
        if eligible(trial["metrics"], baseline, config):
            allowed.append(dict(trial, objective=objective(trial["metrics"], config)))
    # Teacher comes first: an exact tie conserves the frozen reference.
    return max(allowed, key=lambda value: value["objective"])


def promotion(metrics, baseline, config):
    gate = config["promotion"]
    return (eligible(metrics, baseline, config)
            and objective(metrics, config) >= objective(baseline, config) + gate["robust_gain"]
            and metrics["clean"]["short_recall"] >= baseline["clean"]["short_recall"] + gate["short_recall_gain"])
