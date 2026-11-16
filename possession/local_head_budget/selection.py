"""Own validation only, previous safeguards plus explicitly typed HA events."""
from possession.duration_balanced.selection import objective, promotion


def choose(trials, teacher, spec):
    original = trials["8"]
    gates = {}
    for cap in spec["budgets"][1:]:
        name = str(cap)
        check = promotion(trials[name], original, teacher, spec)
        check["checks"]["direct_HA_false_transitions"] = all(
            trials[name][scenario]["false_transitions_per_minute"] <=
            original[scenario]["false_transitions_per_minute"] + spec["promotion"]["false_transitions_per_minute_increase_max"]
            for scenario in spec["scenarios"])
        check["passed"] = all(check["checks"].values())
        gates[name] = check
    eligible = ["8"] + [str(cap) for cap in spec["budgets"][1:] if gates[str(cap)]["passed"]]
    # Stable cap order gives the smaller budget priority in ties.
    winner = max(eligible, key=lambda name: objective(trials[name], spec))
    best = max([str(cap) for cap in spec["budgets"]], key=lambda name: objective(trials[name], spec))
    return {"winner": int(winner), "unconstrained_best": int(best), "gates": gates,
            "objectives": {str(cap): objective(trials[str(cap)], spec) for cap in spec["budgets"]},
            "test_used": False}
