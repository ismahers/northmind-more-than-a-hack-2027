"""Pure aggregate diagnostics of frozen geometry, fixed fusion and saved HMM.

Truth is used only in these post-hoc diagnostics, never to modify predictions.
"""
import numpy as np
from possession.football_predictive import predict, combine
from possession.long_diagnosis import GlobalMatcher, matrix


def lift(context, values):
    """Use the SAME 10Hz-to-canonical mapping as the official scoring context."""
    out = np.full(len(context.gt), np.nan)
    for selected, mapping, output in context.parts:
        out[output] = np.asarray(values)[selected][mapping]
    return out


def measures(probability, home_truth, mask, clip=1e-7, weights=None):
    mask = np.asarray(mask, bool)
    count = int(mask.sum())
    if not count:
        return {"frames": 0, "bce": None, "brier": None, "accuracy": None}
    p = np.asarray(probability)[mask]
    y = np.asarray(home_truth, dtype=float)[mask]
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid conditional probabilities")
    w = np.ones(len(p)) if weights is None else np.asarray(weights)[mask]
    if not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Invalid metric weights")
    w = w/w.sum()
    clipped = np.clip(p, clip, 1-clip)
    return {"frames": count, "bce": float(w@(-y*np.log(clipped)-(1-y)*np.log1p(-clipped))),
            "brier": float(w@((p-y)**2)), "accuracy": float(w@((p >= .5) == (y > .5)))}


def paired(a_correct, b_correct, mask):
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    return {"both_correct": int((mask & a & b).sum()), "gain": int((mask & ~a & b).sum()),
            "loss": int((mask & a & ~b).sum()), "both_wrong": int((mask & ~a & ~b).sum())}


def subsets(context, active, spec):
    live = context.gt < 2
    supported = live & np.isfinite(active) & (active > .5)
    near = np.zeros(len(live), bool)
    radius = round(spec["boundary_radius_s"]*25)
    for a, b in context.runs:
        for i, before, after in context.pm.transitions(context.gt, [(a, b)]):
            if before < 2 and after < 2:
                near[max(a, i-radius):min(b, i+radius+1)] = True
    short = np.zeros(len(live), bool)
    for a, b, _ in context.pm.possession_spells(context.gt, context.runs):
        if b-a < spec["short_spell_s"]*25:
            short[a:b] = True
    return {"all_supported_live": supported, "near_HA_boundary": supported & near,
            "interior": supported & ~near, "short_spells": supported & short,
            "all_live": live}


def conditional(p):
    alive = p[:, :2].sum(-1)
    return np.divide(p[:, 0], alive, out=np.full(len(p), .5), where=alive > 1e-12)


def analyze_case(context, data, models, parent_protocol, spec):
    home_truth = context.gt == 0
    direct = {a: predict(models[a], data["x"]) for a in ("proximity", "football")}
    probability = {a: lift(context, p) for a, p in direct.items()}
    probability["four_view"] = lift(context, conditional(data["four_view_v2"]))
    for a in direct:
        combined = combine(data["four_view_v2"], direct[a], data["active"], parent_protocol)
        probability["blend_"+a] = lift(context, conditional(combined))
    groups = subsets(context, lift(context, data["active"]), spec)
    measures_by_group, pairwise = {}, {}
    for group, mask in groups.items():
        # Standalone classifiers have no geometric answer where support is absent.
        names = probability if group != "all_live" else {k: probability[k] for k in
            ("four_view", "blend_proximity", "blend_football")}
        measures_by_group[group] = {name: measures(probability[name], home_truth, mask, spec["probability_clip"])
                                   for name in names}
        if group == "all_live":
            continue
        correct = {name: (p >= .5) == home_truth for name, p in probability.items()}
        pairwise[group] = {"standalone": paired(correct["proximity"], correct["football"], mask),
            "fixed_fusion": paired(correct["blend_proximity"], correct["blend_football"], mask),
            "HMM": paired(data["hard__proximity"] == context.gt, data["hard__football"] == context.gt, mask)}
        for direction, source in (("standalone_gains", ~correct["proximity"] & correct["football"]),
                                  ("standalone_losses", correct["proximity"] & ~correct["football"])):
            population = mask & source
            pairwise[group][direction] = {
                "frames": int(population.sum()),
                "fusion": paired(correct["blend_proximity"], correct["blend_football"], population),
                "HMM": paired(data["hard__proximity"] == context.gt,
                              data["hard__football"] == context.gt, population)}
        pairwise[group]["standalone_complementarity_vs_base"] = {
            a: paired(correct["four_view"], correct[a], mask) for a in direct}
    matcher = GlobalMatcher(context.pm, context.gt)
    a = matcher.score(data["hard__proximity"], verify=True)
    b = matcher.score(data["hard__football"], verify=True)
    return {"groups": measures_by_group, "paired_frame_counts": pairwise,
            "spells": {"short": matrix(a, b, matcher.short), "long": matrix(a, b, matcher.long)},
            "saved_spellF1": {"proximity": a.spell_f1, "football": b.spell_f1},
            "live_frames": int(groups["all_live"].sum()),
            "supported_live_frames": int(groups["all_supported_live"].sum())}


def decision(cases, spec):
    rows = {}
    for group in ("all_supported_live", "near_HA_boundary"):
        values = [(cases[s]["groups"][group]["proximity"]["bce"],
                   cases[s]["groups"][group]["football"]["bce"]) for s in spec["scenarios"]]
        gain = None if any(a is None or b is None for a, b in values) else float(
            np.dot(spec["weights"], [a-b for a, b in values]))
        rows[group] = {"robust_bce_gain": gain,
                       "meets_minimum": gain is not None and gain >= spec["minimum_bce_gain"]}
    rows["additional_signal_in_frozen_classifier"] = all(v["meets_minimum"] for v in rows.values())
    return rows
