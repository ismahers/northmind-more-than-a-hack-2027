"""Official spell matching plus explicit HA versus D transition diagnostics."""
import numpy as np

from possession.duration_balanced.metrics import measure, analyze, hard_head, matched
from possession.phase6.decoding import DirectionalDecoder


def transition_counts(context, hard):
    pm = context.pm
    truth = pm.transitions(context.gt, context.runs)
    predicted = pm.transitions(hard, context.runs)
    minutes = sum(b-a for a, b in context.runs)/25/60
    result = {}
    for name, predicate in (("all", lambda e: True),
                            ("direct_HA", lambda e: e[1] < 2 and e[2] < 2),
                            ("D_related", lambda e: e[1] == 2 or e[2] == 2)):
        gt, pred = [e for e in truth if predicate(e)], [e for e in predicted if predicate(e)]
        pairs, missed, false = pm.match_transitions(gt, pred, 12, typed="pair")
        result[name] = {"tp": len(pairs), "fp": len(false), "fn": len(missed),
                        "false_per_minute": len(false)/max(minutes, 1e-9)}
    return result


def measured(context, p, prior):
    result = measure(context, p, prior)
    hard = context.decoded(p, DirectionalDecoder(context.view, prior, "hmm", .5))
    result["transitions"] = transition_counts(context, hard)
    result["false_transitions_per_minute"] = result["transitions"]["direct_HA"]["false_per_minute"]
    return result


def short_score(context, hard):
    truth, _, flags = matched(context, hard)
    selected = np.array([(b-a)/25 < 3 for a, b, _ in truth])
    return {"spellF1": float(context.pm.spell_scores(context.gt, hard, context.runs)["mean_f1"]),
            "short_recovered": int(flags[selected].sum()), "short_gt": int(selected.sum())}


def state_metrics(context, hard):
    known = context.gt < 3
    true_D, pred_D = context.gt[known] == 2, hard[known] == 2
    tp = int((true_D & pred_D).sum()); fp = int((~true_D & pred_D).sum()); fn = int((true_D & ~pred_D).sum())
    return {"D_precision": tp/max(1, tp+fp), "D_recall": tp/max(1, tp+fn),
            "D_F1": 2*tp/max(1, 2*tp+fp+fn),
            "false_D_fraction_of_live": fp/max(1, int((~true_D).sum())),
            "false_live_fraction_of_D": fn/max(1, int(true_D.sum()))}


def diagnose_components(context, p, teacher_p, prior):
    raw = hard_head(context, p)
    decoder = DirectionalDecoder(context.view, prior, "hmm", .5)
    hmm = context.decoded(p, decoder)
    detail = analyze(context, p, teacher_p, prior)
    short = [r for r in detail["per_possession"] if r["duration_s"] < 3]
    # Diagnostic labels are used only after both validation selections are sealed.
    # Keep team probabilities conditional, and reuse exactly the same HMM/prior.
    alive = p[:, :2].sum(1)
    conditional = np.divide(p[:, 0], alive, out=np.full(len(p), .5), where=alive > 1e-8)
    true_alive = context.view.alive.astype(np.float32)
    if not np.isin(true_alive, [0, 1]).all():
        raise ValueError("Oracle requires complete public alive labels")
    true_home = np.where(context.view.team >= 0, context.view.team == 0, conditional).astype(np.float32)
    def joint(a, h):
        return np.column_stack((a*h, a*(1-h), 1-a)).astype(np.float32)
    oracles = {"perfect_alive_10hz_same_HMM": joint(true_alive, conditional),
               "perfect_team_10hz_same_HMM": joint(alive, true_home),
               "perfect_both_10hz_same_HMM": joint(true_alive, true_home)}
    by_type = {}
    for kind in ("HA_HA", "D_D", "mixed_HA_D", "censored"):
        rows = [r for r in short if r["boundary_type"] == kind]
        by_type[kind] = {"gt": len(rows), "recovered": sum(r["recovered"] for r in rows),
                         "frozen_dead_all_frames": sum(r["frozen_dead_on_all_frames"] for r in rows)}
    return {"raw": short_score(context, raw), "HMM": short_score(context, hmm),
            "raw_state": state_metrics(context, raw), "HMM_state": state_metrics(context, hmm),
            "raw_transitions": transition_counts(context, raw), "HMM_transitions": transition_counts(context, hmm),
            "short_frozen_dead_all_frames": sum(r["frozen_dead_on_all_frames"] for r in short),
            "short_raw_recovered_lost_by_HMM": sum(r["own_raw_recovered"] and not r["recovered"] for r in short),
            "short_HMM_recovered_not_raw": sum(r["recovered"] and not r["own_raw_recovered"] for r in short),
            "short_by_boundary_type": by_type,
            "oracles": {name: short_score(context, context.decoded(values, decoder)) for name, values in oracles.items()},
            "note": "Oracles use public GT at 10Hz anchors, not a deployable system or a selected recipe. Same HMM, including smoothing/quantization. Frozen alive emissions do not freeze hard D decisions."}
