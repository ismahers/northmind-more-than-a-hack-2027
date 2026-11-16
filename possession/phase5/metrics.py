"""Duration diagnostics use global official greedy matching, never cropped spells."""

import numpy as np


BINS = (("lt1", 0., 1.), ("1to3", 1., 3.), ("3to10", 3., 10.), ("ge10", 10., np.inf))


def event_metrics(truth, prediction, pm, tolerance, typed="none"):
    pairs, missed, false = pm.match_transitions(truth, prediction, round(25 * tolerance), typed=typed)
    tp, fp, fn = len(pairs), len(false), len(missed)
    p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r,
            "f1": 2 * p * r / max(1e-12, p + r),
            "mae_s": float(np.mean([abs(x[0] - y[0]) / 25 for x, y in pairs])) if pairs else None}


def head_events(context, signal, threshold, nms_s=.2):
    out = []
    for selected, mapping, output in context.parts:
        q = signal[selected]
        # Local maxima, confidence-ordered NMS within one run, then time ordered.
        candidates = np.flatnonzero((q >= threshold) & (q >= np.r_[-1., q[:-1]]) & (q > np.r_[q[1:], -1.]))
        chosen = []
        for i in candidates[np.argsort(-q[candidates], kind="stable")]:
            if all(abs(context.view.time[selected[i]] - context.view.time[selected[j]]) > nms_s for j in chosen):
                chosen.append(int(i))
        for i in sorted(chosen):
            native = np.flatnonzero(mapping == i)
            if len(native):
                out.append((output.start + int(native[len(native) // 2]), 0, 1))
    return out


def diagnostics(context, hard, signal=None, threshold=.5):
    pm, gt, runs = context.pm, context.gt, context.runs
    truth, pred = pm.possession_spells(gt, runs), pm.possession_spells(hard, runs)
    overlaps = pm.spell_overlaps(truth, pred)
    td, pd = np.zeros(len(truth), int), np.zeros(len(pred), int)
    for _, i, j in overlaps:
        td[i] += 1
        pd[j] += 1
    durations_gt = np.array([(b - a) / 25 for a, b, _ in truth])
    durations_pred = np.array([(b - a) / 25 for a, b, _ in pred])
    groups = {}
    for name, lo, hi in BINS:
        g, p = (durations_gt >= lo) & (durations_gt < hi), (durations_pred >= lo) & (durations_pred < hi)
        groups[name] = {"gt": int(g.sum()), "pred": int(p.sum()),
                        "fragmented_gt": int(((td > 1) & g).sum()), "fused_predictions": int(((pd > 1) & p).sum()), "iou": {}}
    for tau in (.3, .4, .5, .6, .7):
        mg, mp = np.zeros(len(truth), bool), np.zeros(len(pred), bool)
        for iou, i, j in overlaps:
            if iou < tau:
                break
            if not mg[i] and not mp[j]:
                mg[i] = mp[j] = True
        for name, lo, hi in BINS:
            g, p = (durations_gt >= lo) & (durations_gt < hi), (durations_pred >= lo) & (durations_pred < hi)
            groups[name]["iou"][str(tau)] = {"matched_gt": int((mg & g).sum()), "fn": int((~mg & g).sum()),
                "fp": int((~mp & p).sum()), "matched_pred": int((mp & p).sum()),
                "recall": float((mg & g).sum() / max(1, g.sum())), "precision": float((mp & p).sum() / max(1, p.sum()))}
    gt_events = [e for e in pm.transitions(gt, runs) if e[1] < 2 and e[2] < 2]
    pred_events = [e for e in pm.transitions(hard, runs) if e[1] < 2 and e[2] < 2]
    decoded = {str(t): event_metrics(gt_events, pred_events, pm, t, "pair") for t in (.2, .5, 1.)}
    result = {"duration_groups": groups, "decoded_HA": decoded,
              "fragmented_gt": int((td > 1).sum()), "fused_predictions": int((pd > 1).sum()),
              "gt_spells": len(truth), "pred_spells": len(pred)}
    if signal is not None:
        events = head_events(context, signal, threshold)
        result["head_HA"] = {str(t): event_metrics(gt_events, events, pm, t) for t in (.2, .5, 1.)}
        result["head_threshold"] = threshold
    return result
