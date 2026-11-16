"""Pure, deterministic 25 Hz diagnostics. No file, model or data access.

Truth is fixed; every repair is evaluated on the complete sequence. Public
aggregates and local per-case rows are returned separately. These GT-assisted
repairs are diagnostic interventions, not deployable predictions or bounds.
"""
from dataclasses import dataclass

import numpy as np

IOUS = (.3, .4, .5, .6, .7)
REPAIRS = ("boundaries", "team", "fragmentation", "D", "combined")


@dataclass
class Result:
    effective: np.ndarray
    spells: list
    flags: np.ndarray
    assignments: np.ndarray
    counts: np.ndarray  # threshold x (TP, FP, FN)
    spell_f1: float


class GlobalMatcher:
    """Cache official overlap computations, never crop or rematch subsets.

    A predicted interval's overlaps with fixed GT are independent of the other
    predictions. Reindex cached overlaps, sort by the official tie rule, then
    call the unchanged official greedy matcher for every full-sequence score.
    """
    def __init__(self, pm, truth):
        self.pm = pm
        self.gt = np.asarray(truth, np.int8).copy()
        self.runs = pm.valid_runs(self.gt)
        self.truth = pm.possession_spells(self.gt, self.runs)
        self.long = np.array([b-a >= 250 for a, b, _ in self.truth], bool)
        self.short = np.array([b-a < 75 for a, b, _ in self.truth], bool)
        self.cache = {}

    def score(self, hard, verify=False):
        hard = np.asarray(hard)
        if hard.shape != self.gt.shape or not np.isin(hard, (0, 1, 2, 3)).all():
            raise ValueError("Invalid complete canonical sequence")
        effective = self.pm.fill_abstentions(hard, self.runs)
        spells = self.pm.possession_spells(effective, self.runs)
        overlaps = []
        for j, spell in enumerate(spells):
            key = tuple(spell)
            if key not in self.cache:
                self.cache[key] = self.pm.spell_overlaps(self.truth, [spell])
            overlaps.extend((iou, i, j) for iou, i, _ in self.cache[key])
        overlaps.sort(key=lambda r: (-r[0], r[1], r[2]))
        flags = np.stack([self.pm.matched_gt_flags(overlaps, len(self.truth), t) for t in IOUS])
        counts = np.array([self.pm.match_spells(overlaps, len(self.truth), len(spells), t) for t in IOUS])
        assignments = np.full(flags.shape, -1, int)
        for k, threshold in enumerate(IOUS):
            used = set()
            for iou, i, j in overlaps:
                if iou < threshold:
                    break
                if assignments[k, i] < 0 and j not in used:
                    assignments[k, i] = j
                    used.add(j)
        if not np.array_equal(assignments >= 0, flags):
            raise ValueError("Assignment replay differs from official matcher")
        f1 = float(np.mean(2*counts[:, 0]/max(1, len(self.truth)+len(spells))))
        if verify:
            if overlaps != self.pm.spell_overlaps(self.truth, spells):
                raise ValueError("Cached global overlaps differ")
            full = self.pm.spell_scores(self.gt, effective, self.runs)
            if not np.isclose(f1, full["mean_f1"], atol=1e-14, rtol=0):
                raise ValueError("Official global score differs")
        return Result(effective, spells, flags, assignments, counts, f1)


def iou(left, right):
    a, b, _ = left
    c, d, _ = right
    return max(0, min(b, d)-max(a, c))/max(1, max(b, d)-min(a, c))


def associate(truth, spells, same_team=False):
    candidates = [(iou(truth, pred), j) for j, pred in enumerate(spells)
                  if (not same_team or pred[2] == truth[2]) and iou(truth, pred) > 0]
    # Stable official prediction order resolves equal geometry, regardless of team.
    return min(candidates, key=lambda r: (-r[0], r[1]))[1] if candidates else None


def repair_plan(matcher, base, case_id, kind):
    """Return disjoint constant-value edits fixed from the unmodified baseline.

    Boundary shrinkage vacates the removed tail as D; expansion overwrites the
    colliding neighbour and its effect is charged globally. Fragmentation joins
    first/last overlapping correct-team fragments, including their outer tails.
    D repairs only strict interior D runs bounded immediately by the GT team.
    Combined applies the compatible union, then true boundaries/team; there must
    be an overlapping H/A candidate. X is filled by the scorer, never treated as D.
    """
    if kind not in REPAIRS:
        raise ValueError("Unknown repair")
    a, b, team = matcher.truth[case_id]
    candidate = associate((a, b, team), base.spells)
    y = base.effective
    changed = y.copy()
    touched = np.zeros(len(y), bool)

    def write(left, right, label):
        changed[left:right] = label
        touched[left:right] = True
    fragments = [(l, r, k) for l, r, k in base.spells if k == team and l < b and r > a]
    internal_D = [(l, r) for l, r, k in matcher.pm.segments(y, [(a, b)])
                  if k == 2 and a < l and r < b and y[l-1] == team and y[r] == team]
    applicable = False
    if kind in ("fragmentation", "combined") and len(fragments) >= 2:
        write(fragments[0][0], fragments[-1][1], team)
        applicable = True
    if kind in ("D", "combined") and internal_D:
        for l, r in internal_D:
            write(l, r, team)
        applicable = True
    if kind in ("team", "boundaries", "combined") and candidate is not None:
        l, r, label = base.spells[candidate]
        applicable = True
        if kind == "team":
            write(l, r, team)
        else:
            if kind == "combined" and len(fragments) >= 2:
                l, r = min(l, fragments[0][0]), max(r, fragments[-1][1])
            write(l, min(r, a), 2)
            write(max(l, b), r, 2)
            write(a, b, team if kind == "combined" else label)
    mask = changed != y
    edits = [(l, r, k) for l, r, k in matcher.pm.segments(changed, matcher.pm.valid_runs(
        np.where(mask, 0, 3).astype(np.int8)))]
    # Protect also the unchanged interior a plan intends to preserve. Otherwise
    # two boundary repairs of one fused prediction can silently delete each other.
    constraints = matcher.pm.segments(changed, matcher.pm.valid_runs(np.where(touched, 0, 3).astype(np.int8)))
    if np.any(mask & (matcher.gt == 3)):
        raise ValueError("Repair crossed a run boundary")
    return {"case_id": int(case_id), "kind": kind, "applicable": applicable,
            "changed": bool(mask.any()), "candidate": candidate, "edits": edits, "constraints": constraints}


def apply_plans(original, plans):
    """Atomic joint conflict policy: ascending GT ID, reject conflicting edits.

    Identical writes commute; a conflicting later case is skipped in its entirety.
    Plans are never re-associated on an already repaired sequence.
    """
    result = np.asarray(original).copy()
    claimed = np.full(len(result), -1, np.int8)
    accepted, rejected = [], []
    for plan in sorted(plans, key=lambda p: p["case_id"]):
        constraints = plan.get("constraints", plan["edits"])
        if any(np.any((claimed[a:b] >= 0) & (claimed[a:b] != label)) for a, b, label in constraints):
            rejected.append(plan["case_id"])
            continue
        for a, b, label in plan["edits"]:
            result[a:b] = label
        for a, b, label in constraints:
            claimed[a:b] = label
        accepted.append(plan["case_id"])
    return result, accepted, rejected


def delta(matcher, old, new, selected=(), edits=(), detailed=False):
    gained, lost = new.flags & ~old.flags, old.flags & ~new.flags
    neighbours = np.zeros(len(matcher.truth), bool)
    for i in selected:
        a, b, _ = matcher.truth[i]
        for j, (l, r, _) in enumerate(matcher.truth):
            if j != i and (j in (i-1, i+1) or any(l < d and r > c for c, d, _ in edits)):
                # Adjacent IDs are neighbours only within this valid run.
                if any(start <= min(a, l) and max(b, r) <= stop for start, stop in matcher.runs):
                    neighbours[j] = True
    neighbours[list(selected)] = False
    outside = np.array([not any(a < d and b > c for c, d, _ in edits) for a, b, _ in matcher.truth])
    rows = {}
    for k, t in enumerate(IOUS):
        row = {"delta_tp": int(new.counts[k, 0]-old.counts[k, 0]),
               "delta_fp": int(new.counts[k, 1]-old.counts[k, 1]),
               "delta_fn": int(new.counts[k, 2]-old.counts[k, 2])}
        for name, mask in (("all", np.ones(len(matcher.truth), bool)), ("short", matcher.short),
                           ("long", matcher.long), ("neighbours", neighbours), ("outside_edits", outside)):
            row[name] = {"gained": int(gained[k, mask].sum()), "lost": int(lost[k, mask].sum())}
            if detailed:
                row[name].update(gained_gt_ids=np.flatnonzero(gained[k] & mask).tolist(),
                                 lost_gt_ids=np.flatnonzero(lost[k] & mask).tolist())
        rows[str(t)] = row
    return {"delta_spellF1": new.spell_f1-old.spell_f1, "delta_predictions": len(new.spells)-len(old.spells), "iou": rows}


def matrix(old, new, selected):
    return {str(t): {"retained": int((old.flags[k] & new.flags[k] & selected).sum()),
                    "gained": int((~old.flags[k] & new.flags[k] & selected).sum()),
                    "lost": int((old.flags[k] & ~new.flags[k] & selected).sum()),
                    "never": int((~old.flags[k] & ~new.flags[k] & selected).sum())}
            for k, t in enumerate(IOUS)}


def score_summary(matcher, result):
    return {"spellF1": result.spell_f1, "predictions": len(result.spells), "gt": len(matcher.truth),
            "short_gt": int(matcher.short.sum()), "long_gt": int(matcher.long.sum()),
            "iou": {str(t): {"tp": int(result.counts[k, 0]), "fp": int(result.counts[k, 1]),
                              "fn": int(result.counts[k, 2]), "short_matched": int(result.flags[k, matcher.short].sum()),
                              "long_matched": int(result.flags[k, matcher.long].sum())} for k, t in enumerate(IOUS)}}


def describe_case(matcher, original, result, i, evidence):
    gt = matcher.truth[i]
    a, b, team = gt
    best = associate(gt, result.spells)
    same = associate(gt, result.spells, True)
    fragments = [(j, pred) for j, pred in enumerate(result.spells) if iou(gt, pred) > 0]
    same_fragments = [p for _, p in fragments if p[2] == team]
    spans_D = [(l, r) for l, r, k in matcher.pm.segments(result.effective, [(a, b)]) if k == 2]
    spans_X = [(l, r) for l, r, k in matcher.pm.segments(original, [(a, b)]) if k == 3]
    overlapped = sorted({j for _, p in fragments for j, g in enumerate(matcher.truth) if j != i and iou(p, g) > 0})
    tags = []
    if same is not None and result.spells[same][:2] != gt[:2] and iou(gt, result.spells[same]) < .5:
        tags.append("boundaries")
    if best is not None and result.spells[best][2] != team:
        tags.append("team")
    if len(same_fragments) >= 2:
        tags.append("fragmentation")
    if overlapped:
        tags.append("merged_neighbours")
    if any(a < l and r < b for l, r in spans_D):
        tags.append("D")
    if spans_X:
        tags.append("X")
    if len(tags) > 1:
        tags.append("mixed")
    if not tags and not result.flags[2, i]:
        tags.append("insufficient_explanation")
    row = {"gt_index": i, "duration_s": (b-a)/25, "interval": [a, b], "team": team,
           "flags": result.flags[:, i].tolist(), "assigned_prediction_ids": result.assignments[:, i].tolist(),
           "best_any_iou": iou(gt, result.spells[best]) if best is not None else 0.,
           "best_same_iou": iou(gt, result.spells[same]) if same is not None else 0.,
           "start_error_s": (result.spells[same][0]-a)/25 if same is not None else None,
           "end_error_s": (result.spells[same][1]-b)/25 if same is not None else None,
           "fragments": [{"id": j, "interval": list(p[:2]), "team": p[2]} for j, p in fragments],
           "overlapped_gt_neighbours": overlapped, "D_spans": spans_D, "original_X_spans": spans_X,
           "effective_spans": matcher.pm.segments(result.effective, [(a, b)]), "tags": tags}
    row["evidence"] = {name: float(np.asarray(value)[a:b].mean()) for name, value in evidence.items()}
    if "home_margin" in evidence:
        margin = np.asarray(evidence["home_margin"])[a:b]*(1 if team == 0 else -1)
        row["evidence"].update(signed_margin=float(margin.mean()), correct_team_fraction=float((margin > 0).mean()))
    return row


def analyze(matcher, systems, evidence, metadata, references):
    """Local exhaustive long-case table; public aggregates contain no case IDs."""
    scored = {name: {stage: matcher.score(hard, verify=True) for stage, hard in stages.items()}
              for name, stages in systems.items()}
    local, summaries = [], {}
    ids = np.flatnonzero(matcher.long).tolist()
    for name, stages in systems.items():
        base = scored[name]["hmm"]
        summary = {"scores": {stage: score_summary(matcher, r) for stage, r in scored[name].items()},
                   "postprocessing": "none in the registered external public pipeline",
                   "raw_to_hmm_long": matrix(scored[name]["raw"], base, matcher.long),
                   "vs_controls": {}, "joint_repairs": {}, "individual_repairs": {}, "missed_tags": {}}
        for ref in references:
            summary["vs_controls"][ref] = {"long": matrix(scored[ref]["hmm"], base, matcher.long),
                "global": delta(matcher, scored[ref]["hmm"], base), "lost_tags": {}}
        plans = {kind: [] for kind in REPAIRS}
        effects = {kind: [] for kind in REPAIRS}
        for i in ids:
            row = describe_case(matcher, stages["hmm"], base, i, evidence[name])
            row.update(metadata, system=name, stable_id=f"{metadata['fold']}:{metadata['match']}:{i}",
                       run=next(j for j, (a, b) in enumerate(matcher.runs) if a <= matcher.truth[i][0] < b))
            row["stages"] = {stage: r.flags[:, i].tolist() for stage, r in scored[name].items()}
            row["controls"] = {}
            for ref in references:
                old = scored[ref]["hmm"].flags[:, i]
                new = base.flags[:, i]
                row["controls"][ref] = ["retained" if o and n else "lost" if o else "gained" if n else "never" for o, n in zip(old, new)]
                if old[2] and not new[2]:
                    for tag in row["tags"]:
                        counts = summary["vs_controls"][ref]["lost_tags"]
                        counts[tag] = counts.get(tag, 0)+1
            if not base.flags[2, i]:
                for tag in row["tags"]:
                    summary["missed_tags"][tag] = summary["missed_tags"].get(tag, 0)+1
            row["repairs"] = {}
            for kind in REPAIRS:
                plan = repair_plan(matcher, base, i, kind)
                hard, _, _ = apply_plans(stages["hmm"], [plan])
                after = matcher.score(hard)
                effect = delta(matcher, base, after, [i], plan["edits"], detailed=True)
                row["repairs"][kind] = {"plan": plan, "effect": effect,
                                        "case_flags_after": after.flags[:, i].tolist()}
                plans[kind].append(plan)
                effects[kind].append(effect["delta_spellF1"])
            local.append(row)
        for kind in REPAIRS:
            values = np.asarray(effects[kind])
            summary["individual_repairs"][kind] = {"n": len(values), "positive": int((values > 1e-12).sum()),
                "negative": int((values < -1e-12).sum()), "median_delta": float(np.median(values)) if len(values) else None,
                "min_delta": float(values.min()) if len(values) else None, "max_delta": float(values.max()) if len(values) else None,
                "applicable": sum(p["applicable"] for p in plans[kind]),
                "changed": sum(p["changed"] for p in plans[kind])}
            summary["joint_repairs"][kind] = {}
            for group in ("all_long", "missed_long_at_0.5"):
                selected = [p for p in plans[kind] if group == "all_long" or not base.flags[2, p["case_id"]]]
                hard, accepted, rejected = apply_plans(stages["hmm"], selected)
                edits = [e for p in selected if p["case_id"] in accepted for e in p["edits"]]
                effect = delta(matcher, base, matcher.score(hard, verify=True), [p["case_id"] for p in selected], edits)
                summary["joint_repairs"][kind][group] = dict(effect, accepted=len(accepted), collisions=len(rejected),
                    applicable=sum(p["applicable"] for p in selected), changed=sum(p["changed"] for p in selected))
        summaries[name] = summary
    return local, summaries


def paired_uncertainty(rows, draws=2000, seed=20261028):
    """Resample whole public matches, never concatenate temporal cuts.

    Rank paired robust deltas of *joint* missed-long repairs. Individual oracle
    improvements are deliberately not added. Two matches give low precision.
    """
    folds = sorted({r["fold"] for r in rows})
    if len(folds) != 2:
        raise ValueError("Both registered validation folds required")
    weights = {"clean": .5, "missing40_5s": .25, "missing40_15s": .25}
    by_key = {(r["fold"], r["scenario"]): r for r in rows}
    if len(by_key) != len(rows) or set(by_key) != {(f, s) for f in folds for s in weights}:
        raise ValueError("Incomplete paired scenarios")
    sampled = np.random.default_rng(seed).integers(0, len(folds), (draws, len(folds)))
    result = {}
    for system in rows[0]["systems"]:
        scores = {}
        fold_values = {}
        for kind in REPAIRS:
            per_fold = np.array([sum(w*by_key[f, s]["systems"][system]["joint_repairs"][kind]["missed_long_at_0.5"]["delta_spellF1"]
                                    for s, w in weights.items()) for f in folds])
            distribution = per_fold[sampled].mean(1)
            fold_values[kind] = per_fold
            scores[kind] = {"by_fold": dict(zip(map(str, folds), per_fold.tolist())),
                            "mean_delta": float(per_fold.mean()), "paired_match_percentiles_95": np.quantile(distribution, [.025, .975]).tolist()}
        ordered = sorted(REPAIRS, key=lambda k: (-scores[k]["mean_delta"], REPAIRS.index(k)))
        paired = {}
        for i, first in enumerate(REPAIRS):
            for second in REPAIRS[i+1:]:
                diff = fold_values[first]-fold_values[second]
                paired[first+"_minus_"+second] = {"by_fold": dict(zip(map(str, folds), diff.tolist())),
                    "mean_delta": float(diff.mean()), "paired_match_percentiles_95": np.quantile(diff[sampled].mean(1), [.025, .975]).tolist()}
        result[system] = {"ranking": list(ordered), "repairs": scores,
                          "ranking_by_fold": {str(f): sorted(REPAIRS, key=lambda k: (-fold_values[k][j], REPAIRS.index(k)))
                                              for j, f in enumerate(folds)},
                          "paired_repair_differences": paired,
                          "caution": "Only two reused matches; descriptive conditional uncertainty, not independent confirmation"}
    return result
