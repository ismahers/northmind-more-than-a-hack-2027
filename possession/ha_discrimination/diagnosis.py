"""Operational emission evidence, timing and physics; no training or GT features."""
import time
import numpy as np
import torch

from possession.external_data.schema import file_hash
from possession.interval_probe.data import load
from possession.learning.study import ScoreContext
from possession.learning.temporal import fit_priors, TemporalDecoder
from possession.night.common import atomic_json
from possession.temporal_probe.common import Store
from possession.temporal_probe.windows import predict, emissions
from possession.duration_balanced.training import init_head, paths
from possession.duration_balanced.metrics import hard_head, matched
from possession.phase6.decoding import DirectionalDecoder
from possession.phase8.diagnosis import HA_events, matched_truth
from .common import ROOT, DEST, PARENT, register, verify_ancestry
from .features import geometry


def summary(values):
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    return {"n": len(x), "mean": float(x.mean()), "median": float(np.median(x)),
            "q25": float(np.quantile(x,.25)), "q75": float(np.quantile(x,.75))} if len(x) else {"n": 0}


def group(rows):
    fields = ("p_correct", "p_joint_correct", "signed_logit_margin", "correct_fraction", "best_raw_iou",
              "nearest_correct_m", "nearest_other_m", "proximity_correct_fraction", "ball_available_fraction",
              "ball_speed_m_s", "start_error_s", "end_error_s")
    return {"n": len(rows), "hmm_recovered": sum(r["hmm_recovered"] for r in rows),
            "raw_recovered": sum(r["raw_recovered"] for r in rows),
            "A_strong_emissions_missed": sum(r["A"] for r in rows),
            "B_insufficient_emissions_missed": sum(r["B"] for r in rows),
            "B_no_correct_conditional_frame": sum(r["B_no_frame"] for r in rows),
            "C_raw_recovered_HMM_lost": sum(r["C"] for r in rows),
            "D_physical_conflict": sum(r["physical_conflict"] for r in rows),
            "error_types": {name: sum(r["error_type"]==name for r in rows) for name in
                            ("recovered", "decoder", "classification", "alive_masking", "temporal_extent")},
            "statistics": {field: summary([r[field] for r in rows]) for field in fields}}


def analyze(context, p, logit, physical, prior, spec):
    pm = context.pm
    raw = hard_head(context,p)
    hmm = context.decoded(p,DirectionalDecoder(context.view,prior,"hmm",.5))
    truth, _, raw_flags = matched(context,raw)
    hmm_flags = matched(context,hmm)[2]
    raw_spells = pm.possession_spells(raw,context.runs)
    overlaps = pm.spell_overlaps(truth,raw_spells)
    best = {}
    for iou, i, j in overlaps:
        if i not in best:
            best[i] = (iou,raw_spells[j])
    n = len(context.gt)
    native = {"logit": np.zeros(n), "home": np.zeros(n), "alive": np.zeros(n),
              "available": np.zeros(n,bool), "both": np.zeros(n,bool), "speed": np.zeros(n),
              "velocity_valid": np.zeros(n,bool), "delta": np.zeros(n), "nearest": np.zeros((n,2))}
    alive = p[:,:2].sum(1)
    home = 1/(1+np.exp(-np.clip(logit,-30,30)))
    for selected,mapping,output in context.parts:
        for name,values in dict(logit=logit,home=home,alive=alive,**physical).items():
            if name in native:
                native[name][output] = values[selected][mapping]
    curves = {}; rows = []
    for i,(a,b,team) in enumerate(truth):
        if (b-a)/25 >= 3:
            continue
        sign = 1 if team==0 else -1
        correct = native["home"][a:b] if team==0 else 1-native["home"][a:b]
        old = int(context.gt[a-1]) if a else 3
        new = int(context.gt[b]) if b<n else 3
        kind = "censored" if old==3 or new==3 else "HA_HA" if old<2 and new<2 else "D_D" if old==2 and new==2 else "mixed_HA_D"
        observed = native["both"][a:b]
        advantage = sign*native["delta"][a:b]
        proximity = float((advantage[observed]>=spec["proximity_margin_m"]).mean()) if observed.any() else 0.
        fraction = float((correct>.5).mean())
        strong = float((correct>=spec["strong_probability"]).mean())>=spec["strong_fraction"]
        raw_matched,hmm_matched = bool(raw_flags[i]),bool(hmm_flags[i])
        speed = native["speed"][a:b][native["velocity_valid"][a:b]]
        best_iou, interval = best.get(i,(0.,None))
        conflict = bool(not hmm_matched and fraction<.5 and observed.mean()>=spec["ball_available_fraction"] and proximity>=spec["proximity_fraction"])
        error = "recovered" if hmm_matched else "decoder" if raw_matched else "classification" if fraction<.5 else "alive_masking" if (native["alive"][a:b]>=.5).mean()<.5 else "temporal_extent"
        rows.append({"duration_s":(b-a)/25,"boundary_type":kind,"hmm_recovered":hmm_matched,"raw_recovered":raw_matched,
                     "A":not hmm_matched and strong,"B":not hmm_matched and not strong,
                     "B_no_frame":not hmm_matched and not (correct>.5).any(),"C":raw_matched and not hmm_matched,
                     "physical_conflict":conflict,"error_type":error,
                     "p_correct":float(correct.mean()),"p_joint_correct":float((correct*native["alive"][a:b]).mean()),
                     "signed_logit_margin":float((sign*native["logit"][a:b]).mean()),"correct_fraction":fraction,
                     "best_raw_iou":float(best_iou),
                     "nearest_correct_m":float(native["nearest"][a:b,team][observed].mean()) if observed.any() else None,
                     "nearest_other_m":float(native["nearest"][a:b,1-team][observed].mean()) if observed.any() else None,
                     "proximity_correct_fraction":proximity,"ball_available_fraction":float(native["available"][a:b].mean()),
                     "ball_speed_m_s":float(np.median(speed)) if len(speed) else None,
                     "start_error_s":(interval[0]-a)/25 if interval else None,"end_error_s":(interval[1]-b)/25 if interval else None})
        for side,anchor in (("entry",a),("exit",b)):
            for offset in spec["diagnostic_offsets_s"]:
                j = anchor+round(offset*25)
                # Diagnostic curve never crosses a run separator.
                lo,hi = sorted((max(0,min(n-1,anchor)),j))
                if j<0 or j>=n or (context.gt[lo:hi+1]==3).any():
                    continue
                cp = native["home"][j] if team==0 else 1-native["home"][j]
                key = f"{kind}/{side}/{offset}"
                curves.setdefault(key,[]).append({"probability":float(cp),"signed_margin":float(sign*native["logit"][j])})
    gt_events = HA_events(pm,context.gt,context.runs)
    raw_events = matched_truth(pm,gt_events,HA_events(pm,raw,context.runs))
    hmm_events = matched_truth(pm,gt_events,HA_events(pm,hmm,context.runs))
    return {"all":group(rows),"by_boundary":{k:group([r for r in rows if r["boundary_type"]==k]) for k in ("HA_HA","D_D","mixed_HA_D","censored")},
            "by_emission":{k:group([r for r in rows if r["error_type"]==k]) for k in ("recovered","decoder","classification","alive_masking","temporal_extent")},
            "by_motion":{name:group([r for r in rows if r["ball_speed_m_s"] is not None and lo<=r["ball_speed_m_s"]<hi])
                         for name,lo,hi in (("lt2",0,2),("2to8",2,8),("8to20",8,20),("ge20",20,np.inf))},
            "direct_transitions":{"gt":len(gt_events),"raw_detected":len(raw_events),"HMM_detected":len(hmm_events),
                                  "raw_correct_suppressed":len(raw_events-hmm_events),"HMM_added":len(hmm_events-raw_events)},
            "curves":{key:{"probability":summary([r["probability"] for r in values]),"signed_margin":summary([r["signed_margin"] for r in values])} for key,values in curves.items()},
            "per_possession":rows}


def case(store,record,fold,part):
    spec,recipe = record["protocol"],record["recipe"]
    teacher = record["teachers"][str(fold)]; key = teacher["split"][part]
    view = store.harmonized(key); match = store.match(key)
    context = ScoreContext(match,view)
    prior = fit_priors([store.harmonized(k) for k in teacher["split"]["train"]])
    data = load(paths(teacher,key,recipe)["clean"])
    if not np.array_equal(data["native_index"],view.native_index):
        raise ValueError("Embedding clock mismatch")
    model = init_head(fold,teacher,recipe)
    state = torch.load(ROOT/"evaluation/reports/external/duration-balanced-phase8-v1"/f"training/fold-{fold}/original/last.pt",map_location="cpu",weights_only=False)
    model.load_state_dict(state["model"])
    logits = predict(model,data,recipe["core_frames"],recipe["batch_size"])[0]
    with np.load(PARENT/f"predictions/fold-{fold}/cap-8/{part}/clean.npz",allow_pickle=False) as z:
        p = z["probabilities"]
    if not np.array_equal(emissions(logits,data["probabilities"]),p):
        raise ValueError("Existing local emissions not exactly reproduced")
    _,physical = geometry(view,match.metadata["pitch_m"])
    result = analyze(context,p,logits,physical,prior,spec)
    result.update(fold=fold,part=part)
    atomic_json(DEST/f"cases/{part}/fold-{fold}.json",result)
    return {key:value for key,value in result.items() if key!="per_possession"}


def run():
    torch.use_deterministic_algorithms(True)
    record = register(); torch.set_num_threads(record["protocol"]["cpu_threads"])
    if (DEST/"diagnostic-summary.json").exists():
        return 0
    begin = time.perf_counter(); store = Store(record["records"])
    validation = [case(store,record,fold,"validation") for fold in record["protocol"]["pilots"]]
    atomic_json(DEST/"diagnostic-validation.json",validation)
    threshold = record["protocol"]["minimum_conflicting_short_spells_per_validation"]
    allowed = all(row["all"]["D_physical_conflict"]>=threshold for row in validation)
    gate = {"pilot_justified":allowed,"counts_by_fold":{str(row["fold"]):row["all"]["D_physical_conflict"] for row in validation},
            "minimum_per_fold":threshold,"test_used":False,"validation_sha256":file_hash(DEST/"diagnostic-validation.json")}
    atomic_json(DEST/"hypothesis.json",gate)
    # Descriptive reuse of historical tests only AFTER hypothesis is frozen.
    tests = [case(store,record,fold,"test") for fold in record["protocol"]["pilots"]]
    checks = verify_ancestry(record)
    checks["hypothesis_before_test_diagnosis"] = (DEST/"hypothesis.json").stat().st_mtime_ns < min((DEST/f"cases/test/fold-{i}.json").stat().st_mtime_ns for i in record["protocol"]["pilots"])
    checks["hypothesis_own_validation_only"] = not gate["test_used"]
    atomic_json(DEST/"diagnostic-verification.json",{"passed":all(checks.values()),"count":len(checks),"checks":checks})
    atomic_json(DEST/"diagnostic-summary.json",{"validation":validation,"test":tests,"hypothesis":gate,
                "seconds":time.perf_counter()-begin,"new_training":0,"verification_passed":all(checks.values()),
                "definitions":"A/B partition HMM-missed short spells at p>=.6 in>=50% frames; C is overlapping decoder-loss subset. Lack of operational emission evidence never proves absent latent information. Proximity/motion are associations, not possession ground truth."})
    return 0 if all(checks.values()) else 1
