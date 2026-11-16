"""Post-selection diagnostics only; never feeds training or parameter selection."""

import csv
import json
import time

import numpy as np

from possession.learning.study import ScoreContext
from possession.learning.temporal import TemporalDecoder
from possession.night.common import OLD, atomic_json
from possession.night.decoding_study import load_predictions
from possession.phase5.experiment import DEST, register, source, get_store, load_prediction
from possession.phase5.decoding import AdaptiveDecoder
from possession.phase5.metrics import head_events, BINS
from scripts.train_e0 import manifest


def main():
    begin=time.perf_counter()
    protocol=register()
    data,_=manifest("idsse-only")
    store=get_store(data)
    boundary_counts=[]
    possession_rows=[]
    for i in range(7):
        selection=json.loads((DEST/f"selections/fold-{i}.json").read_text())
        src=source(i)
        key=data["folds"][i]["test"][0]
        view=store.view(key)
        context=ScoreContext(store.match(key),view)
        pm,gt,runs=context.pm,context.gt,context.runs
        gt_events=[e for e in pm.transitions(gt,runs) if e[1]<2 and e[2]<2]
        for case in protocol["scenarios"]:
            p,q=load_prediction(DEST/f"predictions/fold-{i}/boundary/{case}.npz")
            signal=q*p[:,:2].sum(-1)
            peaks=head_events(context,signal,selection["head_threshold"])
            head_pairs,_,head_fp=pm.match_transitions(gt_events,peaks,12,typed="none")
            gt_head={tuple(x) for x,_ in head_pairs}
            paths={"C":context.decoded(p,TemporalDecoder(src["specification"],src["priors"])),
                   "raw_C":context.decoded(p,TemporalDecoder({"kind":"argmax"},{}))}
            trial=selection["grids"]["D"]["winner"]
            paths["D"]=context.decoded(p,AdaptiveDecoder(src["specification"],src["priors"],view,q,trial["strength"],trial["threshold"]))
            _,original=load_predictions(OLD/f"test/seed-2027/fold-{i}/{src['condition']}/{case}/probabilities.npz")
            paths["A"]=context.decoded(original,TemporalDecoder(src["specification"],src["priors"]))
            raw_events=[e for e in pm.transitions(paths["raw_C"],runs) if e[1]<2 and e[2]<2]
            raw_pairs,_,raw_fp=pm.match_transitions(gt_events,raw_events,12,typed="pair")
            gt_raw={tuple(x) for x,_ in raw_pairs}
            for system,hard in paths.items():
                events=[e for e in pm.transitions(hard,runs) if e[1]<2 and e[2]<2]
                pairs,missed,fp=pm.match_transitions(gt_events,events,12,typed="pair")
                gt_decoded={tuple(x) for x,_ in pairs}
                boundary_counts.append({"fold":i,"scenario":case,"system":system,"gt_HA":len(gt_events),
                    "head_detected":len(gt_head),"head_and_decoded":len(gt_head & gt_decoded),
                    "head_only":len(gt_head-gt_decoded),"decoded_only":len(gt_decoded-gt_head),
                    "neither":len(set(gt_events)-gt_head-gt_decoded),"head_fp":len(head_fp),
                    "raw_C_true_changes":len(gt_raw),"decoded_true_changes":len(gt_decoded),
                    "raw_true_removed":len(gt_raw-gt_decoded),"true_added_over_raw":len(gt_decoded-gt_raw),
                    "raw_C_fp":len(raw_fp),"decoded_fp":len(fp)})
                full=json.loads((DEST/f"metrics/fold-{i}/{case}/{system}.json").read_text())["official"]
                for item in full["per_possession"]:
                    a,b=item["start_canonical_frame"],item["stop_canonical_frame"]
                    team=0 if item["team"]=="H" else 1
                    start="HA" if a>0 and gt[a-1]==1-team else "D" if a>0 and gt[a-1]==2 else "censored"
                    end="HA" if b<len(gt) and gt[b]==1-team else "D" if b<len(gt) and gt[b]==2 else "censored"
                    duration=item["duration_s"]
                    group=next(name for name,lo,hi in BINS if lo<=duration<hi)
                    possession_rows.append({"fold":i,"scenario":case,"system":system,"start":a,"stop":b,
                        "duration_s":duration,"duration_group":group,"start_type":start,"end_type":end,
                        "has_HA_edge":start=="HA" or end=="HA","both_D_edges":start=="D" and end=="D",
                        "matched_iou_0_5":item["matched_at_iou_0_5"]})
    for filename,rows in (("boundary_retention.csv",boundary_counts),("possession_error_analysis.csv",possession_rows)):
        with (DEST/filename).open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    retention={}
    for system in ("A","C","D","raw_C"):
        selected=[r for r in boundary_counts if r["system"]==system and r["scenario"]=="clean"]
        retention[system]={k:sum(r[k] for r in selected) for k in selected[0] if k not in ("fold","scenario","system")}
    strata={}
    for system in ("A","C","D","raw_C"):
        strata[system]={}
        for group,_,_ in BINS:
            strata[system][group]={}
            for category in ("has_HA_edge","both_D_edges"):
                selected=[r for r in possession_rows if r["system"]==system and r["scenario"]=="clean" and r["duration_group"]==group and r[category]]
                strata[system][group][category]={"gt":len(selected),"matched":sum(r["matched_iou_0_5"] for r in selected),
                    "recall":sum(r["matched_iou_0_5"] for r in selected)/max(1,len(selected))}
    atomic_json(DEST/"error_analysis.json",{"clean_boundary_retention":retention,"clean_possession_strata":strata,
        "matching_tolerance_s":.48,"note":"head is undirected, decoded/raw transitions typed; descriptive only",
        "seconds":time.perf_counter()-begin})
    return 0


if __name__=="__main__":
    raise SystemExit(main())
