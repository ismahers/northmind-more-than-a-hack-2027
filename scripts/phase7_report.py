"""Measured public results, duration diagnostics, isolated cost and learning curves."""
import csv
import json
from collections import defaultdict
import numpy as np
from possession.phase7.public import DEST,protocol
from possession.night.common import atomic_json


def grouped(rows):
    result={}
    for arm in ("legacy","harmonized","capacity","set_transformer"):
        if not any(r["arm"]==arm for r in rows):continue
        result[arm]={}
        for system in ("fixed_hmm","fixed_hsmm","selected"):
            cases={}
            for scenario in ("clean","missing40_5s","missing40_15s"):
                picked=[r for r in rows if r["arm"]==arm and r["system"]==system and r["scenario"]==scenario]
                if not picked:continue
                metrics={k:{"mean":float(np.mean([r[k] for r in picked])),"sd_population":float(np.std([r[k] for r in picked]))}
                         for k in ("spellF1","macroF1","team_accuracy_true_alive","alive_dead_F1")}
                metrics.update({k:sum(r[k] for r in picked) for k in ("gt_spells","pred_spells","fragmented_gt","fused_predictions","false_D_interruptions","false_team_changes")})
                if "independent" in picked[0]:
                    metrics["raw_conditional_team_accuracy_on_gt_alive"]=float(np.mean([r["independent"]["conditional_team_accuracy_gt_alive"] for r in picked]))
                    metrics["raw_alive_F1_0_5"]=float(np.mean([r["independent"]["head_alive_F1_at_0_5"] for r in picked]))
                bins={}
                for name in ("lt1","1to3","3to10","ge10"):
                    items=[r["duration_groups"][name] for r in picked]
                    g={k:sum(x[k] for x in items) for k in ("gt","pred","fragmented_gt","fused_predictions")}
                    g["iou"]={}
                    for tau in ("0.3","0.4","0.5","0.6","0.7"):
                        z={k:sum(x["iou"][tau][k] for x in items) for k in ("matched_gt","fn","fp","matched_pred")}
                        z["recall"]=z["matched_gt"]/max(1,g["gt"]);g["iou"][tau]=z
                    bins[name]=g
                metrics["duration_groups"]=bins
                for kind in ("head_HA","decoded_HA"):
                    values=[r[kind]["0.5"] for r in picked]
                    totals={k:sum(v[k] for v in values) for k in ("tp","fp","fn")}
                    totals["precision"]=totals["tp"]/max(1,totals["tp"]+totals["fp"])
                    totals["recall"]=totals["tp"]/max(1,totals["tp"]+totals["fn"])
                    metrics[kind+"_0_5s"]=totals
                cases[scenario]=metrics
            result[arm][system]=cases
    return result


def transfer_summary(rows):
    result={}
    for arm in ("legacy","harmonized","capacity","set_transformer"):
        if not any(r["arm"]==arm for r in rows):continue
        result[arm]={}
        for scenario in ("clean","missing40_5s","missing40_15s"):
            games=defaultdict(list)
            for r in rows:
                if r["arm"]==arm and r["scenario"]==scenario:games[r["match"]].append(r["conditional_team"])
            values=[np.mean([x["all"]["accuracy"] for x in group]) for group in games.values()]
            f1=[np.mean([x["all"]["macroF1"] for x in group]) for group in games.values()]
            result[arm][scenario]={"games":len(games),"conditional_accuracy_mean":float(np.mean(values)),
                "conditional_accuracy_sd_across_games":float(np.std(values)),"conditional_macroF1_mean":float(np.mean(f1)),
                "aggregation":"mean two frozen pilot models within game, then equal game weighting"}
    return result


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spec=protocol();rows=json.loads((DEST/"all_rows.json").read_text())
    transfer_rows=json.loads((DEST/"transfer.json").read_text())["rows"]
    capacity=DEST.parent/"phase7-capacity-v1"
    has_control=(capacity/"compute.json").exists()
    if has_control:
        rows+=json.loads((capacity/"all_rows.json").read_text())
        transfer_rows+=json.loads((capacity/"transfer.json").read_text())["rows"]
    summary=grouped(rows);transfer=transfer_summary(transfer_rows)
    audit=json.loads((DEST/"audit.json").read_text());causes=defaultdict(list)
    for r in audit["causal_interventions"]:causes[r["intervention"]].append(r["conditional_team"]["all"]["accuracy"])
    causes={k:float(np.mean(v)) for k,v in causes.items()}
    costs={};fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for row,i in enumerate(spec["pilot_folds"]):
        for arm in (("harmonized","capacity","set_transformer") if has_control else ("harmonized","set_transformer")):
            histories=[];seconds=0
            for stage in ("base","binary"):
                source=capacity/f"training/fold-{i}/{stage}/complete.json" if arm=="capacity" else DEST/f"training/fold-{i}/{arm}/{stage}/complete.json"
                d=json.loads(source.read_text());histories+=d["history"];seconds+=d["seconds"]
            costs[f"{i}/{arm}"]={"seconds":seconds,"parameters":d["parameters"],"training_epoch_s_mean":float(np.mean([x["training_seconds"] for x in histories])),
                "initial_validation_loss":histories[0]["validation"]["loss"],"final_validation_loss":histories[-1]["validation"]["loss"],
                "initial_training_loss":histories[0]["train"]["loss"],"final_training_state_loss":histories[-1]["train"]["state_loss"],
                "final_training_team_loss":histories[-1]["train"]["team_loss"],"final_training_boundary_focal_loss":histories[-1]["train"]["boundary_loss"],
                "peak_rss_bytes_cumulative":d["memory"]["process_peak_rss_bytes"]}
            epoch=[x["global_epoch"] for x in histories]
            trainloss=[x["train"].get("loss",x["train"].get("state_loss",0)+x["train"].get("team_loss",0)) for x in histories]
            axes[row,0].plot(epoch,trainloss,label=arm);axes[row,1].plot(epoch,[x["validation"]["loss"] for x in histories],label=arm)
        for col in range(2):axes[row,col].set_title(f"Fold {i} / "+("train" if col==0 else "inner validation"));axes[row,col].set_ylabel("state + conditional team BCE");axes[row,col].legend();axes[row,col].grid(alpha=.2)
    for ax in axes[-1]:ax.set_xlabel("Epoch (binary auxiliary loss starts at 13)")
    fig.tight_layout();fig.savefig(DEST/"learning_curves.png",dpi=150);plt.close(fig)
    decisions=json.loads((DEST/"independent_architecture_selection.json").read_text())
    result={"IDSSE":summary,"SkillCorner":transfer,"metadata_audit_four_development_games":causes,"training_costs":costs,
        "architecture_decisions_independent":decisions,"compute":json.loads((DEST/"compute.json").read_text()),
        "profile":{a:json.loads((DEST/"profile"/(a+".json")).read_text()) for a in ("harmonized","capacity","set_transformer")},
        "scope":"two exploratory IDSSE outer folds; 16 provider-confirmation games already explored in Phase6; no official assets accessed",
        "no_full_LOMO_new_claim":True}
    if has_control:
        result["capacity_control"]={"compute":json.loads((capacity/"compute.json").read_text()),
            "own_validation":{str(i):json.loads((capacity/f"selections/fold-{i}.json").read_text()) for i in spec["pilot_folds"]},
            "interpretation":"one exploratory follow-up, widths determined by near-equal parameter count; no search and no original main choices overwritten"}
    atomic_json(DEST/"summary.json",result)
    columns=[k for k in rows[0] if not isinstance(rows[0][k],(dict,list))]
    with (DEST/"results.csv").open("w",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=columns,extrasaction="ignore");w.writeheader();w.writerows(rows)
    lines=["# Phase 7 — public-only measured pilots","","All metrics below are measured; no new seven-fold benchmark is claimed.",
        "Each fold's decoder and architecture choice uses only its own calibration/validation. The pooled recommendation file is exploratory and does not choose a fold's system.","",
        "| Fold | System | Case | spellF1 | macroF1 | Fragments | Merges |","|---|---|---|---:|---:|---:|---:|"]
    for r in rows:
        if r["system"]=="selected":lines.append(f"| {r['fold']} | {r['arm']} | {r['scenario']} | {r['spellF1']:.5f} | {r['macroF1']:.5f} | {r['fragmented_gt']} | {r['fused_predictions']} |")
    if has_control:
        lines += ["","Additional parameter-count control, all encoders with identical fixed HMM:","",
            "| Fold | System | Case | spellF1 | macroF1 | Fragments | Merges |","|---|---|---|---:|---:|---:|---:|"]
        for r in rows:
            if r["system"]=="fixed_hmm":lines.append(f"| {r['fold']} | {r['arm']} | {r['scenario']} | {r['spellF1']:.5f} | {r['macroF1']:.5f} | {r['fragmented_gt']} | {r['fused_predictions']} |")
    lines += ["","SkillCorner has partial conditional team labels; spellF1 and D metrics are unavailable.","",
        "| System | Case | Conditional H/A accuracy | Conditional macroF1 |","|---|---|---:|---:|"]
    for arm,cases in transfer.items():
        for case,x in cases.items():lines.append(f"| {arm} | {case} | {x['conditional_accuracy_mean']:.5f} | {x['conditional_macroF1_mean']:.5f} |")
    lines += ["","Detailed IoU .3/.4/.5/.6/.7 duration-bin recall, FP/FN, boundaries, loss curves and costs: summary.json, results.csv, learning_curves.png.",
        "","Reproduce serially with `.venv/bin/python -m scripts.run_local TASK`: test-phase7, audit-phase7, run-phase7 --stage pilots, freeze-phase7, run-phase7 --stage test, run-phase7 --stage transfer, profile-phase7, verify-phase7, report-phase7."]
    (DEST/"REPORT.md").write_text("\n".join(lines)+"\n")
    return 0
