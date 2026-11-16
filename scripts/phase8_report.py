"""Measured aggregates; average seeds within match before reporting variability."""
import csv
import json
from collections import defaultdict
import numpy as np
from possession.phase8.common import DEST,register
from possession.phase8.training import source_folder
from possession.night.common import atomic_json

CASES=("clean","missing40_5s","missing40_15s")
BINS=("lt1","1to3","3to10","ge10")


def read(path):return json.loads(path.read_text())


def stats(values):return {"mean":float(np.mean(values)),"sd_population":float(np.std(values)),"n":len(values)}


def aggregate(rows):
    games=defaultdict(list)
    for r in rows:games[r["fold"]].append(r)
    result={"matches":len(games),"model_runs":len(rows),"aggregation":"average seeds within match; then equally weighted matches; population SD across matches"}
    for k in ("spellF1","macroF1","team_accuracy_true_alive","alive_dead_F1","actual_hidden_fraction"):
        result[k]=stats([np.mean([r[k] for r in g]) for g in games.values()])
    result["counts_sum_matches_mean_seeds"]={k:float(sum(np.mean([r[k] for r in g]) for g in games.values())) for k in
        ("gt_spells","pred_spells","fragmented_gt","fused_predictions","false_D_interruptions","false_team_changes")}
    bins={}
    for b in BINS:
        items=[r["duration_groups"][b] for r in rows];total=sum(x["gt"] for x in items)
        bins[b]={"gt_model_runs":total,"matched_gt_model_runs":sum(x["iou"]["0.5"]["matched_gt"] for x in items),
                 "recall_IoU0_5":sum(x["iou"]["0.5"]["matched_gt"] for x in items)/max(1,total),
                 "recall_by_IoU":{t:sum(x["iou"][t]["matched_gt"] for x in items)/max(1,total) for t in ("0.3","0.4","0.5","0.6","0.7")}}
    result["duration_groups"]=bins
    return result


def transfer_summary(rows):
    result={}
    for condition,arm in (("IDSSE","harmonized"),("IDSSE","set_transformer"),("IDSSE_SK","set_transformer")):
        key=condition+"/"+arm;result[key]={}
        for case in CASES:
            games=defaultdict(list)
            for r in rows:
                if (r["training_condition"],r["arm"],r["scenario"])==(condition,arm,case):games[r["match"]].append(r["conditional_team"])
            result[key][case]={"accuracy":stats([np.mean([x["all"]["accuracy"] for x in g]) for g in games.values()]),
                              "macroF1":stats([np.mean([x["all"]["macroF1"] for x in g]) for g in games.values()]),
                              "models_per_match":sorted(set(map(len,games.values()))),"match_count":len(games)}
    return result


def diagnosis_summary(rows):
    result={}
    for case in CASES:
        picked=[r for r in rows if r["scenario"]==case];spells=[s for r in picked for s in r["per_possession"]]
        d={k:sum(r[k] for r in picked) for k in ("direct_HA_truth","joint_argmax_detected","head_argmax_detected","auxiliary_candidates_detected","network_union_detected", "network_present_HMM_absent","network_present_selected_absent","not_identified_by_operational_network_candidates","HMM_found_without_operational_network_candidate")}
        c={k:sum(r["candidate_metrics"][k] for r in picked) for k in ("tp","fp","fn","minutes")}
        c.update(precision=c["tp"]/max(1,c["tp"]+c["fp"]),recall=c["tp"]/max(1,c["tp"]+c["fn"]),false_per_minute=c["fp"]/max(1e-9,c["minutes"]))
        d["candidates"]=c;d["decoders"]={}
        for name in picked[0]["decoders"]:
            items=[r["decoders"][name] for r in picked];bins={}
            for b in BINS:
                g=[x["duration_groups"][b] for x in items];n=sum(x["gt"] for x in g);tp=sum(x["iou"]["0.5"]["matched_gt"] for x in g)
                bins[b]={"gt":n,"matched":tp,"recall":tp/max(1,n)}
            d["decoders"][name]={"spellF1":stats([x["spellF1"] for x in items]),"duration_groups":bins,
                "fragmented_gt":sum(x["fragmented_gt"] for x in items),"fused_predictions":sum(x["fused_predictions"] for x in items)}
        d["spells"]={}
        for b,lo,hi in (("lt1",0,1),("1to3",1,3),("3to10",3,10),("ge10",10,float("inf"))):
            group=[s for s in spells if lo<=s["duration_s"]<hi]
            d["spells"][b]={"count":len(group),"no_10Hz_anchor":sum(s["model_anchor_centres_inside"]==0 for s in group),
                "both_network_endpoints":sum(s["both_network_endpoints_present"] for s in group),"both_HMM_endpoints":sum(s["both_hmm_endpoints_present"] for s in group),
                "false_D_inside_HMM_spells":sum(s["false_D_frames_hmm"]>0 for s in group),"wrong_team_inside_HMM_spells":sum(s["wrong_team_frames_hmm"]>0 for s in group),
                "endpoint_types":{k:sum(s["start_type"]==k for s in group)+sum(s["end_type"]==k for s in group) for k in ("HA","D_restart","D_stop","censored")}}
        result[case]=d
    return result


def training_costs(spec):
    result={};all_history=defaultdict(list)
    scope=read(DEST/"mixed_scope.json")
    for seed in spec["seeds"]:
        for i in spec["folds"]:
            for condition,arm in [("IDSSE",a) for a in spec["arms"]]+([("IDSSE_SK","set_transformer")] if i in scope["folds"] else []):
                source=source_folder(i,seed,arm,condition);reused=condition=="IDSSE" and seed==2027 and i in (0,6)
                for stage in (("base","binary") if condition=="IDSSE" else ("binary",)):
                    d=read(source/stage/"complete.json");tag=f"{condition}/{i}/{seed}/{arm}/{stage}"
                    result[tag]={k:d[k] for k in ("seconds","parameters","memory")}
                    result[tag]["reused_previous_phase"]=reused
                    result[tag]["history"]=d["history"]
                    if not reused:all_history[(condition,arm,stage)]+=d["history"]
    def comparable_loss(r):
        t=r["train"];return t.get("loss",t.get("state_loss",0)+t.get("team_loss",0))
    summaries={"/".join(k):{"epochs":len(h),"training_epoch_seconds":stats([r["training_seconds"] for r in h]),
            "first_state_team_loss":float(np.mean([comparable_loss(r) for r in h if r["epoch"]==1])),
            "last_state_team_loss":float(np.mean([comparable_loss(r) for r in h if r["epoch"]==max(x["epoch"] for x in h)]))} for k,h in all_history.items()}
    return {"stages":result,"new_epoch_summary":summaries}


def probe_summary():
    dest=DEST.parent/"phase8-interval-v1"
    if not (dest/"all_rows.json").exists():return {"status":"not_completed"}
    rows=read(dest/"all_rows.json");result={"configuration_results":{},"selections":{str(i):read(dest/f"selections/fold-{i}.json") for i in (0,6)},"compute":read(dest/"compute.json")}
    for kind,system in sorted(set((r["loss_kind"],r["system"]) for r in rows)):
        result["configuration_results"][kind+"/"+system]={c:aggregate([r for r in rows if (r["loss_kind"],r["system"],r["scenario"])==(kind,system,c)]) for c in CASES}
    winners=[]
    for i in (0,6):
        choice=result["selections"][str(i)]["winner"]
        for c in CASES:
            winners.append(next(r for r in rows if r["fold"]==i and r["scenario"]==c and ((choice["kind"]=="teacher" and r["system"]=="teacher_hmm") or (r["loss_kind"]==choice["kind"] and r["system"]=="proposal"))))
    result["validation_selected_including_teacher"]={c:aggregate([r for r in winners if r["scenario"]==c]) for c in CASES}
    return result


def main():
    spec=register();rows=read(DEST/"all_rows.json");summary={"primary":{},"seed_results":{},"paired_folds":[]}
    for arm in spec["arms"]:
        summary["primary"][arm]={s:{c:aggregate([r for r in rows if (r["training_condition"],r["arm"],r["system"],r["scenario"])==("IDSSE",arm,s,c)]) for c in CASES} for s in ("fixed_hmm","fixed_hsmm","selected")}
        summary["seed_results"][arm]={str(seed):{s:{c:aggregate([r for r in rows if (r["training_condition"],r["arm"],r["system"],r["scenario"],r["seed"])==("IDSSE",arm,s,c,seed)]) for c in CASES} for s in ("fixed_hmm","selected")} for seed in spec["seeds"]}
    for i in spec["folds"]:
        d={"fold":i};delta=[]
        for seed in spec["seeds"]:
            for a in spec["arms"]:
                r=next(r for r in rows if (r["training_condition"],r["fold"],r["seed"],r["arm"],r["system"],r["scenario"])==("IDSSE",i,seed,a,"fixed_hmm","clean"))
                d[f"{a}_{seed}"]=r["spellF1"];d["match"]=r["match"]
            delta.append(d[f"set_transformer_{seed}"]-d[f"harmonized_{seed}"])
        d["paired_delta_mean"]=float(np.mean(delta));d["paired_delta_per_seed"]=delta;summary["paired_folds"].append(d)
    scope=read(DEST/"mixed_scope.json");summary["mixing"]={"scope":scope,"paired_results":{}}
    best=read(DEST/"best_set_transformer.json")["folds"]
    summary["best_ST_own_validation"]={c:aggregate([r for r in rows if r["scenario"]==c and r["arm"]=="set_transformer" and r["system"]=="selected" and r["seed"]==best[str(r["fold"])]["seed"] and r["training_condition"]==best[str(r["fold"])]["condition"]]) for c in CASES}
    for condition in ("IDSSE","IDSSE_SK"):
        summary["mixing"]["paired_results"][condition]={s:{c:aggregate([r for r in rows if r["fold"] in scope["folds"] and (r["training_condition"],r["arm"],r["system"],r["scenario"])==(condition,"set_transformer",s,c)]) for c in CASES} for s in ("fixed_hmm","selected")}
    summary.update(transfer=transfer_summary(read(DEST/"transfer.json")["rows"]),diagnosis=diagnosis_summary(read(DEST/"diagnosis.json")["rows"]),
        costs=training_costs(spec),compute=read(DEST/"compute.json"),probe=probe_summary(),verification=read(DEST/"verification.json"),seed_variability=read(DEST/"seed_variability.json"))
    atomic_json(DEST/"summary.json",summary)
    columns=[k for k in rows[0] if not isinstance(rows[0][k],(dict,list))]
    with (DEST/"results.csv").open("w",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=columns,extrasaction="ignore");w.writeheader();w.writerows(rows)
    probe_dest=DEST.parent/"phase8-interval-v1"
    if (probe_dest/"all_rows.json").exists():
        probe_rows=read(probe_dest/"all_rows.json")
        with (probe_dest/"results.csv").open("w",newline="") as stream:
            w=csv.DictWriter(stream,fieldnames=[k for k in probe_rows[0] if not isinstance(probe_rows[0][k],(dict,list))]+["proposal_count","proposal_strength"],extrasaction="ignore")
            w.writeheader();w.writerows(probe_rows)
    lines=["# Fase 8 — resultados medidos con datos públicos externos","","Confirmación exploratoria: los siete partidos y la cohorte SkillCorner ya se habían consultado. Dos semillas no son catorce partidos independientes.","",
        "## Comparación emparejada, HMM idéntico, tracking limpio","","| Partido | DS 2027 | DS 2028 | ST 2027 | ST 2028 | Δ media ST−DS |","|---|---:|---:|---:|---:|---:|"]
    for d in summary["paired_folds"]:lines.append(f"| {d['match'].split(':')[1]} | {d['harmonized_2027']:.5f} | {d['harmonized_2028']:.5f} | {d['set_transformer_2027']:.5f} | {d['set_transformer_2028']:.5f} | {d['paired_delta_mean']:+.5f} |")
    lines += ["","## Media de semillas por partido y desviación poblacional entre siete partidos","","| Codificador | Decodificador | Caso | spellF1 ± SD | macroF1 | Fusiones* | Fragmentaciones* |","|---|---|---|---:|---:|---:|---:|"]
    for a,systems in summary["primary"].items():
        for s,cases in systems.items():
            for c,m in cases.items():
                v=m["spellF1"];n=m["counts_sum_matches_mean_seeds"]
                lines.append(f"| {a} | {s} | {c} | {v['mean']:.5f} ± {v['sd_population']:.5f} | {m['macroF1']['mean']:.5f} | {n['fused_predictions']:g} | {n['fragmented_gt']:g} |")
    lines += ["","*Suma por partidos tras promediar semillas; se permiten medios enteros. Conteos y recalls completos por IoU: summary.json.","","## Semillas, tracking limpio","","| Codificador | Semilla | HMM spellF1 ± SD | Selección propia spellF1 ± SD |","|---|---:|---:|---:|"]
    for a,seeds in summary["seed_results"].items():
        for seed,data in seeds.items():
            x,y=[data[s]["clean"]["spellF1"] for s in ("fixed_hmm","selected")]
            lines.append(f"| {a} | {seed} | {x['mean']:.5f} ± {x['sd_population']:.5f} | {y['mean']:.5f} ± {y['sd_population']:.5f} |")
    lines += ["","## Mezcla de proveedores: mismos dos folds y semillas","","| Entrenamiento | Caso | spellF1 selección interna | macroF1 |","|---|---|---:|---:|"]
    for condition,systems in summary["mixing"]["paired_results"].items():
        for c,m in systems["selected"].items():lines.append(f"| {condition} | {c} | {m['spellF1']['mean']:.5f} | {m['macroF1']['mean']:.5f} |")
    lines += ["","La mezcla no pasó el criterio interno de ampliación; no se ejecutaron otros diez entrenamientos mixtos.","","## Transferencia SkillCorner (16 partidos, cuatro modelos por condición)","","| Modelo | Caso | Exactitud H/A | macroF1 condicional |","|---|---|---:|---:|"]
    for a,cases in summary["transfer"].items():
        for c,m in cases.items():lines.append(f"| {a} | {c} | {m['accuracy']['mean']:.5f} | {m['macroF1']['mean']:.5f} |")
    lines += ["","Son métricas condicionales sobre etiquetas parciales, no spellF1 H/A/D.","","## Diagnóstico del ST elegido en cada validación interna","","| Caso | Decodificador | spellF1 | <1 s R@.5 | 1–3 s | 3–10 s | ≥10 s | Fusiones | Fragmentos |","|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c,d in summary["diagnosis"].items():
        for name,m in d["decoders"].items():
            r=[m["duration_groups"][b]["recall"] for b in BINS]
            lines.append(f"| {c} | {name} | {m['spellF1']['mean']:.5f} | {r[0]:.3%} | {r[1]:.3%} | {r[2]:.3%} | {r[3]:.3%} | {m['fused_predictions']} | {m['fragmented_gt']} |")
    lines += ["","La reconstrucción GT25→10→25 usa etiquetas únicamente como diagnóstico; no es un modelo ni una cota estricta de información.","","## Sonda de intervalos congelados","","| Pérdida/sistema | Caso | spellF1 | macroF1 | <1 s R@.5 | 1–3 s |","|---|---|---:|---:|---:|---:|"]
    for name,cases in summary["probe"].get("configuration_results",{}).items():
        for c,m in cases.items():lines.append(f"| {name} | {c} | {m['spellF1']['mean']:.5f} | {m['macroF1']['mean']:.5f} | {m['duration_groups']['lt1']['recall_IoU0_5']:.3%} | {m['duration_groups']['1to3']['recall_IoU0_5']:.3%} |")
    if summary["probe"].get("configuration_results"):
        lines += ["","Selección de sonda por validación, resultados por fold:","","| Fold | Caso | Profesor HMM | Ganador interno | spellF1 ganador |","|---|---|---:|---|---:|"]
        for i in (0,6):
            chosen=summary["probe"]["selections"][str(i)]["winner"]
            for c in CASES:
                teacher=next(r for r in probe_rows if r["fold"]==i and r["scenario"]==c and r["system"]=="teacher_hmm")
                winner=teacher if chosen["kind"]=="teacher" else next(r for r in probe_rows if (r["fold"],r["scenario"],r["loss_kind"],r["system"])==(i,c,chosen["kind"],"proposal"))
                lines.append(f"| {i} | {c} | {teacher['spellF1']:.5f} | {chosen['kind']}, bonus={chosen['strength']:g} | {winner['spellF1']:.5f} |")
    lines += ["","Selecciones de sonda, calibradores, controles sin propuestas, curvas, tiempos y verificaciones: summary.json. No se implementó TriDet completo ni se cargaron pesos PathCRF."]
    (DEST/"REPORT.md").write_text("\n".join(lines)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for row,i in enumerate((0,6)):
        for a in spec["arms"]:
            for seed in spec["seeds"]:
                h=[]
                for stage in ("base","binary"):h+=read(source_folder(i,seed,a)/stage/"complete.json")["history"]
                x=[r["global_epoch"] for r in h]
                # State+team losses share the same meaning before/after auxiliary training.
                axes[row,0].plot(x,[r["train"].get("loss",r["train"].get("state_loss",0)+r["train"].get("team_loss",0)) for r in h],label=f"{a} {seed}")
                axes[row,1].plot(x,[r["validation"]["loss"] for r in h],label=f"{a} {seed}")
        for col in (0,1):axes[row,col].set_title(f"Fold {i}: "+("train state+team" if col==0 else "inner validation state+team"));axes[row,col].grid(alpha=.2);axes[row,col].legend(fontsize=7);axes[row,col].set_xlabel("Epoch; auxiliary head starts at13")
    fig.tight_layout();fig.savefig(DEST/"learning_curves.png",dpi=140);plt.close(fig)
    return 0
