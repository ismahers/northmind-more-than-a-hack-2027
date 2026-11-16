"""Measured public aggregates or explicit missing-parent status, never estimates."""
import csv
import numpy as np
import torch
from possession.external_data.schema import file_hash
from possession.night.common import atomic_json
from .common import ROOT,destination,read_json,register
from .training import head_path


def verify(source):
    record=register(source)
    if record is None:return False
    dest=destination(source); checks={}
    for path,sha in record["sources"].items():checks["source:"+path]=file_hash(ROOT/path)==sha
    for path,sha in record["references"].items():checks["parent:"+path]=file_hash(ROOT/path)==sha
    spec=record["protocol"]; seal=read_json(dest/"selection-seal.json")
    first_test=min(p.stat().st_mtime_ns for p in (dest/"predictions").glob("**/test/*.npz"))
    checks["both_selections_before_new_test"]=(dest/"selection-seal.json").stat().st_mtime_ns<first_test
    for i in spec["pilots"]:
        selected=read_json(dest/f"selections/fold-{i}.json"); teacher=record["teachers"][str(i)]
        checks[f"{i}_no_test_selection"]=not selected["test_used"]
        checks[f"{i}_selection_immutable"]=file_hash(dest/f"selections/fold-{i}.json")==seal["selections"][str(i)]
        steps=[]; rng=[]
        for arm in spec["arms"]:
            complete=read_json(dest/f"training/fold-{i}/{arm}/complete.json")
            steps.append([h["steps"] for h in complete["history"]])
            state=torch.load(head_path(dest,i,arm,teacher),map_location="cpu",weights_only=False)
            rng.append(state["loader_rng"])
            checks[f"{i}/{arm}_8_epochs"]=len(state["history"])==spec["epochs"]
            checks[f"{i}/{arm}_frozen_checkpoint"]=file_hash(head_path(dest,i,arm,teacher))==selected["checkpoint_hashes"][arm]
            checks[f"{i}/{arm}_finite_weights"]=all(bool(torch.isfinite(t).all()) for t in state["model"].values())
        checks[f"{i}_same_updates"]=all(s==steps[0] for s in steps)
        checks[f"{i}_same_sampler_state"]=all(torch.equal(r,rng[0]) for r in rng)
    rows=read_json(dest/"all_rows.json"); checks["24_measured_rows"]=len(rows)==24
    for path in (dest/"predictions").glob("**/*.npz"):
        with np.load(path,allow_pickle=False) as z:
            p=z["probabilities"]
            checks["normalized:"+str(path.relative_to(dest))]=bool(np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=2e-6))
    result={"passed":all(checks.values()),"checks":checks}
    atomic_json(dest/"verification.json",result)
    return result["passed"]


def aggregate(rows):
    result={}
    for arm in sorted(set(r["arm"] for r in rows)):
        result[arm]={}
        for scenario in ("clean","missing40_5s","missing40_15s"):
            chosen=[r for r in rows if r["arm"]==arm and r["scenario"]==scenario]
            ms=[r["metrics"] for r in chosen]; bins={}
            for group in ("lt1","1to3","3to10","ge10"):
                gt=sum(m["duration_groups"][group]["gt"] for m in ms)
                n=sum(m["duration_groups"][group]["iou"]["0.5"]["matched_gt"] for m in ms)
                bins[group]={"gt":gt,"recovered":n,"recall":n/max(1,gt)}
            counts={k:sum(m[k] for m in ms) for k in ("short_recovered","short_gt","false_possessions","fragmented_gt","fused_predictions")}
            details=[d for r in chosen for d in r["analysis"]["per_possession"]]; types={}
            for kind in ("HA_HA","D_D","mixed_HA_D","censored"):
                d=[p for p in details if p["boundary_type"]==kind and p["duration_s"]<3]
                types[kind]={"gt_sub3":len(d),"recovered":sum(p["recovered"] for p in d),
                    "gained_vs_teacher":sum(p["recovered"] and not p["teacher_recovered"] for p in d),
                    "lost_vs_teacher":sum(p["teacher_recovered"] and not p["recovered"] for p in d)}
            result[arm][scenario]={"spellF1_mean":float(np.mean([m["spellF1"] for m in ms])),
                "spellF1_sd_population":float(np.std([m["spellF1"] for m in ms])),
                "macroF1_mean":float(np.mean([m["macroF1"] for m in ms])),"duration_groups":bins,**counts,
                "false_possessions_per_minute":counts["false_possessions"]/sum(m["minutes"] for m in ms),
                "boundary_type_short_spells":types,
                "sub3_frozen_dead_all_frames":sum(p["frozen_dead_on_all_frames"] and p["duration_s"]<3 for p in details),
                "sub3_teacher_raw_HMM_removed":sum(p["teacher_raw_spell_HMM_removed"] and p["duration_s"]<3 for p in details),
                "true_D_predicted_team_oracle_spellF1_mean":float(np.mean([r["analysis"]["true_D_predicted_team_oracle_spellF1"] for r in chosen]))}
    return result


def report(source):
    dest=destination(source)
    if not (dest/"all_rows.json").is_file():
        status=read_json(dest/"readiness.json")
        atomic_json(dest/"summary.json",{"status":"not_executed_missing_parent","measured_rows":0,"readiness":status})
        (dest/"REPORT.md").write_text("# Duration-balanced local head\n\nNo se ha ejecutado entrenamiento ni medido rendimiento real.\nFaltan los artefactos del padre; no se entrenaron nuevos profesores.\n\n"+
            "\n".join("- "+p for p in status["missing"])+"\n",encoding="utf-8")
        return 0
    if not verify(source):return 1
    rows=read_json(dest/"all_rows.json"); record=read_json(dest/"registration.json")
    decisions={str(i):read_json(dest/f"selections/fold-{i}.json") for i in record["protocol"]["pilots"]}
    common=[a for a in record["protocol"]["arms"] if a!="original" and all(v["gates"][a]["passed"] for v in decisions.values())]
    summary={"source":source,"systems":aggregate(rows),"selections":decisions,
        "same_recipe_passes_both_folds":common,"no_automatic_expansion":True,
        "compute":read_json(dest/"compute.json"),"new_teacher_training":0}
    atomic_json(dest/"summary.json",summary)
    fields=["fold","arm","scenario","spellF1","macroF1","short_recall","short_recovered","short_gt","false_possessions_per_minute","fragmented_gt","fused_predictions"]
    with (dest/"results.csv").open("w",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore"); w.writeheader()
        for row in rows:w.writerow({**row,**row["metrics"]})
    lines=["# Duration-balanced local head — resultados medidos","","Referencia: "+source+". Dos pilotos exploratorios; ningún profesor reentrenado.","",
        "| Fold | Pérdida | Caso | spellF1 | macroF1 | Sub3 recuperadas | FP/min | Fusiones | Fragmentos |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        m=row["metrics"]
        lines.append(f"| {row['fold']} | {row['arm']} | {row['scenario']} | {m['spellF1']:.5f} | {m['macroF1']:.5f} | {m['short_recovered']}/{m['short_gt']} | {m['false_possessions_per_minute']:.3f} | {m['fused_predictions']} | {m['fragmented_gt']} |")
    lines+=["","Recetas que pasan los dos folds solo por validación: "+(", ".join(common) if common else "ninguna")+".",
            "No se amplía automáticamente. Oráculos y tipos de frontera son análisis; no seleccionan."]
    (dest/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return 0
