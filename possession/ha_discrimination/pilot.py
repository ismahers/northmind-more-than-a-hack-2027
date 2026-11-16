"""One predeclared geometry/capacity contrast, two own-fold selections."""
import time
import numpy as np
import torch

from possession.external_data.schema import file_hash
from possession.learning.study import ScoreContext
from possession.learning.temporal import fit_priors
from possession.night.common import atomic_json
from possession.temporal_probe.common import Store,Budget,read_json
from possession.temporal_probe.windows import emissions
from possession.duration_balanced.selection import promotion,objective
from possession.local_head_budget.metrics import measured
from .common import ROOT,DEST,PARENT,register,verify_ancestry
from .training import train_fold,make_model,records_for,predict_geometry


def select(trials,teacher,recipe):
    original=trials["original"];gates={}
    for arm in ("geometry","capacity"):
        check=promotion(trials[arm],original,teacher,recipe)
        check["checks"]["direct_HA_false_transitions"]=all(trials[arm][name]["false_transitions_per_minute"]<=
                original[name]["false_transitions_per_minute"]+.1 for name in recipe["scenarios"])
        if arm=="geometry":
            check["checks"]["beats_capacity_by_005"]=objective(trials[arm],recipe)>=objective(trials["capacity"],recipe)+.005
        check["passed"]=all(check["checks"].values());gates[arm]=check
    allowed=["original"]+[arm for arm in ("geometry","capacity") if gates[arm]["passed"]]
    return {"winner":max(allowed,key=lambda arm:objective(trials[arm],recipe)),"gates":gates,
            "objectives":{arm:objective(value,recipe) for arm,value in trials.items()},"test_used":False}


def score_part(store,record,fold,part,budget,digest):
    recipe=record["recipe"];teacher=record["teachers"][str(fold)];key=teacher["split"][part]
    if part=="test":assert_sealed(record,digest)
    data=records_for(store,teacher,key,recipe)
    context=ScoreContext(store.match(key),store.harmonized(key))
    prior=fit_priors([store.harmonized(k) for k in teacher["split"]["train"]])
    teacher_metrics={name:measured(context,value["probabilities"],prior) for name,value in data.items()}
    trials={}
    for arm in record["protocol"]["pilot"]["arms"]:
        model=make_model(fold,teacher,arm,recipe)
        path=DEST/f"pilot/training/fold-{fold}/{arm}/last.pt"
        state=torch.load(path,map_location="cpu",weights_only=False)
        if state["registration_sha256"]!=digest or len(state["history"])!=8:raise ValueError("Wrong pilot checkpoint")
        model.load_state_dict(state["model"]);trials[arm]={}
        for name,value in data.items():
            budget.check();p=emissions(predict_geometry(model,value,recipe,arm),value["probabilities"])
            if not np.allclose(p[:,2],value["probabilities"][:,2],atol=2e-7,rtol=0):raise ValueError("Frozen D emissions changed")
            if arm=="original":
                with np.load(PARENT/f"predictions/fold-{fold}/cap-8/{part}/{name}.npz",allow_pickle=False) as z:
                    if not np.array_equal(p,z["probabilities"]):raise ValueError("Original control predictions differ")
            file=DEST/f"pilot/predictions/fold-{fold}/{arm}/{part}/{name}.npz"
            file.parent.mkdir(parents=True,exist_ok=True)
            with file.with_suffix(".tmp").open("wb") as stream:np.savez_compressed(stream,probabilities=p)
            file.with_suffix(".tmp").replace(file)
            trials[arm][name]=measured(context,p,prior)
    return trials,teacher_metrics


def seal(record,digest):
    value={"registration_sha256":digest,"test_used":False,
           "selections":{str(i):file_hash(DEST/f"pilot/selections/fold-{i}.json") for i in record["protocol"]["pilots"]}}
    path=DEST/"pilot/selection-seal.json"
    if path.exists() and read_json(path)!=value:raise ValueError("Pilot selection seal changed")
    if not path.exists():atomic_json(path,value)


def assert_sealed(record,digest):
    value=read_json(DEST/"pilot/selection-seal.json")
    if value["registration_sha256"]!=digest or value["test_used"]:raise ValueError("Invalid pilot seal")
    for i in record["protocol"]["pilots"]:
        if value["selections"][str(i)]!=file_hash(DEST/f"pilot/selections/fold-{i}.json"):raise ValueError("Pilot selections changed")


def report(record,digest):
    assert_sealed(record,digest)
    selections={str(i):read_json(DEST/f"pilot/selections/fold-{i}.json") for i in record["protocol"]["pilots"]}
    checks=verify_ancestry(record)
    registration=read_json(DEST/"pilot/registration.json")
    checks.update({"pilot_source:"+name:file_hash(ROOT/name)==sha for name,sha in registration["pilot_sources"].items()})
    files=list((DEST/"pilot/predictions").glob("**/test/*.npz"))
    checks["both_selections_before_test"]=bool(files) and (DEST/"pilot/selection-seal.json").stat().st_mtime_ns<min(p.stat().st_mtime_ns for p in files)
    for fold in record["protocol"]["pilots"]:
        rng=[];steps=[]
        for arm in record["protocol"]["pilot"]["arms"]:
            state=torch.load(DEST/f"pilot/training/fold-{fold}/{arm}/last.pt",map_location="cpu",weights_only=False)
            rng.append(state["loader_rng"]);steps.append([r["steps"] for r in state["history"]])
            checks[f"{fold}/{arm}/epochs8"]=len(state["history"])==8
            checks[f"{fold}/{arm}/checkpoint_frozen"]=file_hash(DEST/f"pilot/training/fold-{fold}/{arm}/last.pt")==selections[str(fold)]["checkpoint_hashes"][arm]
        checks[f"{fold}/same_sampler"]=all(torch.equal(rng[0],r) for r in rng)
        checks[f"{fold}/same_updates"]=all(steps[0]==value for value in steps)
    allfiles=list((DEST/"pilot/predictions").glob("**/*.npz"))
    checks["36_pilot_prediction_files"]=len(allfiles)==36
    for path in allfiles:
        with np.load(path,allow_pickle=False) as z:
            p=z["probabilities"];checks["normalized:"+str(path.relative_to(DEST))]=bool(np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=2e-6))
    atomic_json(DEST/"pilot/verification.json",{"passed":all(checks.values()),"count":len(checks),"checks":checks})
    if not all(checks.values()):return 1
    rows=read_json(DEST/"pilot/all_rows.json")
    costs={str(i):{arm:read_json(DEST/f"pilot/training/fold-{i}/{arm}/complete.json") for arm in record["protocol"]["pilot"]["arms"]} for i in record["protocol"]["pilots"]}
    success=all(s["gates"]["geometry"]["passed"] for s in selections.values())
    atomic_json(DEST/"pilot/summary.json",{"selections":selections,"test":rows,"geometry_passes_both_own_validations":success,
                "no_automatic_expansion":True,"verification_passed":True,"verification_count":len(checks),
                "registered_compute":read_json(DEST/"pilot/compute.json"),
                "training_seconds":sum(row["seconds"] for group in costs.values() for value in group.values() for row in value["history"]),
                "parameters":{arm:costs["0"][arm]["parameters"] for arm in record["protocol"]["pilot"]["arms"]},"new_teachers":0,"new_embedding_exports":0})
    lines=["# Physics HA pilot — fixed8 epochs","","| Fold | Arm | Robust validation | Selected |","|---|---|---:|---|"]
    for fold,s in selections.items():
        for arm,value in s["objectives"].items():lines.append(f"| {fold} | {arm} | {value:.5f} | {arm==s['winner']} |")
    lines += ["","| Fold | Arm | Scenario | Test spellF1 | Short recovered | Long recall | Fused | False HA/min |","|---|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        m=row["metrics"];lines.append(f"| {row['fold']} | {row['arm']} | {row['scenario']} | {m['spellF1']:.5f} | {m['short_recovered']}/{m['short_gt']} | {m['long_recall']:.4f} | {m['fused_predictions']} | {m['false_transitions_per_minute']:.3f} |")
    (DEST/"pilot/REPORT.md").write_text("\n".join(lines)+"\n")
    return 0


def run(stage="pilot"):
    torch.use_deterministic_algorithms(True);record=register();torch.set_num_threads(record["protocol"]["cpu_threads"])
    hypothesis=read_json(DEST/"hypothesis.json")
    if not hypothesis["pilot_justified"] or hypothesis["test_used"]:raise ValueError("No validation-justified physics hypothesis")
    paths=["possession/ha_discrimination/"+name+".py" for name in ("model","training","pilot")]
    registration={"diagnostic_registration_sha256":file_hash(DEST/"registration.json"),"hypothesis_sha256":file_hash(DEST/"hypothesis.json"),
                  "protocol":record["protocol"]["pilot"],"recipe":record["recipe"],"pilot_sources":{p:file_hash(ROOT/p) for p in paths}}
    marker=DEST/"pilot/registration.json"
    if marker.exists() and read_json(marker)!=registration:raise ValueError("Immutable physics pilot changed")
    if not marker.exists():atomic_json(marker,registration)
    digest=file_hash(marker)
    if stage=="report":return report(record,digest)
    if (DEST/"pilot/summary.json").exists():return report(record,digest)
    lock=DEST/"pilot/running.lock"
    with lock.open("x") as stream:stream.write("One fixed physics pilot only.\n")
    budget=Budget(record["protocol"]["max_compute_seconds"],DEST/"pilot");status="failed";store=Store(record["records"])
    try:
        for fold in record["protocol"]["pilots"]:
            path=DEST/f"pilot/selections/fold-{fold}.json"
            if path.exists():continue
            teacher=record["teachers"][str(fold)]
            train_fold(store,fold,teacher,record["recipe"],record["protocol"],budget,digest)
            trials,baseline=score_part(store,record,fold,"validation",budget,digest)
            choice=select(trials,baseline,record["recipe"])
            atomic_json(path,dict(choice,trials=trials,teacher=baseline,checkpoint_hashes={arm:file_hash(DEST/f"pilot/training/fold-{fold}/{arm}/last.pt") for arm in record["protocol"]["pilot"]["arms"]}))
        seal(record,digest)
        rows=[]
        for fold in record["protocol"]["pilots"]:
            trials,baseline=score_part(store,record,fold,"test",budget,digest)
            for arm,metrics in dict(trials,teacher=baseline).items():
                rows.extend({"fold":fold,"arm":arm,"scenario":name,"metrics":value} for name,value in metrics.items())
        atomic_json(DEST/"pilot/all_rows.json",rows);status="complete"
    finally:
        budget.close(stage,status);lock.unlink()
    return report(record,digest)
