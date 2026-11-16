"""Two-fold train/select seal before held-out inference; fixed existing decoder."""
from pathlib import Path
import time
import numpy as np
import torch
from possession.external_data.schema import file_hash
from possession.interval_probe.data import load
from possession.learning.study import ScoreContext
from possession.learning.temporal import fit_priors
from possession.night.common import atomic_json
from possession.temporal_probe.common import Store, Budget, read_json
from possession.temporal_probe.windows import predict,emissions
from .common import destination,register
from .training import run_training,load_head,paths,head_path
from .selection import select
from .metrics import measure,analyze


def probabilities(dest,fold,arm,teacher,key,spec,part,budget):
    model=load_head(dest,fold,arm,teacher,spec); result={}
    for scenario,path in paths(teacher,key,spec).items():
        budget.check()
        file=dest/f"predictions/fold-{fold}/{arm}/{part}/{scenario}.npz"
        data=load(path)
        if file.is_file():
            with np.load(file,allow_pickle=False) as z:p=z["probabilities"]
        else:
            team,_,_=predict(model,data,spec["core_frames"],spec["batch_size"])
            p=emissions(team,data["probabilities"])
            if not (np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=2e-6)
                    and np.allclose(p[:,2],data["probabilities"][:,2],atol=2e-7)):
                raise ValueError("Frozen live/dead probabilities changed")
            file.parent.mkdir(parents=True,exist_ok=True); tmp=file.with_suffix(".tmp")
            with tmp.open("wb") as stream:
                np.savez_compressed(stream,probabilities=p,time=data["time"],run=data["run"],native_index=data["native_index"])
            tmp.replace(file)
        result[scenario]=(p,data["probabilities"])
    return result


def priors(store,teacher):
    return fit_priors([store.harmonized(k) for k in teacher["split"]["train"]])


def selection(store,dest,fold,teacher,spec,budget,digest):
    path=dest/f"selections/fold-{fold}.json"
    if path.is_file():return
    run_training(store,dest,fold,teacher,spec,budget,digest)
    key=teacher["split"]["validation"]
    context=ScoreContext(store.match(key),store.harmonized(key)); prior=priors(store,teacher)
    trials={}; baseline=None
    for arm in spec["arms"]:
        pack=probabilities(dest,fold,arm,teacher,key,spec,"validation",budget)
        trials[arm]={n:measure(context,p,prior) for n,(p,_) in pack.items()}
        if baseline is None:baseline={n:measure(context,t,prior) for n,(_,t) in pack.items()}
    choice=select(trials,baseline,spec)
    atomic_json(path,dict(choice,trials=trials,teacher=baseline,teacher_descriptor=teacher,
                checkpoint_hashes={a:file_hash(head_path(dest,fold,a,teacher)) for a in spec["arms"]},
                registration_sha256=digest))


def seal(dest,spec,digest):
    value={"selections":{str(i):file_hash(dest/f"selections/fold-{i}.json") for i in spec["pilots"]},
           "registration_sha256":digest,"test_used":False}
    path=dest/"selection-seal.json"
    if path.is_file() and read_json(path)!=value:raise ValueError("Duration selections changed")
    if not path.is_file():atomic_json(path,value)


def test(store,dest,fold,teacher,spec,budget,digest):
    seal(dest,spec,digest)
    path=dest/f"rows/fold-{fold}.json"
    if path.is_file():return read_json(path)
    key=teacher["split"]["test"]
    context=ScoreContext(store.match(key),store.harmonized(key)); prior=priors(store,teacher)
    rows=[]
    for arm in spec["arms"]:
        for name,(p,t) in probabilities(dest,fold,arm,teacher,key,spec,"test",budget).items():
            rows.append({"fold":fold,"arm":arm,"scenario":name,"metrics":measure(context,p,prior),
                         "analysis":analyze(context,p,t,prior)})
            if arm=="original":
                rows.append({"fold":fold,"arm":"teacher","scenario":name,"metrics":measure(context,t,prior),
                             "analysis":analyze(context,t,t,prior)})
    atomic_json(path,rows)
    return rows


def run(source="isma",stage="all"):
    record=register(source)
    dest=destination(source)
    if record is None:
        atomic_json(dest/"status.json",{"status":"blocked_missing_parent_artifacts","trained_teachers":0,"measured_rows":0})
        return 2
    lock=dest/"running.lock"
    with lock.open("x") as stream:stream.write("One duration experiment at a time.\n")
    spec=record["protocol"]; torch.set_num_threads(spec["cpu_threads"])
    budget=Budget(spec["max_compute_seconds"],dest); status="failed"
    store=Store(record["records"]); digest=file_hash(dest/"registration.json")
    try:
        if stage in ("all","select"):
            for fold in spec["pilots"]:
                selection(store,dest,fold,record["teachers"][str(fold)],spec,budget,digest)
            seal(dest,spec,digest)
        if stage in ("all","test"):
            rows=[]
            for fold in spec["pilots"]:
                rows.extend(test(store,dest,fold,record["teachers"][str(fold)],spec,budget,digest))
            atomic_json(dest/"all_rows.json",rows)
        status="complete"
    finally:
        budget.close(stage,status); lock.unlink()
    return 0
