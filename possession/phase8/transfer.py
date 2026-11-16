"""Paired provider comparison on the same fixed pilots and confirmation games."""
import json
import time
import numpy as np
import torch
from possession.external_data.schema import file_hash
from possession.learning.corruption import hide_ball,stable_seed
from possession.night.common import atomic_json
from possession.phase5.experiment import scenarios
from scripts.phase6_transfer import conditional_metrics
from .common import Store,DEST,OLD,register,Budget,progress
from .training import load_model
from .evaluation import predict


def run():
    spec=register();torch.set_num_threads(4);budget=Budget(spec);status="failed";store=Store()
    assert (DEST/"best_set_transformer.json").exists()
    scope={"folds":spec["mixed"]["pilot_folds"],"seeds":spec["seeds"],"matches":spec["skillcorner_confirmation_keys"],
        "source_sha256":file_hash(__file__),"selection_used":False,
        "aggregation":"four seed/fold model accuracies averaged within each match, then sixteen matches equally weighted; no H/A/D spellF1 inferred"}
    path=DEST/"transfer_registration.json"
    if path.exists() and json.loads(path.read_text())!=scope:raise ValueError("Immutable transfer scope changed")
    if not path.exists():atomic_json(path,scope)
    oldrows=json.loads((OLD/"transfer.json").read_text())["rows"]
    mixed=json.loads((DEST/"mixed_scope.json").read_text())["folds"]
    descriptors=[(i,s,a,"IDSSE") for i in scope["folds"] for s in spec["seeds"] for a in spec["arms"]]
    descriptors += [(i,s,"set_transformer","IDSSE_SK") for i in scope["folds"] if i in mixed for s in spec["seeds"]]
    models={};rows=[]
    try:
        for j,key in enumerate(scope["matches"]):
            destination=DEST/"transfer"/(key.replace(":","_")+".json")
            if destination.exists():rows.extend(json.loads(destination.read_text()));continue
            view=store.harmonized(key);current=[];progress("provider_transfer",match=j+1,total=len(scope["matches"]))
            for name,(v,missing) in scenarios(view,key,spec).items():
                for i,seed,arm,condition in descriptors:
                    budget.check()
                    if condition=="IDSSE" and seed==2027:
                        source=next(r for r in oldrows if r["match"]==key and r["fold"]==i and r["arm"]==arm and r["scenario"]==name)
                        current.append({**source,"seed":seed,"training_condition":condition,"reused_phase7_metric":True})
                        continue
                    descriptor=(i,seed,arm,condition)
                    if descriptor not in models:models[descriptor]=load_model(i,seed,arm,condition)
                    start=time.perf_counter();p,q,_=predict(models[descriptor],v)
                    file=DEST/f"transfer_probabilities/{condition}/fold-{i}/seed-{seed}/{arm}"/key.replace(":","_")/(name+".npz")
                    file.parent.mkdir(parents=True,exist_ok=True);temporary=file.with_suffix(".tmp")
                    with temporary.open("wb") as stream:np.savez_compressed(stream,probabilities=p,transition=q,time=v.time,run=v.run)
                    temporary.replace(file)
                    current.append({"match":key,"fold":i,"seed":seed,"arm":arm,"training_condition":condition,"scenario":name,
                                    "conditional_team":conditional_metrics(v,p),"inference_s":time.perf_counter()-start,"reused_phase7_metric":False})
            atomic_json(destination,current);rows.extend(current);store.common.pop(key,None)
        atomic_json(DEST/"transfer.json",{"rows":rows,"scope":scope,"note":"A and B never trained/calibrated on these sixteen matches; research cohort previously explored"})
        status="complete"
    finally:budget.close("provider_transfer",status)
