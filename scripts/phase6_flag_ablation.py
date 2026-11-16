"""One post-hoc exploratory input ablation; frozen models, no parameter search."""

from dataclasses import replace
import json
import time

import numpy as np
import torch

from possession.external_data.schema import Match,file_hash
from possession.external_data.features import build_features
from possession.night.common import atomic_json
from possession.phase6.experiment import DEST,register,load_model,predict,progress
from scripts.phase6_transfer import conditional_metrics
from scripts.train_e0 import manifest


def remove_detection_metadata(view):
    nodes,global_features=view.nodes.copy(),view.global_features.copy()
    nodes[...,4:6]=0
    global_features[:,4:6]=0
    for run in np.unique(view.run):
        last=None
        for i in np.flatnonzero(view.run==run):
            if global_features[i,3]>.5:last=view.time[i]
            global_features[i,6]=1 if last is None else min(10,view.time[i]-last)/10
    return replace(view,nodes=nodes,global_features=global_features)


def main():
    protocol=register();torch.set_num_threads(protocol["cpu_threads"])
    folder=DEST/"provider_flag_ablation";folder.mkdir(parents=True,exist_ok=True)
    spec={"operation":"set player/ball detected and detection_known to unknown/zero; recompute age using finite XY; preserve positions, availability, labels",
          "selection":"no parameters; one frozen ablation on all twenty matches after observing transfer drop; exploratory",
          "source_sha256":file_hash(__file__),"conditions":["clean"],"pilots":protocol["pilot_folds"]}
    frozen=folder/"protocol.json"
    if frozen.exists() and json.loads(frozen.read_text())!=spec:raise ValueError("Ablation changed")
    if not frozen.exists():atomic_json(frozen,spec)
    data,_=manifest("idsse-skillcorner-phase3")
    models={(i,a):load_model(i,a) for i in protocol["pilot_folds"] for a in protocol["arms"]}
    records=sorted((k,v) for k,v in data["artifacts"].items() if k.startswith("skillcorner:"))
    rows=[];start=time.perf_counter()
    for index,(key,record) in enumerate(records):
        path=folder/(key.replace(":","_")+".json")
        if path.exists():rows.extend(json.loads(path.read_text()));continue
        progress("provider_flag_ablation",match=index+1,total=len(records))
        match=Match.load(record["path"]);view=build_features(match);ablated=remove_detection_metadata(view)
        assert np.array_equal(view.team,ablated.team) and np.array_equal(view.node_mask,ablated.node_mask)
        current=[]
        for (i,arm),model in models.items():
            p,_=predict(model,ablated)
            current.append({"match":key,"pilot":i,"arm":arm,"conditional_team":conditional_metrics(ablated,p),
                            "original_player_detection_known":float(view.nodes[...,5][view.node_mask].mean()),
                            "original_ball_detection_known":float(view.global_features[:,5].mean())})
        atomic_json(path,current);rows.extend(current)
    atomic_json(DEST/"provider_flag_ablation.json",{"protocol":spec,"rows":rows,"seconds":time.perf_counter()-start})
    return 0
