"""Frozen robustness extension of the single provider-metadata intervention."""

import json
import time

import torch

from possession.external_data.schema import Match,file_hash
from possession.external_data.features import build_features
from possession.learning.corruption import hide_ball,stable_seed
from possession.night.common import atomic_json
from possession.phase6.experiment import DEST,register,load_model,predict,progress
from scripts.phase6_transfer import conditional_metrics
from scripts.phase6_flag_ablation import remove_detection_metadata
from scripts.train_e0 import manifest


def main():
    protocol=register();torch.set_num_threads(protocol["cpu_threads"])
    folder=DEST/"provider_flags_robustness";folder.mkdir(parents=True,exist_ok=True)
    spec={"intervention":"same fixed unknown-detection-metadata policy, after original 40% ball masking",
          "scenarios":["missing40_5s","missing40_15s"],"selection":False,"post_hoc_exploratory":True,
          "source":file_hash(__file__),"intervention_source":file_hash(remove_detection_metadata.__code__.co_filename)}
    frozen=folder/"protocol.json"
    if frozen.exists() and json.loads(frozen.read_text())!=spec:raise ValueError("Robustness extension changed")
    if not frozen.exists():atomic_json(frozen,spec)
    data,_=manifest("idsse-skillcorner-phase3")
    models={(i,a):load_model(i,a) for i in protocol["pilot_folds"] for a in protocol["arms"]}
    records=sorted((k,v) for k,v in data["artifacts"].items() if k.startswith("skillcorner:"))
    start=time.perf_counter();rows=[]
    for index,(key,record) in enumerate(records):
        path=folder/(key.replace(":","_")+".json")
        if path.exists():rows.extend(json.loads(path.read_text()));continue
        progress("flags_robustness",match=index+1,total=len(records))
        match=Match.load(record["path"]);view=build_features(match);current=[]
        for scenario,length in (("missing40_5s",5.),("missing40_15s",15.)):
            hidden=hide_ball(view,.4,length,stable_seed(key,701))[0]
            aligned=remove_detection_metadata(hidden)
            for (i,arm),model in models.items():
                p,_=predict(model,aligned)
                current.append({"match":key,"pilot":i,"arm":arm,"scenario":scenario,"conditional_team":conditional_metrics(aligned,p)})
        atomic_json(path,current);rows.extend(current)
    atomic_json(DEST/"provider_flags_robustness.json",{"rows":rows,"seconds":time.perf_counter()-start,"protocol":spec})
    return 0
