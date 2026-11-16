"""Frozen-model conditional-team transfer on public SkillCorner; no H/A/D claims."""

import json
import time

import numpy as np
import torch

from possession.external_data.features import build_features
from possession.external_data.schema import Match
from possession.learning.corruption import hide_ball,stable_seed
from possession.night.common import atomic_json
from possession.phase6.experiment import DEST,register,load_model,predict,progress
from scripts.train_e0 import manifest


def conditional_metrics(view,p):
    valid=(view.team>=0)&(view.confidence[:,1]>0)
    hard=(p[:,1]>p[:,0]).astype(int)
    result={"labeled_frames":int(valid.sum()),"label_coverage":float(valid.mean())}
    groups={"all":valid,"ball_coordinates_available":valid&(view.global_features[:,3]>.5),
            "ball_coordinates_missing":valid&(view.global_features[:,3]<.5)}
    for name,mask in groups.items():
        cm=np.zeros((2,2),int)
        np.add.at(cm,(view.team[mask].astype(int),hard[mask]),1)
        tp=cm.diagonal();recall=tp/np.maximum(1,cm.sum(1));precision=tp/np.maximum(1,cm.sum(0))
        result[name]={"n":int(mask.sum()),"accuracy":float(tp.sum()/max(1,cm.sum())),
            "macroF1":float(np.mean(2*precision*recall/np.maximum(1e-12,precision+recall))),
            "balanced_accuracy":float(recall.mean()),"confusion":cm.tolist()}
    return result


def main():
    protocol=register();torch.set_num_threads(protocol["cpu_threads"])
    for i in protocol["pilot_folds"]:
        assert (DEST/f"selections/fold-{i}.json").exists()
    data,_=manifest("idsse-skillcorner-phase3")
    records={k:v for k,v in data["artifacts"].items() if k.startswith("skillcorner:")}
    models={(i,arm):load_model(i,arm) for i in protocol["pilot_folds"] for arm in protocol["arms"]}
    started=time.perf_counter();rows=[]
    folder=DEST/"skillcorner_transfer";folder.mkdir(parents=True,exist_ok=True)
    for index,(key,record) in enumerate(sorted(records.items())):
        path=folder/(key.replace(":","_")+".json")
        if path.exists():
            rows.extend(json.loads(path.read_text()));continue
        progress("skillcorner_transfer",match=index+1,total=len(records))
        match=Match.load(record["path"]);view=build_features(match)
        current=[]
        for scenario,length in (("clean",0.),("missing40_5s",5.),("missing40_15s",15.)):
            degraded=view if not length else hide_ball(view,.4,length,stable_seed(key,701))[0]
            for (i,arm),model in models.items():
                p,_=predict(model,degraded)
                current.append({"match":key,"pilot":i,"arm":arm,"scenario":scenario,
                    "conditional_team":conditional_metrics(degraded,p),
                    "native_ball_available_fraction":float(match.ball_available.mean()),
                    "native_player_observation_count_mean":float(match.player_available.sum((1,2)).mean())})
        atomic_json(path,current);rows.extend(current)
        del match,view,degraded
    atomic_json(DEST/"skillcorner_transfer.json",{"rows":rows,"matches":len(records),
        "seconds":time.perf_counter()-started,"models_and_calibrators_never_saw_SkillCorner":True,
        "cohort_seen_in_previous_research":True,"metrics_not_used_for_selection":True,
        "semantics":"conditional team on provider-valid partial labels only; no alive/dead, event or spellF1 ground truth assumed"})
    return 0
