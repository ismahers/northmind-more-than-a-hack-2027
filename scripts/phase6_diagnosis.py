"""Validation-only spatial/temporal diagnostics. Never chooses model parameters."""

import json

import numpy as np

from possession.night.common import atomic_json
from possession.learning.study import ScoreContext
from possession.phase5.experiment import get_store,scenarios
from possession.phase6.experiment import DEST,register
from possession.phase6.model import transition_targets
from scripts.train_e0 import manifest
from scripts.phase3 import extra_diagnostics
from possession.learning.temporal import TemporalDecoder


def accuracy(truth,pred,mask):
    return {"n":int(mask.sum()),"accuracy":float((truth[mask]==pred[mask]).mean()) if mask.any() else None}


def main():
    protocol=register();data,_=manifest("idsse-only");store=get_store(data);rows=[]
    for i in protocol["pilot_folds"]:
        selection=json.loads((DEST/f"selections/fold-{i}.json").read_text())
        key=selection["split"]["validation"]
        view=store.view(key);match=store.match(key)
        boundary=transition_targets(match,view,2,.5)[0].max(-1)>0
        for arm in protocol["arms"]:
            for case,(degraded,_) in scenarios(view,key,protocol).items():
                path=DEST/f"predictions/fold-{i}/{arm}/validation/{case}.npz"
                with np.load(path,allow_pickle=False) as z:p=z["probabilities"]
                hard=p[:,:2].argmax(-1)
                valid=(view.alive==1)&(view.team>=0)&(view.confidence[:,1]>0)
                observed=degraded.global_features[:,3]>.5
                distance=np.linalg.norm(degraded.nodes[...,2:4]*np.array([52.5,34.]),axis=-1)
                distance[~degraded.node_mask]=np.inf
                nearest=distance.min(-1)
                geometric=nearest.argmin(-1);close=nearest.min(-1)<=2.
                strata={"alive":valid,"near_HA_boundary":valid&boundary,"interior":valid&~boundary,
                        "coordinates_missing":valid&~observed,"observed_within_2m":valid&observed&close,
                        "observed_farther_than_2m":valid&observed&~close}
                measured={name:{"TCN":accuracy(view.team,hard,mask),
                          "nearest_geometry":accuracy(view.team,geometric,mask&observed)} for name,mask in strata.items()}
                shifts={}
                for offset in (-10,-5,0,5,10):
                    shifted=hard.copy()
                    for run in np.unique(view.run):
                        ix=np.flatnonzero(view.run==run)
                        shifted[ix]=hard[ix[np.clip(np.arange(len(ix))+offset,0,len(ix)-1)]]
                    shifts[str(offset/10)]=accuracy(view.team,shifted,valid)
                oracle=extra_diagnostics(match,view,p,TemporalDecoder({"kind":"argmax"},{}))["oracles_analysis_only"]
                rows.append({"fold":i,"arm":arm,"scenario":case,"strata":measured,"temporal_shift_s":shifts,"oracles_analysis_only":oracle})
    atomic_json(DEST/"validation_diagnosis.json",{"rows":rows,"used_for_model_selection":False,
        "note":"Nearest geometry is a heuristic, shifts/oracles are diagnostics, not deployable choices or proof of causal capacity limitations"})
    return 0
