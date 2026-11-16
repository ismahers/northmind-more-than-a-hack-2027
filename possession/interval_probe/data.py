"""Frozen public-only teacher exports, with native full-possession regression targets."""
import json
import numpy as np
import torch
from possession.external_data.audit import joint_labels,segments
from possession.external_data.schema import file_hash
from possession.phase5.experiment import scenarios
from possession.night.common import atomic_json
from possession.phase8.common import DEST as MAIN
from possession.phase8.evaluation import predict,selection_path
from possession.phase8.training import load_model,source_folder
from scripts.external_data import REPORTS,ROOT

DEST=REPORTS/"phase8-interval-v1"


def native_targets(match,view):
    labels=joint_labels(match);y=labels[view.native_index].astype(np.int64)
    if (y<0).any():raise ValueError("Native complete labels required for interval targets")
    distance=np.zeros((len(y),2),np.float32);weight=np.zeros(len(y),np.float32);valid=np.zeros(len(y),bool)
    for run,(a,b) in enumerate(segments(match)):
        changes=np.r_[a,np.flatnonzero(labels[a+1:b]!=labels[a:b-1])+a+1,b]
        selected=np.flatnonzero(view.run==run)
        for left,right in zip(changes[:-1],changes[1:]):
            if labels[left]>=2 or labels[left]<0:continue
            lo,hi=np.searchsorted(view.native_index[selected],[left,right])
            ix=selected[lo:hi]
            start,end=match.time[left],match.time[right-1]+1/match.metadata["native_fps"]
            distance[ix,0]=np.maximum(0,view.time[ix]-start);distance[ix,1]=np.maximum(0,end-view.time[ix])
            valid[ix]=(view.time[ix]>=start)&(view.time[ix]<end)
            weight[ix]=1/max(.1,end-start)
    return {"label":y,"distance":distance,"regression_weight":weight,"regression_valid":valid}


def teacher(i,spec):
    choices=[json.loads(selection_path(i,s,"set_transformer").read_text()) for s in spec["seeds"]]
    best=max(choices,key=lambda x:x["winner"]["objective"])
    return {"seed":best["seed"],"split":best["split"],"checkpoint":str(source_folder(i,best["seed"],"set_transformer")/"binary/last.pt")}


def export(store,key,i,chosen,main_spec):
    result={};view=store.harmonized(key);targets=None;model=None
    for name,(v,missing) in scenarios(view,key,main_spec).items():
        path=DEST/f"embeddings/fold-{i}"/key.replace(":","_")/(name+".npz")
        if not path.exists():
            if model is None:model=load_model(i,chosen["seed"],"set_transformer")
            if targets is None:targets=native_targets(store.match(key),view)
            p,q,h=predict(model,v,embeddings=True)
            if not (np.isfinite(h).all() and np.allclose(p.sum(1),1,atol=2e-6)):raise ValueError("Invalid teacher export")
            path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(".tmp")
            with temporary.open("wb") as stream:np.savez_compressed(stream,hidden=h,probabilities=p,transition=q,time=v.time,run=v.run,native_index=v.native_index,actual_hidden_fraction=np.asarray(missing.mean()),**targets)
            temporary.replace(path)
        result[name]=path
    return result


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
