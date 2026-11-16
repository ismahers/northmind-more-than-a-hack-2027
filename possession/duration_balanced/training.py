"""Reuse exact local control when available; change only target loss weights."""
import time
from pathlib import Path
import numpy as np
import torch
from possession.external_data.schema import file_hash
from possession.interval_probe.data import load
from possession.learning.training import seed_everything
from possession.night.common import atomic_json
from possession.temporal_probe.training import make_head, train_epoch, save_state, restore_state
from possession.temporal_probe.windows import EmbeddingWindows
from possession.temporal_probe.teacher import load_teacher
from .weights import train_durations, raw_weights, normalize_train
from .common import read_json


def paths(teacher, key, spec):
    root = Path(teacher["export_root"])/key.replace(":", "_")
    return {s:root/(s+".npz") for s in spec["scenarios"]}


def init_head(fold, teacher, spec):
    return make_head(fold, teacher, "local_classification", spec)


def head_path(dest, fold, arm, teacher):
    return Path(teacher["original_head"]) if arm=="original" and teacher["original_head"] else dest/f"training/fold-{fold}/{arm}/last.pt"


def load_head(dest, fold, arm, teacher, spec):
    model = init_head(fold,teacher,spec)
    state = torch.load(head_path(dest,fold,arm,teacher),map_location="cpu",weights_only=False)
    model.load_state_dict(state["model"])
    return model


def run_training(store, dest, fold, teacher, spec, budget, digest):
    train = teacher["split"]["train"]; data={}; durations={}; label_by_key={}
    for key in train:
        match, view = store.match(key), store.harmonized(key)
        durations[key] = train_durations(match,view,train)
        for name,path in paths(teacher,key,spec).items():
            value=load(path)
            if not np.array_equal(value["native_index"],view.native_index):
                raise ValueError("Parent training clock mismatch")
            data[key+"/"+name]=value
            label_by_key[key]=value["label"]
    for arm in spec["arms"]:
        folder=dest/f"training/fold-{fold}/{arm}"
        if (folder/"complete.json").is_file():
            continue
        cap=spec["duration"]["caps"][arm]
        raw=[raw_weights(durations[k],label_by_key[k],spec["duration"],cap) for k in train]
        normalized,mean=normalize_train(raw,[label_by_key[k] for k in train])
        weights=dict(zip(train,normalized)); records={}
        for identity,value in data.items():
            key,name=identity.rsplit("/",1)
            records[identity]=dict(value,condition_weight=weights[key]*spec["condition_weights"][name])
        dataset=EmbeddingWindows(records,spec["core_frames"],spec["training_halo_frames"])
        weight_record={"native_durations_train_only":True,"raw_cap":cap,"raw_mean":mean,
                       "supervised_frames":{k:int(((y>=0)&(y<2)).sum()) for k,y in label_by_key.items()},
                       "windows":len(dataset),"steps_per_epoch":(len(dataset)+spec["batch_size"]-1)//spec["batch_size"]}
        atomic_json(folder/"weighting.json",weight_record)
        if arm=="original" and teacher["original_head"]:
            previous=read_json(teacher["original_complete"])
            if len(previous["history"])!=spec["epochs"] or any(r["steps"]!=weight_record["steps_per_epoch"] for r in previous["history"]):
                raise ValueError("Original local schedule differs")
            atomic_json(folder/"complete.json",dict(previous,reused_original_checkpoint=teacher["original_head"],
                         reused_checkpoint_sha256=file_hash(teacher["original_head"]),new_training_seconds=0.))
            continue
        seed_everything(spec["seed"]); model=init_head(fold,teacher,spec)
        optimizer=torch.optim.AdamW(model.parameters(),lr=spec["learning_rate"],weight_decay=spec["weight_decay"])
        generator=torch.Generator().manual_seed(spec["seed"])
        path=folder/"last.pt"
        history=restore_state(path,model,optimizer,generator,digest) if path.is_file() else []
        begin=time.perf_counter()
        for epoch in range(len(history)+1,spec["epochs"]+1):
            budget.check(); atomic_json(dest/"progress.json",{"stage":"training","fold":fold,"arm":arm,"epoch":epoch})
            t=time.perf_counter(); metrics=train_epoch(model,optimizer,dataset,generator,spec,budget)
            history.append(dict(metrics,epoch=epoch,seconds=time.perf_counter()-t))
            save_state(path,model,optimizer,generator,history,digest)
        atomic_json(folder/"complete.json",{"history":history,"parameters":sum(p.numel() for p in model.parameters()),
                    "loader_rng":generator.get_state().tolist(),"registration_sha256":digest,
                    "new_training_seconds":time.perf_counter()-begin,"reused_original_checkpoint":None})
