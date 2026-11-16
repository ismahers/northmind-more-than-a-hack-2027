"""Immutable training protocol, public roots and cumulative compute budget."""
from dataclasses import fields
from pathlib import Path
import json
import time
import numpy as np
from possession.external_data.features import Features
from possession.external_data.schema import file_hash
from possession.night.common import atomic_json
from possession.phase7.public import Store as OriginalStore,DEST as OLD
from possession.phase7.features import harmonize
from possession.phase7.experiment import register as verify_parent
from scripts.external_data import ROOT,REPORTS

DEST=REPORTS/"phase8-v1"


def register():
    parent=verify_parent()
    profile=json.loads((REPORTS/"phase8-profile.json").read_text())
    if not profile["passed"]:raise ValueError("Compaction equivalence not verified")
    spec=json.loads((ROOT/"configs/phase8_protocol.json").read_text())
    spec["skillcorner_training_keys"]=parent["skillcorner_development_keys"]
    spec["skillcorner_confirmation_keys"]=parent["skillcorner_confirmation_keys"]
    sources=[ROOT/p for p in ("possession/phase8/common.py","possession/phase8/packing.py","possession/phase8/training.py","possession/phase8/mixing.py")]
    record={"protocol":spec,"sources":{str(p.relative_to(ROOT)):file_hash(p) for p in sources},"parent_registration":file_hash(OLD/"registration.json")}
    path=DEST/"registration.json"
    if path.exists() and json.loads(path.read_text())!=record:raise ValueError("Immutable Phase8 training changed")
    if not path.exists():atomic_json(path,record)
    return spec


class Store(OriginalStore):
    def harmonized(self,key):
        old=OLD/"feature_cache"/key.replace(":","_")/"complete.json"
        if old.exists():return super().harmonized(key)
        if key not in self.common:
            folder=DEST/"feature_cache"/key.replace(":","_")
            if not (folder/"complete.json").exists():
                value=harmonize(self.view(key));folder.mkdir(parents=True,exist_ok=True)
                for f in fields(Features):np.save(folder/(f.name+".npy"),getattr(value,f.name),allow_pickle=False)
                atomic_json(folder/"complete.json",{"artifact":self.records[key]["sha256"],"version":"common-xy-v1"})
            self.common[key]=Features(**{f.name:np.load(folder/(f.name+".npy"),mmap_mode="c",allow_pickle=False) for f in fields(Features)})
        return self.common[key]


def progress(stage,**kwargs):atomic_json(DEST/"progress.json",{"stage":stage,**kwargs})


class Budget:
    def __init__(self,spec):
        self.start=time.perf_counter();self.limit=spec["max_compute_seconds"];self.path=DEST/"compute.json"
        self.saved=json.loads(self.path.read_text()) if self.path.exists() else {"seconds":0.,"sessions":[]}
    def remaining(self):return self.limit-self.saved["seconds"]-time.perf_counter()+self.start
    def check(self):
        if self.remaining()<0:raise TimeoutError("Phase8 cumulative budget reached; checkpoints preserved")
    def close(self,stage,status):
        seconds=time.perf_counter()-self.start;self.saved["seconds"]+=seconds
        self.saved["sessions"].append({"stage":stage,"status":status,"seconds":seconds});atomic_json(self.path,self.saved)
