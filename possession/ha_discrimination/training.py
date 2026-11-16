"""One fixed8-epoch paired experiment; label-free geometry in a separate input."""
import time
import numpy as np
import torch

from possession.external_data.schema import file_hash
from possession.interval_probe.data import load
from possession.learning.training import seed_everything
from possession.night.common import atomic_json
from possession.phase5.experiment import scenarios
from possession.temporal_probe.windows import EmbeddingWindows,collate
from possession.temporal_probe.model import loss_for_core
from possession.temporal_probe.training import save_state,restore_state
from possession.duration_balanced.training import init_head,paths
from possession.local_head_budget.training import state_equal
from .common import DEST,ROOT
from .features import geometry
from .model import PhysicalHead


class GeoWindows(EmbeddingWindows):
    def __getitem__(self,index):
        item=super().__getitem__(index)
        key,left,right,_,_=self.windows[index]
        item["targets"]["geometry"]=self.records[key]["geometry"][left:right]
        return item


def records_for(store,teacher,key,recipe):
    view=store.harmonized(key);match=store.match(key);result={}
    for name,(degraded,missing) in scenarios(view,key,recipe).items():
        value=load(paths(teacher,key,recipe)[name])
        if not np.array_equal(value["native_index"],view.native_index):
            raise ValueError("Frozen feature clock differs")
        if abs(float(value["actual_hidden_fraction"])-float(missing.mean()))>1e-10:
            raise ValueError("Geometry occlusion must match teacher embedding")
        value["geometry"]=geometry(degraded,match.metadata["pitch_m"])[0]
        value["condition_weight"]=np.full(len(value["hidden"]),recipe["condition_weights"][name],np.float32)
        result[name]=value
    return result


def make_model(fold,teacher,arm,recipe):
    original=init_head(fold,teacher,recipe)
    return original if arm=="original" else PhysicalHead(original,arm=="geometry",64 if arm=="geometry" else 83,recipe["seed"])


def output(model,hidden,targets,arm):
    return model(hidden) if arm=="original" else model(hidden,targets["geometry"])


def train_epoch(model,optimizer,dataset,generator,recipe,arm,budget):
    model.train();order=torch.randperm(len(dataset),generator=generator).tolist();total=0.;count=0
    for start in range(0,len(order),recipe["batch_size"]):
        budget.check()
        hidden,core,targets=collate([dataset[i] for i in order[start:start+recipe["batch_size"]]])
        optimizer.zero_grad(set_to_none=True)
        loss=loss_for_core(output(model,hidden,targets,arm),targets,core,False)
        if not torch.isfinite(loss):raise ValueError("Nonfinite physics pilot loss")
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),recipe["gradient_clip"],error_if_nonfinite=True)
        optimizer.step();total+=float(loss.detach())*int(core.sum());count+=int(core.sum())
    return {"loss":total/max(1,count),"steps":(len(order)+recipe["batch_size"]-1)//recipe["batch_size"]}


def train_fold(store,fold,teacher,recipe,spec,budget,digest):
    records={}
    for key in teacher["split"]["train"]:
        for name,value in records_for(store,teacher,key,recipe).items():records[key+"/"+name]=value
    dataset=GeoWindows(records,recipe["core_frames"],recipe["training_halo_frames"])
    for arm in spec["pilot"]["arms"]:
        folder=DEST/f"pilot/training/fold-{fold}/{arm}"
        if (folder/"complete.json").exists():continue
        seed_everything(recipe["seed"]);model=make_model(fold,teacher,arm,recipe)
        optimizer=torch.optim.AdamW(model.parameters(),lr=recipe["learning_rate"],weight_decay=recipe["weight_decay"])
        generator=torch.Generator().manual_seed(recipe["seed"]);history=[];path=folder/"last.pt"
        if path.exists():history=restore_state(path,model,optimizer,generator,digest)
        for epoch in range(len(history)+1,9):
            budget.check();atomic_json(DEST/"pilot/progress.json",{"stage":"training","fold":fold,"arm":arm,"epoch":epoch})
            start=time.perf_counter();metrics=train_epoch(model,optimizer,dataset,generator,recipe,arm,budget)
            history.append(dict(metrics,epoch=epoch,seconds=time.perf_counter()-start))
            save_state(path,model,optimizer,generator,history,digest)
        if arm=="original":
            old=torch.load(ROOT/f"evaluation/reports/external/duration-balanced-phase8-v1/training/fold-{fold}/original/last.pt",map_location="cpu",weights_only=False)
            if not state_equal(model.state_dict(),old["model"]) or not state_equal(optimizer.state_dict(),old["optimizer"]) or not torch.equal(generator.get_state(),old["loader_rng"]):
                raise ValueError("Identical no-physics control not exactly reproduced")
        atomic_json(folder/"complete.json",{"history":history,"parameters":sum(p.numel() for p in model.parameters()),
                    "training_windows":len(dataset),"steps_per_epoch":history[0]["steps"],
                    "original_exactly_reproduced":arm=="original","registration_sha256":digest})


def predict_geometry(model,data,recipe,arm):
    # Explicit whitelist: no labels, duration, GT events or loss weights in inputs.
    inputs={k:data[k] for k in ("hidden","time","run","geometry")}
    dataset=GeoWindows({"input":inputs},recipe["core_frames"],0)
    logits=np.empty(len(data["hidden"]),np.float32);model.eval()
    with torch.no_grad():
        for start in range(0,len(dataset),recipe["batch_size"]):
            items=[dataset[i] for i in range(start,min(start+recipe["batch_size"],len(dataset)))]
            hidden,_,targets=collate(items);result=output(model,hidden,targets,arm)[0]
            for row,item in enumerate(items):
                a,b,left=item["a"],item["b"],item["left"]
                logits[a:b]=result[row,a-left:b-left].numpy()
    return logits
