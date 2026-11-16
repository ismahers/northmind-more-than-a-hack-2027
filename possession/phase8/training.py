"""Seven-match, two-seed paired study; exact 12+8 schedule and reusable pilots."""
from pathlib import Path
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from possession.external_data.schema import file_hash
from possession.learning.data import Windows
from possession.learning.training import seed_everything,epoch_loss,memory_usage
from possession.night.common import atomic_json
from possession.phase7.model import SpatialE0
from possession.phase6.experiment import split,corrupt,save_state,restore,epoch_pass
from possession.phase6.model import transition_targets,TransitionWindows
from scripts.train_e0 import manifest
from .packing import compact_collate,compact_transition_collate
from .mixing import SkillBatches,mixed_epoch
from .common import Store,DEST,OLD,register,Budget,progress


def folder(i,seed,arm,condition="IDSSE"):
    return DEST/f"training/{condition}/fold-{i}/seed-{seed}/{arm}"


def source_folder(i,seed,arm,condition="IDSSE"):
    if condition=="IDSSE" and seed==2027 and i in (0,6):return OLD/f"training/fold-{i}/{arm}"
    return folder(i,seed,arm,condition)


def load_model(i,seed,arm,condition="IDSSE"):
    model=SpatialE0(arm,True)
    path=source_folder(i,seed,arm,condition)/"binary/last.pt"
    model.load_state_dict(torch.load(path,map_location="cpu",weights_only=False)["model"]);return model


def train(store,parts,i,seed,arm,spec,budget,condition="IDSSE"):
    dest=folder(i,seed,arm,condition)
    if condition=="IDSSE" and source_folder(i,seed,arm)!=dest:
        source=source_folder(i,seed,arm)
        hashes={}
        for stage,n in (("base",12),("binary",8)):
            state=torch.load(source/stage/"last.pt",map_location="cpu",weights_only=False)
            assert state["split"]==parts and len(state["history"])==n
            hashes[stage]=file_hash(source/stage/"last.pt")
        atomic_json(dest/"reused.json",{"source":str(source),"sha256":hashes,"split":parts,"seed":seed,"arm":arm})
        return
    views={k:store.harmonized(k) for k in parts["train"]};validation=store.harmonized(parts["validation"])
    stages=(("base",12,.001,0),("binary",8,.0003,12)) if condition=="IDSSE" else (("binary",8,.0003,12),)
    for stage,epochs,lr,offset in stages:
        stagefolder=dest/stage;stagefolder.mkdir(parents=True,exist_ok=True)
        if (stagefolder/"complete.json").exists():continue
        seed_everything(seed);model=SpatialE0(arm,stage=="binary")
        if offset:
            base=(source_folder(i,seed,arm)/"base/last.pt" if condition!="IDSSE" else dest/"base/last.pt")
            state=torch.load(base,map_location="cpu",weights_only=False)
            assert state["split"]==parts
            missing,unexpected=model.load_state_dict(state["model"],strict=False)
            assert set(missing)=={"boundary_head.weight","boundary_head.bias"} and not unexpected
            seed_everything(seed+6000)
        initial=torch.get_rng_state().clone();opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001)
        generator=torch.Generator().manual_seed(seed+(6000 if offset else 0));scaler=torch.amp.GradScaler("cuda",enabled=False)
        targets={k:transition_targets(store.match(k),v,1,.12) for k,v in views.items()} if offset else None
        if targets:
            positive=sum((y*v[:,None]).sum(0) for y,v in targets.values());n=sum(v.sum() for _,v in targets.values())
            weights=torch.tensor(np.minimum(40.,(n-positive)/np.maximum(1,positive)),dtype=torch.float32)
        history=[];start=time.perf_counter()
        if (stagefolder/"last.pt").exists():history=restore(torch.load(stagefolder/"last.pt",map_location="cpu",weights_only=False),model,opt,generator)
        val_loader=DataLoader(Windows({"val":validation},256,15),batch_size=4,collate_fn=compact_collate)
        skviews={k:store.harmonized(k) for k in spec["skillcorner_training_keys"]} if condition!="IDSSE" else None
        # Corruption depends only on key/epoch/seed, independent of architecture and source.
        local=dict(spec,seed=seed)
        for epoch in range(len(history)+1,epochs+1):
            budget.check();progress("training",condition=condition,fold=i,seed=seed,arm=arm,stage_name=stage,epoch=epoch)
            t=time.perf_counter();degraded=corrupt(views,offset+epoch,local)
            ds=TransitionWindows(degraded,targets) if offset else Windows(degraded,256,15)
            loader=DataLoader(ds,batch_size=4,shuffle=True,generator=generator,collate_fn=compact_transition_collate if offset else compact_collate)
            if condition!="IDSSE":
                degraded_sk=corrupt(skviews,offset+epoch,local);skill=SkillBatches(degraded_sk,seed+100000*epoch)
                sk_rng=torch.Generator().manual_seed(seed+90000+epoch).get_state()
                stats,_=mixed_epoch(model,loader,opt,weights,local,skill,sk_rng)
            elif offset:stats=epoch_pass(model,loader,opt,weights,"binary",local)
            else:stats=epoch_loss(model,loader,torch.device("cpu"),opt,scaler)
            train_seconds=time.perf_counter()-t;val=epoch_loss(model,val_loader,torch.device("cpu"))
            history.append({"epoch":epoch,"global_epoch":offset+epoch,"train":stats,"validation":val,"training_seconds":train_seconds,"seconds":time.perf_counter()-t})
            save_state(stagefolder/"last.pt",model,opt,generator,history,{"split":parts,"arm":arm,"condition":condition,
                       "seed":seed,"initial_rng":initial,"protocol":spec,"packing":True})
        atomic_json(stagefolder/"complete.json",{"history":history,"seconds":time.perf_counter()-start,"split":parts,
            "parameters":sum(p.numel() for p in model.parameters()),"memory":memory_usage(torch.device("cpu")),"seed":seed,"condition":condition})


def run_primary():
    spec=register();torch.set_num_threads(spec["cpu_threads"]);data,_=manifest("idsse-only");store=Store();budget=Budget(spec);status="failed"
    try:
        # Both architectures for every paired seed/fold; no new outer-test scores read.
        for seed in spec["seeds"]:
            for i in spec["folds"]:
                parts=split(data,i)
                for arm in spec["arms"]:train(store,parts,i,seed,arm,spec,budget)
        status="complete";progress("primary_training",status=status)
    finally:budget.close("primary_training",status)
