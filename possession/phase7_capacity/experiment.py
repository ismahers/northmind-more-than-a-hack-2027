"""Single near-parameter-matched control; no new hyperparameter search.

Motivated after the main comparison: explicitly exploratory. Widths are fixed
by parameter count, never by external labels. All train/val/test decisions and
the 12+8 schedule remain identical to the main two pilots.
"""
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from possession.phase7.public import Store,DEST as MAIN,protocol
from possession.phase7.experiment import register as check_main,progress
from possession.phase7_capacity.model import CapacityE0
from possession.external_data.schema import file_hash
from possession.learning.training import seed_everything,epoch_loss,memory_usage
from possession.learning.data import Windows,collate
from possession.learning.study import ScoreContext
from possession.learning.metrics import score_match
from possession.phase5.experiment import scenarios
from possession.phase5.metrics import diagnostics
from possession.phase6.experiment import split,corrupt,save_state,restore,epoch_pass,predict
from possession.phase6.model import transition_targets,TransitionWindows,transition_collate
from possession.phase6.decoding import DirectionalDecoder
from possession.night.common import atomic_json
from scripts.external_data import ROOT,REPORTS
from scripts.train_e0 import manifest
from scripts.phase3 import metric_row
from scripts.phase6_transfer import conditional_metrics

DEST=REPORTS/"phase7-capacity-v1"


def train(store,parts,i,spec,begin):
    views={k:store.harmonized(k) for k in parts["train"]};validation=store.harmonized(parts["validation"])
    for stage,epochs,lr,offset in (("base",12,.001,0),("binary",8,.0003,12)):
        folder=DEST/f"training/fold-{i}/{stage}";folder.mkdir(parents=True,exist_ok=True)
        if (folder/"complete.json").exists():continue
        seed_everything(spec["seed"]);model=CapacityE0(stage=="binary")
        if stage=="binary":
            base=torch.load(folder.parent/"base/last.pt",map_location="cpu",weights_only=False)
            missing,unexpected=model.load_state_dict(base["model"],strict=False)
            assert set(missing)=={"boundary_head.weight","boundary_head.bias"} and not unexpected
            seed_everything(spec["seed"]+6000)
        initial=torch.get_rng_state().clone();opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001)
        generator=torch.Generator().manual_seed(spec["seed"]+(6000 if offset else 0))
        scaler=torch.amp.GradScaler("cuda",enabled=False);targets=None;weights=None
        if offset:
            targets={k:transition_targets(store.match(k),v,1,.12) for k,v in views.items()}
            positive=sum((y*v[:,None]).sum(0) for y,v in targets.values());n=sum(v.sum() for _,v in targets.values())
            weights=torch.tensor(np.minimum(40.,(n-positive)/np.maximum(1,positive)),dtype=torch.float32)
        history=[];start=time.perf_counter()
        if (folder/"last.pt").exists():history=restore(torch.load(folder/"last.pt",weights_only=False,map_location="cpu"),model,opt,generator)
        val_loader=DataLoader(Windows({"val":validation},256,15),batch_size=4,collate_fn=collate)
        for epoch in range(len(history)+1,epochs+1):
            if time.perf_counter()-begin>900:raise TimeoutError("Capacity control budget; checkpoints preserved")
            progress("capacity_control",fold=i,stage_name=stage,epoch=epoch);t=time.perf_counter()
            degraded=corrupt(views,offset+epoch,spec)
            dataset=TransitionWindows(degraded,targets) if offset else Windows(degraded,256,15)
            loader=DataLoader(dataset,batch_size=4,shuffle=True,generator=generator,collate_fn=transition_collate if offset else collate)
            stats=epoch_pass(model,loader,opt,weights,"binary",spec) if offset else epoch_loss(model,loader,torch.device("cpu"),opt,scaler)
            training_seconds=time.perf_counter()-t;val=epoch_loss(model,val_loader,torch.device("cpu"))
            history.append({"epoch":epoch,"global_epoch":offset+epoch,"train":stats,"validation":val,"training_seconds":training_seconds,"seconds":time.perf_counter()-t})
            save_state(folder/"last.pt",model,opt,generator,history,{"split":parts,"initial_rng":initial,"protocol":spec})
        atomic_json(folder/"complete.json",{"history":history,"seconds":time.perf_counter()-start,"parameters":sum(p.numel() for p in model.parameters()),"memory":memory_usage(torch.device("cpu"))})


def model_for(i):
    model=CapacityE0(True);model.load_state_dict(torch.load(DEST/f"training/fold-{i}/binary/last.pt",map_location="cpu",weights_only=False)["model"]);return model


def cases(store,key,i,part,spec):
    view=store.harmonized(key);model=None;out={}
    for name,(v,missing) in scenarios(view,key,spec).items():
        path=DEST/f"predictions/fold-{i}/{part}/{name}.npz"
        if path.exists():
            with np.load(path) as z:p,q=z["probabilities"],z["transition"]
        else:
            if model is None:model=model_for(i)
            p,q=predict(model,v);path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,probabilities=p,transition=q,time=view.time,run=view.run,native_index=view.native_index)
        out[name]=(v,p,q,missing)
    return out


def run():
    spec=check_main();torch.set_num_threads(4);begin=time.perf_counter();data,_=manifest("idsse-only");store=Store()
    record={"parent_registration_sha256":file_hash(MAIN/"registration.json"),"encoder":"6-96-128-32 ReLU MLP; team mean/max; 92163 total parameters vs attention 92387",
        "sources":{str(p.relative_to(ROOT)):file_hash(p) for p in (ROOT/"possession/phase7_capacity").glob("*.py")},
        "schedule":"identical fixed 12+8 epochs, same four training matches, initialization of common layers, dropout/sampler RNG, corruptions, losses and fixed HMM0.5",
        "scope":"exploratory follow-up control after seeing main study; no new HPO; no original main choices overwritten",
        "rule":"compare own-validation robust fixed-HMM objective; external scores do not select weights or parameters",
        "max_additional_seconds":900}
    marker=DEST/"registration.json"
    if marker.exists() and json.loads(marker.read_text())!=record:raise ValueError("Immutable control changed")
    if not marker.exists():atomic_json(marker,record)
    for i in spec["pilot_folds"]:
        path=DEST/f"selections/fold-{i}.json"
        if path.exists():continue
        parts=split(data,i);train(store,parts,i,spec,begin)
        with np.load(MAIN/f"training/fold-{i}/priors.npz") as z:priors=dict(z)
        result={}
        for name,(v,p,q,missing) in cases(store,parts["validation"],i,"validation",spec).items():
            context=ScoreContext(store.match(parts["validation"]),v)
            result[name]=context.spell(p,DirectionalDecoder(v,priors,"hmm",.5))
        atomic_json(path,{"split":parts,"fixed_hmm_validation":result,"objective":sum(w*result[n] for w,n in zip(spec["score_weights"],spec["scenarios"])),"outer_test_used":False})
    for i in spec["pilot_folds"]:assert (DEST/f"selections/fold-{i}.json").exists()
    rows=[]
    for i in spec["pilot_folds"]:
        path=DEST/f"rows/fold-{i}.json"
        if path.exists():rows.extend(json.loads(path.read_text()));continue
        parts=split(data,i);match=store.match(parts["test"])
        with np.load(MAIN/f"training/fold-{i}/priors.npz") as z:priors=dict(z)
        current=[]
        for name,(v,p,q,missing) in cases(store,parts["test"],i,"test",spec).items():
            context=ScoreContext(match,v);decoder=DirectionalDecoder(v,priors,"hmm",.5)
            full=score_match(match,v,p,decoder);detail=diagnostics(context,context.decoded(p,decoder),q[:,0])
            row=metric_row(full,i,"capacity",name,"fixed_hmm",2027,20,20,float(missing.mean()))
            row.update(arm="capacity",system="fixed_hmm",duration_groups=detail["duration_groups"],decoded_HA=detail["decoded_HA"],head_HA=detail["head_HA"])
            current.append(row)
        atomic_json(path,current);rows.extend(current)
    atomic_json(DEST/"all_rows.json",rows)
    models={i:model_for(i) for i in spec["pilot_folds"]};transfer=[]
    for key in spec["skillcorner_confirmation_keys"]:
        path=DEST/"transfer"/(key.replace(":","_")+".json")
        if path.exists():transfer.extend(json.loads(path.read_text()));continue
        v=store.harmonized(key);current=[]
        for scenario,(view,missing) in scenarios(v,key,spec).items():
            for i,model in models.items():
                p,_=predict(model,view);current.append({"match":key,"fold":i,"arm":"capacity","scenario":scenario,"conditional_team":conditional_metrics(view,p)})
        atomic_json(path,current);transfer.extend(current);store.common.pop(key,None)
    atomic_json(DEST/"transfer.json",{"rows":transfer})
    atomic_json(DEST/"compute.json",{"seconds":time.perf_counter()-begin,"status":"complete"})
