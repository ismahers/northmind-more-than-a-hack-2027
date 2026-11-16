"""Bounded spatial comparison: immutable public data, identical temporal pipeline."""
from dataclasses import asdict
import json
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from possession.external_data.schema import file_hash
from possession.learning.corruption import hide_ball,stable_seed
from possession.learning.data import Windows,collate
from possession.learning.training import seed_everything,epoch_loss,atomic_checkpoint,memory_usage
from possession.learning.study import ScoreContext
from possession.learning.temporal import fit_priors
from possession.learning.metrics import SCORER_PATH,score_match
from possession.night.common import atomic_json
from possession.phase5.experiment import scenarios
from possession.phase5.metrics import diagnostics
from possession.phase6.experiment import (split,corrupt,save_state,restore,epoch_pass,predict,
    make_candidates,score,objective,guard,DEST as PREVIOUS,load_model as old_model)
from possession.phase6.model import transition_targets,TransitionWindows,transition_collate
from possession.phase6.candidates import fit_logistic,candidate_metrics
from possession.phase6.decoding import DirectionalDecoder,candidate_bonuses
from scripts.train_e0 import manifest
from scripts.external_data import ROOT
from scripts.phase3 import metric_row,extra_diagnostics
from .public import Store,DEST,protocol
from .model import SpatialE0


def progress(stage,**fields):
    atomic_json(DEST/"progress.json",{"stage":stage,**fields})


def register():
    spec=protocol();sources=[]
    for folder in ("phase7","phase6","phase5","night","learning","external_data"):
        sources.extend((ROOT/"possession"/folder).glob("*.py"))
    record={"protocol":spec,"source":{str(p.relative_to(ROOT)):file_hash(p) for p in sources},
            "official_scorer":file_hash(SCORER_PATH)}
    path=DEST/"registration.json"
    if path.exists() and json.loads(path.read_text())!=record:raise ValueError("Registered source changed; use a new version")
    if not path.exists():atomic_json(path,record)
    return spec


class Budget:
    def __init__(self,spec):
        self.path=DEST/"compute.json";self.limit=spec["max_compute_seconds"];self.start=time.perf_counter()
        self.saved=json.loads(self.path.read_text()) if self.path.exists() else {"seconds":0.,"sessions":[]}
    def check(self):
        if self.saved["seconds"]+time.perf_counter()-self.start>=self.limit:raise TimeoutError("Pilot budget exhausted; resume checkpoints")
    def close(self,stage,status):
        seconds=time.perf_counter()-self.start;self.saved["seconds"]+=seconds
        self.saved["sessions"].append({"stage":stage,"status":status,"seconds":seconds});atomic_json(self.path,self.saved)


def train(store,parts,i,arm,spec,budget):
    if arm=="legacy":return
    views={k:store.harmonized(k) for k in parts["train"]}
    validation=store.harmonized(parts["validation"])
    for stage,epochs,lr,offset in (("base",12,.001,0),("binary",8,.0003,12)):
        folder=DEST/f"training/fold-{i}/{arm}/{stage}";folder.mkdir(parents=True,exist_ok=True)
        if (folder/"complete.json").exists():continue
        seed_everything(spec["seed"])
        model=SpatialE0(arm,stage=="binary")
        if stage=="binary":
            base=torch.load(folder.parent/"base/last.pt",map_location="cpu",weights_only=False)
            missing,unexpected=model.load_state_dict(base["model"],strict=False)
            assert set(missing)=={"boundary_head.weight","boundary_head.bias"} and not unexpected
            seed_everything(spec["seed"]+6000)
        initial_rng=torch.get_rng_state().clone()
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001)
        generator=torch.Generator().manual_seed(spec["seed"]+(6000 if stage=="binary" else 0))
        scaler=torch.amp.GradScaler("cuda",enabled=False)
        targets={k:transition_targets(store.match(k),v,1,spec["target_tolerance_s"]) for k,v in views.items()} if stage=="binary" else None
        if targets:
            positive=sum((y*valid[:,None]).sum(0) for y,valid in targets.values());n=sum(v.sum() for _,v in targets.values())
            weights=torch.tensor(np.minimum(40.,(n-positive)/np.maximum(1,positive)),dtype=torch.float32)
        history=[];begin=time.perf_counter()
        if (folder/"last.pt").exists():history=restore(torch.load(folder/"last.pt",map_location="cpu",weights_only=False),model,optimizer,generator)
        val_loader=DataLoader(Windows({"validation":validation},256,15),batch_size=4,collate_fn=collate)
        for epoch in range(len(history)+1,epochs+1):
            budget.check();progress("training",fold=i,arm=arm,stage_name=stage,epoch=epoch)
            start=time.perf_counter();degraded=corrupt(views,offset+epoch,spec)
            if stage=="base":
                loader=DataLoader(Windows(degraded,256,15),batch_size=4,shuffle=True,generator=generator,collate_fn=collate)
                stats=epoch_loss(model,loader,torch.device("cpu"),optimizer,scaler)
            else:
                loader=DataLoader(TransitionWindows(degraded,targets),batch_size=4,shuffle=True,generator=generator,collate_fn=transition_collate)
                stats=epoch_pass(model,loader,optimizer,weights,"binary",spec)
            train_seconds=time.perf_counter()-start
            val=epoch_loss(model,val_loader,torch.device("cpu"))
            history.append({"epoch":epoch,"global_epoch":offset+epoch,"train":stats,"validation":val,
                            "training_seconds":train_seconds,"seconds":time.perf_counter()-start})
            save_state(folder/"last.pt",model,optimizer,generator,history,{"split":parts,"arm":arm,"stage":stage,"initial_rng":initial_rng,"protocol":spec})
        atomic_json(folder/"complete.json",{"history":history,"seconds":time.perf_counter()-begin,"split":parts,
            "parameters":sum(p.numel() for p in model.parameters()),"parameter_bytes_float32":4*sum(p.numel() for p in model.parameters()),
            "memory":memory_usage(torch.device("cpu")),"memory_scope":"process cumulative RSS, not isolated encoder allocation"})


def load_model(i,arm):
    if arm=="legacy":return old_model(i,"binary")
    model=SpatialE0(arm,True)
    model.load_state_dict(torch.load(DEST/f"training/fold-{i}/{arm}/binary/last.pt",map_location="cpu",weights_only=False)["model"])
    return model


def cases(store,key,i,arm,part,spec):
    view=store.view(key) if arm=="legacy" else store.harmonized(key)
    model=None;out={}
    for name,(v,missing) in scenarios(view,key,spec).items():
        path=(PREVIOUS/f"predictions/fold-{i}/binary/{part}/{name}.npz" if arm=="legacy"
              else DEST/f"predictions/fold-{i}/{arm}/{part}/{name}.npz")
        if path.exists():
            with np.load(path,allow_pickle=False) as z:p,q=z["probabilities"],z["transition"]
        else:
            if arm=="legacy":raise ValueError("Reusable legacy predictions absent")
            if model is None:model=load_model(i,arm)
            start=time.perf_counter();p,q=predict(model,v);seconds=time.perf_counter()-start
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,probabilities=p,transition=q,time=view.time,run=view.run,native_index=view.native_index)
            atomic_json(path.with_suffix(".timing.json"),{"seconds":seconds,"frames":len(p)})
        out[name]=(v,p,q,missing)
    return out


def select(store,data,i,spec,budget):
    path=DEST/f"selections/fold-{i}.json"
    if path.exists():return
    parts=split(data,i);priors=fit_priors([store.view(k) for k in parts["train"]])
    (DEST/f"training/fold-{i}").mkdir(parents=True,exist_ok=True)
    np.savez(DEST/f"training/fold-{i}/priors.npz",**priors)
    selected={}
    for arm in spec["arms"]:
        train(store,parts,i,arm,spec,budget)
        calibration=cases(store,parts["calibration"],i,arm,"calibration",spec)
        validation=cases(store,parts["validation"],i,arm,"validation",spec)
        cal_context=ScoreContext(store.match(parts["calibration"]),next(iter(calibration.values()))[0])
        val_context=ScoreContext(store.match(parts["validation"]),next(iter(validation.values()))[0])
        packs=[make_candidates(cal_context,v,"binary","nms") for v in calibration.values()]
        x=np.concatenate([v["x"][:,:1] for v in packs]);y=np.concatenate([v["labels"] for v in packs])
        fitted=fit_logistic(x,y);prevalence=float((y.sum()+.5)/(len(y)+1))
        packs={name:make_candidates(val_context,v,"binary","platt",fitted) for name,v in validation.items()}
        threshold_trials=[]
        for threshold in spec["candidate_thresholds"]:
            metrics={name:candidate_metrics(val_context,v["indices"],v["directions"],v["confidence"],threshold) for name,v in packs.items()}
            threshold_trials.append({"threshold":threshold,"objective":sum(w*metrics[n]["f1"] for w,n in zip(spec["score_weights"],spec["scenarios"]))})
        threshold=max(threshold_trials,key=lambda r:r["objective"])["threshold"]
        trials=[]
        for family in spec["decoder_families"]:
            baseline={n:score(val_context,c,priors,family) for n,c in validation.items()}
            trials.append({"family":family,"strength":0.,"threshold":0.,"method":"fixed","metrics":baseline,"eligible":True,"objective":objective(baseline,spec)})
            metrics={n:score(val_context,c,priors,family,packs[n],threshold,prevalence,.5) for n,c in validation.items()}
            trials.append({"family":family,"strength":.5,"threshold":threshold,"method":"platt","metrics":metrics,"eligible":guard(metrics,baseline),"objective":objective(metrics,spec)})
        winner=max((r for r in trials if r["eligible"]),key=lambda r:r["objective"])
        selected[arm]={"winner":winner,"trials":trials,"calibrator":fitted,"prevalence":prevalence,"threshold_trials":threshold_trials}
        progress("selection",fold=i,arm=arm)
    atomic_json(path,{"fold":i,"split":parts,"arms":selected,"outer_test_used":False})


def choose_architecture(spec):
    path=DEST/"architecture_selection.json"
    if path.exists():return json.loads(path.read_text())
    selections=[json.loads((DEST/f"selections/fold-{i}.json").read_text()) for i in spec["pilot_folds"]]
    a=[s["arms"]["harmonized"]["winner"] for s in selections];b=[s["arms"]["set_transformer"]["winner"] for s in selections]
    delta=float(np.mean([y["objective"]-x["objective"] for x,y in zip(a,b)]))
    occlusion={n:float(np.mean([y["metrics"][n]["spellF1"]-x["metrics"][n]["spellF1"] for x,y in zip(a,b)])) for n in spec["scenarios"][1:]}
    long=float(np.mean([y["metrics"]["clean"]["long_recall"]-x["metrics"]["clean"]["long_recall"] for x,y in zip(a,b)]))
    promoted=delta>=.01 and min(occlusion.values())>=-.02 and long>=-.03
    result={"selected_architecture":"set_transformer" if promoted else "harmonized","set_transformer_robust_gain":delta,
        "occlusion_gain":occlusion,"long_recall_gain":long,"rule":spec["architecture_rule"],"test_used":False,
        "SkillCorner_confirmation_used":False,"selection_folds":spec["pilot_folds"]}
    atomic_json(path,result);return result


def test(store,data,i,spec,budget):
    path=DEST/f"rows/fold-{i}.json"
    if path.exists():return json.loads(path.read_text())
    assert (DEST/"architecture_selection.json").exists()
    selection=json.loads((DEST/f"selections/fold-{i}.json").read_text());parts=selection["split"]
    match=store.match(parts["test"])
    with np.load(DEST/f"training/fold-{i}/priors.npz") as z:priors=dict(z)
    rows=[]
    for arm in spec["arms"]:
        current=cases(store,parts["test"],i,arm,"test",spec);chosen=selection["arms"][arm]
        finals={"selected":chosen["winner"]}
        for family in spec["decoder_families"]:finals["fixed_"+family]=next(r for r in chosen["trials"] if r["method"]=="fixed" and r["family"]==family)
        for name,case in current.items():
            view,p,q,missing=case;context=ScoreContext(match,view)
            pack=make_candidates(context,case,"binary","platt",chosen["calibrator"])
            for label,trial in finals.items():
                budget.check();progress("external_test",fold=i,arm=arm,scenario=name,decoder=label)
                bonuses=None if trial["method"]=="fixed" else candidate_bonuses(view,pack["indices"],pack["directions"],pack["confidence"],trial["threshold"],chosen["prevalence"],trial["strength"])
                decoder=DirectionalDecoder(view,priors,trial["family"],.5,bonuses)
                full=score_match(match,view,p,decoder);detail=diagnostics(context,context.decoded(p,decoder),q[:,0],.5)
                row=metric_row(full,i,arm,name,label,2027,20,20,float(missing.mean()),label=="selected")
                row.update(arm=arm,system=label,duration_groups=detail["duration_groups"],decoded_HA=detail["decoded_HA"],head_HA=detail["head_HA"],
                    independent=extra_diagnostics(match,view,p,decoder,missing),match=parts["test"])
                rows.append(row)
    atomic_json(path,rows);return rows


def transfer(store,spec,budget):
    from scripts.phase6_transfer import conditional_metrics
    assert (DEST/"architecture_selection.json").exists()
    models={(i,a):load_model(i,a) for i in spec["pilot_folds"] for a in spec["arms"]}
    result=[];begin=time.perf_counter()
    for index,key in enumerate(spec["skillcorner_confirmation_keys"]):
        path=DEST/"transfer"/(key.replace(":","_")+".json")
        if path.exists():result.extend(json.loads(path.read_text()));continue
        original=store.view(key);common=store.harmonized(key);rows=[]
        progress("SkillCorner_confirmation",match=index+1,total=16)
        for arm in spec["arms"]:
            for name,(view,missing) in scenarios(original if arm=="legacy" else common,key,spec).items():
                for i in spec["pilot_folds"]:
                    budget.check();start=time.perf_counter();p,q=predict(models[i,arm],view)
                    rows.append({"match":key,"fold":i,"arm":arm,"scenario":name,"conditional_team":conditional_metrics(view,p),"inference_s":time.perf_counter()-start})
        atomic_json(path,rows);result.extend(rows)
        # Public SK features are bounded: release per-match mappings between games.
        store.common.pop(key,None)
    atomic_json(DEST/"transfer.json",{"rows":result,"seconds":time.perf_counter()-begin,"selection_used":False,
        "semantics":"conditional H/A on partial native provider labels; D/spellF1 unavailable, null not D"})


def run(stage):
    spec=register();torch.set_num_threads(spec["cpu_threads"])
    data,_=manifest("idsse-only");store=Store();budget=Budget(spec);status="failed"
    try:
        if stage in ("pilots","all"):
            for i in spec["pilot_folds"]:select(store,data,i,spec,budget)
            choose_architecture(spec)
        if stage in ("test","all"):
            rows=[]
            for i in spec["pilot_folds"]:rows.extend(test(store,data,i,spec,budget))
            atomic_json(DEST/"all_rows.json",rows)
        if stage in ("transfer","all"):transfer(store,spec,budget)
        status="complete";progress(stage,status=status)
    finally:budget.close(stage,status)
