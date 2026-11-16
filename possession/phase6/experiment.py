"""Strict train/calibration/selection/test separation for two bounded pilots."""

from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from possession.external_data.schema import file_hash
from possession.learning.corruption import hide_ball, stable_seed
from possession.learning.data import Windows,collate
from possession.learning.model import masked_losses
from possession.learning.training import (TrainConfig,new_model,seed_everything,epoch_loss,forward,atomic_checkpoint,memory_usage)
from possession.learning.study import ScoreContext
from possession.learning.temporal import fit_priors,TemporalDecoder
from possession.learning.metrics import SCORER_PATH,score_match
from possession.phase5.experiment import get_store,scenarios
from possession.phase5.model import boundary_loss
from possession.phase5.metrics import diagnostics
from possession.night.common import atomic_json
from scripts.train_e0 import manifest
from scripts.external_data import ROOT,REPORTS
from scripts.phase3 import metric_row,extra_diagnostics
from .model import TransitionE0,TransitionWindows,transition_collate,transition_targets
from .candidates import (direction_scores,context_features,generate,candidate_labels,candidate_metrics,
                         fit_logistic,apply_logistic,calibration_metrics)
from .decoding import DirectionalDecoder,candidate_bonuses

DEST=REPORTS/"phase6-v1"


def register():
    protocol=json.loads((ROOT/"configs/phase6_protocol.json").read_text())
    sources=list((ROOT/"possession/phase6").glob("*.py"))+[ROOT/"scripts/phase6_experiments.py"]
    # Freeze old dependencies too, without changing their sources/fingerprints.
    for directory in ("possession/learning","possession/external_data","possession/night","possession/phase5"):
        sources.extend((ROOT/directory).glob("*.py"))
    record={"protocol":protocol,"source":{str(p.relative_to(ROOT)):file_hash(p) for p in sources},"official_scorer":file_hash(SCORER_PATH)}
    path=DEST/"registration.json"
    if path.exists() and json.loads(path.read_text())!=record:
        raise ValueError("Immutable phase6 experiment changed; use a new version")
    if not path.exists():
        atomic_json(path,record)
    return protocol


class Budget:
    def __init__(self,protocol):
        self.path=DEST/"compute.json"
        self.saved=json.loads(self.path.read_text()) if self.path.exists() else {"seconds":0.,"sessions":[]}
        self.start=time.perf_counter();self.limit=protocol["max_compute_seconds"]

    def check(self):
        if time.perf_counter()-self.start+self.saved["seconds"]>=self.limit:
            raise TimeoutError("Bounded pilot budget exhausted; checkpoints preserved")

    def close(self,stage,status):
        elapsed=time.perf_counter()-self.start
        self.saved["seconds"]+=elapsed
        self.saved["sessions"].append({"stage":stage,"status":status,"seconds":elapsed})
        atomic_json(self.path,self.saved)


def progress(stage,**kwargs):
    atomic_json(DEST/"progress.json",{"stage":stage,**kwargs})


def split(data,i):
    keys=sorted(data["artifacts"])
    outer=data["folds"][i]
    calibration=keys[(i+1)%len(keys)]
    validation=outer["validation"][0]
    test=outer["test"][0]
    train=[k for k in keys if k not in (calibration,validation,test)]
    assert len(train)==4 and len(set(train+[calibration,validation,test]))==7
    return {"train":train,"calibration":calibration,"validation":validation,"test":test}


def corrupt(views,epoch,protocol):
    length=protocol["corruption_block_seconds"][(epoch-1)%3]
    return {k:hide_ball(v,.4,length,stable_seed(k,protocol["seed"]+1000*epoch))[0] for k,v in views.items()}


def save_state(path,model,optimizer,generator,history,extra):
    atomic_checkpoint(path,{"model":model.state_dict(),"optimizer":optimizer.state_dict(),"loader_rng":generator.get_state(),
        "torch_rng":torch.get_rng_state(),"numpy_rng":np.random.get_state(),"python_rng":random.getstate(),"history":history,**extra})


def restore(state,model,optimizer,generator):
    model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"])
    generator.set_state(state["loader_rng"]);torch.set_rng_state(state["torch_rng"])
    np.random.set_state(state["numpy_rng"]);random.setstate(state["python_rng"])
    return state["history"]


def pretrain(store,parts,i,protocol,budget):
    folder=DEST/f"training/fold-{i}/base";folder.mkdir(parents=True,exist_ok=True)
    if (folder/"complete.json").exists():
        return
    seed_everything(protocol["seed"])
    model=new_model(TrainConfig())
    optimizer=torch.optim.AdamW(model.parameters(),lr=protocol["base_learning_rate"],weight_decay=.0001)
    generator=torch.Generator().manual_seed(protocol["seed"])
    scaler=torch.amp.GradScaler("cuda",enabled=False)
    views={k:store.view(k) for k in parts["train"]}
    history=[];begin=time.perf_counter()
    if (folder/"last.pt").exists():
        history=restore(torch.load(folder/"last.pt",weights_only=False,map_location="cpu"),model,optimizer,generator)
    for epoch in range(len(history)+1,protocol["base_epochs"]+1):
        budget.check();progress("base",fold=i,epoch=epoch)
        start=time.perf_counter()
        loader=DataLoader(Windows(corrupt(views,epoch,protocol),256,model.context),batch_size=4,shuffle=True,generator=generator,collate_fn=collate)
        stats=epoch_loss(model,loader,torch.device("cpu"),optimizer,scaler)
        history.append({"epoch":epoch,"train":stats,"seconds":time.perf_counter()-start})
        save_state(folder/"last.pt",model,optimizer,generator,history,{"split":parts,"protocol":protocol})
    atomic_json(folder/"complete.json",{"history":history,"seconds":time.perf_counter()-begin,"split":parts})


def epoch_pass(model,loader,optimizer,weights,arm,protocol):
    model.train();total=np.zeros(4);auxsum=0.;norms=[]
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        out=forward(model,batch)
        base,stats=masked_losses(out,batch["alive"],batch["team"],batch["confidence"],batch["core"])
        aux=boundary_loss(out["boundary_logit"],batch["transition_target"],
                          (batch["transition_valid"]&batch["core"])[...,None],weights,"focal",protocol["focal_gamma"])
        loss=base+(0. if arm=="control" else protocol["boundary_weight"])*aux
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite pilot loss")
        loss.backward();norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)))
        optimizer.step();total+=stats.numpy();auxsum+=float(aux.detach())
    return {"state_loss":float(total[0]/max(1,total[1])),"team_loss":float(total[2]/max(1,total[3])),
            "boundary_loss":auxsum/max(1,len(loader)),"gradient_norm_max":max(norms),"updates":len(loader)}


def train_arm(store,parts,i,arm,protocol,budget):
    folder=DEST/f"training/fold-{i}/{arm}";folder.mkdir(parents=True,exist_ok=True)
    if (folder/"complete.json").exists():
        return
    seed_everything(protocol["seed"])
    directions=2 if arm=="directional" else 1
    model=TransitionE0(directions=directions)
    base=torch.load(DEST/f"training/fold-{i}/base/last.pt",weights_only=False,map_location="cpu")
    missing,unexpected=model.load_state_dict(base["model"],strict=False)
    assert set(missing)=={"boundary_head.weight","boundary_head.bias"} and not unexpected
    # Identical dropout and sampler RNG irrespective of one/two head initialization.
    seed_everything(protocol["seed"]+6000)
    optimizer=torch.optim.AdamW(model.parameters(),lr=protocol["finetune_learning_rate"],weight_decay=.0001)
    generator=torch.Generator().manual_seed(protocol["seed"]+6000)
    views={k:store.view(k) for k in parts["train"]}
    targets={k:transition_targets(store.match(k),v,directions,protocol["target_tolerance_s"]) for k,v in views.items()}
    positives=sum((y*valid[:,None]).sum(0) for y,valid in targets.values())
    n=sum(valid.sum() for _,valid in targets.values())
    weights=torch.tensor(np.minimum(protocol["positive_weight_cap_per_binary_head"]*directions,(n-positives)/np.maximum(1,positives)),dtype=torch.float32)
    history=[];begin=time.perf_counter()
    if (folder/"last.pt").exists():
        history=restore(torch.load(folder/"last.pt",weights_only=False,map_location="cpu"),model,optimizer,generator)
    for epoch in range(len(history)+1,protocol["finetune_epochs"]+1):
        budget.check();progress("finetune",fold=i,arm=arm,epoch=epoch)
        start=time.perf_counter()
        loader=DataLoader(TransitionWindows(corrupt(views,protocol["base_epochs"]+epoch,protocol),targets),
                          batch_size=4,shuffle=True,generator=generator,collate_fn=transition_collate)
        stats=epoch_pass(model,loader,optimizer,weights,arm,protocol)
        history.append({"epoch":epoch,"train":stats,"seconds":time.perf_counter()-start})
        save_state(folder/"last.pt",model,optimizer,generator,history,{"split":parts,"arm":arm,"protocol":protocol})
    atomic_json(folder/"complete.json",{"history":history,"seconds":time.perf_counter()-begin,"weights":weights.tolist(),
                                      "positive_frames":positives.tolist(),"valid_frames":int(n),"split":parts,"memory":memory_usage(torch.device("cpu"))})


def load_model(i,arm):
    model=TransitionE0(directions=2 if arm=="directional" else 1)
    model.load_state_dict(torch.load(DEST/f"training/fold-{i}/{arm}/last.pt",weights_only=False,map_location="cpu")["model"])
    return model


def predict(model,view):
    ds=Windows({"public":view},256,model.context)
    p=np.zeros((len(view.time),3),np.float32);q=np.zeros((len(view.time),model.directions),np.float32)
    model.eval()
    with torch.no_grad():
        for a in range(0,len(ds),8):
            items=[ds[i] for i in range(a,min(a+8,len(ds)))]
            out=forward(model,collate(items))
            for item,pp,qq in zip(items,out["probabilities"].numpy(),out["boundary_logit"].sigmoid().numpy()):
                left,start,stop=item["left"],item["a"],item["b"]
                p[start:stop],q[start:stop]=pp[start-left:stop-left],qq[start-left:stop-left]
    return p,q


def prediction_cases(store,key,i,arm,part,protocol):
    model=load_model(i,arm);view=store.view(key)
    result={}
    for name,(degraded,missing) in scenarios(view,key,protocol).items():
        path=DEST/f"predictions/fold-{i}/{arm}/{part}/{name}.npz"
        if path.exists():
            with np.load(path,allow_pickle=False) as z:
                p,q=z["probabilities"],z["transition"]
        else:
            p,q=predict(model,degraded);path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,probabilities=p,transition=q,time=view.time,run=view.run,native_index=view.native_index)
        result[name]=(degraded,p,q,missing)
    return result


def make_candidates(context,case,arm,method,calibrator=None):
    view,p,q,_=case
    scores=direction_scores(view,p,q,arm)
    x,confirmation=context_features(view,p,scores)
    ix,d=generate(view,scores,0. if method=="local" else .2)
    raw=scores[ix,d]
    selected=x[ix,d]
    if method=="confirmed":
        confidence=confirmation[ix,d]
    elif method in ("platt","context","geometry"):
        width={"platt":1,"context":13,"geometry":18}[method]
        confidence=apply_logistic(calibrator,selected[:,:width])
    else:
        confidence=raw
    return {"indices":ix,"directions":d,"confidence":confidence,"x":selected,
            "labels":candidate_labels(context,ix,d)}


def score(context,case,priors,family,pack=None,threshold=0.,prevalence=.1,strength=0.):
    view,p,_,_=case
    bonuses=None if pack is None or strength==0 else candidate_bonuses(view,pack["indices"],pack["directions"],pack["confidence"],threshold,prevalence,strength)
    decoder=DirectionalDecoder(view,priors,family,.5,bonuses)
    hard=context.decoded(p,decoder)
    diag=diagnostics(context,hard)
    cm=context.pm.scored_confusion(context.pm.confusion(context.gt,hard))
    small=[diag["duration_groups"][g] for g in ("lt1","1to3")]
    return {"spellF1":context.pm.spell_scores(context.gt,hard,context.runs)["mean_f1"],"macroF1":context.pm.macro_f1(cm),
            "short_recall":sum(g["iou"]["0.5"]["matched_gt"] for g in small)/max(1,sum(g["gt"] for g in small)),
            "long_recall":diag["duration_groups"]["ge10"]["iou"]["0.5"]["recall"],
            "fragmented_gt":diag["fragmented_gt"],"fused_predictions":diag["fused_predictions"]}


def objective(metrics,protocol):
    return sum(w*metrics[name]["spellF1"] for w,name in zip(protocol["score_weights"],protocol["scenarios"]))


def guard(metrics,baseline):
    return (all(metrics[k]["spellF1"]>=baseline[k]["spellF1"]-.02 for k in ("missing40_5s","missing40_15s")) and
            metrics["clean"]["long_recall"]>=baseline["clean"]["long_recall"]-.03)


def select(store,data,i,protocol,budget):
    destination=DEST/f"selections/fold-{i}.json"
    if destination.exists():
        return
    parts=split(data,i)
    pretrain(store,parts,i,protocol,budget)
    priors=fit_priors([store.view(k) for k in parts["train"]])
    np.savez(DEST/f"training/fold-{i}/priors.npz",**priors)
    selections={}
    for arm in protocol["arms"]:
        train_arm(store,parts,i,arm,protocol,budget)
        calibration=prediction_cases(store,parts["calibration"],i,arm,"calibration",protocol)
        validation=prediction_cases(store,parts["validation"],i,arm,"validation",protocol)
        cal_context=ScoreContext(store.match(parts["calibration"]),store.view(parts["calibration"]))
        val_context=ScoreContext(store.match(parts["validation"]),store.view(parts["validation"]))
        cal_packs=[make_candidates(cal_context,case,arm,"nms") for case in calibration.values()]
        x=np.concatenate([v["x"] for v in cal_packs]);y=np.concatenate([v["labels"] for v in cal_packs])
        calibrators={name:fit_logistic(x[:,:width],y) for name,width in (("platt",1),("context",13),("geometry",18))}
        prevalence=float((y.sum()+.5)/(len(y)+1))
        packs={};candidate_trials={};thresholds={}
        for method in protocol["candidate_methods"]:
            packs[method]={name:make_candidates(val_context,case,arm,method,calibrators.get(method)) for name,case in validation.items()}
            trials=[]
            for threshold in protocol["candidate_thresholds"]:
                metrics={name:candidate_metrics(val_context,v["indices"],v["directions"],v["confidence"],threshold) for name,v in packs[method].items()}
                obj=sum(w*metrics[name]["f1"] for w,name in zip(protocol["score_weights"],protocol["scenarios"]))
                trials.append({"threshold":threshold,"objective":obj,"metrics":metrics})
            candidate_trials[method]=trials
            thresholds[method]=max(trials,key=lambda row:row["objective"])["threshold"]
        trials=[]
        for family in protocol["decoder_families"]:
            baseline={name:score(val_context,case,priors,family) for name,case in validation.items()}
            trials.append({"family":family,"method":"fixed","strength":0.,"threshold":0.,"metrics":baseline,"eligible":True,"objective":objective(baseline,protocol)})
            for method in protocol["candidate_methods"]:
                for strength in protocol["adaptive_strengths"]:
                    budget.check();progress("candidate_selection",fold=i,arm=arm,method=method,family=family)
                    metrics={name:score(val_context,case,priors,family,packs[method][name],thresholds[method],prevalence,strength) for name,case in validation.items()}
                    trials.append({"family":family,"method":method,"strength":strength,"threshold":thresholds[method],"metrics":metrics,
                                   "eligible":guard(metrics,baseline),"objective":objective(metrics,protocol)})
        winner=max([r for r in trials if r["eligible"]],key=lambda r:r["objective"])
        calibration_quality={method:{name:calibration_metrics(v["labels"],v["confidence"]) for name,v in pack.items()}
                             for method,pack in packs.items() if method!="local"}
        selections[arm]={"winner":winner,"trials":trials,"candidate_trials":candidate_trials,"thresholds":thresholds,
                         "calibrators":calibrators,"prevalence":prevalence,"calibration_quality":calibration_quality,
                         "fitting_match":parts["calibration"],"selection_match":parts["validation"]}
    atomic_json(destination,{"fold":i,"split":parts,"arms":selections,"outer_test_used":False})


def test(store,data,i,protocol,budget):
    destination=DEST/f"rows/fold-{i}.json"
    if destination.exists():
        return json.loads(destination.read_text())
    for j in protocol["pilot_folds"]:
        assert (DEST/f"selections/fold-{j}.json").exists()
    selection=json.loads((DEST/f"selections/fold-{i}.json").read_text());parts=selection["split"]
    view=store.view(parts["test"]);match=store.match(parts["test"]);context=ScoreContext(match,view)
    with np.load(DEST/f"training/fold-{i}/priors.npz") as z:
        priors=dict(z)
    rows=[]
    for arm in protocol["arms"]:
        chosen=selection["arms"][arm]
        cases=prediction_cases(store,parts["test"],i,arm,"test",protocol)
        # Evaluate one own-validation finalist per method, not the full search.
        finalists={"selected":chosen["winner"]}
        for family in protocol["decoder_families"]:
            finalists["fixed_"+family]=next(r for r in chosen["trials"] if r["family"]==family and r["method"]=="fixed")
        for method in protocol["candidate_methods"]:
            admissible=[r for r in chosen["trials"] if r["method"]==method and r["eligible"]]
            if admissible:
                finalists[method]=max(admissible,key=lambda r:r["objective"])
        for name,case in cases.items():
            progress("test",fold=i,arm=arm,scenario=name)
            packs={method:make_candidates(context,case,arm,method,chosen["calibrators"].get(method)) for method in protocol["candidate_methods"]}
            for label,trial in finalists.items():
                budget.check();start=time.perf_counter()
                method=trial["method"];pack=None if method=="fixed" else packs[method]
                bonuses=None if pack is None else candidate_bonuses(view,pack["indices"],pack["directions"],pack["confidence"],
                    trial["threshold"],chosen["prevalence"],trial["strength"])
                decoder=DirectionalDecoder(view,priors,trial["family"],.5,bonuses)
                full=score_match(match,view,case[1],decoder)
                hard=context.decoded(case[1],decoder);detail=diagnostics(context,hard)
                row=metric_row(full,i,arm,name,label,2027,8,8,float(case[3].mean()),label=="selected")
                row.update(arm=arm,system=label,method=method,family=trial["family"],duration_groups=detail["duration_groups"],
                    decoded_HA=detail["decoded_HA"],candidate_metrics={m:candidate_metrics(context,v["indices"],v["directions"],v["confidence"],chosen["thresholds"][m]) for m,v in packs.items()},
                    calibration_quality={m:calibration_metrics(v["labels"],v["confidence"]) for m,v in packs.items() if m!="local"},
                    independent=extra_diagnostics(match,view,case[1],decoder,case[3]),evaluation_s=time.perf_counter()-start)
                rows.append(row)
    atomic_json(destination,rows)
    return rows


def run(stage):
    protocol=register();torch.set_num_threads(protocol["cpu_threads"])
    data,_=manifest("idsse-only");store=get_store(data);budget=Budget(protocol);status="failed"
    try:
        if stage in ("pilots","all"):
            for i in protocol["pilot_folds"]:
                select(store,data,i,protocol,budget)
        if stage in ("test","all"):
            rows=[]
            for i in protocol["pilot_folds"]:
                rows.extend(test(store,data,i,protocol,budget))
            atomic_json(DEST/"all_rows.json",rows)
        status="complete";progress(stage,status=status)
    finally:
        budget.close(stage,status)
