"""Own-fold calibration/selection; no external results enter a selector."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from possession.external_data.schema import file_hash
from possession.learning.data import Windows
from possession.learning.training import forward
from possession.learning.study import ScoreContext
from possession.learning.temporal import fit_priors
from possession.learning.metrics import score_match
from possession.night.common import atomic_json
from possession.phase5.experiment import scenarios
from possession.phase5.metrics import diagnostics
from possession.phase6.experiment import split,make_candidates,score,objective,guard
from possession.phase6.candidates import fit_logistic
from possession.phase6.decoding import DirectionalDecoder,candidate_bonuses
from scripts.train_e0 import manifest
from scripts.phase3 import metric_row,extra_diagnostics
from .common import Store,DEST,OLD,register,Budget,progress
from .training import train,load_model,source_folder
from .packing import compact_collate


def seal():
    path=DEST/"evaluation_registration.json"
    record={"source_sha256":file_hash(__file__),"training_registration_sha256":file_hash(DEST/"registration.json")}
    if path.exists() and json.loads(path.read_text())!=record:raise ValueError("Immutable evaluator wrapper changed")
    if not path.exists():atomic_json(path,record)


def selection_path(i,seed,arm,condition="IDSSE"):
    return DEST/f"selections/{condition}/fold-{i}/seed-{seed}/{arm}.json"


def priors_for(store,parts,i):
    path=DEST/f"priors/fold-{i}.npz"
    if path.exists():
        with np.load(path,allow_pickle=False) as z:return dict(z)
    prior=fit_priors([store.harmonized(k) for k in parts["train"]])
    path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,**prior)
    return prior


def predict(model,view,embeddings=False):
    dataset=Windows({"public":view},256,15);p=np.zeros((len(view.time),3),np.float32);q=np.zeros((len(view.time),1),np.float32)
    hidden=np.zeros((len(view.time),64),np.float32) if embeddings else None
    captured=[]
    hook=model.alive_head.register_forward_pre_hook(lambda module,args:captured.append(args[0].transpose(1,2).detach())) if embeddings else None
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0,len(dataset),8):
                items=[dataset[i] for i in range(start,min(start+8,len(dataset)))];captured.clear()
                out=forward(model,compact_collate(items));pp=out["probabilities"].numpy();qq=out["boundary_logit"].sigmoid().numpy()
                hh=captured[0].numpy() if embeddings else None
                for j,item in enumerate(items):
                    a,b,left=item["a"],item["b"],item["left"]
                    p[a:b],q[a:b]=pp[j,a-left:b-left],qq[j,a-left:b-left]
                    if embeddings:hidden[a:b]=hh[j,a-left:b-left]
    finally:
        if hook:hook.remove()
    return p,q,hidden


def cases(store,key,i,seed,arm,part,spec,condition="IDSSE"):
    view=store.harmonized(key);model=None;result={}
    for name,(degraded,missing) in scenarios(view,key,spec).items():
        path=DEST/f"predictions/{condition}/fold-{i}/seed-{seed}/{arm}/{part}/{name}.npz"
        reusable=condition=="IDSSE" and seed==2027 and i in (0,6)
        source=OLD/f"predictions/fold-{i}/{arm}/{part}/{name}.npz" if reusable else path
        if source.exists():
            with np.load(source,allow_pickle=False) as z:p,q=z["probabilities"],z["transition"]
        else:
            if reusable:raise ValueError("Expected frozen probability cache absent")
            if model is None:model=load_model(i,seed,arm,condition)
            t=time.perf_counter();p,q,_=predict(model,degraded)
            path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(".tmp")
            with temporary.open("wb") as stream:np.savez_compressed(stream,probabilities=p,transition=q,time=view.time,run=view.run,native_index=view.native_index)
            temporary.replace(path)
            atomic_json(path.with_suffix(".timing.json"),{"seconds":time.perf_counter()-t,"frames":len(p)})
        result[name]=(degraded,p,q,missing)
    return result


def select(store,data,i,seed,arm,spec,budget,condition="IDSSE"):
    path=selection_path(i,seed,arm,condition)
    if path.exists():return json.loads(path.read_text())
    budget.check();parts=split(data,i);prior=priors_for(store,parts,i)
    calibration=cases(store,parts["calibration"],i,seed,arm,"calibration",spec,condition)
    validation=cases(store,parts["validation"],i,seed,arm,"validation",spec,condition)
    cc=ScoreContext(store.match(parts["calibration"]),store.harmonized(parts["calibration"]))
    vc=ScoreContext(store.match(parts["validation"]),store.harmonized(parts["validation"]))
    packs=[make_candidates(cc,v,"binary","nms") for v in calibration.values()]
    x=np.concatenate([p["x"][:,:1] for p in packs]);y=np.concatenate([p["labels"] for p in packs])
    fitted=fit_logistic(x,y);prevalence=float((y.sum()+.5)/(len(y)+1))
    packs={n:make_candidates(vc,c,"binary","platt",fitted) for n,c in validation.items()};thresholds=[]
    from possession.phase6.candidates import candidate_metrics
    for threshold in spec["candidate_thresholds"]:
        metrics={n:candidate_metrics(vc,p["indices"],p["directions"],p["confidence"],threshold) for n,p in packs.items()}
        thresholds.append({"threshold":threshold,"objective":sum(w*metrics[n]["f1"] for w,n in zip(spec["score_weights"],spec["scenarios"]))})
    threshold=max(thresholds,key=lambda r:r["objective"])["threshold"];trials=[]
    for family in spec["decoder_families"]:
        baseline={n:score(vc,c,prior,family) for n,c in validation.items()}
        trials.append({"family":family,"method":"fixed","threshold":0.,"strength":0.,"metrics":baseline,"eligible":True,"objective":objective(baseline,spec)})
        metrics={n:score(vc,c,prior,family,packs[n],threshold,prevalence,.5) for n,c in validation.items()}
        trials.append({"family":family,"method":"platt","threshold":threshold,"strength":.5,"metrics":metrics,"eligible":guard(metrics,baseline),"objective":objective(metrics,spec)})
    result={"fold":i,"seed":seed,"arm":arm,"condition":condition,"split":parts,
        "winner":max((r for r in trials if r["eligible"]),key=lambda r:r["objective"]),"trials":trials,
        "calibrator":fitted,"prevalence":prevalence,"threshold_trials":thresholds,"outer_test_used":False}
    atomic_json(path,result);progress("internal_selection",fold=i,seed=seed,arm=arm,condition=condition)
    return result


def freeze_best(spec):
    path=DEST/"best_set_transformer.json"
    if path.exists():return
    scope=json.loads((DEST/"mixed_scope.json").read_text());best={}
    for i in spec["folds"]:
        candidates=[]
        for seed in spec["seeds"]:
            a=json.loads(selection_path(i,seed,"set_transformer").read_text());candidates.append(a)
            if i in scope["folds"]:
                b=json.loads(selection_path(i,seed,"set_transformer","IDSSE_SK").read_text())
                if guard(b["winner"]["metrics"],a["winner"]["metrics"]):candidates.append(b)
        choice=max(candidates,key=lambda r:r["winner"]["objective"])
        best[str(i)]={k:choice[k] for k in ("seed","condition","arm","split")}
    atomic_json(path,{"folds":best,"test_used":False,"note":"each fold uses only its own calibration/validation; condition availability recorded; no pooling other folds for own parameter selection"})


def evaluate_run(store,data,i,seed,arm,spec,budget,condition="IDSSE"):
    path=DEST/f"rows/{condition}/fold-{i}/seed-{seed}/{arm}.json"
    if path.exists():return json.loads(path.read_text())
    assert (DEST/"best_set_transformer.json").exists()
    chosen=json.loads(selection_path(i,seed,arm,condition).read_text());parts=chosen["split"]
    view=store.harmonized(parts["test"]);match=store.match(parts["test"]);context=ScoreContext(match,view)
    prior=priors_for(store,parts,i);rows=[]
    finals={"selected":chosen["winner"]}
    for family in spec["decoder_families"]:finals["fixed_"+family]=next(t for t in chosen["trials"] if t["family"]==family and t["method"]=="fixed")
    for name,case in cases(store,parts["test"],i,seed,arm,"test",spec,condition).items():
        budget.check();p,q,missing=case[1:];pack=make_candidates(context,case,"binary","platt",chosen["calibrator"])
        for system,trial in finals.items():
            bonuses=None if trial["method"]=="fixed" else candidate_bonuses(view,pack["indices"],pack["directions"],pack["confidence"],trial["threshold"],chosen["prevalence"],trial["strength"])
            decoder=DirectionalDecoder(view,prior,trial["family"],.5,bonuses)
            full=score_match(match,view,p,decoder);detail=diagnostics(context,context.decoded(p,decoder),q[:,0])
            row=metric_row(full,i,condition+"/"+arm,name,system,seed,20,20,float(missing.mean()),system=="selected")
            row.update(arm=arm,training_condition=condition,system=system,match=parts["test"],duration_groups=detail["duration_groups"],
                       decoded_HA=detail["decoded_HA"],head_HA=detail["head_HA"],independent=extra_diagnostics(match,view,p,decoder,missing))
            rows.append(row)
    atomic_json(path,rows);progress("external_evaluation",fold=i,seed=seed,arm=arm,condition=condition)
    return rows


def primary_selection():
    spec=register();seal();torch.set_num_threads(4);store=Store();data,_=manifest("idsse-only");budget=Budget(spec);status="failed"
    try:
        for seed in spec["seeds"]:
            for i in spec["folds"]:
                for arm in spec["arms"]:select(store,data,i,seed,arm,spec,budget)
        status="complete"
    finally:budget.close("primary_selection",status)


def mixed_training():
    spec=register();seal();torch.set_num_threads(4);store=Store();data,_=manifest("idsse-only");budget=Budget(spec);status="failed"
    try:
        pilots=spec["mixed"]["pilot_folds"];gains=[];occlusions={i:[] for i in pilots};times=[]
        for seed in spec["seeds"]:
            for i in pilots:
                start=time.perf_counter();parts=split(data,i)
                train(store,parts,i,seed,"set_transformer",spec,budget,"IDSSE_SK")
                b=select(store,data,i,seed,"set_transformer",spec,budget,"IDSSE_SK")
                a=json.loads(selection_path(i,seed,"set_transformer").read_text());times.append(time.perf_counter()-start)
                gains.append(b["winner"]["objective"]-a["winner"]["objective"])
                occlusions[i].append([b["winner"]["metrics"][n]["spellF1"]-a["winner"]["metrics"][n]["spellF1"] for n in spec["scenarios"][1:]])
        expanded=(np.mean(gains)>=.005 and sum(x>0 for x in gains)>=3 and all(np.asarray(v).mean(0).min()>=-.02 for v in occlusions.values())
                  and budget.remaining()>np.mean(times)*(len(spec["folds"])-len(pilots))*len(spec["seeds"])+300)
        scope={"folds":spec["folds"] if expanded else pilots,"expanded":bool(expanded),"pilot_validation_gains":gains,
               "pilot_occlusion_gains":occlusions,"remaining_compute_seconds":budget.remaining(),"outer_test_used":False,
               "gate_scope":"compute allocation only; model recipe fixed in advance; own selectors never use another fold"}
        frozen=DEST/"mixed_scope.json"
        if frozen.exists():scope=json.loads(frozen.read_text())
        else:atomic_json(frozen,scope)
        for seed in spec["seeds"]:
            for i in scope["folds"]:
                if i in pilots:continue
                train(store,split(data,i),i,seed,"set_transformer",spec,budget,"IDSSE_SK")
                select(store,data,i,seed,"set_transformer",spec,budget,"IDSSE_SK")
        freeze_best(spec);status="complete"
    finally:budget.close("mixed_training_selection",status)


def tests():
    spec=register();seal();torch.set_num_threads(4);store=Store();data,_=manifest("idsse-only");budget=Budget(spec);status="failed"
    try:
        scope=json.loads((DEST/"mixed_scope.json").read_text());rows=[]
        for seed in spec["seeds"]:
            for i in spec["folds"]:
                for arm in spec["arms"]:rows.extend(evaluate_run(store,data,i,seed,arm,spec,budget))
                if i in scope["folds"]:rows.extend(evaluate_run(store,data,i,seed,"set_transformer",spec,budget,"IDSSE_SK"))
        atomic_json(DEST/"all_rows.json",rows);status="complete"
    finally:budget.close("external_tests",status)
