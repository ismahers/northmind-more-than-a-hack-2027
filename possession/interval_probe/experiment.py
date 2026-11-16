"""Two small frozen-teacher probes; no complete detector or CRF is integrated."""
import json
import time
import numpy as np
import torch
from possession.external_data.schema import file_hash
from possession.learning.training import seed_everything,atomic_checkpoint
from possession.learning.study import ScoreContext
from possession.phase5.metrics import diagnostics
from possession.phase6.candidates import fit_logistic,logit
from possession.phase6.decoding import DirectionalDecoder
from possession.learning.metrics import score_match
from possession.night.common import atomic_json
from possession.phase8.common import Store,register,Budget,progress
from possession.phase8.evaluation import priors_for
from possession.phase8.training import load_model
from scripts.external_data import ROOT
from scripts.phase3 import metric_row
from .data import DEST,teacher,export,load
from .model import IntervalHead,interval_loss
from .proposals import head_predictions,generate,calibration_targets,bonuses
from .decoding import ProposalDecoder


def make_head(i,chosen):
    model=load_model(i,chosen["seed"],"set_transformer")
    return IntervalHead(model.team_head.weight,model.team_head.bias)


def train_head(store,i,chosen,config,main_spec,kind,budget):
    folder=DEST/f"training/fold-{i}/{kind}";folder.mkdir(parents=True,exist_ok=True)
    if (folder/"complete.json").exists():return
    data=[]
    for key in chosen["split"]["train"]:
        for name,path in export(store,key,i,chosen,main_spec).items():
            value=load(path);value["condition_weight"]=np.full(len(value["label"]),config["condition_weights"][name],np.float32);data.append(value)
    fields=("hidden","label","distance","regression_weight","regression_valid","condition_weight")
    tensors=[torch.from_numpy(np.concatenate([d[f] for d in data])) for f in fields]
    del data
    seed_everything(config["seed"]);model=make_head(i,chosen)
    opt=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=.0001)
    generator=torch.Generator().manual_seed(config["seed"]);history=[];start=time.perf_counter()
    if (folder/"last.pt").exists():
        state=torch.load(folder/"last.pt",map_location="cpu",weights_only=False);model.load_state_dict(state["model"]);opt.load_state_dict(state["optimizer"])
        generator.set_state(state["loader_rng"]);torch.set_rng_state(state["torch_rng"]);history=state["history"]
    for epoch in range(len(history)+1,config["epochs"]+1):
        budget.check();progress("interval_head",fold=i,loss=kind,epoch=epoch);t=time.perf_counter();total=[]
        order=torch.randperm(len(tensors[0]),generator=generator)
        for a in range(0,len(order),config["batch_size"]):
            index=order[a:a+config["batch_size"]]
            batch={field:value[index] for field,value in zip(fields,tensors)};opt.zero_grad(set_to_none=True)
            loss,stats=interval_loss(model(batch["hidden"]),batch,kind)
            if not torch.isfinite(loss):raise ValueError("Nonfinite interval head loss")
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step();total.append(float(loss.detach()))
        history.append({"epoch":epoch,"loss":float(np.mean(total)),"seconds":time.perf_counter()-t})
        atomic_checkpoint(folder/"last.pt",{"model":model.state_dict(),"optimizer":opt.state_dict(),"loader_rng":generator.get_state(),"torch_rng":torch.get_rng_state(),"history":history,"teacher":chosen})
    atomic_json(folder/"complete.json",{"history":history,"seconds":time.perf_counter()-start,"teacher":chosen,"parameters":sum(p.numel() for p in model.parameters())})


def load_head(i,chosen,kind):
    model=make_head(i,chosen);model.load_state_dict(torch.load(DEST/f"training/fold-{i}/{kind}/last.pt",weights_only=False,map_location="cpu")["model"]);return model


def packs(store,key,i,chosen,config,main_spec,kind):
    model=load_head(i,chosen,kind);view=store.harmonized(key);result={}
    for name,path in export(store,key,i,chosen,main_spec).items():
        data=load(path);p,distance,confidence=head_predictions(model,data);box,raw=generate(view,distance,confidence,config)
        result[name]={"p":p,"boxes":box,"raw":raw,"teacher_p":data["probabilities"],"actual_hidden_fraction":float(data["actual_hidden_fraction"])}
    return result


def evaluate(context,prior,pack,calibrator,strength):
    reward,q=bonuses(calibrator,pack["raw"],strength)
    decoder=ProposalDecoder(context.view,prior,pack["boxes"],reward)
    hard=context.decoded(pack["p"],decoder);detail=diagnostics(context,hard)
    return {"spellF1":context.pm.spell_scores(context.gt,hard,context.runs)["mean_f1"],
            "long_recall":detail["duration_groups"]["ge10"]["iou"]["0.5"]["recall"]}


def select(store,i,chosen,config,main_spec,budget):
    path=DEST/f"selections/fold-{i}.json"
    if path.exists():return
    parts=chosen["split"];prior=priors_for(store,parts,i);view=store.harmonized(parts["validation"])
    vc=ScoreContext(store.match(parts["validation"]),view);cc=ScoreContext(store.match(parts["calibration"]),store.harmonized(parts["calibration"]))
    teacher_cases=export(store,parts["validation"],i,chosen,main_spec)
    baseline={}
    for name,file in teacher_cases.items():
        hard=vc.decoded(load(file)["probabilities"],DirectionalDecoder(view,prior,"hmm",.5));detail=diagnostics(vc,hard)
        baseline[name]={"spellF1":vc.pm.spell_scores(vc.gt,hard,vc.runs)["mean_f1"],"long_recall":detail["duration_groups"]["ge10"]["iou"]["0.5"]["recall"]}
    teacher_trial={"kind":"teacher","strength":0.,"metrics":baseline,"objective":sum(w*baseline[n]["spellF1"] for w,n in zip(main_spec["score_weights"],main_spec["scenarios"]))}
    modes={};trials=[teacher_trial]
    for kind in config["losses"]:
        train_head(store,i,chosen,config,main_spec,kind,budget)
        cal=packs(store,parts["calibration"],i,chosen,config,main_spec,kind)
        x=np.concatenate([logit(p["raw"]) for p in cal.values()])[:,None]
        y=np.concatenate([calibration_targets(cc,p["boxes"]) for p in cal.values()])
        calibrator=fit_logistic(x,y) if len(y) else {"mean":[0.],"std":[1.],"weights":[float(logit(.01)),0.],"candidate_prevalence":.01,"n":0,"positives":0}
        val=packs(store,parts["validation"],i,chosen,config,main_spec,kind);own=[]
        for strength in config["bonus_strengths"]:
            metrics={n:evaluate(vc,prior,p,calibrator,strength) for n,p in val.items()}
            own.append({"kind":kind,"strength":strength,"metrics":metrics,"objective":sum(w*metrics[n]["spellF1"] for w,n in zip(main_spec["score_weights"],main_spec["scenarios"]))})
        zero=own[0]
        for trial in own:
            trial["eligible"]=all(trial["metrics"][n]["spellF1"]>=baseline[n]["spellF1"]-.02 for n in main_spec["scenarios"][1:]) and trial["metrics"]["clean"]["long_recall"]>=baseline["clean"]["long_recall"]-.03
        eligible=[t for t in own if t["eligible"]]
        best=max(eligible,key=lambda t:t["objective"]) if eligible else zero
        modes[kind]={"calibrator":calibrator,"trials":own,"winner":best}
        if best["eligible"]:trials.append(best)
    atomic_json(path,{"teacher":chosen,"modes":modes,"winner":max(trials,key=lambda t:t["objective"]),"test_used":False})


def test(store,i,chosen,config,main_spec,budget):
    path=DEST/f"rows/fold-{i}.json"
    if path.exists():return json.loads(path.read_text())
    for j in config["pilots"]:assert (DEST/f"selections/fold-{j}.json").exists()
    chosen_head=json.loads((DEST/f"selections/fold-{i}.json").read_text());parts=chosen["split"]
    match=store.match(parts["test"]);view=store.harmonized(parts["test"]);context=ScoreContext(match,view);prior=priors_for(store,parts,i)
    rows=[]
    teacher_cases=export(store,parts["test"],i,chosen,main_spec)
    for name,file in teacher_cases.items():
        data=load(file)
        for family in ("hmm","hsmm"):
            decoder=DirectionalDecoder(view,prior,family,.5);full=score_match(match,view,data["probabilities"],decoder);detail=diagnostics(context,context.decoded(data["probabilities"],decoder))
            row=metric_row(full,i,"teacher",name,"fixed_"+family,chosen["seed"],20,20,float(data["actual_hidden_fraction"]))
            row.update(system="teacher_"+family,loss_kind="teacher",duration_groups=detail["duration_groups"]);rows.append(row)
    for kind in config["losses"]:
        trial=chosen_head["modes"][kind];current=packs(store,parts["test"],i,chosen,config,main_spec,kind)
        for name,pack in current.items():
            for system in ("rehead_hmm","rehead_hsmm","proposal"):
                budget.check()
                if system=="proposal":
                    strength=trial["winner"]["strength"];reward,q=bonuses(trial["calibrator"],pack["raw"],strength)
                    decoder=ProposalDecoder(view,prior,pack["boxes"],reward)
                else:decoder=DirectionalDecoder(view,prior,system.split("_")[1],.5)
                full=score_match(match,view,pack["p"],decoder);detail=diagnostics(context,context.decoded(pack["p"],decoder))
                row=metric_row(full,i,kind,name,system,chosen["seed"],8,8,pack["actual_hidden_fraction"])
                row.update(system=system,loss_kind=kind,duration_groups=detail["duration_groups"],proposal_count=len(pack["boxes"]),
                    proposal_strength=trial["winner"]["strength"] if system=="proposal" else 0.)
                rows.append(row)
    atomic_json(path,rows);return rows


def run():
    main_spec=register();config=json.loads((ROOT/"configs/interval_probe_protocol.json").read_text());torch.set_num_threads(4)
    parent_budget=Budget(main_spec);store=Store();status="failed";begin=time.perf_counter()
    class LocalBudget:
        def check(self):
            parent_budget.check()
            if time.perf_counter()-begin>config["max_seconds"]:raise TimeoutError("Probe time budget; checkpoints preserved")
    budget=LocalBudget()
    chosen={str(i):teacher(i,main_spec) for i in config["pilots"]}
    sources={str(p.relative_to(ROOT)):file_hash(p) for p in (ROOT/"possession/interval_probe").glob("*.py")}
    record={"config":config,"teachers":chosen,"source":sources,"teacher_hashes":{i:file_hash(v["checkpoint"]) for i,v in chosen.items()}}
    marker=DEST/"registration.json"
    if marker.exists() and json.loads(marker.read_text())!=record:raise ValueError("Frozen interval probe changed")
    if not marker.exists():atomic_json(marker,record)
    try:
        for i in config["pilots"]:select(store,i,chosen[str(i)],config,main_spec,budget)
        rows=[]
        for i in config["pilots"]:rows.extend(test(store,i,chosen[str(i)],config,main_spec,budget))
        atomic_json(DEST/"all_rows.json",rows);status="complete"
    finally:
        atomic_json(DEST/"compute.json",{"seconds":time.perf_counter()-begin,"status":status})
        parent_budget.close("frozen_interval_probe",status)
