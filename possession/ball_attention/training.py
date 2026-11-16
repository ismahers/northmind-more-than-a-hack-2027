"""Paired fresh trajectories, original frozen losses/schedule and complete RNG."""
import copy
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from possession.learning.data import Windows
from possession.learning.training import seed_everything, epoch_loss, memory_usage, forward
from possession.phase5.model import boundary_loss
from possession.learning.model import masked_losses
from possession.phase6.experiment import corrupt, save_state, restore, epoch_pass
from possession.phase6.model import transition_targets, TransitionWindows
from possession.phase7.model import SpatialE0
from possession.phase8.packing import compact_collate, compact_transition_collate
from possession.night.common import atomic_json
from possession.temporal_probe.common import read_json
from .model import QueryE0
from .common import DEST


def equal(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


class CheckedLoader:
    def __init__(self, loader, budget):
        self.loader, self.budget, self.dataset = loader, budget, loader.dataset
    def __len__(self):
        return len(self.loader)
    def __iter__(self):
        for batch in self.loader:
            self.budget.check()
            yield batch


def folder(fold, arm, stage):
    return DEST/f"training/fold-{fold}/{arm}/{stage}"


def load_model(fold, arm):
    model = QueryE0(arm, True)
    state = torch.load(folder(fold, arm, "binary")/"last.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return model


def equivalence(store, record):
    """Execute BEFORE fitting: exact original forward/gradient/AdamW and resume."""
    checks = {}; maxima = {}
    for fold in record["protocol"]["pilots"]:
        teacher = record["teachers"][str(fold)]; seed = teacher["seed"]
        key = teacher["split"]["train"][0]; view = store.harmonized(key)
        for auxiliary in (False, True):
            name = f"fold-{fold}/"+("binary" if auxiliary else "base")
            seed_everything(seed); old = SpatialE0("set_transformer", auxiliary)
            rng = torch.get_rng_state().clone()
            seed_everything(seed); new = QueryE0("original", auxiliary)
            checks[name+"/init"] = equal(old.state_dict(), new.state_dict())
            checks[name+"/constructor_rng"] = torch.equal(rng, torch.get_rng_state())
            degraded = corrupt({key:view}, 13 if auxiliary else 1, dict(record["phase8_recipe"], seed=seed))
            targets = {key:transition_targets(store.match(key), view, 1, .12)} if auxiliary else None
            ds = TransitionWindows(degraded, targets) if auxiliary else Windows(degraded, 256, 15)
            batch = (compact_transition_collate if auxiliary else compact_collate)([ds[i] for i in range(4)])
            # Exact checks include dropout, base/aux losses, gradients and two updates.
            opts = [torch.optim.AdamW(m.parameters(), lr=.0003 if auxiliary else .001, weight_decay=.0001) for m in (old,new)]
            last_rng = None
            for step in range(2):
                rng = torch.get_rng_state().clone(); outputs = []; grads = []; losses = []; rngs = []
                for m, opt in zip((old,new), opts):
                    torch.set_rng_state(rng); m.train(); opt.zero_grad(set_to_none=True)
                    out = forward(m, batch)
                    loss = masked_losses(out,batch["alive"],batch["team"],batch["confidence"],batch["core"])[0]
                    if auxiliary:
                        loss = loss + .5*boundary_loss(out["boundary_logit"],batch["transition_target"],
                            (batch["transition_valid"]&batch["core"])[...,None],torch.tensor([40.]),"focal",2.)
                    loss.backward(); outputs.append({k:v.detach() for k,v in out.items()})
                    grads.append({k:p.grad.clone() for k,p in m.named_parameters() if p.grad is not None})
                    losses.append(loss.detach()); rngs.append(torch.get_rng_state().clone())
                    torch.nn.utils.clip_grad_norm_(m.parameters(),5.,error_if_nonfinite=True);opt.step()
                for label,a,b in (("output",outputs[0],outputs[1]),("loss",losses[0],losses[1]),
                                  ("gradient",grads[0],grads[1]),("weights",old.state_dict(),new.state_dict()),
                                  ("optimizer",opts[0].state_dict(),opts[1].state_dict()),("rng",rngs[0],rngs[1])):
                    checks[f"{name}/step{step}/{label}"] = equal(a,b)
                maxima[name+f"/step{step}"] = float((outputs[0]["probabilities"]-outputs[1]["probabilities"]).abs().max())
                if step == 0:
                    saved = copy.deepcopy(new.state_dict()), copy.deepcopy(opts[1].state_dict()), rngs[1]
                last_rng = rngs[1]
            resumed = QueryE0("original",auxiliary); resumed.load_state_dict(saved[0])
            ropt = torch.optim.AdamW(resumed.parameters(), lr=.0003 if auxiliary else .001, weight_decay=.0001)
            ropt.load_state_dict(saved[1]);torch.set_rng_state(saved[2])
            # The same second update, using the historical epoch implementation.
            if auxiliary:
                epoch_pass(resumed,[batch],ropt,torch.tensor([40.]),"binary",record["phase8_recipe"])
            else:
                epoch_loss(resumed,[batch],torch.device("cpu"),ropt,torch.amp.GradScaler("cuda",enabled=False))
            checks[name+"/exact_resume_weights"] = equal(resumed.state_dict(),new.state_dict())
            checks[name+"/exact_resume_optimizer"] = equal(ropt.state_dict(),opts[1].state_dict())
            checks[name+"/exact_resume_rng"] = torch.equal(last_rng,torch.get_rng_state())
    result = {"passed":all(checks.values()),"count":len(checks),"checks":checks,"maximum_probability_differences":maxima,
              "scope":"Real train windows only, two optimizer steps base/aux per fold before pilot, original wrapper versus unmodified SpatialE0. No historical checkpoints changed."}
    atomic_json(DEST/"equivalence.json",result)
    if not result["passed"]:
        raise ValueError("Original control equivalence failed")


def train(store, record, fold, arm, budget, digest):
    spec = record["phase8_recipe"]; teacher = record["teachers"][str(fold)]
    parts, seed = teacher["split"], teacher["seed"]
    views = {k:store.harmonized(k) for k in parts["train"]}
    validation = store.harmonized(parts["validation"])
    for stage, epochs, lr, offset in (("base",12,.001,0),("binary",8,.0003,12)):
        dest = folder(fold,arm,stage); dest.mkdir(parents=True,exist_ok=True)
        if (dest/"complete.json").exists():
            continue
        seed_everything(seed); model = QueryE0(arm, bool(offset))
        if offset:
            base = torch.load(folder(fold,arm,"base")/"last.pt",map_location="cpu",weights_only=False)
            missing,extra = model.load_state_dict(base["model"],strict=False)
            if set(missing)!={"boundary_head.weight","boundary_head.bias"} or extra:
                raise ValueError("Base handoff mismatch")
            seed_everything(seed+6000)
        initial = torch.get_rng_state().clone()
        opt = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001)
        generator = torch.Generator().manual_seed(seed+(6000 if offset else 0))
        scaler = torch.amp.GradScaler("cuda",enabled=False)
        targets = {k:transition_targets(store.match(k),v,1,.12) for k,v in views.items()} if offset else None
        if targets:
            positive = sum((y*v[:,None]).sum(0) for y,v in targets.values()); n = sum(v.sum() for _,v in targets.values())
            weights = torch.tensor(np.minimum(40.,(n-positive)/np.maximum(1,positive)),dtype=torch.float32)
        history = []; start = time.perf_counter()
        if (dest/"last.pt").exists():
            state = torch.load(dest/"last.pt",map_location="cpu",weights_only=False)
            if state["registration_sha256"] != digest:
                raise ValueError("Resume provenance changed")
            history = restore(state,model,opt,generator)
        val_loader = DataLoader(Windows({"val":validation},256,15),batch_size=4,collate_fn=compact_collate)
        for epoch in range(len(history)+1,epochs+1):
            budget.check(); atomic_json(DEST/"progress.json",{"fold":fold,"arm":arm,"stage":stage,"epoch":epoch})
            t = time.perf_counter(); degraded = corrupt(views,offset+epoch,dict(spec,seed=seed))
            ds = TransitionWindows(degraded,targets) if offset else Windows(degraded,256,15)
            loader = CheckedLoader(DataLoader(ds,batch_size=4,shuffle=True,generator=generator,
                       collate_fn=compact_transition_collate if offset else compact_collate),budget)
            stats = epoch_pass(model,loader,opt,weights,"binary",spec) if offset else epoch_loss(model,loader,torch.device("cpu"),opt,scaler)
            train_seconds = time.perf_counter()-t
            val = epoch_loss(model,CheckedLoader(val_loader,budget),torch.device("cpu"))
            history.append({"epoch":epoch,"global_epoch":offset+epoch,"train":stats,"validation":val,
                            "training_seconds":train_seconds,"seconds":time.perf_counter()-t,"windows":len(ds)})
            save_state(dest/"last.pt",model,opt,generator,history,{"split":parts,"seed":seed,"arm":arm,
                "registration_sha256":digest,"initial_rng":initial,"packing":True})
        atomic_json(dest/"complete.json",{"history":history,"seconds":time.perf_counter()-start,
            "parameters":sum(p.numel() for p in model.parameters()),"memory":memory_usage(torch.device("cpu")),
            "split":parts,"seed":seed,"registration_sha256":digest})
