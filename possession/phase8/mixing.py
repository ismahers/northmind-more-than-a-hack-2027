"""Extra partial team supervision without replacing a single IDSSE exposure."""
import numpy as np
import torch
from possession.learning.data import Windows
from possession.learning.model import masked_losses
from possession.learning.training import forward
from possession.phase5.model import boundary_loss
from .packing import compact_collate


class SkillBatches:
    def __init__(self,views,seed):
        self.dataset=Windows(views,256,15);self.keys=sorted(views);self.groups={k:[] for k in self.keys}
        self.generator=np.random.default_rng(seed)
        for i,(key,_,_,a,b) in enumerate(self.dataset.windows):
            v=views[key]
            if ((v.team[a:b]>=0)&(v.confidence[a:b,1]>0)).any():self.groups[key].append(i)
        if any(not x for x in self.groups.values()):raise ValueError("Missing partial team supervision")
    def next(self):
        items=[self.dataset[int(self.generator.choice(self.groups[k]))] for k in self.keys]
        return compact_collate(items)


def mixed_epoch(model,loader,optimizer,weights,spec,skill,sk_rng):
    model.train();total=np.zeros(4);extra=np.zeros(2);auxsum=0.;norms=[]
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        out=forward(model,batch);base,stats=masked_losses(out,batch["alive"],batch["team"],batch["confidence"],batch["core"])
        aux=boundary_loss(out["boundary_logit"],batch["transition_target"],
                          (batch["transition_valid"]&batch["core"])[...,None],weights,"focal",spec["focal_gamma"])
        loss=base+spec["boundary_weight"]*aux
        if not torch.isfinite(loss):raise ValueError("Nonfinite IDSSE loss")
        loss.backward()
        id_rng=torch.get_rng_state();torch.set_rng_state(sk_rng)
        sk=skill.next();skout=forward(model,sk)
        if not ((sk["alive"]==-1).all() and (sk["confidence"][...,0]==0).all()):raise ValueError("SK state must stay unknown")
        team,skstats=masked_losses(skout,sk["alive"],sk["team"],sk["confidence"],sk["core"])
        if not torch.isfinite(team):raise ValueError("Nonfinite partial team loss")
        (spec["mixed"]["weight"]*team).backward()
        sk_rng=torch.get_rng_state();torch.set_rng_state(id_rng)
        norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)))
        optimizer.step();total+=stats.numpy();extra+=skstats.numpy()[2:];auxsum+=float(aux.detach())
    return {"state_loss":float(total[0]/max(1,total[1])),"team_loss":float(total[2]/max(1,total[3])),
            "boundary_loss":auxsum/max(1,len(loader)),"skillcorner_team_loss":float(extra[0]/max(1,extra[1])),
            "skillcorner_team_weight":float(extra[1]),"gradient_norm_max":max(norms),"updates":len(loader),
            "idsse_windows":len(loader.dataset),
            "skillcorner_windows":4*len(loader)},sk_rng
