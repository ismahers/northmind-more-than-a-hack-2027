"""Verify/profiling of public-only padding compaction; no model selection."""
import time
import numpy as np
import torch
from possession.phase7.model import SpatialE0
from possession.phase8.packing import pack
from possession.learning.training import seed_everything,forward
from possession.night.common import atomic_json
from scripts.external_data import REPORTS


def main():
    torch.set_num_threads(4);seed_everything(2027)
    x=torch.randn(4,286,2,18,6);mask=torch.zeros(4,286,2,18,dtype=torch.bool)
    mask[...,:11]=True;mask[0,0]=False;mask[0,1,0]=False
    batch={"nodes":x,"node_mask":mask,"global_features":torch.randn(4,286,7),"valid":torch.ones(4,286,dtype=torch.bool)}
    packed=pack(batch);rows=[]
    for arm in ("harmonized","set_transformer"):
        seed_everything(2027);model=SpatialE0(arm,True)
        model.eval()
        with torch.no_grad():
            a,b=[forward(model,v)["probabilities"] for v in (batch,packed)]
        error=float((a-b).abs().max())
        gradients=[]
        for data in (batch,packed):
            model.zero_grad(set_to_none=True);out=forward(model,data)
            sum(out[k].square().mean() for k in ("team_logit","alive_logit","boundary_logit")).backward()
            gradients.append({n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None})
        grad=max(float((g-gradients[1][n]).abs().max()) for n,g in gradients[0].items())
        times={}
        for name,data in (("padded",batch),("compacted",packed)):
            values=[];model.train()
            for j in range(6):
                start=time.perf_counter();model.zero_grad(set_to_none=True);out=forward(model,data)
                sum(out[k].square().mean() for k in ("team_logit","alive_logit","boundary_logit")).backward()
                if j>=2:values.append(time.perf_counter()-start)
            times[name]=float(np.median(values))
        rows.append({"arm":arm,"max_probability_error":error,"max_gradient_abs_error_eval":grad,"step_seconds_median":times,
                     "speedup":times["padded"]/times["compacted"],"passed":error<2e-6 and grad<2e-5})
    atomic_json(REPORTS/"phase8-profile.json",{"rows":rows,"packing_only":True,"padded_players":18,"compacted_players":11,"passed":all(r["passed"] for r in rows)})
    return 0 if all(r["passed"] for r in rows) else 1
