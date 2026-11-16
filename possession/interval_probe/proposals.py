"""Class-aware NMS, native-IoU calibration labels and positive interval rewards."""
import numpy as np
import torch
from possession.phase6.candidates import logit,apply_logistic
from possession.external_data.features import nearest_indices


def head_predictions(model,data):
    outputs=[];model.eval()
    with torch.no_grad():
        for a in range(0,len(data["hidden"]),8192):
            outputs.append(tuple(x.numpy() for x in model(torch.from_numpy(data["hidden"][a:a+8192]))))
    team,distance,quality=[np.concatenate([v[i] for v in outputs]) for i in range(3)]
    h=1/(1+np.exp(-np.clip(team,-30,30)));alive=np.clip(data["probabilities"][:,:2].sum(1),0,1)
    p=np.column_stack((alive*h,alive*(1-h),1-alive)).astype(np.float32)
    score=np.column_stack((h,1-h))*np.sqrt(1/(1+np.exp(-np.clip(quality,-30,30))))*alive[:,None]
    return p,distance,score


def generate(view,distance,score,config):
    result=[];confidence=[]
    for run in np.unique(view.run):
        ix=np.flatnonzero(view.run==run);time=view.time[ix]
        boundary=np.r_[time[0],(time[:-1]+time[1:])/2,time[-1]+.05]
        for team in (0,1):
            value=score[ix,team];chosen=np.flatnonzero(value>=config["proposal_floor"])
            chosen=chosen[np.argsort(-value[chosen],kind="stable")[:config["pre_topk_per_run_team"]]]
            start=nearest_indices(boundary,time[chosen]-distance[ix[chosen],team,0])
            end=nearest_indices(boundary,time[chosen]+distance[ix[chosen],team,1])
            start=start.clip(0,len(ix)-1);end=np.maximum(start+1,end.clip(1,len(ix)))
            keep=[];remaining=np.arange(len(chosen))
            while len(remaining) and len(keep)<config["post_topk_per_run_team"]:
                j=remaining[0];keep.append(j);others=remaining[1:]
                overlap=np.maximum(0,np.minimum(end[j],end[others])-np.maximum(start[j],start[others]))
                union=(end[j]-start[j])+(end[others]-start[others])-overlap
                remaining=others[overlap/np.maximum(1,union)<=config["nms_iou"]]
            for j in keep:
                result.append([int(ix[0]+start[j]),int(ix[0]+end[j]),team]);confidence.append(float(value[chosen[j]]))
    return np.asarray(result,np.int64).reshape(-1,3),np.asarray(confidence,float)


def calibration_targets(context,boxes):
    truth=context.pm.possession_spells(context.gt,context.runs);labels=np.zeros(len(boxes),float)
    for j,(a,b,team) in enumerate(boxes):
        for selected,mapping,output in context.parts:
            if not selected[0]<=a<=b-1<=selected[-1]:continue
            start=output.start+np.searchsorted(mapping,a-selected[0],"left")
            end=output.start+np.searchsorted(mapping,b-selected[0],"left")
            best=0.
            for lo,hi,state in truth:
                if state!=team:continue
                intersection=max(0,min(end,hi)-max(start,lo));union=max(end,hi)-min(start,lo)
                best=max(best,intersection/max(1,union))
            labels[j]=best>=.5
    return labels


def bonuses(calibrator,raw,strength):
    q=apply_logistic(calibrator,logit(raw)[:,None])
    reward=strength*np.clip(logit(q)-logit(calibrator["candidate_prevalence"]),0,5)
    return reward,q
