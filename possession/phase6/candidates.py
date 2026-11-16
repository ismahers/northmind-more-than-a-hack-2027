"""OOF candidate generation, ridge logistic calibration and event matching."""

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


def logit(x):
    x = np.clip(x, 1e-5, 1-1e-5)
    return np.log(x/(1-x))


def means(x, radius, before):
    prefix = np.r_[0., np.cumsum(x)]
    t = np.arange(len(x))
    lo, hi = (np.maximum(0,t-radius),t) if before else (t,np.minimum(len(x),t+radius))
    return (prefix[hi]-prefix[lo])/np.maximum(1,hi-lo)


def direction_scores(view, p, q, arm):
    out = np.zeros((len(p),2), np.float64)
    if arm == "directional":
        return q.astype(float)
    h = p[:,0]/np.maximum(1e-7,p[:,:2].sum(-1))
    for run in np.unique(view.run):
        ix = np.flatnonzero(view.run==run)
        delta = means(h[ix],3,True)-means(h[ix],3,False)
        if arm == "binary":
            out[ix,0],out[ix,1] = q[ix,0]*(delta>=0),q[ix,0]*(delta<0)
        else:
            out[ix,0],out[ix,1] = np.maximum(0,delta),np.maximum(0,-delta)
    return out


def context_features(view, p, scores):
    h = p[:,0]/np.maximum(1e-7,p[:,:2].sum(-1))
    alive = p[:,:2].sum(-1)
    available = view.global_features[:,3]
    # Comparable canonical-pitch distance. Missing coordinates never imply D.
    distance = np.linalg.norm(view.nodes[...,2:4]*np.array([52.5,34.]),axis=-1)
    distance[~view.node_mask] = np.inf
    nearest = distance.min(-1)
    nearest = np.where(available[:,None]>.5,np.minimum(nearest,50.),50.)
    geom = np.tanh((nearest[:,1]-nearest[:,0])/3.)
    features = np.zeros((len(p),2,18),float)
    confirmation = np.zeros((len(p),2),float)
    for run in np.unique(view.run):
        ix = np.flatnonzero(view.run==run)
        for direction in (0,1):
            old = h[ix] if direction==0 else 1-h[ix]
            new = 1-old
            left,right = means(old,3,True),means(new,3,False)
            gl = geom[ix] if direction==0 else -geom[ix]
            columns=[logit(scores[ix,direction]),np.full(len(ix),direction),left,right,left+right-1,
                     means(old,10,True),means(new,10,False),
                     means(alive[ix],3,True),means(alive[ix],3,False),
                     means(available[ix],3,True),means(available[ix],3,False),
                     abs(h[ix]-.5),scores[ix,1-direction],
                     means(gl,3,True),-means(gl,3,False),
                     np.minimum(nearest[ix,direction],50.)/50.,
                     np.minimum(nearest[ix,1-direction],50.)/50.,available[ix]]
            features[ix,direction]=np.column_stack(columns)
            confirmation[ix,direction]=scores[ix,direction]*left*right*np.minimum(columns[7],columns[8])
    return features,confirmation


def generate(view,scores,nms_s=.2,floor=1e-5):
    indices,directions=[],[]
    for run in np.unique(view.run):
        ix=np.flatnonzero(view.run==run)
        for direction in (0,1):
            q=scores[ix,direction]
            local=np.flatnonzero((q>=floor)&(q>=np.r_[-1.,q[:-1]])&(q>np.r_[q[1:],-1.]))
            chosen=[]
            blocked=np.zeros(len(ix),bool)
            for i in local[np.argsort(-q[local],kind="stable")]:
                if not blocked[i]:
                    chosen.append(int(i))
                    lo,hi=np.searchsorted(view.time[ix],[view.time[ix[i]]-nms_s,view.time[ix[i]]+nms_s],side="left")
                    blocked[lo:hi]=True
                    blocked[i]=True
            indices.extend(ix[chosen].tolist());directions.extend([direction]*len(chosen))
    order=np.lexsort((np.array(directions),np.array(indices)))
    return np.asarray(indices,int)[order],np.asarray(directions,int)[order]


def canonical_events(context,indices,directions):
    lookup=getattr(context,"phase6_native_lookup",None)
    if lookup is None:
        lookup={}
        for selected,mapping,output in context.parts:
            local=np.arange(len(selected))
            lo,hi=np.searchsorted(mapping,local,"left"),np.searchsorted(mapping,local,"right")
            valid=hi>lo
            native=output.start+(lo+hi)//2
            lookup.update(zip(selected[valid].tolist(),native[valid].tolist()))
        context.phase6_native_lookup=lookup
    return [(lookup[int(i)],int(d),1-int(d)) for i,d in zip(indices,directions) if int(i) in lookup]


def candidate_labels(context,indices,directions,tolerance=12):
    events=canonical_events(context,indices,directions)
    gt=[e for e in context.pm.transitions(context.gt,context.runs) if e[1]<2 and e[2]<2]
    pairs,missed,false=context.pm.match_transitions(gt,events,tolerance,typed="pair")
    matched={tuple(p) for _,p in pairs}
    return np.array([tuple(e) in matched for e in events],float)


def candidate_metrics(context,indices,directions,confidence,threshold):
    choose=confidence>=threshold
    events=canonical_events(context,indices[choose],directions[choose])
    gt=[e for e in context.pm.transitions(context.gt,context.runs) if e[1]<2 and e[2]<2]
    pairs,missed,false=context.pm.match_transitions(gt,events,12,typed="pair")
    tp,fp,fn=len(pairs),len(false),len(missed)
    precision,recall=tp/max(1,tp+fp),tp/max(1,tp+fn)
    minutes=sum(b-a for a,b in context.runs)/25/60
    return {"tp":tp,"fp":fp,"fn":fn,"precision":precision,"recall":recall,
            "f1":2*precision*recall/max(1e-12,precision+recall),"false_per_minute":fp/max(1e-9,minutes),
            "candidates":len(events),"minutes":minutes}


def fit_logistic(x,y,penalty=1.):
    x,y=np.asarray(x,float),np.asarray(y,float)
    if not len(y):
        raise ValueError("No OOF calibration candidates")
    mean,std=x.mean(0),np.maximum(x.std(0),.01)
    design=np.column_stack((np.ones(len(x)),(x-mean)/std))
    weights=np.zeros(design.shape[1]);weights[0]=logit((y.sum()+.5)/(len(y)+1))
    ridge=np.r_[1e-6,np.full(x.shape[1],penalty)]
    for _ in range(35):
        p=sigmoid(design@weights)
        gradient=design.T@(p-y)+ridge*weights
        hessian=design.T@(design*np.maximum(1e-6,p*(1-p))[:,None])+np.diag(ridge)
        step=np.linalg.solve(hessian,gradient)
        weights-=np.clip(step,-2,2)
        if np.max(abs(step))<1e-7:
            break
    return {"mean":mean.tolist(),"std":std.tolist(),"weights":weights.tolist(),
            "candidate_prevalence":float((y.sum()+.5)/(len(y)+1)),"n":len(y),"positives":int(y.sum())}


def apply_logistic(model,x):
    design=np.column_stack((np.ones(len(x)),(x-np.array(model["mean"]))/np.array(model["std"])))
    return sigmoid(design@np.array(model["weights"]))


def calibration_metrics(y,q):
    q=np.clip(q,1e-7,1-1e-7)
    ece=0.
    for a,b in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        mask=(q>=a)&(q<b)
        if mask.any():
            ece+=mask.mean()*abs(q[mask].mean()-y[mask].mean())
    return {"brier":float(np.mean((q-y)**2)),"log_loss":float(-np.mean(y*np.log(q)+(1-y)*np.log1p(-q))),"ece10":float(ece)}
