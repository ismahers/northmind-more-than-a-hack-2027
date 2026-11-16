"""Direction-specific H/A potentials, frozen D costs, exact unbounded HSMM tail."""

import numpy as np
from numba import njit

from possession.learning.temporal import TemporalDecoder
from possession.night.decoders import dynamic_viterbi
from .candidates import logit


@njit(cache=True)
def directional_hsmm(emission,jumps,duration,survival,tail,mass,bonuses):
    n,cap=len(emission),duration.shape[1]
    prefix=np.zeros((n+1,3))
    for t in range(n):
        prefix[t+1]=prefix[t]+emission[t]
    values=np.full((n+1,3),-np.inf)
    entry=np.full((n+1,3),-np.inf);entry[0]=-np.log(3.)
    before=np.full((n+1,3),-1,np.int8)
    start=np.zeros((n+1,3),np.int32)
    tail_best,tail_start=np.full(3,-np.inf),np.zeros(3,np.int32)
    for t in range(1,n+1):
        for state in range(3):
            best,winning=-np.inf,0
            for d in range(1,min(cap-1,t)+1):
                u=t-d
                score=entry[u,state]+prefix[t,state]-prefix[u,state]+(survival[state,d] if t==n else duration[state,d])
                if score>best:
                    best,winning=score,u
            u=t-cap
            if u>=0:
                score=entry[u,state]-prefix[u,state]-u*tail[state,0]
                if score>tail_best[state]:
                    tail_best[state],tail_start[state]=score,u
                score=prefix[t,state]+(t-cap)*tail[state,0]+mass[state]+tail_best[state]+(0. if t==n else tail[state,1])
                if score>best:
                    best,winning=score,tail_start[state]
            values[t,state],start[t,state]=best,winning
        for state in range(3):
            best,old_state=-np.inf,0
            for old in range(3):
                bonus=bonuses[t,old] if t<n and old<2 and state<2 and old!=state else 0.
                score=values[t,old]+jumps[old,state]+bonus
                if score>best:
                    best,old_state=score,old
            entry[t,state],before[t,state]=best,old_state
    state=np.argmax(values[n]);t=n
    result=np.empty(n,np.int8)
    while t>0:
        u=start[t,state];result[u:t]=state
        state,t=before[u,state],u
    return result


def candidate_bonuses(view,indices,directions,confidence,threshold,prevalence,strength):
    result=np.zeros((len(view.time),2),float)
    for i,d,q in zip(indices,directions,confidence):
        if q<threshold:
            continue
        reward=strength*float(np.clip(logit(q)-logit(prevalence),0,5))
        for offset,factor in ((-1,.5),(0,1.),(1,.5)):
            j=i+offset
            if 0<=j<len(result) and view.run[j]==view.run[i]:
                result[j,d]=max(result[j,d],reward*factor)
    return result


class DirectionalDecoder:
    def __init__(self,view,priors,family="hmm",strength=.5,bonuses=None):
        self.priors,self.family,self.strength=priors,family,strength
        self.original=TemporalDecoder({"kind":family,"strength":strength,"boundary_strength":0.},priors)
        self.bonuses=bonuses
        self.indices={}
        for run in np.unique(view.run):
            ix=np.flatnonzero(view.run==run)
            self.indices[(float(view.time[ix[0]]),float(view.time[ix[-1]]),len(ix))]=ix

    def decode(self,p,time):
        if self.bonuses is None or not np.any(self.bonuses):
            return self.original.decode(p,time)
        ix=self.indices[(float(time[0]),float(time[-1]),len(time))]
        bonus=np.ascontiguousarray(self.bonuses[ix]).copy();bonus[0]=0
        e=np.log(np.clip(p.astype(float),1e-7,1))
        prior,s=self.priors,self.strength
        if self.family=="hmm":
            return dynamic_viterbi(e,prior["hmm_log_transition"]*s,bonus[:,0],bonus[:,1])
        return directional_hsmm(e,prior["jump_log_transition"]*s,prior["log_duration"]*s,
                                prior["log_survival"]*s,prior["tail"]*s,prior["tail_mass"]*s,bonus)
