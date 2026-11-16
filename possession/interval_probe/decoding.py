"""Sparse complete-possession arcs plus HMM frame fallback, with independent D.

Each proposal must start and end a maximal H/A segment. D is never inferred
from absence of proposals: the original three-state frame arcs always remain.
Zero bonus returns the original HMM exactly. No minimum spell length is added.
"""
import numpy as np
from numba import njit
from possession.learning.temporal import hmm_viterbi


@njit(cache=True)
def interval_viterbi(emission,transition,starts,ends,classes,bonus):
    n=len(emission);prefix=np.zeros((n+1,3))
    for t in range(n):prefix[t+1]=prefix[t]+emission[t]
    order=np.argsort(starts);offset=np.zeros(n+2,np.int64)
    for s in starts:offset[s+1]+=1
    for t in range(n+1):offset[t+1]+=offset[t]
    values=np.full((n+1,3,2),-np.inf)
    previous_time=np.full((n+1,3,2),-1,np.int32)
    previous_state=np.full((n+1,3,2),-1,np.int8)
    previous_lock=np.zeros((n+1,3,2),np.int8)
    for c in range(3):values[1,c,0]=emission[0,c]-np.log(3.);previous_time[1,c,0]=0
    for t in range(n):
        if t>0:
            for c in range(3):
                for old in range(3):
                    for lock in range(2):
                        if lock==1 and c==old:continue
                        candidate=values[t,old,lock]+transition[old,c]+emission[t,c]
                        if candidate>values[t+1,c,0]:
                            values[t+1,c,0]=candidate;previous_time[t+1,c,0]=t
                            previous_state[t+1,c,0]=old;previous_lock[t+1,c,0]=lock
        for pointer in range(offset[t],offset[t+1]):
            j=order[pointer];e,c=ends[j],classes[j]
            span=prefix[e,c]-prefix[t,c]+(e-t-1)*transition[c,c]+bonus[j]
            if t==0:
                candidate=span-np.log(3.)
                if candidate>values[e,c,1]:values[e,c,1]=candidate;previous_time[e,c,1]=0
            else:
                for old in range(3):
                    if old==c:continue
                    for lock in range(2):
                        candidate=values[t,old,lock]+transition[old,c]+span
                        if candidate>values[e,c,1]:
                            values[e,c,1]=candidate;previous_time[e,c,1]=t
                            previous_state[e,c,1]=old;previous_lock[e,c,1]=lock
    winner=np.argmax(values[n]);state,lock=winner//2,winner%2;t=n
    result=np.empty(n,np.int8)
    while t>0:
        before=previous_time[t,state,lock];result[before:t]=state
        old,was_locked=previous_state[t,state,lock],previous_lock[t,state,lock]
        state,lock,t=old,was_locked,before
    return result


class ProposalDecoder:
    def __init__(self,view,prior,proposals,bonuses):
        self.transition=prior["hmm_log_transition"]*.5;self.runs={}
        for run in np.unique(view.run):
            ix=np.flatnonzero(view.run==run);selected=(proposals[:,0]>=ix[0])&(proposals[:,1]<=ix[-1]+1)
            self.runs[(float(view.time[ix[0]]),float(view.time[ix[-1]]),len(ix))]=(proposals[selected],bonuses[selected],ix[0])

    def decode(self,p,time):
        proposal,bonus,left=self.runs[(float(time[0]),float(time[-1]),len(time))]
        e=np.log(np.clip(p.astype(float),1e-7,1))
        use=bonus>0
        if not use.any():return hmm_viterbi(e,self.transition)
        box=proposal[use].astype(np.int64)
        return interval_viterbi(e,self.transition,box[:,0]-left,box[:,1]-left,box[:,2],bonus[use])
