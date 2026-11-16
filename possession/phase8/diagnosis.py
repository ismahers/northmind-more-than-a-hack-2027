"""Post-selection diagnostics only: network evidence, decoder losses, grid limits."""
import json
import numpy as np
import torch
from possession.external_data.audit import joint_labels
from possession.external_data.schema import file_hash
from possession.learning.study import ScoreContext
from possession.learning.temporal import TemporalDecoder
from possession.phase5.metrics import diagnostics,event_metrics
from possession.phase6.experiment import make_candidates
from possession.phase6.candidates import canonical_events,candidate_metrics
from possession.phase6.decoding import DirectionalDecoder,candidate_bonuses
from possession.night.common import atomic_json
from scripts.train_e0 import manifest
from .common import Store,DEST,register,Budget,progress
from .evaluation import cases,selection_path,priors_for


def matched_truth(pm,truth,prediction,tolerance=12):
    pairs,_,_=pm.match_transitions(truth,prediction,tolerance,typed="pair")
    return {tuple(t) for t,_ in pairs}


def HA_events(pm,hard,runs):return [e for e in pm.transitions(hard,runs) if e[1]<2 and e[2]<2]


def flags(pm,truth,hard,runs):
    predicted=pm.possession_spells(hard,runs);overlap=pm.spell_overlaps(truth,predicted)
    return pm.matched_gt_flags(overlap,len(truth),.5)


def state_binary(y):return np.where(y<2,0,np.where(y==2,1,3)).astype(np.int8)


def analyze(context,case,prior,selection):
    view,p,q,missing=case;pm=context.pm
    raw=context.decoded(p,TemporalDecoder({"kind":"argmax"},None))
    head_label=np.where(p[:,:2].sum(1)>=.5,(p[:,1]>p[:,0]).astype(int),2)
    head=context.decoded(np.eye(3)[head_label],TemporalDecoder({"kind":"argmax"},None))
    hmm=context.decoded(p,DirectionalDecoder(view,prior,"hmm",.5))
    hsmm=context.decoded(p,DirectionalDecoder(view,prior,"hsmm",.5))
    pack=make_candidates(context,case,"binary","platt",selection["calibrator"])
    # Threshold was selected on the own validation candidate F1, before test.
    threshold=max(selection["threshold_trials"],key=lambda r:r["objective"])["threshold"]
    selected=pack["confidence"]>=threshold
    candidate_events=canonical_events(context,pack["indices"][selected],pack["directions"][selected])
    truth=HA_events(pm,context.gt,context.runs)
    raw_detected=matched_truth(pm,truth,HA_events(pm,raw,context.runs))
    head_detected=matched_truth(pm,truth,HA_events(pm,head,context.runs))
    aux_detected=matched_truth(pm,truth,candidate_events)
    network=head_detected|aux_detected
    hmm_detected=matched_truth(pm,truth,HA_events(pm,hmm,context.runs))
    trial=selection["winner"]
    bonuses=None if trial["method"]=="fixed" else candidate_bonuses(view,pack["indices"],pack["directions"],pack["confidence"],trial["threshold"],selection["prevalence"],trial["strength"])
    chosen=context.decoded(p,DirectionalDecoder(view,prior,trial["family"],.5,bonuses))
    chosen_detected=matched_truth(pm,truth,HA_events(pm,chosen,context.runs))
    native=joint_labels_for_context(context,view)
    sampling=context.decoded(np.eye(3)[native],TemporalDecoder({"kind":"argmax"},None))
    binary_truth=pm.transitions(state_binary(context.gt),context.runs)
    binary_network=matched_truth(pm,binary_truth,pm.transitions(state_binary(head),context.runs))
    binary_hmm=matched_truth(pm,binary_truth,pm.transitions(state_binary(hmm),context.runs))
    spells=pm.possession_spells(context.gt,context.runs)
    matched={name:flags(pm,spells,hard,context.runs) for name,hard in (("joint_argmax",raw),("head_argmax",head),("hmm",hmm),("selected",chosen),("sampling_reference",sampling))}
    # Place native sample indices on the scorer timeline without interpolating labels.
    anchors=[]
    for indices,mapping,output in context.parts:
        local=np.arange(len(indices));lo,hi=np.searchsorted(mapping,local,"left"),np.searchsorted(mapping,local,"right")
        anchors.extend((output.start+(lo+hi)//2)[hi>lo].tolist())
    anchors=np.asarray(anchors)
    details=[]
    for j,(a,b,team) in enumerate(spells):
        old=int(context.gt[a-1]) if a else 3;new=int(context.gt[b]) if b<len(context.gt) else 3
        start="censored" if old==3 else "D_restart" if old==2 else "HA"
        end="censored" if new==3 else "D_stop" if new==2 else "HA"
        begin_ok=(True if start=="censored" else (a,old,team) in network if start=="HA" else (a,1,0) in binary_network)
        end_ok=(True if end=="censored" else (b,team,new) in network if end=="HA" else (b,0,1) in binary_network)
        begin_hmm=(True if start=="censored" else (a,old,team) in hmm_detected if start=="HA" else (a,1,0) in binary_hmm)
        end_hmm=(True if end=="censored" else (b,team,new) in hmm_detected if end=="HA" else (b,0,1) in binary_hmm)
        details.append({"duration_s":(b-a)/25,"team":int(team),"start_type":start,"end_type":end,
            "model_anchor_centres_inside":int(((anchors>=a)&(anchors<b)).sum()),
            "both_network_endpoints_present":bool(begin_ok and end_ok),"both_hmm_endpoints_present":bool(begin_hmm and end_hmm),
            "false_D_frames_hmm":int((hmm[a:b]==2).sum()),"wrong_team_frames_hmm":int((hmm[a:b]==1-team).sum()),
            "matched_IoU0_5":{name:bool(value[j]) for name,value in matched.items()}})
    return {"direct_HA_truth":len(truth),"joint_argmax_detected":len(raw_detected),"head_argmax_detected":len(head_detected),
        "auxiliary_candidates_detected":len(aux_detected),"network_union_detected":len(network),
        "network_present_HMM_absent":len(network-hmm_detected),"network_present_selected_absent":len(network-chosen_detected),
        "not_identified_by_operational_network_candidates":len(set(map(tuple,truth))-network),
        "HMM_found_without_operational_network_candidate":len(hmm_detected-network),
        "candidate_metrics":candidate_metrics(context,pack["indices"],pack["directions"],pack["confidence"],threshold),
        "candidate_threshold":threshold,"tolerance_frames_25Hz":12,
        "decoders":{name:{"spellF1":pm.spell_scores(context.gt,hard,context.runs)["mean_f1"],
                           **diagnostics(context,hard)} for name,hard in (("joint_argmax",raw),("head_argmax",head),("hmm",hmm),("hsmm",hsmm),("selected",chosen),("GT25_to10_to25_reference",sampling))},
        "per_possession":details,
        "interpretation":"presence at frozen threshold is evidence, not proof a decoder can recover it without false positives; missing operational candidates does not prove hidden embeddings lack information"}


def joint_labels_for_context(context,view):
    # Complete public IDSSE labels, analysis-only; never called by a model/selector.
    labels=np.where(view.alive==0,2,view.team)
    if (labels<0).any():raise ValueError("Sampling reference requires full public labels")
    return labels.astype(int)


def run():
    spec=register();torch.set_num_threads(4);store=Store();data,_=manifest("idsse-only");budget=Budget(spec);status="failed"
    best=json.loads((DEST/"best_set_transformer.json").read_text())
    record={"source_sha256":file_hash(__file__),"best_selection_sha256":file_hash(DEST/"best_set_transformer.json"),
            "scope":"post-selection analysis only; no oracle feeds predictions or selection; GT reconstruction is a reference, not an information-theoretic upper bound"}
    marker=DEST/"diagnosis_registration.json"
    if marker.exists() and json.loads(marker.read_text())!=record:raise ValueError("Immutable diagnosis changed")
    if not marker.exists():atomic_json(marker,record)
    rows=[]
    try:
        for i in spec["folds"]:
            selected=best["folds"][str(i)];seed=selected["seed"];condition=selected["condition"]
            selection=json.loads(selection_path(i,seed,"set_transformer",condition).read_text());parts=selection["split"]
            v=store.harmonized(parts["test"]);ctx=ScoreContext(store.match(parts["test"]),v);prior=priors_for(store,parts,i)
            for name,case in cases(store,parts["test"],i,seed,"set_transformer","test",spec,condition).items():
                destination=DEST/f"diagnosis/fold-{i}/{name}.json"
                if destination.exists():rows.append(json.loads(destination.read_text()));continue
                budget.check();progress("short_spell_diagnosis",fold=i,scenario=name)
                result={"fold":i,"seed":seed,"condition":condition,"scenario":name,**analyze(ctx,case,prior,selection)}
                atomic_json(destination,result);rows.append(result)
        atomic_json(DEST/"diagnosis.json",{"rows":rows,"registration":record});status="complete"
    finally:budget.close("short_spell_diagnosis",status)
