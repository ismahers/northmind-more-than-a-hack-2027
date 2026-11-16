"""Local gated-target audit. Never import or open private ground truth.

All entry points must run inside scripts.run_local with output captured from
process creation. Generated target arrays/reports stay restricted and local.
"""

import csv
import json
from pathlib import Path

import numpy as np

from games import GAME,GAMES,asset_paths,source_frames,half_video_blocks
from starter.baseline import team_of_each_track
from possession.external_data.schema import empty_match
from possession.external_data.features import build_features,nearest_indices
from possession.learning.metrics import official_scorer


def xy_metres(normalized,pitch):
    return np.asarray(normalized)*np.asarray(pitch)-np.asarray(pitch)/2


def load_target():
    paths=asset_paths();n=source_frames();pitch=GAMES[GAME]["pitch"]
    if not all(Path(paths[name]).is_file() for name in ("atd_meta","atd_raw")):
        return None
    teams=team_of_each_track(paths["atd_meta"])
    periods=np.zeros(n,np.int16)
    for half,first,last in half_video_blocks(GAME):
        periods[first:last+1]=half
    selected=np.flatnonzero(periods>0)
    frame_to_row=np.full(n,-1,int);frame_to_row[selected]=np.arange(len(selected))
    rosters=[["slot-"+str(i) for i in np.flatnonzero(teams==side)] for side in (0,1)]
    match=empty_match("target",GAME,selected,periods[selected],np.full(len(selected),""),selected/25,
                      rosters,pitch,25,[])
    count=0
    with open(paths["atd_raw"],encoding="utf-8") as stream:
        for frame,line in enumerate(stream):
            groups=line.strip().split(":")
            if len(groups)!=3 or frame>=n or int(groups[0])!=frame+1:
                raise ValueError("Target frame contract failed")
            row=frame_to_row[frame]
            if row>=0:
                players=np.fromstring(groups[1].replace(";",","),sep=",")
                ball=np.fromstring(groups[2],sep=",")
                if players.size!=3*len(teams) or ball.size!=3:
                    raise ValueError("Target slot dimensions failed")
                players=players.reshape(-1,3)
                for side in (0,1):
                    xy=xy_metres(players[teams==side,:2],pitch)
                    match.players[row,side,:len(xy)]=xy
                match.ball[row,:2]=xy_metres(ball[:2],pitch)
                # ATD's third value is speed, NOT height. Leave z unknown.
            count+=1
    if count!=n:
        raise ValueError("Target file incomplete")
    match.player_available=np.isfinite(match.players).all(-1)
    match.ball_available=np.isfinite(match.ball[:,:2]).all(-1)
    match.metadata["slot_identity_semantics"]="tracklets; no verified person identities"
    match.metadata["ball_height_semantics"]="unknown; ATD third value is speed"
    return match.validate()


def missing_runs(mask,run,fps):
    lengths=[]
    for identity in np.unique(run):
        v=mask[run==identity]
        edges=np.diff(np.r_[0,v.astype(int),0])
        lengths.extend(((np.flatnonzero(edges==-1)-np.flatnonzero(edges==1))/fps).tolist())
    return {"count":len(lengths),"mean_s":float(np.mean(lengths)) if lengths else 0.,
            "p50_s":float(np.quantile(lengths,.5)) if lengths else 0.,
            "p95_s":float(np.quantile(lengths,.95)) if lengths else 0.,"max_s":max(lengths,default=0.)}


def audit_match(match,view):
    differences=np.diff(match.players,axis=0)
    dt=np.diff(match.time)
    same=(match.period[1:]==match.period[:-1])&(dt>0)&(dt<.1)
    valid=match.player_available[1:]&match.player_available[:-1]&same[:,None,None]
    speed=np.linalg.norm(differences,axis=-1)/np.maximum(1e-6,dt[:,None,None])
    finite_speed=speed[valid]
    starts=match.player_available[1:]&~match.player_available[:-1]&same[:,None,None]
    return {"frames":len(match.frame),"native_hz":25,"model_hz":10,
            "ball_missing_fraction":float((~match.ball_available).mean()),
            "ball_missing_blocks_native":missing_runs(~match.ball_available,match.period,25),
            "ball_missing_blocks_10hz":missing_runs(view.global_features[:,3]<.5,view.run,10),
            "observed_players_mean":float(match.player_available.sum((1,2)).mean()),
            "observed_players_quantiles":np.quantile(match.player_available.sum((1,2)),[0,.05,.5,.95,1]).tolist(),
            "player_reappearances":int(starts.sum()),"slot_jumps_above_12m_per_s":int((finite_speed>12).sum()),
            "identity_note":"reappearance/jump proxies only; no identity ground truth or confirmed ID-switch count",
            "pitch_m":match.metadata["pitch_m"],"coordinate_mapping":"0..1 to centred metres; no GT-based rotation",
            "observed_ball_xy_quantiles_m":np.nanquantile(match.ball[:,:2],[.01,.5,.99],axis=0).tolist(),
            "z_missing_fraction":float((~np.isfinite(match.ball[:,2])).mean()),
            "feature_mean":view.global_features.mean(0).tolist(),"feature_std":view.global_features.std(0).tolist(),
            "private_GT_accessed":False}


def run_target():
    # Imports after output redirection in the reviewed runner.
    from possession.phase6.experiment import DEST,register,load_model,predict
    from possession.phase6.candidates import direction_scores,context_features,generate,apply_logistic
    from possession.phase6.decoding import candidate_bonuses,DirectionalDecoder
    from possession.night.common import atomic_json
    from scripts.external_data import ROOT
    from possession.external_data.schema import file_hash
    output=ROOT/"evaluation/reports/phase6-target"
    output.mkdir(parents=True,exist_ok=True)
    paths=asset_paths()
    missing=[Path(paths[name]).name for name in ("atd_meta","atd_raw") if not Path(paths[name]).is_file()]
    if missing:
        atomic_json(output/"status.json",{"status":"MISSING_ASSETS","required_missing":missing,
            "next":"Put the official tactical XML/TXT in data/tactical or set MTAH_DATA_DIR; rerun audit-phase6-target",
            "assets_opened":False,"private_GT_accessed":False})
        return 0
    protocol=register()
    # Fixed pilot 0; arm selected on its own inner validation, before target read.
    selection=json.loads((DEST/"selections/fold-0.json").read_text())
    arm=max(selection["arms"],key=lambda name:selection["arms"][name]["winner"]["objective"])
    choice=selection["arms"][arm];trial=choice["winner"]
    atomic_json(output/"frozen_selection.json",{"pilot":0,"arm":arm,"trial":trial,"target_labels_used":False,
                                              "target_adapter_sha256":file_hash(Path(__file__))})
    match=load_target();view=build_features(match)
    audit=audit_match(match,view)
    # IDSSE reference distributions use only this pilot's training matches.
    from scripts.train_e0 import manifest
    from possession.phase5.experiment import get_store
    data,_=manifest("idsse-only");store=get_store(data)
    references=[store.view(k) for k in selection["split"]["train"]]
    audit["idsse_training_feature_mean"]=np.concatenate([v.global_features for v in references]).mean(0).tolist()
    audit["idsse_training_ball_missing_fraction"]=float(np.mean([np.mean(v.global_features[:,3]<.5) for v in references]))
    atomic_json(output/"audit.json",audit)
    model=load_model(0,arm);p,q=predict(model,view)
    bonus=None;method=trial["method"]
    if method!="fixed":
        scores=direction_scores(view,p,q,arm);x,confirmation=context_features(view,p,scores)
        ix,d=generate(view,scores,0. if method=="local" else .2)
        confidence=scores[ix,d]
        if method=="confirmed":confidence=confirmation[ix,d]
        if method in ("platt","context","geometry"):
            width={"platt":1,"context":13,"geometry":18}[method]
            confidence=apply_logistic(choice["calibrators"][method],x[ix,d,:width])
        bonus=candidate_bonuses(view,ix,d,confidence,trial["threshold"],choice["prevalence"],trial["strength"])
    with np.load(DEST/"training/fold-0/priors.npz") as z:priors=dict(z)
    decoder=DirectionalDecoder(view,priors,trial["family"],.5,bonus)
    prediction=np.full(source_frames(),3,np.int8)
    for run in np.unique(view.run):
        ix=np.flatnonzero(view.run==run);native=np.flatnonzero(match.period==run+1)
        hard=decoder.decode(p[ix],view.time[ix])
        prediction[match.frame[native]]=hard[nearest_indices(view.time[ix],match.time[native])]
    with (output/"preliminary.csv").open("w",newline="") as stream:
        writer=csv.writer(stream);writer.writerow(["frame","label"])
        writer.writerows((i,"HADX"[code]) for i,code in enumerate(prediction))
    # Diagnostic only. This is the ONLY target GT path ever opened.
    public_path=ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv"
    truth=np.full(source_frames(),3,np.int8)
    with public_path.open() as stream:
        for row in csv.DictReader(stream):
            truth[int(row["frame"])]=dict(H=0,A=1,D=2,X=3)[row["label"]]
    result=official_scorer().evaluate(truth,prediction,bootstrap_n=0)
    atomic_json(output/"public_diagnostic.json",{"metrics":result,"hyperparameter_selection":False,
        "public_GT_only":True,"warning":"five public minutes are a transfer diagnostic, not a tuning set"})
    atomic_json(output/"status.json",{"status":"COMPLETE","private_GT_accessed":False,"reports_restricted_local":True})
    return 0
