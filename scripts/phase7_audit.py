"""Aggregate audit of ONLY public external artifacts, and frozen input interventions."""
import json
import time
import numpy as np
import torch
from possession.external_data.audit import segments
from possession.external_data.schema import file_hash
from possession.phase7.public import Store,DEST,protocol
from possession.phase7.features import intervene
from possession.phase6.experiment import load_model,predict
from scripts.phase6_transfer import conditional_metrics
from possession.night.common import atomic_json


def quantiles(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return np.quantile(x,[0,.01,.5,.99,1]).tolist() if len(x) else None


def match_audit(match,view):
    dt=np.diff(match.time); same=np.diff(match.period)==0
    xy=view.nodes[...,:2][view.node_mask]
    masked=view.nodes[view.node_mask]
    counts=view.node_mask.sum((1,2))
    adjacent=match.player_available[1:] & match.player_available[:-1] & same[:,None,None]
    displacement=np.linalg.norm(np.diff(match.players,axis=0),axis=-1)
    speeds=displacement[adjacent]/np.broadcast_to(dt[:,None,None],adjacent.shape)[adjacent]
    starts=match.player_available.copy()
    starts[1:] &= ~(match.player_available[:-1] & same[:,None,None])
    return {"key":match.key,"pitch_m":match.metadata["pitch_m"],"native_fps":match.metadata["native_fps"],
        "frames_native":len(match.time),"frames_model":len(view.time),"runs":len(segments(match)),
        "native_dt_quantiles_s":quantiles(dt[same]),"coordinate_scale":"centered metres / half pitch dimensions",
        "player_xy_normalized_quantiles":[quantiles(xy[:,j]) for j in range(2)],
        "player_outside_pitch_fraction":float((np.abs(xy)>1).any(1).mean()),
        "nodes_quantiles":[quantiles(masked[:,j]) for j in range(6)],
        "global_quantiles":[quantiles(view.global_features[:,j]) for j in range(7)],
        "player_count_quantiles":quantiles(counts),"player_mask_fraction":float(view.node_mask.mean()),
        "player_detection_known_fraction":float(masked[:,5].mean()),"player_detected_fraction":float(masked[:,4].mean()),
        "ball_available_fraction":float(match.ball_available.mean()),
        "ball_detection_known_fraction":float(match.ball_detection_known.mean()),"ball_detected_fraction":float(match.ball_detected.mean()),
        "ball_height_known_fraction":float(np.isfinite(match.ball[:,2]).mean()),"height_quantiles_m":quantiles(match.ball[:,2]),
        "conditional_label_coverage":float((view.team>=0).mean()),"state_label_coverage":float((view.alive>=0).mean()),
        "roster_slots":int(np.prod(match.player_ids.shape)),"available_tracklet_starts":int(starts.sum()),
        "adjacent_identity_speed_quantiles_m_s":quantiles(speeds),"adjacent_speed_over_15m_s_fraction":float((speeds>15).mean()),
        "velocities_used_by_model":False,"persistent_ids_used_by_model":False}


def main():
    spec=protocol();torch.set_num_threads(spec["cpu_threads"]);store=Store();start=time.perf_counter()
    dest=DEST/"audit.json"
    if dest.exists():return 0
    audit=[]
    for key in sorted(store.records):
        m=store.match(key);v=store.view(key);audit.append(match_audit(m,v))
        atomic_json(DEST/"progress.json",{"stage":"public_feature_audit","processed":len(audit),"total":len(store.records)})
    models={i:load_model(i,"directional") for i in spec["pilot_folds"]}
    norms={str(i):{"node_metadata_column_norms":model.encoder[0].weight.detach()[:,4:6].norm(dim=0).tolist(),
                   "global_metadata_input_column_norms":model.input.weight.detach()[:,-3:-1,0].norm(dim=0).tolist()}
           for i,model in models.items()}
    conditions=["original","player_flags","ball_flags","known_only","detected_only","age","all_flags","harmonized"]
    rows=[]
    for key in spec["skillcorner_development_keys"]:
        view=store.view(key)
        for condition in conditions:
            v=intervene(view,condition)
            for i,model in models.items():
                p,_=predict(model,v)
                x=torch.from_numpy(view.nodes[:1000]);x0=x.clone();x0[...,4:6]=0
                with torch.no_grad():
                    a=model.encoder[0](x);b=model.encoder[0](x0)
                rows.append({"match":key,"fold":i,"intervention":condition,"conditional_team":conditional_metrics(v,p),
                    "first_layer_metadata_abs_shift":float((a-b).abs().mean()),
                    "first_layer_relu_sign_change_fraction":float(((a>0)!=(b>0)).float().mean())})
    atomic_json(dest,{"protocol":spec,"matches":audit,"causal_interventions":rows,"frozen_weight_norms":norms,
        "seconds":time.perf_counter()-start,"code_sha256":file_hash(__file__),
        "scope":"public IDSSE and SkillCorner only; no hackathon assets, events, labels or discovery",
        "causal_limits":"interventions hold weights/geometry/labels fixed; four prespecified development games; architecture confirmation games not scored"})
    return 0
