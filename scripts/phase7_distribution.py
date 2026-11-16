"""Supplemental common-10Hz observability audit, exclusively public artifacts."""
import time
import numpy as np
from possession.phase7.public import Store,DEST,protocol
from possession.night.common import atomic_json


def missing_intervals(missing,run):
    lengths=[]
    for r in np.unique(run):
        values=missing[run==r]
        changes=np.diff(np.r_[False,values,False].astype(int))
        lengths.extend(((np.flatnonzero(changes==-1)-np.flatnonzero(changes==1))/10).tolist())
    return {"count":len(lengths),"duration_quantiles_s":np.quantile(lengths,[0,.5,.9,.99,1]).tolist() if lengths else None,
            "total_seconds":float(sum(lengths))}


def main():
    protocol();path=DEST/"common_grid_audit.json"
    if path.exists():return 0
    store=Store();rows=[];start=time.perf_counter()
    for key in sorted(store.records):
        match=store.match(key);view=store.view(key);same=np.diff(view.run)==0
        starts=view.node_mask.copy();starts[1:] &= ~(view.node_mask[:-1] & same[:,None,None])
        present=view.node_mask;known=view.nodes[...,5]>.5;detected=view.nodes[...,4]>.5
        g=view.global_features;available=g[:,3]>.5;ballknown=g[:,5]>.5;balldetected=g[:,4]>.5
        # Source roster identities are used ONLY for this quality diagnostic.
        adjacent=present[1:] & present[:-1] & same[:,None,None]
        pitch=np.asarray(match.metadata["pitch_m"])/2
        displacement=np.linalg.norm(np.diff(view.nodes[...,:2]*pitch,axis=0),axis=-1)
        valid_speed=displacement[adjacent]/.1
        rows.append({"key":key,"parser_issues":match.metadata.get("issues",{}),
            "max_nearest_native_time_error_s":float(np.abs(match.time[view.native_index]-view.time).max()),
            "native_indices_strictly_increasing":bool((np.diff(view.native_index)>0).all()),
            "max_grid_dt_error_s":float(np.abs(np.diff(view.time)[same]-.1).max()),
            "ball_missing":missing_intervals(~available,view.run),"all_players_missing":missing_intervals(~present.any((1,2)),view.run),
            "player_tracklet_starts_at_10Hz":int(starts.sum()),
            "identity_switch_GT_available":False,"identity_diagnostic":"availability tracklets, not verified identity switches",
            "player_available_extrapolated_fraction":float((present & known & ~detected).sum()/max(1,present.sum())),
            "ball_available_extrapolated_fraction":float((available & ballknown & ~balldetected).sum()/max(1,available.sum())),
            "adjacent_available_speed_p99_m_s":float(np.quantile(valid_speed,.99)),
            "adjacent_available_speed_over_15m_s":int((valid_speed>15).sum()),
            "null_team_labels":int((view.team<0).sum()),"null_state_labels":int((view.alive<0).sum()),
            "null_state_not_inferred_as_dead":bool(not key.startswith("skillcorner:") or (view.alive==-1).all()),
            "missing_XY_values_zero_with_explicit_mask":bool((g[~available,:2]==0).all() and (view.nodes[~present]==0).all())})
    atomic_json(path,{"rows":rows,"seconds":time.perf_counter()-start,
        "scope":"quality diagnostics only; no fitting, no official assets, no private paths"})
    return 0
