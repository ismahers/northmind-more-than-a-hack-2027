"""Paired analysis of the originally frozen error cohorts; no new training."""
import time
import numpy as np
import torch
from possession.external_data.schema import file_hash
from possession.interval_probe.data import load
from possession.learning.study import ScoreContext
from possession.learning.temporal import fit_priors
from possession.night.common import atomic_json
from possession.temporal_probe.common import Store,read_json
from possession.temporal_probe.windows import emissions
from possession.duration_balanced.training import paths
from possession.ha_discrimination.common import ROOT,DEST,register,verify_ancestry
from possession.ha_discrimination.features import geometry
from possession.ha_discrimination.diagnosis import analyze
from possession.ha_discrimination.training import make_model,predict_geometry


def main():
    torch.use_deterministic_algorithms(True);record=register();torch.set_num_threads(4)
    if not read_json(DEST/"pilot/verification.json")["passed"]:raise ValueError("Verified fixed pilot required")
    start=time.perf_counter();store=Store(record["records"]);rows=[]
    for fold in record["protocol"]["pilots"]:
        teacher=record["teachers"][str(fold)];recipe=record["recipe"]
        prior=fit_priors([store.harmonized(k) for k in teacher["split"]["train"]])
        selected=read_json(DEST/f"pilot/selections/fold-{fold}.json")
        for part in ("validation","test"):
            key=teacher["split"][part];view=store.harmonized(key);match=store.match(key)
            context=ScoreContext(match,view)
            data=load(paths(teacher,key,recipe)["clean"])
            data["geometry"],physical=geometry(view,match.metadata["pitch_m"])
            original=read_json(DEST/f"cases/{part}/fold-{fold}.json")["per_possession"]
            for arm in ("geometry","capacity"):
                path=DEST/f"pilot/training/fold-{fold}/{arm}/last.pt"
                if file_hash(path)!=selected["checkpoint_hashes"][arm]:raise ValueError("Pilot checkpoint changed")
                model=make_model(fold,teacher,arm,recipe)
                model.load_state_dict(torch.load(path,map_location="cpu",weights_only=False)["model"])
                logit=predict_geometry(model,data,recipe,arm);p=emissions(logit,data["probabilities"])
                with np.load(DEST/f"pilot/predictions/fold-{fold}/{arm}/{part}/clean.npz",allow_pickle=False) as z:
                    if not np.array_equal(p,z["probabilities"]):raise ValueError("Pilot predictions changed")
                new=analyze(context,p,logit,physical,prior,record["protocol"])["per_possession"]
                if len(new)!=len(original) or any(a["duration_s"]!=b["duration_s"] or a["boundary_type"]!=b["boundary_type"] for a,b in zip(original,new)):
                    raise ValueError("Cohort possession alignment mismatch")
                cohorts={}
                for name,predicate in (("all",lambda r:True),("physical_conflict",lambda r:r["physical_conflict"]),
                                       ("strong_missed",lambda r:r["A"]),("insufficient_missed",lambda r:r["B"]),
                                       ("no_correct_frame",lambda r:r["B_no_frame"]),("decoder_loss",lambda r:r["C"])):
                    pairs=[(a,b) for a,b in zip(original,new) if predicate(a)]
                    cohorts[name]={"n":len(pairs),"original_recovered":sum(a["hmm_recovered"] for a,b in pairs),
                                   "new_recovered":sum(b["hmm_recovered"] for a,b in pairs),
                                   "gains":sum(b["hmm_recovered"] and not a["hmm_recovered"] for a,b in pairs),
                                   "losses":sum(a["hmm_recovered"] and not b["hmm_recovered"] for a,b in pairs),
                                   "p_correct_change_mean":float(np.mean([b["p_correct"]-a["p_correct"] for a,b in pairs])) if pairs else None,
                                   "new_any_correct_frame":sum(b["correct_fraction"]>0 for a,b in pairs),
                                   "new_majority_correct":sum(b["correct_fraction"]>=.5 for a,b in pairs)}
                rows.append({"fold":fold,"part":part,"arm":arm,"cohorts_frozen_from_original":True,"cohorts":cohorts})
    checks=verify_ancestry(record)
    pilot=read_json(DEST/"pilot/registration.json")
    checks.update({"pilot_source:"+path:file_hash(ROOT/path)==sha for path,sha in pilot["pilot_sources"].items()})
    atomic_json(DEST/"pilot/mechanism.json",{"rows":rows,"seconds":time.perf_counter()-start,
                "passed":all(checks.values()),"check_count":len(checks),"training":0,"selection_changed":False,
                "script_sha256":file_hash(ROOT/"scripts/ha_pilot_mechanism.py")})
    return 0 if all(checks.values()) else 1
