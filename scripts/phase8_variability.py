"""Decide whether an additional seed is warranted using own validations only."""
import json
import numpy as np
from possession.phase8.common import DEST,register
from possession.phase8.evaluation import selection_path
from possession.night.common import atomic_json


def main():
    spec=register();path=DEST/"seed_variability.json"
    if path.exists():return 0
    assert not (DEST/"all_rows.json").exists()
    folds=[]
    for i in spec["folds"]:
        gains=[]
        for seed in spec["seeds"]:
            a=json.loads(selection_path(i,seed,"harmonized").read_text())["winner"]["objective"]
            b=json.loads(selection_path(i,seed,"set_transformer").read_text())["winner"]["objective"]
            gains.append(b-a)
        folds.append({"fold":i,"paired_validation_gains":gains,"mean":float(np.mean(gains)),"seed_sd":float(np.std(gains))})
    gains=[g for r in folds for g in r["paired_validation_gains"]];mean=float(np.mean(gains));noise=float(np.mean([r["seed_sd"] for r in folds]))
    required=sum(g>0 for g in gains)<10 or abs(mean)<noise
    atomic_json(path,{"folds":folds,"mean_paired_validation_gain":mean,"mean_within_fold_seed_sd":noise,
        "positive_pairs":sum(g>0 for g in gains),"pairs":len(gains),"third_seed_justified":bool(required),"test_used":False,
        "rule":"additional seed warranted if fewer than10 of14 own-validation paired gains positive, or mean gain smaller than mean within-fold seed SD; no external scores consulted"})
    return 0
