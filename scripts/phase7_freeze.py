"""Freeze per-fold architecture decisions; never pool a different fold's labels.

Pilot inner-validation games overlap other pilots' outer tests. A pooled
recommendation is exploratory only and CANNOT define a nested-CV winner.
This addendum makes independent decisions using each fold's own validation.
"""
import json
from possession.phase7.public import DEST,protocol
from possession.night.common import atomic_json


def main():
    spec=protocol();path=DEST/"independent_architecture_selection.json"
    if path.exists():return 0
    assert not (DEST/"all_rows.json").exists() and not (DEST/"transfer.json").exists()
    decisions={}
    for i in spec["pilot_folds"]:
        own=json.loads((DEST/f"selections/fold-{i}.json").read_text())
        a,b=[own["arms"][k]["winner"] for k in ("harmonized","set_transformer")]
        gain=b["objective"]-a["objective"]
        occlusion={n:b["metrics"][n]["spellF1"]-a["metrics"][n]["spellF1"] for n in spec["scenarios"][1:]}
        long=b["metrics"]["clean"]["long_recall"]-a["metrics"]["clean"]["long_recall"]
        promote=gain>=.01 and min(occlusion.values())>=-.02 and long>=-.03
        decisions[str(i)]={"selected":"set_transformer" if promote else "harmonized","robust_gain":gain,
            "occlusion_gain":occlusion,"long_recall_gain":long,"split":own["split"],"used_test":False}
    atomic_json(path,{"folds":decisions,"rule":"same gain/robustness thresholds applied independently to each own validation; never pool labels across outer folds",
        "pooled_architecture_selection_json":"exploratory recommendation only; never used to select a fold's evaluated system",
        "reason":"other folds' inner validation can overlap this fold's test; independent decisions preserve exclusion",
        "new_outer_predictions_created":False})
    return 0
