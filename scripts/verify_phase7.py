"""Verify public-only provenance, same budget/RNG, predictions and sealed decisions."""
import json
import numpy as np
import torch
from possession.phase7.public import DEST,protocol,Store
from possession.phase7.experiment import register
from possession.phase7.model import SpatialE0
from possession.learning.metrics import SCORER_PATH
from possession.external_data.schema import file_hash
from possession.night.common import atomic_json


def main():
    spec=register();store=Store();checks={}
    frozen=json.loads((DEST/"registration.json").read_text())
    checks["official_scorer_unchanged"]=file_hash(SCORER_PATH)=="341bad48904b489d6e875397b27687f5d6a4d22a397f3dc8af3a00d48f16874d"
    checks["old_dependencies_unchanged"]=all(file_hash(p)==v for p,v in frozen["source"].items())
    independent=DEST/"independent_architecture_selection.json"
    checks["decisions_frozen_before_new_test"]=independent.exists() and independent.stat().st_mtime<min((DEST/"rows").glob("*.json"),key=lambda p:p.stat().st_mtime).stat().st_mtime
    choices=json.loads(independent.read_text())["folds"]
    checks["SK_split_disjoint"]=not(set(spec["skillcorner_development_keys"])&set(spec["skillcorner_confirmation_keys"]))
    for i in spec["pilot_folds"]:
        selection=json.loads((DEST/f"selections/fold-{i}.json").read_text());split=selection["split"]
        checks[f"fold{i}_disjoint"]=len(set(split["train"]+[split[k] for k in ("calibration","validation","test")]))==7
        checks[f"fold{i}_own_choice"]=choices[str(i)]["split"]==split and not choices[str(i)]["used_test"]
        states={}
        for stage,n in (("base",12),("binary",8)):
            for arm in ("harmonized","set_transformer"):
                path=DEST/f"training/fold-{i}/{arm}/{stage}/last.pt"
                state=torch.load(path,map_location="cpu",weights_only=False);states[(arm,stage)]=state
                checks[f"fold{i}_{arm}_{stage}_budget"]=len(state["history"])==n and state["split"]==split
                checks[f"fold{i}_{arm}_{stage}_finite"]=all(torch.isfinite(t).all() for t in state["model"].values())
            a,b=[states[(arm,stage)] for arm in ("harmonized","set_transformer")]
            checks[f"fold{i}_{stage}_same_rng"]=torch.equal(a["initial_rng"],b["initial_rng"]) and torch.equal(a["torch_rng"],b["torch_rng"]) and torch.equal(a["loader_rng"],b["loader_rng"])
            def steps(s):return [r["train"].get("updates",r["train"].get("optimizer_steps")) for r in s["history"]]
            checks[f"fold{i}_{stage}_same_updates"]=steps(a)==steps(b)
        for arm in ("harmonized","set_transformer"):
            for part in ("calibration","validation","test"):
                for case in spec["scenarios"]:
                    with np.load(DEST/f"predictions/fold-{i}/{arm}/{part}/{case}.npz") as z:
                        p=z["probabilities"];checks[f"{i}_{arm}_{part}_{case}_probabilities"]=bool(np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=2e-6))
                    key=split[part];view=store.harmonized(key)
                    checks[f"{key}_canonical_features"]=bool((view.nodes[...,4:6]==0).all() and (view.global_features[:,[2,4,5]]==0).all())
    rows=json.loads((DEST/"all_rows.json").read_text());checks["54_test_rows"]=len(rows)==54
    transfer=json.loads((DEST/"transfer.json").read_text());checks["288_external_transfer_rows"]=len(transfer["rows"])==288
    checks["only_confirmation_SK"]=set(r["match"] for r in transfer["rows"])==set(spec["skillcorner_confirmation_keys"])
    checks["no_SK_D_or_spellF1"]=all("spellF1" not in r and r["conditional_team"]["label_coverage"]<1 for r in transfer["rows"])
    atomic_json(DEST/"verification.json",{"checks":{k:bool(v) for k,v in checks.items()},"passed":all(checks.values()),"scope":"public external data only"})
    return 0 if all(checks.values()) else 1
