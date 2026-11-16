"""Verify parameter-count control against the frozen harmonized training budget."""
import json
import torch
import numpy as np
from possession.phase7.public import DEST as MAIN,protocol
from possession.phase7_capacity.experiment import DEST
from possession.external_data.schema import file_hash
from possession.night.common import atomic_json


def main():
    spec=protocol();checks={}
    frozen=json.loads((DEST/"registration.json").read_text())
    checks["sources_unchanged"]=all(file_hash(p)==h for p,h in frozen["sources"].items())
    for i in spec["pilot_folds"]:
        for stage,n in (("base",12),("binary",8)):
            a=torch.load(DEST/f"training/fold-{i}/{stage}/last.pt",map_location="cpu",weights_only=False)
            b=torch.load(MAIN/f"training/fold-{i}/harmonized/{stage}/last.pt",map_location="cpu",weights_only=False)
            checks[f"{i}_{stage}_same_data_budget"]=a["split"]==b["split"] and len(a["history"])==n
            checks[f"{i}_{stage}_same_RNG"]=all(torch.equal(a[k],b[k]) for k in ("initial_rng","torch_rng","loader_rng"))
            def steps(state):return [r["train"].get("updates",r["train"].get("optimizer_steps")) for r in state["history"]]
            checks[f"{i}_{stage}_same_updates"]=steps(a)==steps(b)
        selections=[DEST/f"selections/fold-{j}.json" for j in spec["pilot_folds"]]
        checks[f"{i}_all_selections_before_test"]=max(p.stat().st_mtime for p in selections)<min((DEST/f"predictions/fold-{i}/test").glob("*.npz"),key=lambda p:p.stat().st_mtime).stat().st_mtime
        for path in (DEST/f"predictions/fold-{i}").glob("*/*.npz"):
            with np.load(path) as z:
                checks[f"{i}_{path.parent.name}_{path.stem}_finite"]=bool(np.isfinite(z["probabilities"]).all() and np.allclose(z["probabilities"].sum(1),1,atol=2e-6))
    checks["six_test_rows"]=len(json.loads((DEST/"all_rows.json").read_text()))==6
    checks["96_confirmation_rows"]=len(json.loads((DEST/"transfer.json").read_text())["rows"])==96
    atomic_json(DEST/"verification.json",{"passed":all(checks.values()),"checks":checks})
    return 0 if all(checks.values()) else 1
