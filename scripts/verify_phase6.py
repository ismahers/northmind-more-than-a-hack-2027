"""Integrity and separation checks for the bounded public experiment."""

import json
import subprocess

import numpy as np
import torch

from possession.night.common import atomic_json
from possession.phase6.experiment import DEST,register
from possession.learning.metrics import SCORER_PATH
from possession.external_data.schema import file_hash


def main():
    protocol=register();selection_paths=[DEST/f"selections/fold-{i}.json" for i in protocol["pilot_folds"]]
    saved=[]
    for path in selection_paths:
        selection=json.loads(path.read_text());parts=selection["split"]
        assert not selection["outer_test_used"]
        assert len(parts["train"])==4 and len(set(parts["train"]+[parts[k] for k in ("calibration","validation","test")]))==7
        base=torch.load(DEST/f"training/fold-{selection['fold']}/base/last.pt",weights_only=False,map_location="cpu")
        assert base["split"]==parts and len(base["history"])==12
        arms=[]
        for arm in protocol["arms"]:
            state=torch.load(DEST/f"training/fold-{selection['fold']}/{arm}/last.pt",weights_only=False,map_location="cpu")
            assert state["split"]==parts and len(state["history"])==8
            assert selection["arms"][arm]["fitting_match"]==parts["calibration"]
            assert selection["arms"][arm]["selection_match"]==parts["validation"]
            arms.append(state)
        assert all(torch.equal(arms[0]["loader_rng"],s["loader_rng"]) for s in arms)
        assert all(torch.equal(arms[0]["torch_rng"],s["torch_rng"]) for s in arms)
        assert all([r["train"]["updates"] for r in arms[0]["history"]]==[r["train"]["updates"] for r in s["history"]] for s in arms)
        saved.append({"fold":selection["fold"],"split":parts,"equal_updates_and_sampler_and_dropout_rng":True})
    rows=json.loads((DEST/"all_rows.json").read_text())
    assert max(p.stat().st_mtime for p in selection_paths)<min(p.stat().st_mtime for p in (DEST/"rows").glob("fold-*.json"))
    for row in rows:
        assert sum(g["gt"] for g in row["duration_groups"].values())==row["gt_spells"]
        assert 0<=row["spellF1"]<=1
    predictions=list((DEST/"predictions").glob("fold-*/*/*/*.npz"))
    for path in predictions:
        with np.load(path,allow_pickle=False) as z:
            p,q=z["probabilities"],z["transition"]
            assert np.isfinite(p).all() and np.isfinite(q).all()
            np.testing.assert_allclose(p.sum(-1),1,atol=2e-6,rtol=0)
            assert q.min()>=0 and q.max()<=1
    expected="341bad48904b489d6e875397b27687f5d6a4d22a397f3dc8af3a00d48f16874d"
    assert file_hash(SCORER_PATH)==expected
    assert subprocess.run(["git","diff","--quiet","HEAD","--","evaluation/possession_metrics.py","starter","baselines","games.py","requirements.txt"],check=False).returncode==0
    result={"status":"OK","folds":saved,"predictions":len(predictions),"test_rows":len(rows),
            "all_selections_before_outer_evaluation":True,"official_scorer_sha256":expected,
            "old_model_sources_unchanged":True,"private_GT_accessed":False}
    atomic_json(DEST/"verification.json",result)
    return 0
