"""Verify public experiment provenance, metrics and exact checkpoint continuation."""

import json
import subprocess
import platform
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from possession.external_data.schema import file_hash
from possession.learning.corruption import hide_ball, stable_seed
from possession.night.common import atomic_json
from possession.night.common import OLD
from possession.night.training import equal_state
from possession.phase5.experiment import (DEST, ROOT, register, source, get_store, make_model, epoch_pass, predict)
from possession.phase5.model import BoundaryWindows, boundary_collate, boundary_targets
from possession.learning.metrics import SCORER_PATH
from scripts.train_e0 import manifest


def verify_resume(store,data,protocol):
    # Predeclared pilot: reproduce epoch 8 of fold 0 focal from its epoch 7 save.
    i,kind,start,epoch=0,"focal",7,8
    folder=DEST/f"training/fold-{i}/{kind}"
    if not (folder/f"epoch-{epoch:02d}.pt").exists():
        return {"performed":False,"reason":"predeclared representative trajectory stopped before epoch 8"}
    state=torch.load(folder/f"epoch-{start:02d}.pt",map_location="cpu",weights_only=False)
    expected=torch.load(folder/f"epoch-{epoch:02d}.pt",map_location="cpu",weights_only=False)
    summary=json.loads((folder/"training.json").read_text())
    views={k:store.view(k) for k in state["train_keys"]}
    targets={k:boundary_targets(store.match(k),v,protocol["boundary_tolerance_s"]) for k,v in views.items()}
    val_key=state["validation_key"]
    validation=store.view(val_key)
    val_targets={val_key:boundary_targets(store.match(val_key),validation,protocol["boundary_tolerance_s"])}
    val_loader=DataLoader(BoundaryWindows({val_key:validation},val_targets,256,15),batch_size=4,collate_fn=boundary_collate)
    length=protocol["training_block_seconds"][(epoch-1)%3]
    degraded={k:hide_ball(v,.4,length,stable_seed(k,protocol["seed"]+epoch*1000))[0] for k,v in views.items()}
    model=make_model(source(i),protocol["seed"])
    model.load_state_dict(state["model"])
    optimizer=torch.optim.AdamW(model.parameters(),lr=protocol["learning_rate"],weight_decay=.0001)
    optimizer.load_state_dict(state["optimizer"])
    generator=torch.Generator()
    generator.set_state(state["loader_rng"])
    torch.set_rng_state(state["torch_rng"])
    loader=DataLoader(BoundaryWindows(degraded,targets,256,15),batch_size=4,shuffle=True,generator=generator,collate_fn=boundary_collate)
    epoch_pass(model,loader,optimizer,summary["positive_weight"],kind,protocol)
    epoch_pass(model,val_loader,None,summary["positive_weight"],kind,protocol)
    checks={"model_bit_exact":equal_state(model.state_dict(),expected["model"]),
            "optimizer_bit_exact":equal_state(optimizer.state_dict(),expected["optimizer"]),
            "torch_rng_bit_exact":torch.equal(torch.get_rng_state(),expected["torch_rng"]),
            "loader_rng_bit_exact":torch.equal(generator.get_state(),expected["loader_rng"])}
    assert all(checks.values())
    return {"performed":True,"fold":i,"kind":kind,"from_epoch":start,"to_epoch":epoch,**checks}


def main():
    protocol=register()
    torch.set_num_threads(protocol["cpu_threads"])
    data,_=manifest("idsse-only")
    rows=json.loads((DEST/"all_rows.json").read_text())
    assert len(rows)==7*3*8
    assert len({(r["fold"],r["scenario"],r["system"]) for r in rows})==len(rows)
    for r in rows:
        assert sum(g["gt"] for g in r["duration_groups"].values())==r["gt_spells"]
        assert sum(g["pred"] for g in r["duration_groups"].values())==r["pred_spells"]
        for group in r["duration_groups"].values():
            for v in group["iou"].values():
                assert v["fn"]+v["matched_gt"]==group["gt"]
                assert v["fp"]+v["matched_pred"]==group["pred"]
        assert np.isfinite(r["spellF1"]) and 0<=r["spellF1"]<=1
    expected="341bad48904b489d6e875397b27687f5d6a4d22a397f3dc8af3a00d48f16874d"
    assert file_hash(SCORER_PATH)==expected
    unchanged=subprocess.run(["git","diff","--quiet","HEAD","--","evaluation/possession_metrics.py","baselines","starter_pack","games.py","requirements.txt"],check=False).returncode==0
    assert unchanged
    selections=[DEST/f"selections/fold-{i}.json" for i in range(7)]
    assert all(not json.loads(p.read_text())["outer_test_used"] for p in selections)
    assert max(p.stat().st_mtime for p in selections)<min(p.stat().st_mtime for p in (DEST/"metrics").glob("fold-*/*/*.json"))
    for i in range(7):
        fold=data["folds"][i]
        src=source(i)
        for r in rows:
            if r["fold"]==i and r["system"]=="A":
                old=json.loads((OLD/f"test/seed-2027/fold-{i}/{src['condition']}/{r['scenario']}/{src['specification']['kind']}.json").read_text())
                assert abs(r["spellF1"]-old["spell_f1_mean"])<1e-12
                assert abs(r["macroF1"]-old["macro_f1"])<1e-12
        for kind in protocol["losses"]:
            t=json.loads((DEST/f"training/fold-{i}/{kind}/training.json").read_text())
            assert not set(t["train_keys"])&set(fold["validation"]+fold["test"])
            assert len(t["train_keys"])==5
    store=get_store(data)
    resume=verify_resume(store,data,protocol)
    key=data["folds"][0]["validation"][0]
    view=store.view(key)
    for length in (5.,15.):
        hidden,_=hide_ball(view,.4,length,stable_seed(key,701))
        for name in ("alive","team","confidence","time","run","native_index"):
            np.testing.assert_array_equal(getattr(view,name),getattr(hidden,name))
    prediction_files=list((DEST/"predictions").glob("fold-*/*/*.npz"))
    for path in prediction_files:
        with np.load(path,allow_pickle=False) as z:
            p,q=z["probabilities"],z["boundary"]
            assert np.isfinite(p).all() and np.isfinite(q).all()
            np.testing.assert_allclose(p.sum(-1),1,atol=2e-6,rtol=0)
            assert q.min()>=0 and q.max()<=1
    # Inference must be exactly unchanged when all target arrays are replaced.
    from dataclasses import replace
    model=make_model(source(0),protocol["seed"])
    a=predict(model,view)
    fake=replace(view,alive=np.full_like(view.alive,-1),team=np.full_like(view.team,-1),confidence=np.zeros_like(view.confidence))
    b=predict(model,fake)
    assert all(np.array_equal(x,y) for x,y in zip(a,b))
    result={"status":"OK","rows":len(rows),"prediction_files":len(prediction_files),"official_scorer_sha256":expected,
            "official_files_unchanged":unchanged,"all_selections_before_first_test":True,"five_disjoint_training_matches":True,
            "missingness_preserves_GT":True,"inference_independent_of_target_arrays":True,"duration_counts_conserved":True,
            "A_reproduces_original_phase3_scores":True,
            "complete_checkpoint_resume":resume}
    dependencies=[]
    for directory in ("possession/learning","possession/external_data","possession/night"):
        dependencies.extend((ROOT/directory).glob("*.py"))
    dependencies.extend(ROOT/name for name in ("scripts/phase3.py","scripts/train_e0.py","scripts/external_data.py"))
    result["dependency_sha256"]={str(p.relative_to(ROOT)):file_hash(p) for p in sorted(dependencies)}
    result["environment"]={"python":sys.version,"torch":torch.__version__,"numpy":np.__version__,
                           "platform":platform.platform(),"cpu_threads":torch.get_num_threads()}
    atomic_json(DEST/"verification.json",result)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
