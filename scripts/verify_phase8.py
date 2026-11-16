"""Read-only verification of public experiments; never discover target assets."""
import json
from pathlib import Path
import numpy as np
import torch
from possession.external_data.schema import file_hash
from possession.learning.metrics import SCORER_PATH
from possession.night.common import atomic_json
from possession.phase8.common import DEST,OLD,ROOT,Store,register
from possession.phase8.training import folder,source_folder
from possession.phase8.evaluation import seal,selection_path
from possession.phase8.packing import compact_collate
from possession.learning.data import Windows,collate
from possession.learning.training import forward
from possession.phase8.training import load_model


def read(path):return json.loads(path.read_text())


def main():
    spec=register();seal();torch.set_num_threads(4);store=Store();checks={}
    checks["official_scorer_unchanged"]=file_hash(SCORER_PATH)=="341bad48904b489d6e875397b27687f5d6a4d22a397f3dc8af3a00d48f16874d"
    checks["public_artifact_roots"]=all(Path(r["path"]).resolve().is_relative_to((ROOT/"data/external").resolve()) for r in store.records.values())
    checks["SK_training_confirmation_disjoint"]=not(set(spec["skillcorner_training_keys"])&set(spec["skillcorner_confirmation_keys"]))
    checks["SK_four_training_sixteen_confirmation"]=len(spec["skillcorner_training_keys"])==4 and len(spec["skillcorner_confirmation_keys"])==16
    packing_errors={}
    for key in (sorted(k for k in store.records if k.startswith("idsse:"))[0],spec["skillcorner_training_keys"][0]):
        item=Windows({key:store.harmonized(key)},256,15)[0]
        model=load_model(0,2027,"set_transformer").eval()
        with torch.no_grad():
            original=forward(model,collate([item]))["probabilities"]
            compact=forward(model,compact_collate([item]))["probabilities"]
        error=float((original-compact).abs().max());packing_errors[key]=error
        checks[key+"_real_packing_equivalence"]=error<2e-6
    for key,record in store.records.items():
        cache=(OLD/"feature_cache"/key.replace(":","_"))
        if not (cache/"complete.json").exists():cache=DEST/"feature_cache"/key.replace(":","_")
        checks[key+"_cache_provenance"]=read(cache/"complete.json")["artifact"]==record["sha256"]
        view=store.harmonized(key)
        checks[key+"_reserved_metadata_zero"]=bool((view.nodes[...,4:6]==0).all() and (view.global_features[:,[2,4,5]]==0).all())
        if key.startswith("skillcorner:"):
            checks[key+"_no_state_labels"]=bool((view.alive==-1).all() and (view.confidence[:,0]==0).all())
        store.common.pop(key,None)
    scope=read(DEST/"mixed_scope.json");best=read(DEST/"best_set_transformer.json")
    checks["scope_and_choices_no_test"]=not scope["outer_test_used"] and not best["test_used"]
    latest_selector=max(p.stat().st_mtime for p in (DEST/"selections").rglob("*.json"))
    earliest_new_test=min(p.stat().st_mtime for p in (DEST/"predictions").glob("**/test/*.npz"))
    checks["all_selectors_before_new_test"]=latest_selector<earliest_new_test
    checks["best_before_new_test"]=(DEST/"best_set_transformer.json").stat().st_mtime<earliest_new_test
    logical=0;reused=0
    for i in spec["folds"]:
        for seed in spec["seeds"]:
            states={}
            descriptors=[(a,"IDSSE") for a in spec["arms"]]
            if i in scope["folds"]:descriptors.append(("set_transformer","IDSSE_SK"))
            for arm,condition in descriptors:
                logical+=1;selection=read(selection_path(i,seed,arm,condition));parts=selection["split"]
                tag=f"{i}/{seed}/{arm}/{condition}"
                checks[tag+"_split"]=len(parts["train"])==4 and len(set(parts["train"]+[parts[k] for k in ("calibration","validation","test")]))==7
                checks[tag+"_own_selection"]=not selection["outer_test_used"] and selection["fold"]==i and selection["seed"]==seed
                src=source_folder(i,seed,arm,condition)
                marker=folder(i,seed,arm,condition)/"reused.json"
                if marker.exists():
                    reused+=1;ref=read(marker)
                    checks[tag+"_reused_hash"]=all(file_hash(Path(ref["source"])/stage/"last.pt")==v for stage,v in ref["sha256"].items())
                for stage,n in (("base",12),("binary",8)) if condition=="IDSSE" else (("binary",8),):
                    s=torch.load(src/stage/"last.pt",map_location="cpu",weights_only=False);states[(arm,condition,stage)]=s
                    checks[tag+"/"+stage+"_budget"]=len(s["history"])==n and s["split"]==parts
                    checks[tag+"/"+stage+"_finite"]=all(bool(torch.isfinite(v).all()) for v in s["model"].values())
                for part in ("calibration","validation","test"):
                    for case in spec["scenarios"]:
                        f=DEST/f"predictions/{condition}/fold-{i}/seed-{seed}/{arm}/{part}/{case}.npz"
                        if condition=="IDSSE" and seed==2027 and i in (0,6):f=OLD/f"predictions/fold-{i}/{arm}/{part}/{case}.npz"
                        with np.load(f,allow_pickle=False) as z:
                            p,q=z["probabilities"],z["transition"]
                            good=np.isfinite(p).all() and np.isfinite(q).all() and np.allclose(p.sum(1),1,atol=2e-6) and p.min()>=0 and q.min()>=0 and q.max()<=1
                            if "time" in z:
                                good=good and all((np.diff(z["time"][z["run"]==r])>0).all() for r in np.unique(z["run"]))
                        checks[tag+"/"+part+"/"+case+"_probabilities"]=bool(good)
            for stage in ("base","binary"):
                a,b=[states[(arm,"IDSSE",stage)] for arm in spec["arms"]]
                checks[f"{i}/{seed}/{stage}_paired_RNG"]=all(torch.equal(a[k],b[k]) for k in ("initial_rng","torch_rng","loader_rng"))
                def updates(s):return [r["train"].get("updates",r["train"].get("optimizer_steps")) for r in s["history"]]
                checks[f"{i}/{seed}/{stage}_paired_updates"]=updates(a)==updates(b)
            if i in scope["folds"]:
                a,b=[states[("set_transformer",c,"binary")] for c in ("IDSSE","IDSSE_SK")]
                checks[f"{i}/{seed}_mixed_same_ID_RNG"]=all(torch.equal(a[k],b[k]) for k in ("initial_rng","torch_rng","loader_rng"))
                checks[f"{i}/{seed}_mixed_same_updates"]=updates(a)==updates(b)
                checks[f"{i}/{seed}_extra_SK_windows"]=all(r["train"]["skillcorner_windows"]==4*r["train"]["updates"] for r in b["history"])
    checks["32_logical_models_4_reused"]=logical==28+2*len(scope["folds"]) and reused==4
    rows=read(DEST/"all_rows.json")
    checks["expected_test_rows"]=len(rows)==252+18*len(scope["folds"])
    checks["finite_metrics"]=all(np.isfinite(r["spellF1"]) and 0<=r["spellF1"]<=1 and 0<=r["macroF1"]<=1 for r in rows)
    transfer=read(DEST/"transfer.json");reg=read(DEST/"transfer_registration.json")
    checks["transfer_source_unchanged"]=reg["source_sha256"]==file_hash(ROOT/"possession/phase8/transfer.py")
    checks["576_SK_rows"]=len(transfer["rows"])==576
    checks["only_confirmation_SK"]=set(r["match"] for r in transfer["rows"])==set(spec["skillcorner_confirmation_keys"])
    checks["no_SK_spellF1"]=all("spellF1" not in r for r in transfer["rows"])
    diagnosis=read(DEST/"diagnosis.json")
    checks["21_diagnoses"]=len(diagnosis["rows"])==21
    checks["diagnosis_source_unchanged"]=diagnosis["registration"]["source_sha256"]==file_hash(ROOT/"possession/phase8/diagnosis.py")
    probe=DEST.parent/"phase8-interval-v1"
    if (probe/"all_rows.json").exists():
        reg=read(probe/"registration.json")
        checks["probe_config_unchanged"]=read(ROOT/"configs/interval_probe_protocol.json")==reg["config"]
        checks["probe_sources_unchanged"]=all(file_hash(ROOT/p)==h for p,h in reg["source"].items())
        checks["probe_teachers_unchanged"]=all(file_hash(reg["teachers"][i]["checkpoint"])==h for i,h in reg["teacher_hashes"].items())
        checks["48_probe_rows"]=len(read(probe/"all_rows.json"))==48
        last=max(p.stat().st_mtime for p in (probe/"selections").glob("*.json"))
        first=min(p.stat().st_mtime for p in (probe/"rows").glob("*.json"))
        checks["probe_choices_before_test_rows"]=last<first
        for i,c in reg["teachers"].items():
            test_exports=probe/f"embeddings/fold-{i}"/c["split"]["test"].replace(":","_")
            checks[f"probe{i}_no_early_test_export"]=last<min(p.stat().st_mtime for p in test_exports.glob("*.npz"))
    atomic_json(DEST/"verification.json",{"passed":all(checks.values()),"checks":{k:bool(v) for k,v in checks.items()},"real_packing_max_errors":packing_errors,"scope":"explicit public external artifacts only; no official assets accessed"})
    return 0 if all(checks.values()) else 1
