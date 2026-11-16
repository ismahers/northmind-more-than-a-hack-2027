"""Bounded public train/validation pilot; immutable recipe and portable replay."""
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
import zipfile
from dataclasses import fields

import numpy as np
import torch

from possession import st_tcn_target as transfer
from possession.evidence_first_validation import FoldStore, assert_metrics
from possession.external_data.features import Features
from possession.external_data.schema import file_hash
from possession.hsmm_pilot import save_context, load_context, score
from possession.learning.data import Windows
from possession.learning.model import masked_losses
from possession.learning.training import seed_everything, forward
from possession.learning.temporal import fit_priors, TemporalDecoder
from possession.learning.study import ScoreContext
from possession.learning.corruption import hide_ball, stable_seed
from possession.phase6.experiment import save_state, restore
from possession.phase7.model import SpatialE0
from possession.phase5.experiment import scenarios
from possession.temporal_probe.common import Budget
from possession.temporal_probe.windows import emissions
from possession.night.common import atomic_json
from possession.transfer_validation import gates
from possession.trajectory_adapter import adapted, predict, track_collate

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "evaluation/reports/external/trajectory-adapter-v2"
PROTOCOL = ROOT / "configs/trajectory_adapter_v2.json"
PUBLIC = ROOT / "docs/public-results/2026-10-28"
PARENT = ROOT / "evaluation/reports/external/hsmm-pilot-v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sources():
    paths = list((ROOT / "possession").rglob("*.py"))
    paths += [ROOT / n for n in ("scripts/run_local.py", "scripts/trajectory_adapter.py",
        "tests/test_trajectory_adapter.py", "evaluation/possession_metrics.py", "configs/trajectory_adapter_v2.json")]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(
        p.read_text(encoding="utf-8-sig").replace("\r\n", "\n").encode()).hexdigest() for p in sorted(paths)}


def validate(p):
    if (p["version"] != "trajectory-adapter-v2" or p["folds"] != [0, 6]
        or p["arms"] != ["instant", "trajectory"] or p["epochs"] != 6
        or p["seed"] != 2029 or p["halo"] != 25 or p["core_frames"] != 256
        or p["scenarios"] != ["clean", "missing40_5s", "missing40_15s"]
        or any(p[k] for k in ("test_access", "target_access", "parameter_search"))):
        raise ValueError("Only fixed public screening protocol allowed")


def parent_model(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = SpatialE0("set_transformer", True)
    model.load_state_dict(state["model"])
    return model.eval(), state


def register():
    p = read(PROTOCOL)
    validate(p)
    _, _, members, dependencies = transfer._transfer_spec()
    manifest = read(ROOT / "evaluation/reports/external/folds/idsse-only.json")
    # Pin public replay references to their already published result/seal identities.
    old_record, old_seal = read(PARENT / "registration.json"), read(PARENT / "result-seal.json")
    published = read(PUBLIC / "hsmm-pilot-v1.json")
    if (file_hash(PARENT / "registration.json") != published["registration_sha256"]
            or old_seal["registration_sha256"] != published["registration_sha256"]
            or read(PARENT / "summary.json") != published):
        raise ValueError("Published public parent changed")
    for name, digest in old_seal["outputs"].items():
        if file_hash(PARENT / name) != digest:
            raise ValueError("Public reference artifact changed")
    parents, splits = {}, {}
    for member in members:
        fold = member["fold"]
        model, state = parent_model(member["teacher"])
        parts = state["split"]
        expected = next(x for x in old_record["partitions"] if x["fold"] == fold)
        if parts != {k: expected[k] for k in ("train", "calibration", "validation", "test")}:
            raise ValueError("Teacher/reference partition mismatch")
        splits[str(fold)] = parts
        parents[str(fold)] = str(member["teacher"].relative_to(ROOT).as_posix())
        store = FoldStore(manifest["artifacts"], parts)
        for key in store.allowed:
            record = manifest["artifacts"][key]
            if file_hash(record["path"]) != record["sha256"]:
                raise ValueError("Public data hash mismatch")
            dependencies[record["path"]] = record["sha256"]
    record = {"protocol": p, "sources": sources(), "splits": splits, "parents": parents,
        "dependencies": {Path(k).relative_to(ROOT).as_posix(): h for k, h in dependencies.items()},
        "public_records": {k: dict(v, path=Path(v["path"]).relative_to(ROOT).as_posix())
                           for k, v in manifest["artifacts"].items()},
        "reference_summary_sha256": file_hash(PUBLIC / "hsmm-pilot-v1.json"),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}}
    marker = DEST / "registration.json"
    if marker.exists() and read(marker) != record:
        raise ValueError("Immutable experiment changed; new version required")
    if not marker.exists():
        atomic_json(marker, record)
    return record


def train_one(parent, views, fold, arm, p, fingerprint, budget):
    folder = DEST / f"training/fold-{fold}/{arm}"
    folder.mkdir(parents=True, exist_ok=True)
    seed_everything(p["seed"])
    model = adapted(parent, arm == "trajectory").eval()
    trainable = [v for v in model.parameters() if v.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=p["learning_rate"], weight_decay=p["weight_decay"])
    generator = torch.Generator().manual_seed(p["seed"])
    path = folder / "last.pt"
    history = []
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["registration_sha256"] != fingerprint:
            raise ValueError("Checkpoint registration changed")
        history = restore(state, model, optimizer, generator)
    for epoch in range(len(history), p["epochs"]):
        budget.check()
        # Balanced two cycles of clean/5s/15s; identical corruption in paired arms.
        condition = epoch % 3
        degraded = views if condition == 0 else {k: hide_ball(v, .4, (5., 15.)[condition-1],
            stable_seed(k, p["seed"]+1000*epoch))[0] for k, v in views.items()}
        data = Windows(degraded, p["core_frames"], p["halo"])
        order = torch.randperm(len(data), generator=generator).tolist()
        total, count, steps = 0., 0, 0
        start = time.perf_counter()
        for a in range(0, len(order), p["batch_size"]):
            budget.check()
            batch = track_collate([data[i] for i in order[a:a+p["batch_size"]]])
            optimizer.zero_grad(set_to_none=True)
            output = forward(model, batch)
            loss, _ = masked_losses(output, batch["alive"], batch["team"], batch["confidence"], batch["core"])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, p["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            n = int(batch["core"].sum())
            total += float(loss.detach())*n
            count += n
            steps += 1
        history.append({"epoch": epoch+1, "loss": total/count, "steps": steps,
            "seconds": time.perf_counter()-start, "condition": p["scenarios"][condition]})
        save_state(path, model, optimizer, generator, history, {"registration_sha256": fingerprint,
            "arm": arm, "fold": fold, "trainable_parameters": sum(v.numel() for v in trainable)})
        atomic_json(DEST / "public-progress.json", {"stage": "training", "fold": fold, "arm": arm,
            "epoch": epoch+1, "epoch_seconds": history[-1]["seconds"]})
    # Parent remains byte-equivalent in tensor space after all optimizer steps.
    for name, tensor in parent.state_dict().items():
        mapped = "encoder.original."+name[len("encoder."):] if name.startswith("encoder.") else name
        torch.testing.assert_close(tensor, model.state_dict()[mapped], rtol=0, atol=0)
    return model, history


def four_view(model, view, p):
    logits, alive = np.zeros(len(view.time)), np.zeros(len(view.time))
    for swap, reflect in p["views"]:
        value = predict(model, transfer._transform_view(view, swap, reflect))
        z = np.log(value[:, 0].clip(1e-7))-np.log(value[:, 1].clip(1e-7))
        logits += (-1 if swap else 1)*z/4
        alive += value[:, :2].sum(1)/4
    return emissions(logits, np.column_stack((alive, np.zeros(len(alive)), 1-alive)))


def decisions(metrics, p):
    checks = {name: gates(metrics["trajectory"], metrics[name], p)
              for name in ("parent", "instant", *p["references"])}
    significant = all(checks[name]["candidate_objective"] >= checks[name]["baseline_objective"]+p["meaningful_gain"]
                      for name in ("parent", "instant", "four_view_v2"))
    return {"comparisons": checks, "meaningful_gain": significant,
            "continue_line": significant and all(v["passed"] for v in checks.values())}


def run():
    DEST.mkdir(parents=True, exist_ok=True)
    with (DEST / "running.lock").open("x") as stream:
        stream.write("Exclusive fixed public experiment\n")
    budget = None
    status = "failed"
    try:
        record = register()
        p = record["protocol"]
        if (DEST / "result-seal.json").exists():
            return verify()
        torch.set_num_threads(p["cpu_threads"])
        seed_everything(p["seed"])
        budget = Budget(p["max_seconds"], DEST)
        fingerprint = file_hash(DEST / "registration.json")
        raw_records = {k: dict(v, path=str(ROOT/v["path"])) for k, v in record["public_records"].items()}
        summaries = {}
        # Train BOTH folds before any candidate validation metric is computed.
        for fold in p["folds"]:
            parts = record["splits"][str(fold)]
            store = FoldStore(raw_records, parts)
            parent, _ = parent_model(ROOT / record["parents"][str(fold)])
            root = DEST / f"replay/fold-{fold}"
            root.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / record["parents"][str(fold)], root / "parent.pt")
            views = {k: store.harmonized(k) for k in parts["train"]}
            for arm in p["arms"]:
                train_one(parent, views, fold, arm, p, fingerprint, budget)
        for fold in p["folds"]:
            budget.check()
            parts = record["splits"][str(fold)]
            store = FoldStore(raw_records, parts)
            parent, _ = parent_model(ROOT / record["parents"][str(fold)])
            models = {"parent": parent}
            histories = {}
            for arm in p["arms"]:
                model = adapted(parent, arm == "trajectory")
                state = torch.load(DEST/f"training/fold-{fold}/{arm}/last.pt", map_location="cpu", weights_only=False)
                model.load_state_dict(state["model"])
                models[arm] = model.eval()
                histories[arm] = state["history"]
            if [r["steps"] for r in histories["instant"]] != [r["steps"] for r in histories["trajectory"]]:
                raise ValueError("Unpaired optimizer exposure")
            root = DEST / f"replay/fold-{fold}"
            view = store.harmonized(parts["validation"])
            np.savez_compressed(root/"view.npz", **{f.name: getattr(view, f.name) for f in fields(Features)})
            context = ScoreContext(store.match(parts["validation"]), view)
            save_context(root/"context.npz", context)
            prior = fit_priors([store.harmonized(k) for k in parts["train"]])
            np.savez_compressed(root/"priors.npz", **prior)
            metrics = {name: {} for name in (*models, *p["references"])}
            for scenario, (degraded, _) in scenarios(view, parts["validation"], {}).items():
                budget.check()
                with np.load(PARENT/f"replay/fold-{fold}/{scenario}-emissions.npz", allow_pickle=False) as z:
                    probabilities = {ref: z[ref].copy() for ref in p["references"]}
                for name, model in models.items():
                    budget.check()
                    probabilities[name] = four_view(model, degraded, p)
                for name, value in probabilities.items():
                    metrics[name][scenario] = score(context, context.decoded(value, TemporalDecoder(p["decoder"], prior)))
                np.savez_compressed(root/f"{scenario}-emissions.npz", **probabilities)
                atomic_json(DEST/"public-progress.json", {"stage": "validation", "fold": fold, "scenario": scenario})
            expected = read(PUBLIC/"hsmm-pilot-v1.json")["folds"][str(fold)]["metrics"]
            for ref in p["references"]:
                assert_metrics(metrics[ref], expected[ref])
            summaries[str(fold)] = {"metrics": metrics, "decision": decisions(metrics, p),
                "history": histories, "reference_parity": True}
        if sources() != record["sources"]:
            raise ValueError("Sources changed during execution")
        result = {"folds": summaries, "continue_line": all(x["decision"]["continue_line"] for x in summaries.values()),
            "automatic_promotion": False, "test_accessed": False, "target_accessed": False,
            "registration_sha256": fingerprint, "seconds": budget.saved["seconds"]+time.perf_counter()-budget.start}
        atomic_json(DEST/"summary.json", result)
        outputs = [x for d in (DEST/"training", DEST/"replay") for x in d.rglob("*") if x.is_file()]
        outputs.append(DEST/"summary.json")
        atomic_json(DEST/"result-seal.json", {"registration_sha256": fingerprint,
            "outputs": {x.relative_to(DEST).as_posix(): file_hash(x) for x in outputs}})
        verify()
        status = "complete"
        return 0
    finally:
        if budget:
            budget.close("train_validation", status)
        (DEST/"running.lock").unlink()


def verify():
    record, seal = read(DEST/"registration.json"), read(DEST/"result-seal.json")
    validate(record["protocol"])
    if sources() != record["sources"] or file_hash(DEST/"registration.json") != seal["registration_sha256"]:
        raise ValueError("Immutable source/registration changed")
    expected = {"summary.json"}
    for f in record["protocol"]["folds"]:
        expected.update(f"training/fold-{f}/{a}/last.pt" for a in record["protocol"]["arms"])
        expected.update(f"replay/fold-{f}/{n}" for n in ("parent.pt", "view.npz", "context.npz", "priors.npz"))
        expected.update(f"replay/fold-{f}/{s}-emissions.npz" for s in record["protocol"]["scenarios"])
    if set(seal["outputs"]) != expected:
        raise ValueError("Incomplete capsule")
    for name, digest in seal["outputs"].items():
        if file_hash(DEST/name) != digest:
            raise ValueError("Output changed")
    result = read(DEST/"summary.json")
    if (set(result["folds"]) != {"0", "6"} or result["registration_sha256"] != seal["registration_sha256"]
            or any(result[k] for k in ("automatic_promotion", "test_accessed", "target_accessed"))):
        raise ValueError("Invalid scope")
    for row in result["folds"].values():
        if row["decision"] != decisions(row["metrics"], record["protocol"]):
            raise ValueError("Decision changed")
    if result["continue_line"] != all(x["decision"]["continue_line"] for x in result["folds"].values()):
        raise ValueError("Overall decision changed")
    return 0


def replay(inference=True):
    verify()
    p = read(DEST/"registration.json")["protocol"]
    seed_everything(p["seed"])
    torch.set_num_threads(p["cpu_threads"])
    result = read(DEST/"summary.json")
    for fold in p["folds"]:
        root = DEST/f"replay/fold-{fold}"
        context = load_context(root/"context.npz")
        with np.load(root/"priors.npz", allow_pickle=False) as z:
            prior = dict(z)
        with np.load(root/"view.npz", allow_pickle=False) as z:
            view = Features(**dict(z))
        parent, _ = parent_model(root/"parent.pt")
        models = {"parent": parent}
        for arm in p["arms"]:
            model = adapted(parent, arm == "trajectory")
            state = torch.load(DEST/f"training/fold-{fold}/{arm}/last.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"])
            models[arm] = model.eval()
        split = read(DEST/"registration.json")["splits"][str(fold)]
        for scenario, (degraded, _) in scenarios(view, split["validation"], {}).items():
            with np.load(root/f"{scenario}-emissions.npz", allow_pickle=False) as z:
                probabilities = dict(z)
            if inference:
                for name, model in models.items():
                    np.testing.assert_array_equal(four_view(model, degraded, p), probabilities[name])
            for name, value in probabilities.items():
                metrics = score(context, context.decoded(value, TemporalDecoder(p["decoder"], prior)))
                assert_metrics({scenario: metrics}, result["folds"][str(fold)]["metrics"][name])
    return 0


def export():
    verify()
    names = sorted([*read(DEST/"result-seal.json")["outputs"], "registration.json", "result-seal.json", "compute.json"])
    manifest = {name: file_hash(DEST/name) for name in names}
    archive = ROOT/"evaluation/reports/handoff/northmind-trajectory-adapter-v2.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in names:
            z.write(DEST/name, "evaluation/reports/external/trajectory-adapter-v2/"+name)
        z.writestr("TRAJECTORY-MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("Archive CRC error")
        for name, digest in manifest.items():
            if hashlib.sha256(z.read("evaluation/reports/external/trajectory-adapter-v2/"+name)).hexdigest() != digest:
                raise ValueError("Archive member changed")
    atomic_json(PUBLIC/"trajectory-adapter-v2.json", read(DEST/"summary.json"))
    atomic_json(PUBLIC/"trajectory-adapter-v2-package.json", {"archive": archive.name, "sha256": file_hash(archive),
        "bytes": archive.stat().st_size, "file_count": len(names), "verified": True,
        "scope": "Own checkpoints/optimizers, public IDSSE validation geometry/labels, priors, emissions and immutable protocol",
        "raw_data_required_for_replay": False, "training_data": "Pinned public IDSSE download required to retrain"})
    return 0
