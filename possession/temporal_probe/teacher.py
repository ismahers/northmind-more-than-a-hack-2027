"""Self-contained public pilot teachers; historical Phase8 remains read only."""
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from possession.external_data.schema import file_hash
from possession.learning.data import Windows
from possession.learning.study import ScoreContext
from possession.learning.temporal import fit_priors
from possession.learning.training import seed_everything, epoch_loss
from possession.night.common import atomic_json
from possession.phase5.experiment import scenarios
from possession.phase6.experiment import split, corrupt, save_state, restore, epoch_pass
from possession.phase6.model import transition_targets, TransitionWindows
from possession.phase7.model import SpatialE0
from possession.phase8.evaluation import predict
from possession.phase8.packing import compact_collate, compact_transition_collate

from .common import ROOT, REPORTS, Store, Budget, read_json, splits_valid, public_path

TEACHERS = REPORTS / "temporal-teachers-v1"


def load_teacher(teacher):
    checkpoint = Path(teacher["checkpoint"]).resolve()
    if not checkpoint.is_relative_to(REPORTS.resolve()):
        raise ValueError("Teacher checkpoint outside public experiment root")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["split"] != teacher["split"]:
        raise ValueError("Teacher checkpoint split mismatch")
    model = SpatialE0("set_transformer", True)
    model.load_state_dict(state["model"])
    return model


def train_teacher(store, parts, fold, seed, spec, budget, digest):
    views = {key: store.harmonized(key) for key in parts["train"]}
    folder = TEACHERS / f"training/fold-{fold}/seed-{seed}"
    for stage, epochs, lr, offset in (("base", 12, .001, 0), ("binary", 8, .0003, 12)):
        dest = folder / stage
        dest.mkdir(parents=True, exist_ok=True)
        if (dest / "complete.json").is_file():
            continue
        seed_everything(seed)
        model = SpatialE0("set_transformer", stage == "binary")
        if offset:
            base = torch.load(folder / "base/last.pt", map_location="cpu", weights_only=False)
            missing, extra = model.load_state_dict(base["model"], strict=False)
            if set(missing) != {"boundary_head.weight", "boundary_head.bias"} or extra:
                raise ValueError("Teacher base mismatch")
            seed_everything(seed+6000)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.0001)
        generator = torch.Generator().manual_seed(seed+(6000 if offset else 0))
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        targets = {key: transition_targets(store.match(key), value, 1, .12) for key, value in views.items()} if offset else None
        if targets:
            positive = sum((y*valid[:, None]).sum(0) for y, valid in targets.values())
            n = sum(valid.sum() for _, valid in targets.values())
            weights = torch.tensor(np.minimum(40., (n-positive)/np.maximum(1, positive)), dtype=torch.float32)
        path = dest / "last.pt"
        history = []
        if path.is_file():
            state = torch.load(path, map_location="cpu", weights_only=False)
            if state["registration_sha256"] != digest:
                raise ValueError("Fresh teacher registration changed")
            history = restore(state, model, optimizer, generator)
        for epoch in range(len(history)+1, epochs+1):
            budget.check()
            start = time.perf_counter()
            degraded = corrupt(views, offset+epoch, dict(spec, seed=seed))
            dataset = TransitionWindows(degraded, targets) if offset else Windows(degraded, 256, 15)
            loader = DataLoader(dataset, batch_size=4, shuffle=True, generator=generator,
                                collate_fn=compact_transition_collate if offset else compact_collate)
            stats = epoch_pass(model, loader, optimizer, weights, "binary", spec) if offset else epoch_loss(
                model, loader, torch.device("cpu"), optimizer, scaler)
            history.append({"epoch": epoch, "train": stats, "seconds": time.perf_counter()-start})
            save_state(path, model, optimizer, generator, history,
                       {"split": parts, "seed": seed, "registration_sha256": digest})
            atomic_json(TEACHERS / "progress.json", {"fold": fold, "seed": seed, "stage": stage, "epoch": epoch})
        atomic_json(dest / "complete.json", {"history": history, "split": parts, "registration_sha256": digest})


def bootstrap():
    # Imports here avoid a circular import when the probe consumes a teacher.
    from .experiment import metrics
    from .selection import objective
    config = read_json(ROOT / "configs/temporal_teacher_protocol.json")
    spec = read_json(ROOT / "configs/phase8_protocol.json")
    manifest_path = REPORTS / "folds/idsse-only.json"
    manifest = read_json(manifest_path)
    records = manifest["artifacts"]
    if len(records) != 7 or not all(key.startswith("idsse:") for key in records):
        raise ValueError("Seven explicit public IDSSE records required")
    for record in records.values():
        public_path(record["path"])
    parts = {str(fold): split(manifest, fold) for fold in config["pilots"]}
    if not all(splits_valid(value) for value in parts.values()):
        raise ValueError("Fresh teacher partitions overlap")
    directories = ("temporal_probe", "external_data", "learning", "phase5", "phase6", "phase7", "phase8", "night", "interval_probe")
    sources = {str(p.relative_to(ROOT)): file_hash(p) for directory in directories
               for p in sorted((ROOT / "possession" / directory).glob("*.py"))}
    sources["evaluation/possession_metrics.py"] = file_hash(ROOT / "evaluation/possession_metrics.py")
    record = {"protocol": spec, "bootstrap": config, "parts": parts, "sources": sources,
              "manifest_sha256": file_hash(manifest_path), "test_used": False,
              "environment": {"torch": torch.__version__, "numpy": np.__version__}}
    marker = TEACHERS / "registration.json"
    if marker.is_file() and read_json(marker) != record:
        raise ValueError("Fresh teacher recipe changed; use a new version")
    if not marker.is_file():
        atomic_json(marker, record)
    digest = file_hash(marker)
    torch.set_num_threads(spec["cpu_threads"])
    store, budget, status = Store(records), Budget(config["bootstrap_max_compute_seconds"], TEACHERS), "failed"
    try:
        for fold in config["pilots"]:
            selection = TEACHERS / f"selections/fold-{fold}.json"
            if selection.is_file():
                continue
            partition = parts[str(fold)]
            prior = fit_priors([store.harmonized(key) for key in partition["train"]])
            key = partition["validation"]
            context = ScoreContext(store.match(key), store.harmonized(key))
            trials = []
            for seed in config["seeds"]:
                train_teacher(store, partition, fold, seed, spec, budget, digest)
                checkpoint = TEACHERS / f"training/fold-{fold}/seed-{seed}/binary/last.pt"
                teacher = {"seed": seed, "split": partition, "checkpoint": str(checkpoint), "source": "fresh_public_pilots"}
                model = load_teacher(teacher)
                measured = {}
                for scenario, (view, _) in scenarios(context.view, key, spec).items():
                    budget.check()
                    p, _, _ = predict(model, view)
                    measured[scenario] = metrics(context, p, prior)
                trials.append({"teacher": teacher, "metrics": measured, "objective": objective(measured, spec),
                               "checkpoint_sha256": file_hash(checkpoint)})
            winner = max(trials, key=lambda value: value["objective"])
            atomic_json(selection, {"teacher": winner["teacher"], "trials": trials, "test_used": False,
                                    "registration_sha256": digest})
        status = "complete"
    finally:
        budget.close("bootstrap", status)
