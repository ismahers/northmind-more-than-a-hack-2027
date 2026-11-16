"""Identical window order and optimizer updates, exact per-epoch continuation."""
import random
import time
import numpy as np
import torch

from possession.interval_probe.data import load
from possession.learning.training import atomic_checkpoint, seed_everything
from possession.night.common import atomic_json

from .common import DEST, read_json
from .data import get_exports
from .model import TemporalHead, loss_for_core
from .teacher import load_teacher
from .windows import EmbeddingWindows, collate


def make_head(fold, teacher, arm, config):
    parent = load_teacher(teacher)
    return TemporalHead(parent.team_head.weight, parent.team_head.bias,
                        temporal=arm.startswith("multiscale"), intervals=arm.endswith("intervals"),
                        dilations=config["dilations"], kernel_size=config["kernel_size"], seed=config["seed"])


def train_epoch(model, optimizer, dataset, generator, config, budget):
    model.train()
    order = torch.randperm(len(dataset), generator=generator).tolist()
    total, count = 0., 0
    for start in range(0, len(order), config["batch_size"]):
        budget.check()
        hidden, core, targets = collate([dataset[i] for i in order[start:start+config["batch_size"]]])
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for_core(model(hidden), targets, core, model.intervals)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite temporal probe loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        total += float(loss.detach()) * int(core.sum())
        count += int(core.sum())
    return {"loss": total/max(1, count), "steps": (len(order)+config["batch_size"]-1)//config["batch_size"]}


def save_state(path, model, optimizer, generator, history, registration_hash):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_checkpoint(path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "loader_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
                            "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                            "history": history, "registration_sha256": registration_hash})


def restore_state(path, model, optimizer, generator, registration_hash):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["registration_sha256"] != registration_hash:
        raise ValueError("Checkpoint belongs to a different immutable experiment")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    generator.set_state(state["loader_rng"])
    torch.set_rng_state(state["torch_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    return state["history"]


def train(store, fold, teacher, config, spec, budget, registration_hash):
    records = {}
    for key in teacher["split"]["train"]:
        for scenario, path in get_exports(store, key, fold, teacher, spec, budget).items():
            value = load(path)
            value["condition_weight"] = np.full(len(value["hidden"]), config["condition_weights"][scenario], np.float32)
            records[key+"/"+scenario] = value
    # Same halo and core windows for all four arms; only the temporal head reads halo.
    dataset = EmbeddingWindows(records, config["core_frames"], config["added_context_frames_per_side"])
    for arm in config["arms"]:
        folder = DEST / f"training/fold-{fold}" / arm
        if (folder / "complete.json").is_file():
            continue
        seed_everything(config["seed"])
        model = make_head(fold, teacher, arm, config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        generator = torch.Generator().manual_seed(config["seed"])
        path = folder / "last.pt"
        history = restore_state(path, model, optimizer, generator, registration_hash) if path.is_file() else []
        for epoch in range(len(history)+1, config["epochs"]+1):
            start = time.perf_counter()
            metrics = train_epoch(model, optimizer, dataset, generator, config, budget)
            history.append(dict(metrics, epoch=epoch, seconds=time.perf_counter()-start))
            save_state(path, model, optimizer, generator, history, registration_hash)
        atomic_json(folder / "complete.json", {"arm": arm, "history": history,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "loader_rng": generator.get_state().tolist(), "registration_sha256": registration_hash})


def load_head(fold, teacher, arm, config):
    model = make_head(fold, teacher, arm, config)
    state = torch.load(DEST / f"training/fold-{fold}/{arm}/last.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return model
