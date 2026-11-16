"""Extend exact unweighted original trajectories, preserving parent checkpoints."""
import copy
import random
import time

import numpy as np
import torch

from possession.interval_probe.data import load
from possession.night.common import atomic_json
from possession.temporal_probe.training import train_epoch, save_state, restore_state
from possession.temporal_probe.windows import EmbeddingWindows
from possession.duration_balanced.training import init_head, paths
from .common import DEST, parent_checkpoint, cap_path


def dataset_for(teacher, spec):
    records = {}
    for key in teacher["split"]["train"]:
        for scenario, path in paths(teacher, key, spec).items():
            value = load(path)
            # Original BCE: scenario weights only; no duration-derived weights.
            value["condition_weight"] = np.full(len(value["hidden"]), spec["condition_weights"][scenario], np.float32)
            records[key + "/" + scenario] = value
    return EmbeddingWindows(records, spec["core_frames"], spec["training_halo_frames"])


def state_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(state_equal(x, y) for x, y in zip(a, b))
    return a == b


def fingerprint(model, optimizer, generator):
    return copy.deepcopy({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "loader_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
                          "numpy_rng": np.random.get_state(), "python_rng": random.getstate()})


def continue_fold(fold, teacher, spec, budget, digest, parent_digest):
    folder = DEST / f"training/fold-{fold}"
    complete = folder / "complete.json"
    if complete.is_file():
        return
    dataset = dataset_for(teacher, spec)
    model = init_head(fold, teacher, spec)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["learning_rate"], weight_decay=spec["weight_decay"])
    generator = torch.Generator()
    last = folder / "last.pt"
    if last.is_file():
        history = restore_state(last, model, optimizer, generator, digest)
    else:
        history = restore_state(parent_checkpoint(fold), model, optimizer, generator, parent_digest)
    steps = (len(dataset) + spec["batch_size"]-1) // spec["batch_size"]
    if any(row["steps"] != steps for row in history):
        raise ValueError("Continuation exposure differs from original")
    start = time.perf_counter(); training_seconds = 0.; proof_seconds = 0.
    proof_path = folder / "resume-proof.json"
    for epoch in range(len(history)+1, max(spec["budgets"])+1):
        budget.check()
        atomic_json(DEST / "progress.json", {"stage": "training", "fold": fold, "epoch": epoch})
        begin = time.perf_counter()
        metrics = train_epoch(model, optimizer, dataset, generator, spec, budget)
        seconds = time.perf_counter()-begin; training_seconds += seconds
        history.append(dict(metrics, epoch=epoch, seconds=seconds))
        save_state(last, model, optimizer, generator, history, digest)
        if epoch in spec["budgets"]:
            save_state(cap_path(fold, epoch), model, optimizer, generator, history, digest)
        if epoch == 13:
            # Compare uninterrupted12→13 with checkpoint-restored12→13, including
            # optimizer moments and every RNG. This extra epoch is proof, not exposure.
            continuous = fingerprint(model, optimizer, generator)
            begin = time.perf_counter()
            other = init_head(fold, teacher, spec)
            other_optimizer = torch.optim.AdamW(other.parameters(), lr=spec["learning_rate"], weight_decay=spec["weight_decay"])
            other_generator = torch.Generator()
            restore_state(cap_path(fold, 12), other, other_optimizer, other_generator, digest)
            replay_metrics = train_epoch(other, other_optimizer, dataset, other_generator, spec, budget)
            replay = fingerprint(other, other_optimizer, other_generator)
            same = state_equal(continuous, replay) and metrics == replay_metrics
            proof_seconds += time.perf_counter()-begin
            atomic_json(proof_path, {"exact": bool(same), "epoch": 13, "all_model_optimizer_rng_equal": bool(same),
                                     "extra_proof_updates": steps, "seconds": proof_seconds})
            if not same:
                raise ValueError("Exact real continuation proof failed")
            del other, other_optimizer, replay, continuous
    if not proof_path.is_file():
        raise ValueError("Real continuation proof missing")
    atomic_json(complete, {"epochs": len(history), "history": history, "steps_per_epoch": steps,
                           "training_windows": len(dataset),
                           "new_training_epochs": 8, "new_training_seconds_this_session": training_seconds,
                           "resume_proof_seconds_this_session": proof_seconds,
                           "total_seconds_this_session": time.perf_counter()-start,
                           "parent_epochs_reused": 8, "parameters": sum(p.numel() for p in model.parameters()),
                           "no_duration_weighting": True, "registration_sha256": digest})
