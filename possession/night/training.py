"""Exact continuation of public E0 checkpoints; old phase3 files are read-only."""

from dataclasses import asdict
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from possession.learning.corruption import hide_ball, stable_seed
from possession.learning.data import Windows, collate
from possession.learning.metrics import score_match
from possession.learning.study import ScoreContext, save_probabilities
from possession.learning.temporal import TemporalDecoder
from possession.learning.training import (TrainConfig, new_model, seed_everything, epoch_loss,
    implementation_hashes, atomic_checkpoint, memory_usage, predict)
from .cache import PublicStore, predict_batched
from .common import DEST, OLD, atomic_json, progress, source_hashes, objective, robust_eligible


def old_directory(i, condition):
    return OLD / "training" / condition / "seed-2027" / ("fold-" + str(i))


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b


def reconstructed_loader_rng(seed, count, batch_size, epochs):
    """DataLoader base seed + RandomSampler permutations; independent of model RNG."""
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(range(count), batch_size=batch_size, shuffle=True, generator=generator)
    for _ in range(epochs):
        for _ in loader:
            pass
    return generator.get_state()


def early_state(history, settings):
    best, stale = np.inf, 0
    for row in history:
        loss = row["validation"]["loss"]
        if loss < best - settings["min_delta"]:
            best, stale = loss, 0
        else:
            stale += 1
    return best, stale


def profile(store, data, protocol):
    path = DEST / "profile.json"
    if path.exists():
        return json.loads(path.read_text())
    i = protocol["training_folds"][0]
    validation = data["folds"][i]["validation"][0]
    start = time.perf_counter()
    view = store.view(validation)
    preparation = time.perf_counter() - start
    state = torch.load(old_directory(i, "A_clean") / "cap-12.pt", map_location="cpu", weights_only=False)
    model = new_model(TrainConfig.from_dict(state["config"]))
    model.load_state_dict(state["model"])
    start = time.perf_counter()
    slow = predict(model, view, 256, torch.device("cpu"))
    slow_s = time.perf_counter() - start
    start = time.perf_counter()
    fast = predict_batched(model, view, batch_size=protocol["inference_batch"])
    fast_s = time.perf_counter() - start
    difference = float(np.max(np.abs(slow - fast)))
    match = store.match(validation)
    context = ScoreContext(match, view)
    selected = json.loads((OLD / f"selections/fold-{i}.json").read_text())
    with np.load(old_directory(i, "A_clean") / "priors.npz") as z:
        priors = dict(z)
    same = {}
    for family in ("argmax", "hmm", "hsmm"):
        decoder = TemporalDecoder(selected["conditions"]["A_clean"]["by_family"][family]["specification"], priors)
        same[family] = bool(np.array_equal(context.decoded(slow, decoder), context.decoded(fast, decoder)))
    result = {"feature_preparation_s": preparation, "single_window_inference_s": slow_s,
        "batched_inference_s": fast_s, "speedup": slow_s / fast_s, "maximum_probability_difference": difference,
        "same_native_decoded_paths": same, "enabled": difference <= 2e-6 and all(same.values()),
        "store_timing": store.times, "memory": memory_usage(torch.device("cpu")),
        "note": "Batching changes floating-point reduction order; frozen reference path checked on inner validation only"}
    atomic_json(path, result)
    return result


def infer(model, view, protocol):
    profile_result = json.loads((DEST / "profile.json").read_text())
    return predict_batched(model, view, batch_size=protocol["inference_batch"]) if profile_result["enabled"] else predict(
        model, view, 256, torch.device("cpu"))


def continue_trajectory(store, data, i, condition, protocol, budget):
    folder = DEST / "training" / condition / f"fold-{i}"
    folder.mkdir(parents=True, exist_ok=True)
    finished = folder / "training.json"
    fold = data["folds"][i]
    old = json.loads((old_directory(i, condition) / "training.json").read_text())
    fingerprint = {"old_fingerprint": old["fingerprint"], "night_protocol": protocol,
        "source": source_hashes(["possession/night/training.py", "possession/night/cache.py", "possession/night/common.py"])}
    if old["fingerprint"]["implementation"] != implementation_hashes():
        raise ValueError("Phase3 implementation changed")
    if finished.exists():
        result = json.loads(finished.read_text())
        if result["fingerprint"] != fingerprint:
            raise ValueError("Night continuation provenance changed")
        return result
    begin = time.perf_counter()
    train_keys = list(fold["train"])
    assert not set(train_keys) & set(fold["test"] + fold["validation"])
    views = {key: store.view(key) for key in train_keys}
    validation_key = fold["validation"][0]
    validation = store.view(validation_key)
    context = ScoreContext(store.match(validation_key), validation)
    config = TrainConfig(seed=protocol["seed"], epochs=max(protocol["caps"]), device="cpu")
    seed_everything(config.seed)
    model = new_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    generator = torch.Generator()
    windows = Windows(views, config.core_frames, model.context)
    val_loader = DataLoader(Windows({validation_key: validation}, config.core_frames, model.context),
                            batch_size=config.batch_size, collate_fn=collate)
    latest = folder / "last.json"
    if latest.exists():
        state = torch.load(folder / json.loads(latest.read_text())["file"], map_location="cpu", weights_only=False)
        if state["fingerprint"] != fingerprint:
            raise ValueError("Partial continuation differs from current run")
    else:
        # Recover the missing generator and verify recovery against a complete original checkpoint.
        baseline = torch.load(OLD / f"baseline/fold-{i}/last.pt", map_location="cpu", weights_only=False)
        recovered = reconstructed_loader_rng(config.seed, len(windows), config.batch_size, baseline["epoch"])
        if not torch.equal(recovered, baseline["loader_rng"]):
            raise ValueError("DataLoader RNG reconstruction did not match the complete original checkpoint")
        source = torch.load(old_directory(i, condition) / "cap-6.pt", map_location="cpu", weights_only=False)
        history = old["history"][:source["epoch"]]
        stop_best, stale = early_state(history, protocol["early_stopping"])
        state = {"model": source["model"], "optimizer": source["optimizer"], "epoch": source["epoch"],
            "torch_rng": source["rng"], "loader_rng": reconstructed_loader_rng(config.seed, len(windows), config.batch_size, source["epoch"]),
            "numpy_rng": np.random.get_state(), "python_rng": random.getstate(), "history": history,
            "best_epoch": source["epoch"], "best_loss": source["history_loss"] if "history_loss" in source else min(
                row["validation"]["loss"] for row in history), "stop_best": stop_best, "stale": stale,
            "fingerprint": fingerprint, "config": asdict(config), "prefix_verified": False}
        atomic_checkpoint(folder / f"epoch-{state['epoch']:03d}.pt", state)
        atomic_json(latest, {"file": f"epoch-{state['epoch']:03d}.pt"})
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    generator.set_state(state["loader_rng"])
    torch.set_rng_state(state["torch_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    history, best_loss, best_epoch = state["history"], state["best_loss"], state["best_epoch"]
    stop_best, stale, prefix_verified = state["stop_best"], state["stale"], state["prefix_verified"]
    expected = torch.load(old_directory(i, condition) / "cap-12.pt", map_location="cpu", weights_only=False)
    caps_path = folder / "caps.json"
    caps = json.loads(caps_path.read_text()) if caps_path.exists() else []
    # An interruption between the durable epoch save and cap scoring must not lose that cap.
    for cap in protocol["caps"]:
        if cap <= state["epoch"] and not any(row["cap"] == cap for row in caps):
            prefix = history[:cap]
            chosen = min(prefix, key=lambda row: row["validation"]["loss"])
            rng = torch.get_rng_state()
            caps.append(cap_result(folder, i, condition, cap, cap, chosen["epoch"],
                        chosen["validation"]["loss"], context, validation, protocol))
            atomic_json(caps_path, caps)
            torch.set_rng_state(rng)
    stopped = state["epoch"] >= protocol["early_stopping"]["min_epochs"] and stale >= protocol["early_stopping"]["patience"]
    replay_s, training_s = 0., 0.
    end_epoch = state["epoch"] if stopped else max(protocol["caps"])
    for epoch in range(state["epoch"] + 1, end_epoch + 1):
        budget.check()
        progress("training", fold=i, condition=condition, epoch=epoch)
        started = time.perf_counter()
        epoch_views = views
        if condition.endswith("blocks"):
            length = (1., 5., 15.)[(epoch - 1) % 3]
            epoch_views = {key: hide_ball(view, .4, length, stable_seed(key, config.seed + 1000 * epoch))[0]
                           for key, view in views.items()}
        loader = DataLoader(Windows(epoch_views, config.core_frames, model.context), batch_size=config.batch_size,
                            shuffle=True, generator=generator, collate_fn=collate)
        train_start = time.perf_counter()
        training = epoch_loss(model, loader, torch.device("cpu"), optimizer, scaler)
        train_elapsed = time.perf_counter() - train_start
        val_start = time.perf_counter()
        validation_loss = epoch_loss(model, val_loader, torch.device("cpu"))
        elapsed = time.perf_counter() - started
        history.append({"epoch": epoch, "train": training, "validation": validation_loss,
            "elapsed_s": elapsed, "train_s": train_elapsed, "validation_s": time.perf_counter() - val_start,
            "replayed_prefix": epoch <= 12})
        if validation_loss["loss"] < best_loss:
            best_loss, best_epoch = validation_loss["loss"], epoch
        if validation_loss["loss"] < stop_best - protocol["early_stopping"]["min_delta"]:
            stop_best, stale = validation_loss["loss"], 0
        else:
            stale += 1
        if epoch == expected["epoch"]:
            comparisons = {"model": equal_state(model.state_dict(), expected["model"]),
                "optimizer": equal_state(optimizer.state_dict(), expected["optimizer"]),
                "torch_rng": torch.equal(torch.get_rng_state(), expected["rng"])}
            atomic_json(folder / "prefix_verification.json", {"epoch": epoch, **comparisons,
                "loader_rng_verified_against_complete_baseline": True})
            if not all(comparisons.values()):
                raise ValueError("Replay differs from phase3; do not claim exact continuation")
            prefix_verified = True
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
            "torch_rng": torch.get_rng_state(), "loader_rng": generator.get_state(),
            "numpy_rng": np.random.get_state(), "python_rng": random.getstate(), "history": history,
            "best_epoch": best_epoch, "best_loss": best_loss, "stop_best": stop_best, "stale": stale,
            "fingerprint": fingerprint, "config": asdict(config), "prefix_verified": prefix_verified}
        atomic_checkpoint(folder / f"epoch-{epoch:03d}.pt", state)
        atomic_json(latest, {"file": f"epoch-{epoch:03d}.pt"})
        atomic_json(folder / "history.json", history)
        if epoch <= 12:
            replay_s += elapsed
        else:
            training_s += elapsed
        stopped = epoch >= protocol["early_stopping"]["min_epochs"] and stale >= protocol["early_stopping"]["patience"]
        if epoch in protocol["caps"] or stopped:
            cap = epoch if epoch in protocol["caps"] else next(c for c in protocol["caps"] if c >= epoch)
            if not prefix_verified:
                raise ValueError("Extension requires verified original prefix")
            random_state = torch.get_rng_state()
            caps.append(cap_result(folder, i, condition, cap, epoch, best_epoch, best_loss, context, validation, protocol))
            atomic_json(caps_path, caps)
            torch.set_rng_state(random_state)
        if stopped:
            break
    for cap in protocol["caps"]:
        if not any(c["cap"] == cap for c in caps):
            caps.append({**caps[-1], "cap": cap, "reused_early_stopped_result": True})
    result = {"fingerprint": fingerprint, "caps": caps, "history": history, "epochs_executed": len(history),
        "stopped_early": stopped, "prefix_verified": prefix_verified, "replay_s_this_session": replay_s,
        "new_epochs_s_this_session": training_s, "total_s_this_session": time.perf_counter() - begin,
        "memory": memory_usage(torch.device("cpu")), "store_timing": store.times}
    atomic_json(finished, result)
    return result


def cap_result(folder, i, condition, cap, epochs_executed, best_epoch, best_loss, ctx, validation, protocol):
    selected = json.loads((OLD / f"selections/fold-{i}.json").read_text())["conditions"][condition]
    fixed = max(selected["by_family"].items(), key=lambda pair: pair[1]["objective"])[0]
    families = dict(selected["by_family"])
    with np.load(old_directory(i, condition) / "priors.npz") as z:
        priors = dict(z)
    state = torch.load(folder / f"epoch-{best_epoch:03d}.pt", map_location="cpu", weights_only=False)
    model = new_model(TrainConfig.from_dict(state["config"]))
    model.load_state_dict(state["model"])
    scores = {family: {} for family in ("argmax", "hmm", "hsmm", fixed)}
    timing = 0.
    for case in json.loads((OLD / "protocol.json").read_text())["corruptions"]:
        if case["name"] not in protocol["primary_scenarios"]:
            continue
        path = folder / "validation_predictions" / f"epoch-{best_epoch}" / (case["name"] + ".npz")
        if path.exists():
            with np.load(path) as z:
                p = z["probabilities"]
        elif cap == 12 or best_epoch == json.loads((old_directory(i, condition) / "training.json").read_text())["chosen"]["selected_epoch"]:
            source = OLD / f"validation_predictions/fold-{i}" / condition / (case["name"] + ".npz")
            with np.load(source) as z:
                p = z["probabilities"]
                np.testing.assert_array_equal(z["time"], validation.time)
            save_probabilities(path, validation, p)
        else:
            corrupted, _ = hide_ball(validation, case["fraction"], case["block_seconds"], stable_seed(ctx_key(i), 701))
            started = time.perf_counter()
            p = infer(model, corrupted, protocol)
            timing += time.perf_counter() - started
            save_probabilities(path, validation, p)
        for family in scores:
            scores[family][case["name"]] = ctx.spell(p, TemporalDecoder(families[family]["specification"], priors))
    return {"cap": cap, "epochs_executed": epochs_executed, "selected_epoch": best_epoch, "validation_loss": best_loss,
        "frozen_family": fixed, "validation_scores": scores, "objective": objective(scores[fixed], protocol),
        "inference_s": timing, "checkpoint": str(folder / f"epoch-{best_epoch:03d}.pt")}


def ctx_key(i):
    data = json.loads((OLD.parent / "folds/idsse-only.json").read_text())
    return data["folds"][i]["validation"][0]
