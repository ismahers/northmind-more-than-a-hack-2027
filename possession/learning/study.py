"""Phase 3 training reuses E0/loss/windows; only internal validation makes choices."""

from dataclasses import asdict
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from possession.external_data.audit import segments, joint_labels
from possession.external_data.features import build_features, nearest_indices
from possession.external_data.schema import Match, file_hash
from .corruption import hide_ball, stable_seed
from .data import Windows, collate
from .metrics import official_scorer, score_match, SCORER_PATH
from .temporal import TemporalDecoder, fit_priors
from .training import (TrainConfig, new_model, seed_everything, epoch_loss, predict,
                       implementation_hashes, atomic_checkpoint, memory_usage)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def save_probabilities(path, view, probabilities):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, probabilities=probabilities, time=view.time,
                        run=view.run, native_index=view.native_index)


class ScoreContext:
    """Cache 25Hz mapping; selection calls the official spell scorer, unchanged."""
    def __init__(self, match, view):
        self.pm = official_scorer()
        self.parts, gt, offset = [], [], 0
        labels = joint_labels(match)
        if np.any(labels < 0):
            raise ValueError("Primary IDSSE study requires complete native H/A/D")
        for run_id, (a, b) in enumerate(segments(match)):
            times = match.time[a] + np.arange(int(round((match.time[b - 1] - match.time[a]) * 25)) + 1) / 25
            selected = np.flatnonzero(view.run == run_id)
            mapping = nearest_indices(view.time[selected], times)
            gt.extend((labels[a + nearest_indices(match.time[a:b], times)], np.full(26, 3)))
            self.parts.append((selected, mapping, slice(offset, offset + len(times))))
            offset += len(times) + 26
        self.gt = np.concatenate(gt).astype(np.int8)
        self.runs = self.pm.valid_runs(self.gt)
        self.view = view

    def decoded(self, p, decoder):
        pred = np.full(len(self.gt), 3, dtype=np.int8)
        for selected, mapping, output in self.parts:
            pred[output] = decoder.decode(p[selected], self.view.time[selected])[mapping]
        return pred

    def spell(self, p, decoder):
        return self.pm.spell_scores(self.gt, self.decoded(p, decoder), self.runs)["mean_f1"]


class FixedSourceBatches(Sampler):
    """Half source windows per minibatch, equal update budget to IDSSE-only."""
    def __init__(self, dataset, updates, batch_size, seed):
        self.groups = [np.array([i for i, window in enumerate(dataset.windows)
                                if window[0].startswith(prefix)], dtype=int)
                       for prefix in ("idsse:", "skillcorner:")]
        if not all(len(group) for group in self.groups) or batch_size % 2:
            raise ValueError("Balanced source batches need both sources and an even batch size")
        self.updates, self.batch_size, self.seed = updates, batch_size, seed

    def __len__(self):
        return self.updates

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.updates):
            batch = np.r_[*(rng.choice(group, self.batch_size // 2, replace=True) for group in self.groups)]
            rng.shuffle(batch)
            yield batch.tolist()


def fit_trajectory(records, fold, protocol, directory, condition="A_clean", seed=2027, max_epochs=12):
    """No test arrays loaded. Cap selection uses this fold's one internal validation match."""
    directory = Path(directory)
    completion = directory / "training.json"
    fingerprint = {"implementation": implementation_hashes(), "protocol": protocol,
                   "seed": seed, "condition": condition, "max_epochs": max_epochs,
                   "fold": fold, "data": {k: r["sha256"] for k, r in records.items()}}
    if completion.exists():
        result = json.loads(completion.read_text())
        if result["fingerprint"] != fingerprint:
            raise ValueError("Study training artifact differs; use a new study directory")
        return result
    directory.mkdir(parents=True, exist_ok=True)
    config = TrainConfig(seed=seed, epochs=max_epochs, device="cpu")
    torch.set_num_threads(config.cpu_threads)
    seed_everything(seed)
    model = new_model(config)
    train_keys = list(fold["train"])
    if condition.startswith("A"):
        train_keys = [key for key in train_keys if key.startswith("idsse:")]
    if set(train_keys) & set(fold["test"]):
        raise ValueError("Outer-test leakage")
    views = {key: build_features(Match.load(records[key]["path"])) for key in train_keys}
    validation_key = fold["validation"][0]
    val_match = Match.load(records[validation_key]["path"])
    validation = build_features(val_match)
    context = ScoreContext(val_match, validation)
    argmax = TemporalDecoder({"kind": "argmax"}, None)
    priors = fit_priors([view for key, view in views.items() if key.startswith("idsse:")],
                        protocol["decoder_grid"]["hsmm_duration_cap"])
    np.savez_compressed(directory / "priors.npz", **priors)
    base_windows = Windows({k: v for k, v in views.items() if k.startswith("idsse:")}, config.core_frames, model.context)
    updates = int(np.ceil(len(base_windows) / config.batch_size))
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    val_loader = DataLoader(Windows({validation_key: validation}, config.core_frames, model.context),
                            batch_size=config.batch_size, collate_fn=collate)
    history, caps, best_loss, stop_best, stale, best_epoch = [], [], np.inf, np.inf, 0, 0
    settings = protocol["early_stopping"]
    beginning = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        epoch_views = views
        if condition.endswith("blocks"):
            length = protocol["training_corruption"]["block_seconds_choices"][(epoch - 1) % 3]
            epoch_views = {key: hide_ball(view, 0.4, length, stable_seed(key, seed + 1000 * epoch))[0]
                           for key, view in views.items()}
        dataset = Windows(epoch_views, config.core_frames, model.context)
        if condition.startswith("B"):
            loader = DataLoader(dataset, batch_sampler=FixedSourceBatches(dataset, updates, config.batch_size,
                                seed + epoch), collate_fn=collate)
        else:
            loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                                generator=generator, collate_fn=collate)
        training = epoch_loss(model, loader, torch.device("cpu"), optimizer, scaler)
        validation_loss = epoch_loss(model, val_loader, torch.device("cpu"))
        history.append({"epoch": epoch, "train": training, "validation": validation_loss,
                        "elapsed_s": time.perf_counter() - epoch_start})
        if validation_loss["loss"] < best_loss:
            best_loss, best_epoch = validation_loss["loss"], epoch
            atomic_checkpoint(directory / "best.pt", {"model": model.state_dict(), "config": asdict(config),
                "epoch": epoch, "fingerprint": fingerprint, "optimizer": optimizer.state_dict(),
                "rng": torch.get_rng_state(), "scorer_sha256": file_hash(SCORER_PATH)})
        if validation_loss["loss"] < stop_best - settings["min_delta"]:
            stop_best, stale = validation_loss["loss"], 0
        else:
            stale += 1
        stop = epoch >= min(max_epochs, settings["min_epochs"]) and stale >= settings["patience"]
        cap = epoch if epoch in protocol["epoch_caps"] else max_epochs if stop else None
        if cap is not None:
            random_state = torch.get_rng_state()
            saved = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
            selected_model = new_model(config)
            selected_model.load_state_dict(saved["model"])
            probabilities = predict(selected_model, validation, config.core_frames, torch.device("cpu"))
            path = directory / ("cap-" + str(cap) + ".pt")
            atomic_checkpoint(path, saved)
            save_probabilities(directory / ("validation-cap-" + str(cap) + ".npz"), validation, probabilities)
            caps.append({"budget": cap, "epochs_executed": epoch, "selected_epoch": best_epoch,
                         "validation_loss": best_loss, "validation_spellF1": context.spell(probabilities, argmax),
                         "checkpoint": path.name})
            torch.set_rng_state(random_state)
        # Restore training mode on next epoch via epoch_loss; predict used a separate model.
        if stop:
            break
    if not any(row["budget"] == max_epochs for row in caps):
        saved = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
        selected_model = new_model(config)
        selected_model.load_state_dict(saved["model"])
        p = predict(selected_model, validation, config.core_frames, torch.device("cpu"))
        atomic_checkpoint(directory / ("cap-" + str(max_epochs) + ".pt"), saved)
        save_probabilities(directory / ("validation-cap-" + str(max_epochs) + ".npz"), validation, p)
        caps.append({"budget": max_epochs, "epochs_executed": len(history), "selected_epoch": best_epoch,
            "validation_loss": best_loss, "validation_spellF1": context.spell(p, argmax),
            "checkpoint": "cap-" + str(max_epochs) + ".pt"})
    chosen = max(caps, key=lambda row: (row["validation_spellF1"], -row["budget"]))
    result = {"fingerprint": fingerprint, "caps": caps, "chosen": chosen, "history": history,
        "total_s": time.perf_counter() - beginning, "updates_per_epoch": updates,
        "train_keys": train_keys, "validation_key": validation_key, "memory": memory_usage(torch.device("cpu"))}
    save_json(completion, result)
    return result


def load_study_model(directory, cap):
    state = torch.load(Path(directory) / ("cap-" + str(cap) + ".pt"), map_location="cpu", weights_only=False)
    model = new_model(TrainConfig.from_dict(state["config"]))
    model.load_state_dict(state["model"])
    return model, state
