"""Small reproducible E0 trainer. Validation only for selection; test is an explicit later action."""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from possession.external_data.features import FEATURE_VERSION, build_features, nearest_indices
from possession.external_data.audit import segments
from possession.external_data.schema import Match, file_hash
from .data import Windows, collate
from .metrics import SCORER_PATH, score_match
from .model import E0, masked_losses
from .protocol import Fold, inventory


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 2027
    fps: float = 10.0
    core_frames: int = 256
    batch_size: int = 4
    epochs: int = 2
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    spatial_dim: int = 32
    channels: int = 64
    dilations: tuple = (1, 2, 4, 8)
    dropout: float = 0.1
    accumulation_steps: int = 1
    amp: bool = False
    device: str = "auto"
    cpu_threads: int = 4
    max_train_batches: int | None = None

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if "dilations" in values:
            values["dilations"] = tuple(values["dilations"])
        result = cls(**values)
        if min(result.core_frames, result.batch_size, result.epochs, result.accumulation_steps) < 1:
            raise ValueError("Positive training counts required")
        if result.cpu_threads < 1 or (result.max_train_batches is not None and result.max_train_batches < 1):
            raise ValueError("Positive thread and optional pilot batch limits required")
        return result


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False


def select_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested not in ("cpu", "cuda"):
        raise ValueError("Supported devices: auto, cpu, cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(requested)


def new_model(config):
    return E0(config.spatial_dim, config.channels, config.dilations, config.dropout)


def implementation_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "possession" / "external_data").glob("*.py"))
    paths += sorted((root / "possession" / "learning").glob("*.py"))
    paths.append(root / "configs" / "external_sources.json")
    return {str(path.relative_to(root)): file_hash(path) for path in paths}


def forward(model, batch):
    return model(batch["nodes"], batch["node_mask"], batch["global_features"], batch["valid"])


def epoch_loss(model, loader, device, optimizer=None, scaler=None, accumulation=1, amp=False, max_batches=None):
    model.train(optimizer is not None)
    totals = np.zeros(4, dtype=np.float64)
    batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    grad_norms = []
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(loader):
        if i >= batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(optimizer is not None), torch.autocast(device_type=device.type, enabled=amp):
            output = forward(model, batch)
            loss, stats = masked_losses(output, batch["alive"], batch["team"], batch["confidence"], batch["core"])
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        totals += stats.cpu().numpy()
        if optimizer is not None:
            group_start = (i // accumulation) * accumulation
            group_size = min(accumulation, batches - group_start)
            scaler.scale(loss / group_size).backward()
            if (i + 1) % accumulation == 0 or i + 1 == batches:
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                grad_norms.append(float(norm))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
    if not totals[1] and not totals[3]:
        raise ValueError("No valid supervision in this split")
    return {"loss": float(totals[0] / max(1, totals[1]) + totals[2] / max(1, totals[3])),
            "state_weight": float(totals[1]), "team_weight": float(totals[3]),
            "batches": batches, "optimizer_steps": len(grad_norms),
            "gradient_norm_mean": float(np.mean(grad_norms)) if grad_norms else None,
            "gradient_norm_max": max(grad_norms) if grad_norms else None}


def memory_usage(device):
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = int(rss if platform.system() == "Darwin" else rss * 1024)
    except ImportError:
        peak_bytes = None
    return {"process_peak_rss_bytes": peak_bytes,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}


def predict(model, features, core_frames, device):
    """Halo inference matches full-run TCN inference up to floating-point roundoff."""
    dataset = Windows({"match": features}, core_frames, model.context)
    values = np.zeros((len(features.time), 3), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for item in dataset:
            batch = {k: v.to(device) for k, v in collate([item]).items()}
            result = forward(model, batch)["probabilities"][0].cpu().numpy()
            left, a, b = item["left"], item["a"], item["b"]
            values[a:b] = result[a - left:b - left]
    return values


def atomic_checkpoint(path, state):
    temporary = Path(str(path) + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def train(paths, fold, config, output_dir, resume=None):
    config = TrainConfig.from_dict(asdict(config))
    torch.set_num_threads(config.cpu_threads)
    seed_everything(config.seed)
    device = select_device(config.device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = inventory(paths)
    fold.validate(records)
    # The test artifact is hashed for provenance but is NEVER featured, trained on or selected on here.
    preparation_start = time.perf_counter()
    views = {}
    for key in (*fold.train, *fold.validation):
        match = Match.load(records[key]["path"])
        views[key] = build_features(match, config.fps)
        del match
    preparation_s = time.perf_counter() - preparation_start
    model = new_model(config).to(device)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(Windows({k: views[k] for k in fold.train}, config.core_frames, model.context),
                              batch_size=config.batch_size, shuffle=True, generator=generator, collate_fn=collate)
    val_loader = DataLoader(Windows({k: views[k] for k in fold.validation}, config.core_frames, model.context),
                            batch_size=config.batch_size, shuffle=False, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    provenance = {"artifacts": records, "fold": fold.to_dict(), "feature_version": FEATURE_VERSION,
                  "scorer_sha256": file_hash(SCORER_PATH), "implementation_sha256": implementation_hashes(),
                  "pretrained": None, "normalization": "fixed per-pitch metric scaling; no fitted statistics"}
    start, best, history, telemetry = 0, float("inf"), [], []
    if resume:
        # Only load trusted checkpoints produced by this trainer (optimizer/RNG use Python objects).
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        old_config, requested = dict(saved["config"]), asdict(config)
        for field in ("epochs", "device"):
            old_config.pop(field)
            requested.pop(field)
        if old_config != requested or saved["provenance"] != provenance:
            raise ValueError("Resume requires identical data, split, features and training configuration")
        if saved["amp_enabled"] != amp:
            raise ValueError("Cannot resume across AMP modes; use the same device mode")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        start, best, history = saved["epoch"], saved["best_validation_loss"], saved["history"]
        telemetry = saved.get("telemetry", [])
        generator.set_state(saved["loader_rng"])
        torch.set_rng_state(saved["torch_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        if device.type == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    if start >= config.epochs:
        raise ValueError("Requested epochs must extend the checkpoint")
    for epoch in range(start, config.epochs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        epoch_start = time.perf_counter()
        training = epoch_loss(model, train_loader, device, optimizer, scaler, config.accumulation_steps, amp,
                              config.max_train_batches)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_s = time.perf_counter() - epoch_start
        validation_start = time.perf_counter()
        validation = epoch_loss(model, val_loader, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        telemetry.append({"epoch": epoch + 1, "train_s": training_s,
                          "validation_s": time.perf_counter() - validation_start,
                          "feature_preparation_s_this_session": preparation_s,
                          "train_windows_total": len(train_loader.dataset),
                          "validation_windows_total": len(val_loader.dataset),
                          "bounded_pilot": config.max_train_batches is not None,
                          **memory_usage(device)})
        history.append({"epoch": epoch + 1, "train": training, "validation": validation})
        improved = validation["loss"] < best
        best = min(best, validation["loss"])
        state = {"format_version": 1, "epoch": epoch + 1, "config": asdict(config),
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                 "amp_enabled": amp, "best_validation_loss": best, "history": history,
                 "telemetry": telemetry,
                 "provenance": provenance, "loader_rng": generator.get_state(),
                 "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                 "python_rng": random.getstate(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                 "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                                 "numpy": np.__version__, "device": str(device),
                                 "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None}}
        atomic_checkpoint(output_dir / "last.pt", state)
        if improved:
            atomic_checkpoint(output_dir / "best.pt", state)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (output_dir / "telemetry.json").write_text(json.dumps(telemetry, indent=2), encoding="utf-8")
    return history


def evaluate_checkpoint(paths, checkpoint, split="validation", output_dir=None, allow_partial=False):
    if split not in ("validation", "test"):
        raise ValueError("Evaluation split must be validation or test")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    records = inventory(paths)
    if records != state["provenance"]["artifacts"] or file_hash(SCORER_PATH) != state["provenance"]["scorer_sha256"]:
        raise ValueError("Evaluation inputs/scorer differ from checkpoint provenance")
    if implementation_hashes() != state["provenance"]["implementation_sha256"]:
        raise ValueError("Implementation differs from checkpoint; reproduce with the recorded code")
    config = TrainConfig.from_dict(state["config"])
    torch.set_num_threads(config.cpu_threads)
    device = select_device(config.device)
    model = new_model(config).to(device)
    model.load_state_dict(state["model"])
    results = {}
    for key in state["provenance"]["fold"][split]:
        match = Match.load(records[key]["path"])
        view = build_features(match, config.fps)
        probabilities = predict(model, view, config.core_frames, device)
        results[key] = score_match(match, view, probabilities, require_complete=not allow_partial)
        values = probabilities.astype(np.float64)
        results[key]["probability_diagnostics"] = {
            "mean_HAD": values.mean(0).tolist(), "min_HAD": values.min(0).tolist(),
            "max_HAD": values.max(0).tolist(), "mean_entropy_nats": float(-(values * np.log(values.clip(1e-12))).sum(-1).mean()),
            "predicted_model_frames_HAD": np.bincount(values.argmax(-1), minlength=3).tolist()}
        results[key]["experiment"] = {"fold": state["provenance"]["fold"], "config": state["config"],
                                      "selected_epoch": state["epoch"], "split": split,
                                      "checkpoint_sha256": file_hash(checkpoint),
                                      "data_sources": match.metadata["sources"]}
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            native_probabilities = np.zeros((len(match.frame), 3), dtype=np.float32)
            for run_id, (a, b) in enumerate(segments(match)):
                model_ix = np.flatnonzero(view.run == run_id)
                native_probabilities[a:b] = probabilities[model_ix[nearest_indices(view.time[model_ix], match.time[a:b])]]
            np.savez_compressed(Path(output_dir) / (key.replace(":", "_") + "_probabilities.npz"),
                probabilities=probabilities, time=view.time, native_index=view.native_index, run=view.run,
                native_probabilities=native_probabilities, native_frame=match.frame,
                native_period=match.period, native_time=match.time)
    if output_dir:
        Path(output_dir, "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
