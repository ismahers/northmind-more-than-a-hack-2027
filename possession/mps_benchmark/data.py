"""Public training inputs only. No discovery, download or historical writes."""
from dataclasses import fields
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from possession.external_data.features import Features, build_features
from possession.external_data.schema import Match, file_hash
from possession.learning.corruption import hide_ball, stable_seed
from possession.learning.data import Windows
from possession.phase6.model import TransitionWindows, transition_targets
from possession.phase7.features import harmonize
from possession.phase8.packing import compact_collate, compact_transition_collate

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "evaluation/reports/external"


def public_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to((ROOT / "data/external").resolve()):
        raise ValueError("Input outside explicit public external root")
    return path


def synthetic(spec):
    rng = np.random.default_rng(spec["seed"])
    n = spec["synthetic_windows"] * spec["core_frames"]
    mask = np.zeros((n, 2, 18), bool)
    mask[..., :11] = rng.random((n, 2, 11)) > .04
    # Missing teams and completely empty frames exercise the real attention guard.
    mask[100:103] = False
    mask[200:203, 0] = False
    nodes = rng.uniform(-1, 1, (n, 2, 18, 6)).astype("f4")
    nodes[..., 4:] = 0
    nodes *= mask[..., None]
    global_features = rng.uniform(-1, 1, (n, 7)).astype("f4")
    global_features[:, [2, 4, 5]] = 0
    global_features[:, 3] = 1
    global_features[:, 6] = 0
    label = (np.arange(n) // 17) % 3
    alive = (label != 2).astype("i8")
    team = np.where(alive, label, -1).astype("i8")
    confidence = np.column_stack((np.ones(n), alive)).astype("f4")
    view = Features(nodes, mask, global_features, np.arange(n), np.zeros(n, "i4"),
                    np.arange(n) / 10, alive, team, confidence)
    target = np.zeros((n, 1), "f4")
    changes = np.flatnonzero((label[1:] != label[:-1]) & (label[1:] < 2) & (label[:-1] < 2)) + 1
    for j in changes:
        target[max(0, j-1):j+2, 0] = 1
    target *= alive[:, None]
    return {"synthetic": view}, {"synthetic": (target, np.ones(n, bool))}, {
        "source": "synthetic", "windows": spec["synthetic_windows"], "real_idsse": False}


def idsse(spec):
    marker = REPORTS / "folds" / (spec["manifest"] + ".json")
    if not marker.is_file():
        return None
    manifest = json.loads(marker.read_text())
    records = manifest["artifacts"]
    keys = sorted(records)
    if len(keys) != 7 or not all(k.startswith("idsse:") for k in keys):
        raise ValueError("Expected the seven pinned public IDSSE records")
    fold = spec["fold"]
    test = manifest["folds"][fold]["test"][0]
    validation = manifest["folds"][fold]["validation"][0]
    calibration = keys[(fold+1) % len(keys)]
    train = [k for k in keys if k not in (test, validation, calibration)]
    if len(train) != 4:
        raise ValueError("Strict training split required")
    paths = {k: public_path(records[k]["path"]) for k in train}
    if not all(p.is_file() for p in paths.values()):
        return None
    views, targets, hashes = {}, {}, {}
    for key, path in paths.items():
        if file_hash(path) != records[key]["sha256"]:
            raise ValueError("Public source hash mismatch")
        match = Match.load(path)
        if match.metadata["provider"] != "idsse" or match.key != key:
            raise ValueError("Public source identity mismatch")
        # Prefer existing read-only feature caches; never populate historical folders.
        view = None
        for version in ("phase8-v1", "phase7-v1"):
            cache = REPORTS / version / "feature_cache" / key.replace(":", "_")
            if (cache / "complete.json").is_file():
                record = json.loads((cache / "complete.json").read_text())
                if record.get("artifact") != records[key]["sha256"] or record.get("version") != "common-xy-v1":
                    raise ValueError("Public feature cache provenance mismatch")
                view = Features(**{f.name: np.load(cache / (f.name + ".npy"), mmap_mode="r", allow_pickle=False)
                                   for f in fields(Features)})
                break
        views[key] = view if view is not None else harmonize(build_features(match, 10.))
        targets[key] = transition_targets(match, views[key], 1, .12)
        hashes[key] = records[key]["sha256"]
        del match
    return views, targets, {"source": "IDSSE", "real_idsse": True, "fold": fold,
                            "train_keys": train, "source_sha256": hashes,
                            "manifest_sha256": file_hash(marker), "test_loaded": False}


def weights_for(targets, spec):
    positives = sum((y * mask[:, None]).sum(0) for y, mask in targets.values())
    n = sum(mask.sum() for _, mask in targets.values())
    return torch.tensor(np.minimum(spec["positive_weight_cap"], (n-positives)/np.maximum(1, positives)), dtype=torch.float32)


def loader_for(views, targets, spec, stage, corrupt=True):
    # First base/auxiliary epoch of the historical schedule: 1 s blocks, 40% nominal.
    epoch = 1 if stage == "base" else 13
    degraded = {k: hide_ball(v, .4, 1., stable_seed(k, spec["seed"]+1000*epoch))[0]
                for k, v in views.items()} if corrupt else views
    auxiliary = stage == "auxiliary"
    dataset = TransitionWindows(degraded, targets) if auxiliary else Windows(degraded, spec["core_frames"], spec["context"])
    generator = torch.Generator().manual_seed(spec["seed"] + (6000 if auxiliary else 0))
    return DataLoader(dataset, batch_size=spec["batch_size"], shuffle=True, generator=generator,
                      num_workers=0, collate_fn=compact_transition_collate if auxiliary else compact_collate)


def tensor_hash(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()
