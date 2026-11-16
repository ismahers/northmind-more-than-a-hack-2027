"""Reusable mmap features and bounded native-match cache; no private assets."""

from collections import OrderedDict
from dataclasses import fields
import json
from pathlib import Path
import time

import numpy as np
import torch

from possession.external_data.features import Features, build_features
from possession.external_data.schema import Match, file_hash
from possession.learning.data import Windows, collate
from possession.learning.training import forward
from .common import DEST, atomic_json


class PublicStore:
    def __init__(self, records):
        self.records, self.matches, self.features = records, OrderedDict(), {}
        self.times = {"native_load_s": 0., "feature_build_s": 0., "cache_load_s": 0.}

    def match(self, key):
        if not key.startswith(("idsse:", "skillcorner:")):
            raise ValueError("Only public external providers are allowed")
        if key not in self.matches:
            start = time.perf_counter()
            value = Match.load(self.records[key]["path"])
            if value.key != key:
                raise ValueError("Public artifact identity mismatch")
            self.times["native_load_s"] += time.perf_counter() - start
            self.matches[key] = value
            while len(self.matches) > 2:
                self.matches.popitem(last=False)
        self.matches.move_to_end(key)
        return self.matches[key]

    def view(self, key):
        if key in self.features:
            return self.features[key]
        folder = DEST / "feature_cache" / key.replace(":", "_")
        metadata = folder / "manifest.json"
        stamp = {"artifact": self.records[key]["sha256"],
                 "features_code": file_hash(Path(__file__).resolve().parents[1] / "external_data/features.py")}
        if not metadata.exists():
            start = time.perf_counter()
            view = build_features(self.match(key))
            self.times["feature_build_s"] += time.perf_counter() - start
            folder.mkdir(parents=True, exist_ok=True)
            for field in fields(Features):
                np.save(folder / (field.name + ".npy"), getattr(view, field.name), allow_pickle=False)
            atomic_json(metadata, stamp)
        elif json.loads(metadata.read_text()) != stamp:
            raise ValueError("Cached features have different provenance")
        start = time.perf_counter()
        # Copy-on-write maps remain writable to torch.from_numpy without changing disk.
        view = Features(**{f.name: np.load(folder / (f.name + ".npy"), mmap_mode="c", allow_pickle=False)
                           for f in fields(Features)})
        self.features[key] = view
        self.times["cache_load_s"] += time.perf_counter() - start
        return view


def predict_batched(model, view, core_frames=256, batch_size=8):
    dataset = Windows({"match": view}, core_frames, model.context)
    values = np.zeros((len(view.time), 3), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            items = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            output = forward(model, collate(items))["probabilities"].numpy()
            for item, p in zip(items, output):
                left, a, b = item["left"], item["a"], item["b"]
                values[a:b] = p[a - left:b - left]
    return values
