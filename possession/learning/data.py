"""Run-local windows with a context halo; each central frame is supervised exactly once."""

import numpy as np
import torch
from torch.utils.data import Dataset


class Windows(Dataset):
    def __init__(self, features, core_frames, context):
        if core_frames < 1 or context < 0:
            raise ValueError("Invalid window configuration")
        self.features = features
        self.windows = []
        for key in sorted(features):
            view = features[key]
            for run in np.unique(view.run):
                positions = np.flatnonzero(view.run == run)
                start, stop = int(positions[0]), int(positions[-1]) + 1
                for a in range(start, stop, core_frames):
                    b = min(a + core_frames, stop)
                    self.windows.append((key, max(start, a - context), min(stop, b + context), a, b))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        key, left, right, a, b = self.windows[i]
        view = self.features[key]
        core = np.zeros(right - left, dtype=bool)
        core[a - left:b - left] = True
        return {"key": key, "left": left, "a": a, "b": b,
                "nodes": view.nodes[left:right], "node_mask": view.node_mask[left:right],
                "global_features": view.global_features[left:right], "core": core,
                "alive": view.alive[left:right], "team": view.team[left:right],
                "confidence": view.confidence[left:right]}


def collate(items):
    batch, length = len(items), max(len(x["nodes"]) for x in items)
    players = max(x["nodes"].shape[2] for x in items)
    result = {
        "nodes": torch.zeros(batch, length, 2, players, items[0]["nodes"].shape[-1]),
        "node_mask": torch.zeros(batch, length, 2, players, dtype=torch.bool),
        "global_features": torch.zeros(batch, length, items[0]["global_features"].shape[-1]),
        "valid": torch.zeros(batch, length, dtype=torch.bool),
        "core": torch.zeros(batch, length, dtype=torch.bool),
        "alive": torch.full((batch, length), -1, dtype=torch.int64),
        "team": torch.full((batch, length), -1, dtype=torch.int64),
        "confidence": torch.zeros(batch, length, 2),
    }
    for i, item in enumerate(items):
        n, p = item["nodes"].shape[0], item["nodes"].shape[2]
        result["nodes"][i, :n, :, :p] = torch.from_numpy(item["nodes"])
        result["node_mask"][i, :n, :, :p] = torch.from_numpy(item["node_mask"])
        result["valid"][i, :n] = True
        for field in ("global_features", "core", "alive", "team", "confidence"):
            result[field][i, :n] = torch.from_numpy(item[field])
    return result
