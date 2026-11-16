"""Sequence windows confined to contiguous runs, with identical core schedules."""
import numpy as np
import torch


TARGETS = ("label", "distance", "regression_weight", "regression_valid", "condition_weight")


class EmbeddingWindows:
    def __init__(self, records, core_frames=256, context=32):
        if core_frames < 1 or context < 0:
            raise ValueError("Invalid window size")
        self.records, self.windows = records, []
        for key in sorted(records):
            record = records[key]
            n = len(record["hidden"])
            if not n or record["hidden"].shape != (n, 64):
                raise ValueError("Invalid embedding shape")
            run, time = record["run"], record["time"]
            if run.shape != (n,) or time.shape != (n,):
                raise ValueError("Invalid run timeline")
            boundaries = np.r_[0, np.flatnonzero(run[1:] != run[:-1]) + 1, n]
            seen = set()
            for left, right in zip(boundaries[:-1], boundaries[1:]):
                if int(run[left]) in seen or np.any(np.diff(time[left:right]) <= 0):
                    raise ValueError("Runs must be contiguous with strictly increasing clocks")
                seen.add(int(run[left]))
                for a in range(int(left), int(right), core_frames):
                    b = min(a + core_frames, int(right))
                    self.windows.append((key, max(int(left), a-context), min(int(right), b+context), a, b))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        key, left, right, a, b = self.windows[index]
        record = self.records[key]
        core = np.zeros(right-left, bool)
        core[a-left:b-left] = True
        return {"key": key, "left": left, "a": a, "b": b, "core": core,
                "hidden": record["hidden"][left:right],
                "targets": {name: record[name][left:right] for name in TARGETS if name in record}}


def collate(items):
    n, length = len(items), max(len(item["hidden"]) for item in items)
    hidden = torch.zeros(n, length, 64)
    core = torch.zeros(n, length, dtype=torch.bool)
    targets = {}
    for row, item in enumerate(items):
        size = len(item["hidden"])
        hidden[row, :size] = torch.from_numpy(np.asarray(item["hidden"]).copy())
        core[row, :size] = torch.from_numpy(item["core"])
        for name, array in item["targets"].items():
            value = torch.from_numpy(np.asarray(array).copy())
            if name not in targets:
                targets[name] = torch.zeros((n, length, *value.shape[1:]), dtype=value.dtype)
            targets[name][row, :size] = value
    return hidden, core, targets


def predict(model, data, core_frames=256, batch_size=8):
    """Inference reads only embeddings and clocks, never target fields."""
    inputs = {name: data[name] for name in ("hidden", "time", "run")}
    windows = EmbeddingWindows({"input": inputs}, core_frames, model.context)
    team = np.empty(len(data["hidden"]), np.float32)
    distance = np.empty((len(team), 2, 2), np.float32) if model.intervals else None
    quality = np.empty((len(team), 2), np.float32) if model.intervals else None
    model.eval()
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            items = [windows[i] for i in range(start, min(start+batch_size, len(windows)))]
            hidden, _, _ = collate(items)
            result = model(hidden)
            for row, item in enumerate(items):
                a, b, left = item["a"], item["b"], item["left"]
                team[a:b] = result[0][row, a-left:b-left].numpy()
                if model.intervals:
                    distance[a:b] = result[1][row, a-left:b-left].numpy()
                    quality[a:b] = result[2][row, a-left:b-left].numpy()
    return team, distance, quality


def emissions(team, teacher_p):
    h = 1 / (1 + np.exp(-np.clip(team, -30, 30)))
    alive = np.clip(teacher_p[:, :2].sum(1), 0, 1)
    return np.column_stack((alive*h, alive*(1-h), 1-alive)).astype(np.float32)
