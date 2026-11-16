"""Public-external-only paths, immutable protocol, cumulative compute budget."""

import atexit
from contextlib import contextmanager
import json
from pathlib import Path
import time

from possession.external_data.schema import file_hash
from possession.learning.study import save_json
from scripts.external_data import ROOT, REPORTS

OLD = REPORTS / "phase3"
DEST = REPORTS / "night-2026-10-22-v1"
PROTOCOL = ROOT / "configs/night_protocol.json"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    save_json(temporary, value)
    temporary.replace(path)


def register():
    protocol = json.loads(PROTOCOL.read_text())
    frozen = DEST / "protocol.json"
    if frozen.exists():
        if json.loads(frozen.read_text()) != protocol:
            raise ValueError("Night protocol changed: preserve this run and use a new version")
    else:
        atomic_json(frozen, protocol)
    return protocol


class BudgetStop(Exception):
    pass


class Budget:
    def __init__(self, protocol):
        self.path = DEST / "compute.json"
        self.saved = json.loads(self.path.read_text()) if self.path.exists() else {"seconds": 0., "sessions": []}
        self.start = time.perf_counter()
        self.limit = protocol["compute_budget_seconds"]
        self.reserve = protocol["report_reserve_seconds"]
        self.closed = False

    def check(self):
        if self.saved["seconds"] + time.perf_counter() - self.start >= self.limit - self.reserve:
            raise BudgetStop("Compute budget reached; completed units and per-epoch checkpoints preserved")

    def close(self, stage, status):
        if not self.closed:
            elapsed = time.perf_counter() - self.start
            self.saved["seconds"] += elapsed
            self.saved["sessions"].append({"stage": stage, "status": status, "elapsed_s": elapsed})
            atomic_json(self.path, self.saved)
            self.closed = True


def progress(stage, **fields):
    atomic_json(DEST / "progress.json", {"stage": stage, **fields})


def source_hashes(names):
    return {name: file_hash(ROOT / name) for name in names}


def objective(scores, protocol):
    return sum(weight * scores[name] for name, weight in protocol["score_weights"].items())


def robust_eligible(scores, reference, protocol):
    return all(scores[name] >= reference[name] + protocol["robustness_floor_delta"]
               for name in protocol["score_weights"] if name != "clean")
