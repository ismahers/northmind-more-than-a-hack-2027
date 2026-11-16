"""Own immutable record and budget; all historical experiment artifacts read only."""
from contextlib import contextmanager
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from possession.external_data.features import build_features
from possession.external_data.schema import file_hash
from possession.night.cache import PublicStore
from possession.night.common import atomic_json
from possession.phase7.features import harmonize

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "evaluation/reports/external"
DEST = REPORTS / "temporal-probe-v1"
PROTOCOL = ROOT / "configs/temporal_probe_protocol.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def public_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to((ROOT / "data/external").resolve()):
        raise ValueError("Only explicit public external data paths allowed")
    return path


def splits_valid(parts):
    train = parts["train"]
    keys = train + [parts[name] for name in ("calibration", "validation", "test")]
    return len(train) == 4 and len(set(keys)) == 7 and all(k.startswith("idsse:") for k in keys)


def register():
    config = read_json(PROTOCOL)
    manifest_path = REPORTS / "folds/idsse-only.json"
    manifest = read_json(manifest_path)
    records = manifest["artifacts"]
    if len(records) != 7 or not all(k.startswith("idsse:") for k in records):
        raise ValueError("Exactly seven public IDSSE records required")
    for record in records.values():
        public_path(record["path"])
    fresh = config["teacher_source"] == "fresh_public_pilots"
    parent = REPORTS / ("temporal-teachers-v1/registration.json" if fresh else "phase8-v1/registration.json")
    parent_spec = read_json(parent)["protocol"]
    if parent_spec["seeds"] != [2027, 2028] or parent_spec["scenarios"] != config["scenarios"]:
        raise ValueError("Unexpected teacher protocol")
    teachers, references = {}, {str(parent.relative_to(ROOT)): file_hash(parent),
                                str(manifest_path.relative_to(ROOT)): file_hash(manifest_path)}
    from possession.phase8.training import source_folder
    for fold in config["pilots"]:
        if fresh:
            path = REPORTS / f"temporal-teachers-v1/selections/fold-{fold}.json"
            selected = read_json(path)
            if selected["test_used"] or not splits_valid(selected["teacher"]["split"]):
                raise ValueError("Invalid fresh teacher provenance")
            references[str(path.relative_to(ROOT))] = file_hash(path)
            teachers[str(fold)] = selected["teacher"]
            for seed in parent_spec["seeds"]:
                checkpoint = REPORTS / f"temporal-teachers-v1/training/fold-{fold}/seed-{seed}/binary/last.pt"
                references[str(checkpoint.relative_to(ROOT))] = file_hash(checkpoint)
            continue
        choices = []
        for seed in parent_spec["seeds"]:
            path = REPORTS / f"phase8-v1/selections/IDSSE/fold-{fold}/seed-{seed}/set_transformer.json"
            value = read_json(path)
            if value["outer_test_used"] or not splits_valid(value["split"]):
                raise ValueError("Invalid teacher selection provenance")
            if value["split"]["test"] != manifest["folds"][fold]["test"][0]:
                raise ValueError("Teacher outer fold mismatch")
            references[str(path.relative_to(ROOT))] = file_hash(path)
            choices.append(value)
        winner = max(choices, key=lambda value: value["winner"]["objective"])
        checkpoint = source_folder(fold, winner["seed"], "set_transformer") / "binary/last.pt"
        references[str(checkpoint.relative_to(ROOT))] = file_hash(checkpoint)
        teachers[str(fold)] = {"seed": winner["seed"], "split": winner["split"],
                               "checkpoint": str(checkpoint)}
    # Freeze all imported algorithm sources without calling mutable old register()/Budget().
    sources = {}
    for directory in ("temporal_probe", "interval_probe", "phase8", "phase7", "phase6",
                      "phase5", "learning", "external_data", "night"):
        for path in sorted((ROOT / "possession" / directory).glob("*.py")):
            sources[str(path.relative_to(ROOT))] = file_hash(path)
    for name in ("scripts/temporal_probe.py", "evaluation/possession_metrics.py"):
        sources[name] = file_hash(ROOT / name)
    record = {"protocol": config, "teacher_protocol": parent_spec, "teachers": teachers,
              "sources": sources, "references": references,
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "torch": torch.__version__}}
    marker = DEST / "registration.json"
    if marker.exists() and read_json(marker) != record:
        raise ValueError("Immutable temporal probe changed; use another version")
    if not marker.exists():
        atomic_json(marker, record)
    return record, records


class Store(PublicStore):
    def __init__(self, records):
        super().__init__(records)
        self.common, self.checked = {}, set()

    def match(self, key):
        if key not in self.checked:
            path = public_path(self.records[key]["path"])
            if file_hash(path) != self.records[key]["sha256"]:
                raise ValueError("Public source hash mismatch")
            self.checked.add(key)
        return super().match(key)

    def harmonized(self, key):
        if key not in self.common:
            self.common[key] = harmonize(build_features(self.match(key)))
        return self.common[key]


@contextmanager
def exclusive_run():
    DEST.mkdir(parents=True, exist_ok=True)
    path = DEST / "running.lock"
    # A killed process leaves an explicit lock for local review; never steal a lock.
    with path.open("x", encoding="utf-8") as stream:
        stream.write("One temporal-probe process at a time.\n")
    try:
        yield
    finally:
        path.unlink()


class Budget:
    def __init__(self, limit, directory=DEST):
        self.path = directory / "compute.json"
        self.saved = read_json(self.path) if self.path.exists() else {"seconds": 0., "sessions": []}
        self.start, self.limit = time.perf_counter(), limit

    def check(self):
        if self.saved["seconds"] + time.perf_counter() - self.start >= self.limit:
            raise TimeoutError("Temporal probe budget exhausted; checkpoints preserved")

    def close(self, stage, status):
        elapsed = time.perf_counter() - self.start
        self.saved["seconds"] += elapsed
        self.saved["sessions"].append({"stage": stage, "status": status, "seconds": elapsed})
        atomic_json(self.path, self.saved)
