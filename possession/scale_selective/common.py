"""Immutable registration and read-only provenance checks for scale-selective fusion."""
from contextlib import contextmanager
from pathlib import Path
import platform

import numpy as np
import torch

from possession.external_data.schema import file_hash
from possession.night.common import atomic_json
from possession.temporal_probe.common import (
    ROOT, REPORTS, Store, Budget, read_json, public_path, splits_valid,
)

TEACHERS_DIR = REPORTS / "temporal-teachers-v1"
PROBE_DIR = REPORTS / "temporal-probe-v1"
DEST = REPORTS / "scale-selective-v1"
PROTOCOL = ROOT / "configs/scale_selective_protocol.json"


@contextmanager
def exclusive_run():
    DEST.mkdir(parents=True, exist_ok=True)
    lock = DEST / "running.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write("One scale-selective process at a time.\n")
    try:
        yield
    finally:
        lock.unlink()


def register():
    config = read_json(PROTOCOL)
    manifest_path = REPORTS / "folds/idsse-only.json"
    manifest = read_json(manifest_path)
    records = manifest["artifacts"]
    if len(records) != 7 or not all(k.startswith("idsse:") for k in records):
        raise ValueError("Seven explicit public IDSSE records required")
    for record in records.values():
        public_path(record["path"])

    teacher_reg_path = TEACHERS_DIR / "registration.json"
    probe_reg_path = PROBE_DIR / "registration.json"
    teacher_reg = read_json(teacher_reg_path)
    probe_reg = read_json(probe_reg_path)

    references = {
        str(manifest_path.relative_to(ROOT)): file_hash(manifest_path),
        str(teacher_reg_path.relative_to(ROOT)): file_hash(teacher_reg_path),
        str(probe_reg_path.relative_to(ROOT)): file_hash(probe_reg_path),
    }
    teachers = {}
    alt_teachers = {}
    for fold in config["pilots"]:
        sel_path = TEACHERS_DIR / f"selections/fold-{fold}.json"
        sel = read_json(sel_path)
        if sel["test_used"] or not splits_valid(sel["teacher"]["split"]):
            raise ValueError("Invalid teacher selection provenance")
        references[str(sel_path.relative_to(ROOT))] = file_hash(sel_path)
        primary = sel["teacher"]
        teachers[str(fold)] = primary
        alt_seed = 2027 if primary["seed"] == 2028 else 2028
        alt_ckpt = TEACHERS_DIR / f"training/fold-{fold}/seed-{alt_seed}/binary/last.pt"
        if not alt_ckpt.is_file():
            raise ValueError(f"Missing alternate teacher checkpoint for fold {fold}")
        alt_teachers[str(fold)] = {
            "seed": alt_seed,
            "split": primary["split"],
            "checkpoint": str(alt_ckpt),
            "source": "fresh_public_pilots",
        }
        for seed in (2027, 2028):
            ckpt = TEACHERS_DIR / f"training/fold-{fold}/seed-{seed}/binary/last.pt"
            references[str(ckpt.relative_to(ROOT))] = file_hash(ckpt)
        for arm in ("local_classification", "multiscale_classification"):
            head_ckpt = PROBE_DIR / f"training/fold-{fold}/{arm}/last.pt"
            references[str(head_ckpt.relative_to(ROOT))] = file_hash(head_ckpt)

    sources = {}
    for directory in ("scale_selective", "temporal_probe", "local_head_budget",
                      "duration_balanced", "phase8", "phase7", "phase6", "phase5",
                      "learning", "external_data", "night"):
        folder = ROOT / "possession" / directory
        if folder.is_dir():
            for path in sorted(folder.glob("*.py")):
                sources[str(path.relative_to(ROOT))] = file_hash(path)
    for name in ("scripts/scale_selective.py", "evaluation/possession_metrics.py"):
        if (ROOT / name).is_file():
            sources[name] = file_hash(ROOT / name)

    record = {
        "protocol": config,
        "teacher_protocol": teacher_reg["protocol"],
        "probe_protocol": probe_reg["protocol"],
        "teachers": teachers,
        "alt_teachers": alt_teachers,
        "sources": sources,
        "references": references,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    marker = DEST / "registration.json"
    if marker.exists() and read_json(marker) != record:
        raise ValueError("Immutable scale-selective registration changed; use a new version")
    if not marker.exists():
        atomic_json(marker, record)
    return record, records

