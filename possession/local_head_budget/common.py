"""New immutable registration; never call a historical mutable register/audit."""
import json
import platform
from pathlib import Path

import numpy as np
import torch

from possession.external_data.schema import file_hash
from possession.night.common import atomic_json
from possession.temporal_probe.common import ROOT, REPORTS, read_json, public_path, splits_valid
from possession.duration_balanced.common import SCORER_SHA

DEST = REPORTS / "local-head-budget-phase8-v2"
PARENT = REPORTS / "duration-balanced-phase8-v1"


def report_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(REPORTS.resolve()):
        raise ValueError("Artifact outside explicit public report root")
    return path


def parent_checkpoint(fold):
    return PARENT / f"training/fold-{fold}/original/last.pt"


def cap_path(fold, cap):
    return parent_checkpoint(fold) if cap == 8 else DEST / f"training/fold-{fold}/cap-{cap}.pt"


def register():
    spec = read_json(ROOT / "configs/local_head_budget_v1.json")
    spec.update(read_json(ROOT / "configs/local_head_budget_v2.json"))
    parent_path = PARENT / "registration.json"
    if not parent_path.is_file():
        raise ValueError("Existing Phase8 local heads required; no bootstrap allowed")
    parent = read_json(parent_path)
    if parent["source"] != "phase8" or not read_json(PARENT / "verification.json")["passed"]:
        raise ValueError("Verified Phase8 parent required")
    for name in ("pilots", "seed", "batch_size", "core_frames", "training_halo_frames", "dilations", "kernel_size",
                 "learning_rate", "weight_decay", "gradient_clip", "cpu_threads", "scenarios", "score_weights", "condition_weights"):
        if spec[name] != parent["protocol"][name]:
            raise ValueError("Continuation recipe differs from original local head")
    for name, sha in parent["sources"].items():
        path = (ROOT / name.replace(chr(92), "/")).resolve()
        if not path.is_relative_to(ROOT) or file_hash(path) != sha:
            raise ValueError("Historical algorithm source changed")
    references = {str(parent_path.relative_to(ROOT)): file_hash(parent_path)}
    for name, sha in parent["references"].items():
        path = (ROOT / name.replace(chr(92), "/")).resolve()
        if not (path.is_relative_to(REPORTS.resolve()) or path.is_relative_to((ROOT / "data/external").resolve())):
            raise ValueError("Nonpublic historical reference rejected")
        if file_hash(path) != sha:
            raise ValueError("Historical artifact hash mismatch")
        references[str(path.relative_to(ROOT))] = sha
    for relative in ("selection-seal.json", "verification.json"):
        references[str((PARENT / relative).relative_to(ROOT))] = file_hash(PARENT / relative)
    for record in parent["records"].values():
        public_path(record["path"])
    audit = {}
    for fold in spec["pilots"]:
        teacher = parent["teachers"][str(fold)]
        if teacher["seed"] != spec["teacher_seeds"][str(fold)] or not splits_valid(teacher["split"]):
            raise ValueError("Teacher seed or split differs")
        report_path(teacher["checkpoint"]); report_path(teacher["export_root"])
        checkpoint = parent_checkpoint(fold)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        required = ("model", "optimizer", "loader_rng", "torch_rng", "numpy_rng", "python_rng", "history", "registration_sha256")
        if any(k not in state for k in required) or len(state["history"]) != 8:
            raise ValueError("Exact full-state eight-epoch parent missing; no silent restart")
        if state["registration_sha256"] != file_hash(parent_path):
            raise ValueError("Parent checkpoint registration mismatch")
        group = state["optimizer"]["param_groups"][0]
        if group["lr"] != spec["learning_rate"] or group["weight_decay"] != spec["weight_decay"]:
            raise ValueError("Parent AdamW recipe mismatch")
        if not state["optimizer"]["state"]:
            raise ValueError("Parent AdamW moments missing")
        references[str(checkpoint.relative_to(ROOT))] = file_hash(checkpoint)
        audit[str(fold)] = {"resumable_exactly": True, "epochs": 8,
                            "steps_per_epoch": [r["steps"] for r in state["history"]],
                            "teacher_seed": teacher["seed"], "head_seed": spec["seed"],
                            "optimizer_state_entries": len(state["optimizer"]["state"])}
    sources = {}
    for directory in ("local_head_budget", "duration_balanced", "temporal_probe", "interval_probe", "phase8",
                      "phase7", "phase6", "phase5", "learning", "external_data", "night"):
        for path in sorted((ROOT / "possession" / directory).glob("*.py")):
            sources[str(path.relative_to(ROOT))] = file_hash(path)
    for relative in ("configs/local_head_budget_v1.json", "configs/local_head_budget_v2.json", "configs/duration_balanced_protocol.json",
                     "configs/phase8_protocol.json", "scripts/local_head_budget.py", "evaluation/possession_metrics.py"):
        sources[relative] = file_hash(ROOT / relative)
    if sources["evaluation/possession_metrics.py"] != SCORER_SHA:
        raise ValueError("Official evaluator changed")
    record = {"protocol": spec, "teachers": parent["teachers"], "records": parent["records"],
              "parent_registration_sha256": file_hash(parent_path), "references": references,
              "sources": sources, "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                                         "torch": str(torch.__version__), "device": "cpu",
                                                         "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}, "audit": audit}
    marker = DEST / "registration.json"
    if marker.exists() and read_json(marker) != record:
        raise ValueError("Immutable continuation changed; use a new version")
    if not marker.exists():
        atomic_json(marker, record)
    atomic_json(DEST / "audit.json", {"folds": audit, "teacher_training": 0, "new_embedding_exports": 0,
                                      "historical_sources_and_references_verified": True})
    return record
