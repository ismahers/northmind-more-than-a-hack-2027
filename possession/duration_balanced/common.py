"""Explicit parent audit; never bootstrap teachers or modify parent artifacts."""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import torch
from possession.external_data.schema import file_hash
from possession.night.common import atomic_json
from possession.temporal_probe.common import ROOT, REPORTS, read_json, public_path, splits_valid

SCORER_SHA = "341bad48904b489d6e875397b27687f5d6a4d22a397f3dc8af3a00d48f16874d"


def destination(source="isma"):
    if source not in ("isma", "phase8"):
        raise ValueError("Unknown public parent")
    return REPORTS/("duration-balanced-v1" if source == "isma" else "duration-balanced-phase8-v1")


def _path(relative):
    path = (ROOT/relative).resolve()
    if not path.is_relative_to((ROOT/"evaluation/reports/external").resolve()):
        raise ValueError("Parent artifact outside explicit public report root")
    return path


def audit(source="isma"):
    spec = read_json(ROOT/"configs/duration_balanced_protocol.json")
    dest = destination(source)
    parent = REPORTS/("temporal-probe-v1" if source == "isma" else "phase8-interval-v1")
    required = [REPORTS/"folds/idsse-only.json", parent/"registration.json"]
    if source == "isma":
        required += [parent/"selection-seal.json", parent/"verification.json"]
        required += [REPORTS/"temporal-teachers-v1/registration.json"]
        required += [REPORTS/f"temporal-teachers-v1/selections/fold-{i}.json" for i in spec["pilots"]]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.is_file()]
    result = {"source": source, "ready": False, "missing": missing,
              "scope": "Explicit public external paths; never inspect target assets or captured logs.",
              "teacher_training_allowed": False}
    if missing:
        atomic_json(dest/"readiness.json", result)
        return result, None
    manifest = read_json(required[0]); records = manifest["artifacts"]
    if len(records) != 7 or not all(k.startswith("idsse:") for k in records):
        raise ValueError("Seven public IDSSE artifacts required")
    for record in records.values():
        public_path(record["path"])
    parent_reg = read_json(parent/"registration.json")
    if source == "isma":
        old_spec = parent_reg["protocol"]
        for name in ("seed","epochs","batch_size","core_frames","learning_rate","weight_decay",
                     "gradient_clip","condition_weights","scenarios","score_weights"):
            if old_spec[name] != spec[name]:
                raise ValueError("Original local training/evaluation recipe differs")
        if old_spec["added_context_frames_per_side"] != spec["training_halo_frames"]:
            raise ValueError("Original window halos differ")
    refs = {str(p.relative_to(ROOT)): file_hash(p) for p in required}
    # Phase8 and temporal parents use different registration schemas.
    teachers = parent_reg["teachers"]
    chosen = {}
    for fold in spec["pilots"]:
        original = teachers[str(fold)]
        seed = original["seed"]
        if not splits_valid(original["split"]):
            raise ValueError("Overlapping parent split")
        if original["split"]["test"] != manifest["folds"][fold]["test"][0]:
            raise ValueError("Teacher belongs to a different outer test fold")
        if source == "isma":
            checkpoint = REPORTS/f"temporal-teachers-v1/training/fold-{fold}/seed-{seed}/binary/last.pt"
            local = parent/f"training/fold-{fold}/local_classification/last.pt"
            complete = local.with_name("complete.json")
            selected = parent/f"selections/fold-{fold}.json"
            expected = [checkpoint, local, complete, selected]
        else:
            from possession.phase8.training import source_folder
            checkpoint = source_folder(fold, seed, "set_transformer")/"binary/last.pt"
            local, complete = None, None
            selected = REPORTS/f"phase8-v1/selections/IDSSE/fold-{fold}/seed-{seed}/set_transformer.json"
            expected = [checkpoint, selected]
        expected += [parent/f"embeddings/fold-{fold}"/key.replace(":", "_")/(name+".npz")
                     for key in original["split"]["train"]+[original["split"]["calibration"], original["split"]["validation"], original["split"]["test"]]
                     for name in spec["scenarios"]]
        for path in expected:
            if not path.is_file():
                missing.append(str(path.relative_to(ROOT)))
        if missing:
            continue
        for path in expected:
            refs[str(path.relative_to(ROOT))] = file_hash(path)
            if source == "isma" and path.suffix == ".npz":
                stamp = read_json(path.with_suffix(".provenance.json"))
                if stamp["sha256"] != file_hash(path) or stamp["teacher_sha256"] != file_hash(checkpoint):
                    raise ValueError("Parent embedding provenance mismatch")
        if source == "isma":
            selection = read_json(selected)
            if selection["test_used"] or selection["checkpoints"]["local_classification"] != refs[str(local.relative_to(ROOT))]:
                raise ValueError("Original local head provenance mismatch")
            if not read_json(parent/"verification.json")["passed"]:
                raise ValueError("Parent verification did not pass")
            for name, digest in parent_reg["sources"].items():
                if file_hash(ROOT/name.replace(chr(92),"/")) != digest:
                    raise ValueError("Parent algorithm source differs; preserve original version")
            cp_ref = str(checkpoint.relative_to(ROOT))
            original_refs = {k.replace(chr(92),"/"):v for k,v in parent_reg["references"].items()}
            if original_refs.get(cp_ref) != refs[cp_ref]:
                raise ValueError("Teacher checkpoint hash mismatch")
        elif parent_reg["teacher_hashes"][str(fold)] != file_hash(checkpoint):
            raise ValueError("Phase8 teacher hash mismatch")
        chosen[str(fold)] = dict(original, checkpoint=str(checkpoint),
                                original_head=str(local) if local else None,
                                original_complete=str(complete) if complete else None,
                                export_root=str(parent/f"embeddings/fold-{fold}"))
    result.update(ready=not missing, missing=missing, teacher_seeds={k:v["seed"] for k,v in chosen.items()},
                  note="Exact Isma comparison" if source=="isma" else "Separate Phase8 reference; not the prior temporal teacher or local head")
    atomic_json(dest/"readiness.json", result)
    return result, {"protocol": spec, "teachers": chosen, "records": records, "references": refs,
                    "parent_sources": parent_reg.get("sources", parent_reg.get("source", {}))}


def register(source="isma"):
    readiness, record = audit(source)
    if not readiness["ready"]:
        return None
    dest = destination(source)
    sources = {}
    for directory in ("duration_balanced", "temporal_probe", "interval_probe", "phase8",
                      "phase7", "phase6", "phase5", "learning", "external_data", "night"):
        for path in sorted((ROOT/"possession"/directory).glob("*.py")):
            sources[str(path.relative_to(ROOT))] = file_hash(path)
    for relative in ("scripts/duration_balanced.py", "evaluation/possession_metrics.py"):
        sources[relative] = file_hash(ROOT/relative)
    if sources["evaluation/possession_metrics.py"] != SCORER_SHA:
        raise ValueError("Official evaluator changed")
    record.update(source=source, sources=sources,
                  environment={"python":platform.python_version(), "torch":torch.__version__, "numpy":np.__version__})
    marker = dest/"registration.json"
    if marker.is_file() and read_json(marker) != record:
        raise ValueError("Registered duration experiment changed")
    if not marker.is_file():
        atomic_json(marker, record)
    return record
