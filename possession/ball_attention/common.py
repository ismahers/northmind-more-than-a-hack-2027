"""Immutable provenance, explicit public data and a separate bounded output root."""
import platform
import torch
import numpy as np
from possession.external_data.schema import file_hash
from possession.night.common import atomic_json
from possession.temporal_probe.common import ROOT, REPORTS, read_json, public_path, splits_valid

DEST = REPORTS / "ball-attention-v1"
PARENT = REPORTS / "ha-discrimination-v1"


def register():
    spec = read_json(ROOT/"configs/ball_attention_v1.json")
    parent = read_json(PARENT/"registration.json")
    if not read_json(PARENT/"diagnostic-verification.json")["passed"]:
        raise ValueError("Verified public parent required")
    refs = dict(parent["references"])
    refs[str((PARENT/"registration.json").relative_to(ROOT))] = file_hash(PARENT/"registration.json")
    # Preserve both teacher stages as well as final checkpoints/embeddings.
    for fold in spec["pilots"]:
        teacher = parent["teachers"][str(fold)]
        if teacher["seed"] != spec["seeds_by_fold"][str(fold)] or not splits_valid(teacher["split"]):
            raise ValueError("Prefixed teacher seed/split mismatch")
        for stage in ("base", "binary"):
            path = (ROOT/teacher["checkpoint"]).parent.parent/stage/"last.pt"
            if not path.resolve().is_relative_to(REPORTS.resolve()):
                raise ValueError("Nonpublic checkpoint")
            refs[str(path.relative_to(ROOT))] = file_hash(path)
    for record in parent["records"].values():
        public_path(record["path"])
    sources = dict(parent["sources"])
    paths = list((ROOT/"possession/ball_attention").glob("*.py"))
    paths += [ROOT/"configs/ball_attention_v1.json", ROOT/"configs/phase8_protocol.json",
              ROOT/"scripts/ball_attention.py", ROOT/"scripts/ball_attention_checks.py", ROOT/"tests/test_ball_attention.py"]
    sources.update({str(p.relative_to(ROOT)): file_hash(p) for p in paths})
    record = {"protocol": spec, "phase8_recipe": read_json(ROOT/"configs/phase8_protocol.json"),
              "teachers": parent["teachers"], "records": parent["records"], "sources": sources,
              "references": refs, "environment": {"python": platform.python_version(),
              "torch": str(torch.__version__), "numpy": np.__version__, "device": "cpu"}}
    checks = verify(record)
    if not all(checks.values()):
        raise ValueError("Historical ancestry mismatch")
    path = DEST/"registration.json"
    if path.exists() and read_json(path) != record:
        raise ValueError("Immutable pilot changed; new version required")
    if not path.exists():
        atomic_json(path, record)
    return record


def verify(record):
    checks = {"source:"+p: file_hash(ROOT/p) == sha for p, sha in record["sources"].items()}
    for p, sha in record["references"].items():
        path = (ROOT/p).resolve()
        if not (path.is_relative_to(REPORTS.resolve()) or path.is_relative_to((ROOT/"data/external").resolve())):
            raise ValueError("Nonpublic ancestry rejected")
        checks["reference:"+p] = file_hash(path) == sha
    return checks
