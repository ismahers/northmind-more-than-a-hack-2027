"""Read-only public ancestry and separately immutable diagnostic/pilot records."""
import platform
from pathlib import Path
import torch
import numpy as np
from possession.external_data.schema import file_hash
from possession.night.common import atomic_json
from possession.temporal_probe.common import ROOT, REPORTS, read_json, public_path, splits_valid

DEST = REPORTS / "ha-discrimination-v1"
PARENT = REPORTS / "local-head-budget-phase8-v2"


def register():
    spec = read_json(ROOT / "configs/ha_discrimination_v1.json")
    parent = read_json(PARENT / "registration.json")
    if not read_json(PARENT / "verification.json")["passed"]:
        raise ValueError("Verified original local heads required")
    for name, sha in parent["sources"].items():
        if file_hash(ROOT / name) != sha:
            raise ValueError("Historical algorithm changed")
    refs = dict(parent["references"])
    for name in ("registration.json", "verification.json", "selection-seal.json"):
        path = PARENT / name; refs[str(path.relative_to(ROOT))] = file_hash(path)
    for fold in spec["pilots"]:
        teacher = parent["teachers"][str(fold)]
        if not splits_valid(teacher["split"]):
            raise ValueError("Invalid strict split")
        for part in ("validation", "test"):
            for scenario in parent["protocol"]["scenarios"]:
                path = PARENT / f"predictions/fold-{fold}/cap-8/{part}/{scenario}.npz"
                refs[str(path.relative_to(ROOT))] = file_hash(path)
    for name, sha in refs.items():
        path = (ROOT / name).resolve()
        if not (path.is_relative_to(REPORTS.resolve()) or path.is_relative_to((ROOT / "data/external").resolve())):
            raise ValueError("Nonpublic artifact reference rejected")
        if file_hash(path) != sha:
            raise ValueError("Historical public artifact changed")
    for record in parent["records"].values():
        public_path(record["path"])
    paths = ["configs/ha_discrimination_v1.json", "scripts/ha_discrimination.py"]
    paths += ["possession/ha_discrimination/"+name+".py" for name in ("common", "features", "diagnosis")]
    sources = dict(parent["sources"])
    sources.update({p: file_hash(ROOT / p) for p in paths})
    record = {"protocol": spec, "recipe": parent["protocol"], "teachers": parent["teachers"],
              "records": parent["records"], "references": refs, "sources": sources,
              "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                              "numpy": np.__version__, "device": "cpu"}}
    path = DEST / "registration.json"
    if path.exists() and read_json(path) != record:
        raise ValueError("Immutable diagnosis changed; use another version")
    if not path.exists():
        atomic_json(path, record)
    return record


def verify_ancestry(record):
    checks = {"source:"+p: file_hash(ROOT / p) == sha for p, sha in record["sources"].items()}
    checks.update({"reference:"+p: file_hash(ROOT / p) == sha for p, sha in record["references"].items()})
    return checks
