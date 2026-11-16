"""Post-hoc diagnosis of public capsule only; no fitting, tuning or raw data."""
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from possession.football_mechanism import analyze_case, decision, measures
from possession.football_predictive import predict
from possession.hsmm_pilot import load_context
from possession.external_data.schema import file_hash

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/"evaluation/reports/external/football-mechanism-v1"
PARENT = ROOT/"evaluation/reports/external/football-predictive-v1"
PUBLIC = ROOT/"docs/public-results/2026-10-29"
CONFIG = ROOT/"configs/football_mechanism_v1.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def canonical(path):
    return hashlib.sha256(Path(path).read_text(encoding="utf-8-sig").replace("\r\n", "\n").encode()).hexdigest()


def sources():
    paths = sorted((ROOT/"possession").rglob("*.py")) + [ROOT/n for n in (
        "scripts/run_local.py", "scripts/football_mechanism.py", "tests/test_football_mechanism.py",
        "configs/football_mechanism_v1.json", "evaluation/possession_metrics.py")]
    return {p.relative_to(ROOT).as_posix(): canonical(p) for p in paths}


def parent_inputs():
    published = read(PUBLIC/"football-predictive-v1.json")
    parent = read(PARENT/"registration.json"); seal = read(PARENT/"result-seal.json")
    if (published != read(PARENT/"summary.json") or published["registration_sha256"] != file_hash(PARENT/"registration.json")
            or seal["registration_sha256"] != published["registration_sha256"]):
        raise ValueError("Published capsule identity changed")
    # The historical launcher/source inventory changed as new tasks were added.
    # Verify shared prediction and scoring implementations, not the old launcher.
    for name in ("possession/football_predictive.py", "possession/hsmm_pilot.py",
                 "possession/learning/metrics.py", "possession/learning/study.py", "evaluation/possession_metrics.py"):
        if canonical(ROOT/name) != parent["sources"][name]:
            raise ValueError("Frozen prediction/scoring implementation changed")
    dependencies = {}
    for name, digest in seal["outputs"].items():
        if file_hash(PARENT/name) != digest:
            raise ValueError("Public capsule file changed")
        dependencies[name] = digest
    models_seal = read(PARENT/"models-seal.json")
    for name, digest in models_seal["models"].items():
        if dependencies[name] != digest:
            raise ValueError("Models differ from pre-validation seal")
    dependencies["registration.json"] = file_hash(PARENT/"registration.json")
    dependencies["result-seal.json"] = file_hash(PARENT/"result-seal.json")
    return parent, dependencies


def register():
    spec = read(CONFIG)
    if spec["version"] != "football-mechanism-v1" or any(spec[k] for k in
            ("fit", "weight_search", "raw_data_access", "test_access", "target_access", "promotion")):
        raise ValueError("Frozen diagnosis only")
    parent, deps = parent_inputs()
    stamp = {"protocol": spec, "sources": sources(), "parent_files": deps,
             "parent_protocol": parent["protocol"],
             "environment": {"python": platform.python_version(), "numpy": np.__version__}}
    path = DEST/"registration.json"
    if path.exists() and read(path) != stamp:
        raise ValueError("Immutable diagnostic changed")
    if not path.exists():
        write(path, stamp)
    return stamp


def compute(stamp):
    spec = stamp["protocol"]; folds = {}
    started = time.perf_counter()
    def check():
        if time.perf_counter()-started > spec["max_seconds"]:
            raise TimeoutError("Diagnostic budget exhausted")
    published = read(PUBLIC/"football-predictive-v1.json")
    for fold in spec["folds"]:
        check()
        models = {a: read(PARENT/f"models/fold-{fold}-{a}.json") for a in ("proximity", "football")}
        with np.load(PARENT/f"training/fold-{fold}.npz", allow_pickle=False) as z:
            train = {a: measures(predict(models[a], z["x"]), z["y"], np.ones(len(z["y"]), bool),
                       spec["probability_clip"], z["weight"]) for a in models}
        folder = PARENT/f"validation/fold-{fold}"
        context = load_context(folder/"context.npz")
        cases = {}
        for scenario in spec["scenarios"]:
            check()
            with np.load(folder/f"{scenario}.npz", allow_pickle=False) as z:
                data = {k: z[k] for k in ("x", "active", "four_view_v2", "hard__proximity", "hard__football")}
            cases[scenario] = analyze_case(context, data, models, stamp["parent_protocol"], spec)
            for a in models:
                expected = published["folds"][str(fold)]["metrics"][a][scenario]["spellF1"]
                if not np.isclose(expected, cases[scenario]["saved_spellF1"][a], atol=1e-12, rtol=0):
                    raise ValueError("Frozen SpellF1 parity mismatch")
        folds[str(fold)] = {"train": train, "validation": cases, "decision": decision(cases, spec)}
    check()
    return {"folds": folds, "additional_signal_both_folds": all(
        row["decision"]["additional_signal_in_frozen_classifier"] for row in folds.values()),
        "fit_performed": False, "weight_search_performed": False, "raw_data_accessed": False,
        "test_accessed": False, "target_accessed": False, "promoted": False,
        "post_hoc_exploratory": True, "registration_sha256": file_hash(DEST/"registration.json")}


def run():
    started = time.perf_counter()
    stamp = register()
    result = compute(stamp)
    _, deps = parent_inputs()
    if deps != stamp["parent_files"] or sources() != stamp["sources"]:
        raise ValueError("Dependency changed during diagnostic")
    write(DEST/"summary.json", result)
    write(DEST/"result-seal.json", {"summary_sha256": file_hash(DEST/"summary.json"),
          "registration_sha256": file_hash(DEST/"registration.json"), "seconds": time.perf_counter()-started})
    return verify()


def verify():
    stamp = read(DEST/"registration.json"); seal = read(DEST/"result-seal.json")
    _, deps = parent_inputs()
    if (stamp["sources"] != sources() or stamp["parent_files"] != deps
            or seal["summary_sha256"] != file_hash(DEST/"summary.json")
            or seal["registration_sha256"] != file_hash(DEST/"registration.json")):
        raise ValueError("Diagnostic integrity mismatch")
    result = read(DEST/"summary.json")
    for row in result["folds"].values():
        if row["decision"] != decision(row["validation"], stamp["protocol"]):
            raise ValueError("Diagnostic decision mismatch")
    if result["additional_signal_both_folds"] != all(r["decision"]["additional_signal_in_frozen_classifier"] for r in result["folds"].values()):
        raise ValueError("Fold aggregation mismatch")
    return 0


def replay():
    verify()
    if compute(read(DEST/"registration.json")) != read(DEST/"summary.json"):
        raise ValueError("Diagnostic replay differs")
    return 0


def export():
    verify()
    for source, target in (("summary.json", "football-mechanism-v1.json"),
                           ("registration.json", "football-mechanism-v1-registration.json"),
                           ("result-seal.json", "football-mechanism-v1-seal.json")):
        write(PUBLIC/target, read(DEST/source))
        if file_hash(PUBLIC/target) != file_hash(DEST/source):
            raise ValueError("Public export changed bytes")
    return 0
