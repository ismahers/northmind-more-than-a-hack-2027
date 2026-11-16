"""Fixed public HSMM pilot with a self-contained, path-independent replay capsule."""
import hashlib
import json
from pathlib import Path
import platform
import time
import zipfile

import numpy as np
import torch

from possession import st_tcn_target as transfer
from possession.evidence_first_validation import FoldStore, assert_metrics
from possession.external_data.schema import file_hash
from possession.learning.study import ScoreContext
from possession.learning.temporal import fit_priors
from possession.hsmm_pilot import save_context, load_context, compare
from possession.night.common import atomic_json
from possession.transfer_validation import gates
from scripts.audit_long_diagnosis import normalized_outputs

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "evaluation/reports/external/hsmm-pilot-v1"
PARENT = ROOT / "evaluation/reports/external/long-diagnosis-v3"
PROTOCOL = ROOT / "configs/hsmm_pilot_v1.json"
NEW_TASKS = ("run-hsmm-pilot", "verify-hsmm-pilot", "replay-hsmm-pilot", "test-hsmm-pilot")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_hash(path):
    text = Path(path).read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    return hashlib.sha256(text.encode()).hexdigest()


def sources():
    paths = list((ROOT / "possession").rglob("*.py"))
    paths += [ROOT / name for name in ("scripts/hsmm_pilot.py", "scripts/audit_long_diagnosis.py",
        "scripts/run_local.py", "tests/test_hsmm_pilot.py", "tests/test_phase3.py",
        "evaluation/possession_metrics.py", "configs/hsmm_pilot_v1.json")]
    return {p.relative_to(ROOT).as_posix(): canonical_hash(p) for p in sorted(paths)}


def validate_protocol(p):
    if (p["version"] != "hsmm-pilot-v1" or p["folds"] != [0, 6]
            or p["scenarios"] != ["clean", "missing40_5s", "missing40_15s"]
            or p["weights"] != [.5, .25, .25] or p["references"] != ["registered_multiseed", "four_view_v2"]
            or p["decoder"] != dict(kind="hsmm", strength=.5, boundary_strength=0.)
            or p["control_decoder"] != dict(kind="hmm", strength=.5, boundary_strength=0.)
            or p["duration_cap"] != 300 or p["candidate"] != "four_view_hsmm"
            or p["guardrails"] != dict(robust_delta_min=0, clean_delta_min=0,
                short_recall_delta_min=0, long_recall_delta_min=-.01, macro_f1_delta_min=-.005)
            or any(p[k] for k in ("train", "test_access", "target_access", "parameter_search"))):
        raise ValueError("Only fixed public HSMM protocol allowed")


def check_parent():
    """Byte-exact dependencies, allowing only six reviewed launcher additions."""
    record = read(PARENT / "registration.json")
    audit = read(ROOT / "docs/public-results/2026-10-28/audit.json")
    if (file_hash(PARENT / "registration.json") != audit["registration_sha256"]
            or file_hash(PARENT / "result-seal.json") != audit["result_seal_sha256"]):
        raise ValueError("Published parent identity changed")
    for name, expected in {**{str(ROOT / p): h for p, h in record["sources"].items()},
                            **record["dependencies"]}.items():
        path = Path(name)
        if file_hash(path) == expected:
            continue
        if path != ROOT / "scripts/run_local.py":
            atomic_json(DEST / "public-preflight.json", {"verified": False,
                "stage": "parent_input_changed", "dependency": path.relative_to(ROOT).as_posix()})
            raise ValueError("Parent input changed")
        remove = NEW_TASKS + ("audit-publish-long-diagnosis", "test-long-diagnosis-audit")
        lines = path.read_bytes().splitlines(keepends=True)
        kept = [line for line in lines if not any(line.lstrip().startswith(('"'+k+'":').encode()) for k in remove)]
        if len(lines)-len(kept) != 6 or hashlib.sha256(b"".join(kept)).hexdigest() != expected:
            atomic_json(DEST / "public-preflight.json", {"verified": False, "stage": "parent_launcher_changed"})
            raise ValueError("Parent launcher changed beyond task declarations")
    seal = read(PARENT / "result-seal.json")
    for name, digest in normalized_outputs(seal["outputs"]).items():
        if file_hash(PARENT / name) != digest:
            raise ValueError("Parent output changed")
    atomic_json(DEST / "public-preflight.json", {"verified": True, "stage": "parent_verified"})
    return record


def decisions(metrics, p):
    return {ref: gates(metrics["four_view_hsmm"], metrics[ref], p) for ref in p["references"]}


def run():
    DEST.mkdir(parents=True, exist_ok=True)
    lock = DEST / "running.lock"
    with lock.open("x") as stream:
        stream.write("One fixed pilot process\n")
    try:
        return _run()
    finally:
        lock.unlink()


def _run():
    p = read(PROTOCOL)
    validate_protocol(p)
    parent = check_parent()
    expected = read(ROOT / "evaluation/reports/external/pulse-consensus-validation-v1/summary.json")["folds"]
    stamp = {"protocol": p, "sources": sources(),
        "parent_registration_sha256": file_hash(PARENT / "registration.json"),
        "parent_result_seal_sha256": file_hash(PARENT / "result-seal.json"),
        "reference_metrics": {f: {r: expected[f]["metrics"][r] for r in p["references"]} for f in ("0", "6")},
        "partitions": parent["partitions"],
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}}
    marker = DEST / "registration.json"
    if marker.exists() and read(marker) != stamp:
        raise ValueError("Immutable pilot changed; create another version")
    if not marker.exists():
        atomic_json(marker, stamp)
    if (DEST / "result-seal.json").exists():
        return verify()
    records = read(ROOT / "evaluation/reports/external/folds/idsse-only.json")["artifacts"]
    torch.set_num_threads(4)
    start = time.perf_counter()
    outputs, summaries = [], {}
    for fold in p["folds"]:
        partition = next(s for s in stamp["partitions"] if s["fold"] == fold)
        split = {k: partition[k] for k in ("train", "calibration", "validation", "test")}
        store = FoldStore(records, split)
        prior = fit_priors([store.harmonized(k) for k in split["train"]], cap=p["duration_cap"])
        view = store.harmonized(split["validation"])
        context = ScoreContext(store.match(split["validation"]), view)
        root = DEST / f"replay/fold-{fold}"
        root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(root / "priors.npz", **prior)
        save_context(root / "context.npz", context)
        # Round-trip must reproduce the canonical mapping before scoring.
        context = load_context(root / "context.npz")
        outputs.extend([root / "priors.npz", root / "context.npz"])
        metrics = {r: {} for r in (*p["references"], "four_view_hsmm")}
        changes = {}
        for scenario in p["scenarios"]:
            if time.perf_counter()-start > p["max_seconds"]:
                raise TimeoutError("Pilot budget exhausted")
            source = PARENT / f"cache/fold-{fold}/{scenario}.npz"
            with np.load(source, allow_pickle=False) as z:
                np.testing.assert_array_equal(z["time"], view.time)
                np.testing.assert_array_equal(z["run"], view.run)
                probabilities = {ref: z["p__"+ref].copy() for ref in p["references"]}
            emissions_file = root / f"{scenario}-emissions.npz"
            np.savez_compressed(emissions_file, **probabilities)
            measured, predictions, changes[scenario] = compare(context, prior, probabilities, p)
            for name, row in measured.items():
                metrics[name][scenario] = row
            prediction_file = root / f"{scenario}-predictions.npz"
            np.savez_compressed(prediction_file, **predictions)
            outputs.extend([emissions_file, prediction_file])
            atomic_json(DEST / "public-progress.json", {"fold": fold, "scenario": scenario})
        for ref in p["references"]:
            assert_metrics(metrics[ref], stamp["reference_metrics"][str(fold)][ref])
        summaries[str(fold)] = {"metrics": metrics, "changes": changes,
            "gates": decisions(metrics, p), "reference_parity": True,
            "duration_counts_train": prior["duration_counts"]}
    if sources() != stamp["sources"]:
        raise ValueError("Sources changed during evaluation")
    check_parent()
    result = {"folds": summaries, "passes_both": all(
        all(g["passed"] for g in row["gates"].values()) for row in summaries.values()),
        "automatic_promotion": False, "test_accessed": False, "target_accessed": False,
        "registration_sha256": file_hash(marker), "seconds": time.perf_counter()-start}
    atomic_json(DEST / "summary.json", result)
    outputs.append(DEST / "summary.json")
    atomic_json(DEST / "result-seal.json", {"registration_sha256": file_hash(marker),
        "outputs": {path.relative_to(DEST).as_posix(): file_hash(path) for path in outputs}})
    return verify()


def verify():
    record = read(DEST / "registration.json")
    p = record["protocol"]
    validate_protocol(p)
    if sources() != record["sources"]:
        raise ValueError("Pilot source changed")
    seal = read(DEST / "result-seal.json")
    if seal["registration_sha256"] != file_hash(DEST / "registration.json"):
        raise ValueError("Registration changed")
    expected = {"summary.json"}
    for fold in p["folds"]:
        expected.update(f"replay/fold-{fold}/{name}.npz" for name in ("priors", "context"))
        expected.update(f"replay/fold-{fold}/{s}-{kind}.npz" for s in p["scenarios"] for kind in ("emissions", "predictions"))
    if set(seal["outputs"]) != expected:
        raise ValueError("Incomplete capsule")
    for name, digest in seal["outputs"].items():
        if file_hash(DEST / name) != digest:
            raise ValueError("Capsule file changed")
    result = read(DEST / "summary.json")
    if (set(result["folds"]) != {"0", "6"} or any(result[k] for k in
            ("automatic_promotion", "test_accessed", "target_accessed"))
            or result["registration_sha256"] != seal["registration_sha256"]):
        raise ValueError("Invalid result scope")
    for fold, row in result["folds"].items():
        if row["gates"] != decisions(row["metrics"], p) or not row["reference_parity"]:
            raise ValueError("Invalid gates")
        for ref in p["references"]:
            assert_metrics(row["metrics"][ref], record["reference_metrics"][fold][ref])
    passed = all(all(g["passed"] for g in r["gates"].values()) for r in result["folds"].values())
    if result["passes_both"] != passed:
        raise ValueError("Decision mismatch")
    return 0


def replay():
    """No historical absolute path, model, training set or raw match needed."""
    verify()
    p = read(DEST / "registration.json")["protocol"]
    result = read(DEST / "summary.json")
    for fold in p["folds"]:
        root = DEST / f"replay/fold-{fold}"
        context = load_context(root / "context.npz")
        with np.load(root / "priors.npz", allow_pickle=False) as z:
            prior = dict(z)
        for scenario in p["scenarios"]:
            with np.load(root / f"{scenario}-emissions.npz", allow_pickle=False) as z:
                probabilities = dict(z)
            metrics, predictions, changes = compare(context, prior, probabilities, p)
            expected = result["folds"][str(fold)]
            for name, row in metrics.items():
                assert_metrics({scenario: row}, expected["metrics"][name])
            if changes != expected["changes"][scenario]:
                raise ValueError("Replay matching differs")
            with np.load(root / f"{scenario}-predictions.npz", allow_pickle=False) as z:
                for name, hard in predictions.items():
                    np.testing.assert_array_equal(hard, z[name])
    return 0


def export():
    verify()
    seal = read(DEST / "result-seal.json")
    files = sorted([*seal["outputs"], "registration.json", "result-seal.json"])
    out = ROOT / "evaluation/reports/handoff"
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "northmind-hsmm-pilot-v1.zip"
    manifest = {name: file_hash(DEST / name) for name in files}
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in files:
            z.write(DEST / name, f"evaluation/reports/external/hsmm-pilot-v1/{name}")
        z.writestr("HSMM-MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("Archive CRC mismatch")
        for name, digest in manifest.items():
            if hashlib.sha256(z.read(f"evaluation/reports/external/hsmm-pilot-v1/{name}")).hexdigest() != digest:
                raise ValueError("Archive member changed")
    report = ROOT / "docs/public-results/2026-10-28"
    report.mkdir(parents=True, exist_ok=True)
    atomic_json(report / "hsmm-pilot-v1.json", read(DEST / "summary.json"))
    atomic_json(report / "hsmm-pilot-v1-package.json", {"archive": archive.name,
        "sha256": file_hash(archive), "bytes": archive.stat().st_size, "file_count": len(files),
        "scope": "public IDSSE derived validation labels/mapping, frozen emissions, train-fitted priors and predictions",
        "verified": True, "raw_data_required_for_replay": False})
    return 0
