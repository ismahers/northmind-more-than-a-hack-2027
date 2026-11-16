"""Fixed train-only fitting, frozen models seal, then exploratory validation.

Only public IDSSE data and its public-derived capsule are accessible. No target
loader is called. All entrypoints run after run_local redirects child output.
"""
import hashlib
import json
from pathlib import Path
import platform
import time
import zipfile
import numpy as np
import torch

from possession.football_predictive import features_for, fit, predict, combine
from possession.external_data.schema import Match, file_hash
from possession.external_data.features import build_features
from possession.phase7.features import harmonize
from possession.phase5.experiment import scenarios
from possession.hsmm_pilot import load_context, score
from possession.learning.temporal import TemporalDecoder
from possession.phase5.metrics import diagnostics
from possession.transfer_validation import gates

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/"evaluation/reports/external/football-predictive-v1"
PARENT = ROOT/"evaluation/reports/external/hsmm-pilot-v1"
PUBLIC = ROOT/"docs/public-results/2026-10-29"
PROTOCOL = ROOT/"configs/football_predictive_v1.json"
ROLES = ("train", "calibration", "validation", "test")
SPLITS = {
    "0": {"train": ["J03WOH", "J03WOY", "J03WPY", "J03WQQ"],
          "calibration": "J03WN1", "validation": "J03WR9", "test": "J03WMX"},
    "6": {"train": ["J03WN1", "J03WOH", "J03WOY", "J03WPY"],
          "calibration": "J03WMX", "validation": "J03WQQ", "test": "J03WR9"}}
SPLITS = {f: {r: ["idsse:DFL-MAT-"+x for x in v] if r == "train" else "idsse:DFL-MAT-"+v
              for r, v in split.items()} for f, split in SPLITS.items()}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


def sources():
    paths = sorted((ROOT/"possession").rglob("*.py")) + [ROOT/n for n in
        ("scripts/run_local.py", "scripts/football_predictive.py", "tests/test_football_predictive.py",
         "configs/football_predictive_v1.json", "configs/external_sources.json", "evaluation/possession_metrics.py")]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_text(encoding="utf-8-sig").replace("\r\n", "\n").encode()).hexdigest()
            for p in paths}


def check_split(fold, split):
    if split != SPLITS[str(fold)]:
        raise ValueError("Fixed historical split required")


def model_path(fold, arm):
    return DEST/f"models/fold-{fold}-{arm}.json"


def register():
    p = read(PROTOCOL)
    if (p["version"] != "football-predictive-v1" or p["folds"] != [0, 6]
        or p["arms"] != ["proximity", "football"] or p["lag_seconds"] != [.12, .24]
        or any(p[k] for k in ("parameter_search", "test_access", "target_access", "automatic_promotion"))):
        raise ValueError("Only the registered pilot is allowed")
    published = read(ROOT/"docs/public-results/2026-10-28/hsmm-pilot-v1.json")
    parent = read(PARENT/"registration.json")
    parent_seal = read(PARENT/"result-seal.json")
    if (file_hash(PARENT/"registration.json") != published["registration_sha256"]
        or parent_seal["registration_sha256"] != published["registration_sha256"]
        or read(PARENT/"summary.json") != published):
        raise ValueError("Published reference provenance mismatch")
    dependencies = {}
    for f in p["folds"]:
        partition = next(x for x in parent["partitions"] if x["fold"] == f)
        check_split(f, {r: partition[r] for r in ROLES})
        for name in ("context", "priors", *[s+"-emissions" for s in p["scenarios"]]):
            relative = f"replay/fold-{f}/{name}.npz"
            path = PARENT/relative
            if file_hash(path) != parent_seal["outputs"][relative]:
                raise ValueError("Sealed public dependency changed")
            dependencies[path.relative_to(ROOT).as_posix()] = file_hash(path)
    records = read(ROOT/"evaluation/reports/external/folds/idsse-only.json")["artifacts"]
    allowed = set.union(*(set(s["train"]+[s["validation"]]) for s in SPLITS.values()))
    inputs = {}
    for key in sorted(allowed):
        path = Path(records[key]["path"]).resolve()
        if not path.is_relative_to((ROOT/"data/external/processed").resolve()) or path.suffix != ".npz":
            raise ValueError("Public processed inputs only")
        if file_hash(path) != records[key]["sha256"]:
            raise ValueError("Public source hash mismatch")
        inputs[key] = {"path": path.relative_to(ROOT).as_posix(), "sha256": file_hash(path)}
    stamp = {"protocol": p, "sources": sources(), "splits": SPLITS, "inputs": inputs,
             "dependencies": dependencies, "reference_metrics": {
                 f: {r: published["folds"][f]["metrics"][r] for r in ("registered_multiseed", "four_view_v2")}
                 for f in SPLITS},
             "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}}
    path = DEST/"registration.json"
    if path.exists() and read(path) != stamp:
        raise ValueError("Immutable pilot changed")
    if not path.exists():
        write(path, stamp)
    return stamp


def load_match(stamp, fold, key, role):
    split = stamp["splits"][str(fold)]
    check_split(fold, split)
    allowed = split["train"] if role == "train" else [split["validation"]] if role == "validation" else []
    if key not in allowed:
        raise ValueError("Match is outside this fold's allowed role")
    item = stamp["inputs"][key]
    path = ROOT/item["path"]
    if file_hash(path) != item["sha256"]:
        raise ValueError("Training/validation input changed")
    m = Match.load(path)
    if m.key != key:
        raise ValueError("Public match identity mismatch")
    return m


def objective(metrics, p):
    return sum(w*metrics[s]["spellF1"] for w, s in zip(p["weights"], p["scenarios"]))


def decisions(metrics, p):
    refs = ("proximity", "registered_multiseed", "four_view_v2")
    result = {r: gates(metrics["football"], metrics[r], p) for r in refs}
    result["gain_vs_proximity"] = objective(metrics["football"], p)-objective(metrics["proximity"], p)
    result["gain_vs_four_view"] = objective(metrics["football"], p)-objective(metrics["four_view_v2"], p)
    result["long_fragmentation"] = all(metrics["football"]["clean"]["long_fragmented_gt"]
        <= metrics[r]["clean"]["long_fragmented_gt"]+p["long_fragmentation_delta_max"] for r in refs)
    result["false_HA"] = all(metrics["football"][s]["false_transitions_per_minute"]
        <= metrics[r][s]["false_transitions_per_minute"]+p["false_HA_delta_max"] for r in refs for s in p["scenarios"])
    result["passed"] = (all(result[r]["passed"] for r in refs)
        and result["long_fragmentation"] and result["false_HA"]
        and result["gain_vs_proximity"] >= p["min_gain_vs_proximity"]
        and result["gain_vs_four_view"] >= p["min_gain_vs_four_view"])
    return result


def compare(context, prior, data, models, p):
    predictions = {r: data[r] for r in ("registered_multiseed", "four_view_v2")}
    for arm in p["arms"]:
        predictions[arm] = combine(data["four_view_v2"], predict(models[arm], data["x"]), data["active"], p)
    metrics, hard = {}, {}
    for name, probabilities in predictions.items():
        h = context.decoded(probabilities, TemporalDecoder(p["decoder"], prior))
        metrics[name] = score(context, h)
        diag = diagnostics(context, h)
        metrics[name].update(boundaries_HA=diag["decoded_HA"],
            long_fragmented_gt=diag["duration_groups"]["ge10"]["fragmented_gt"],
            long_fused_predictions=diag["duration_groups"]["ge10"]["fused_predictions"])
        hard[name] = h
    return metrics, hard


def run():
    DEST.mkdir(parents=True, exist_ok=True)
    lock = DEST/"running.lock"
    with lock.open("x") as stream:
        stream.write("One public geometry pilot\n")
    started = time.perf_counter()
    old = read(DEST/"compute.json") if (DEST/"compute.json").exists() else {"seconds": 0., "sessions": []}
    status = "failed"
    try:
        stamp = register(); p = stamp["protocol"]
        if (DEST/"result-seal.json").exists():
            status = "verified_existing"
            return verify()
        def check():
            if old["seconds"]+time.perf_counter()-started > p["max_seconds"]:
                raise TimeoutError("Cumulative predictive pilot budget exhausted")
        torch.set_num_threads(p["cpu_threads"])
        torch.manual_seed(p["seed"])
        torch.use_deterministic_algorithms(True)
        # Fit ALL models and seal them before opening new validation geometry/labels.
        models, feature_cache = {}, {}
        for fold in p["folds"]:
            xs, ys, ws = [], [], []
            for key in SPLITS[str(fold)]["train"]:
                check()
                if key not in feature_cache:
                    m = load_match(stamp, fold, key, "train")
                    v = harmonize(build_features(m))
                    rows = []
                    ix = np.concatenate([np.flatnonzero(v.run == r)[::p["training_stride_10hz"]] for r in np.unique(v.run)])
                    for s, (_, missing) in scenarios(v, key, {}).items():
                        check()
                        x, active = features_for(m, v, missing, p)
                        keep = ix[(v.alive[ix] == 1) & (v.team[ix] >= 0)
                                  & (v.confidence[ix] > 0).all(-1) & active[ix]]
                        if not len(keep):
                            raise ValueError("No training support")
                        rows.append((s, x[keep], (v.team[keep] == 0).astype(float)))
                    feature_cache[key] = rows
                for s, x, y in feature_cache[key]:
                    xs.append(x); ys.append(y)
                    ws.append(np.full(len(y), p["weights"][p["scenarios"].index(s)]/4/len(y)))
            x, y, weight = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
            path = DEST/f"training/fold-{fold}.npz"; path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, x=x, y=y, weight=weight)
            models[str(fold)] = {}
            for arm in p["arms"]:
                model = fit(x, y, weight, arm, p, check)
                model.update(fold=fold, train_keys=SPLITS[str(fold)]["train"], registration_sha256=file_hash(DEST/"registration.json"))
                write(model_path(fold, arm), model)
                models[str(fold)][arm] = model
            write(DEST/"public-progress.json", {"stage": "training", "completed_fold": fold})
        seal_models = {model_path(f, a).relative_to(DEST).as_posix(): file_hash(model_path(f, a))
                       for f in p["folds"] for a in p["arms"]}
        write(DEST/"models-seal.json", {"validation_started": False, "models": seal_models})
        del feature_cache
        summaries = {}
        for fold in p["folds"]:
            check()
            key = SPLITS[str(fold)]["validation"]
            m = load_match(stamp, fold, key, "validation"); v = harmonize(build_features(m))
            folder = DEST/f"validation/fold-{fold}"; folder.mkdir(parents=True, exist_ok=True)
            for name in ("context", "priors"):
                (folder/(name+".npz")).write_bytes((PARENT/f"replay/fold-{fold}/{name}.npz").read_bytes())
            context = load_context(folder/"context.npz")
            np.testing.assert_array_equal(context.view.time, v.time)
            np.testing.assert_array_equal(context.view.run, v.run)
            with np.load(folder/"priors.npz", allow_pickle=False) as z:
                prior = dict(z)
            metrics = {r: {} for r in ("registered_multiseed", "four_view_v2", *p["arms"])}
            for s, (_, missing) in scenarios(v, key, {}).items():
                check()
                x, active = features_for(m, v, missing, p)
                with np.load(PARENT/f"replay/fold-{fold}/{s}-emissions.npz", allow_pickle=False) as z:
                    data = {r: z[r].copy() for r in ("registered_multiseed", "four_view_v2")}
                data.update(x=x, active=active)
                rows, hard = compare(context, prior, data, models[str(fold)], p)
                for name, row in rows.items():
                    metrics[name][s] = row
                np.savez_compressed(folder/f"{s}.npz", **data, **{"hard__"+k: h for k, h in hard.items()})
                write(DEST/"public-progress.json", {"stage": "validation", "fold": fold, "scenario": s})
            for ref in ("registered_multiseed", "four_view_v2"):
                for s in p["scenarios"]:
                    for name, value in stamp["reference_metrics"][str(fold)][ref][s].items():
                        if not np.isclose(metrics[ref][s][name], value, rtol=0, atol=1e-10):
                            raise ValueError("Frozen reference parity failed")
            summaries[str(fold)] = {"metrics": metrics, "gates": decisions(metrics, p), "reference_parity": True}
        check()
        if sources() != stamp["sources"]:
            raise ValueError("Source changed during pilot")
        for name, digest in stamp["dependencies"].items():
            if file_hash(ROOT/name) != digest:
                raise ValueError("Frozen input changed")
        for item in stamp["inputs"].values():
            if file_hash(ROOT/item["path"]) != item["sha256"]:
                raise ValueError("Raw public input changed")
        result = {"folds": summaries, "passes_both": all(r["gates"]["passed"] for r in summaries.values()),
                  "registration_sha256": file_hash(DEST/"registration.json"),
                  "seconds": time.perf_counter()-started, "test_accessed": False, "target_accessed": False,
                  "automatic_promotion": False, "validation_exploratory": True,
                  "neural_training_predictions_used": False}
        write(DEST/"summary.json", result)
        names = ["summary.json", "models-seal.json"] + [n.relative_to(DEST).as_posix()
            for sub in ("models", "training", "validation") for n in sorted((DEST/sub).rglob("*")) if n.is_file()]
        write(DEST/"result-seal.json", {"registration_sha256": result["registration_sha256"],
              "outputs": {n: file_hash(DEST/n) for n in names}})
        status = "complete"
        return verify()
    finally:
        old["seconds"] += time.perf_counter()-started
        old["sessions"].append({"status": status, "seconds": time.perf_counter()-started})
        write(DEST/"compute.json", old)
        lock.unlink()


def verify():
    stamp, seal = read(DEST/"registration.json"), read(DEST/"result-seal.json")
    if stamp["sources"] != sources() or seal["registration_sha256"] != file_hash(DEST/"registration.json"):
        raise ValueError("Pilot provenance changed")
    for name, digest in seal["outputs"].items():
        if file_hash(DEST/name) != digest:
            raise ValueError("Sealed output changed")
    result = read(DEST/"summary.json"); p = stamp["protocol"]
    model_seal = read(DEST/"models-seal.json")
    expected_models = {model_path(f, a).relative_to(DEST).as_posix()
                       for f in p["folds"] for a in p["arms"]}
    if model_seal["validation_started"] or set(model_seal["models"]) != expected_models:
        raise ValueError("Invalid model seal")
    for name, digest in model_seal["models"].items():
        if file_hash(DEST/name) != digest:
            raise ValueError("Model changed after pre-validation seal")
    for fold, row in result["folds"].items():
        if row["gates"] != decisions(row["metrics"], p) or not row["reference_parity"]:
            raise ValueError("Decision mismatch")
    if (result["passes_both"] != all(r["gates"]["passed"] for r in result["folds"].values())
        or any(result[k] for k in ("test_accessed", "target_accessed", "automatic_promotion", "neural_training_predictions_used"))):
        raise ValueError("Scope mismatch")
    return 0


def replay():
    verify()
    stamp = read(DEST/"registration.json"); p = stamp["protocol"]
    result = read(DEST/"summary.json")
    torch.set_num_threads(p["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    for fold in p["folds"]:
        models = {a: read(model_path(fold, a)) for a in p["arms"]}
        with np.load(DEST/f"training/fold-{fold}.npz", allow_pickle=False) as z:
            for arm in p["arms"]:
                restored = fit(z["x"], z["y"], z["weight"], arm, p)
                np.testing.assert_allclose(restored["theta"], models[arm]["theta"], atol=1e-10, rtol=0)
                np.testing.assert_array_equal(restored["scale"], models[arm]["scale"])
        folder = DEST/f"validation/fold-{fold}"
        context = load_context(folder/"context.npz")
        with np.load(folder/"priors.npz", allow_pickle=False) as z:
            prior = dict(z)
        for s in p["scenarios"]:
            with np.load(folder/f"{s}.npz", allow_pickle=False) as z:
                rows, hard = compare(context, prior, dict(z), models, p)
                for arm in hard:
                    np.testing.assert_array_equal(hard[arm], z["hard__"+arm])
                    if rows[arm] != result["folds"][str(fold)]["metrics"][arm][s]:
                        raise ValueError("Replayed metrics differ")
    return 0


def export():
    verify()
    seal = read(DEST/"result-seal.json")
    names = sorted([*seal["outputs"], "registration.json", "result-seal.json", "compute.json"])
    archive = ROOT/"evaluation/reports/handoff/northmind-football-predictive-v1.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    manifest = {n: {"sha256": file_hash(DEST/n), "bytes": (DEST/n).stat().st_size} for n in names}
    prefix = "evaluation/reports/external/football-predictive-v1/"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for name in names:
            info = zipfile.ZipInfo(prefix+name, (2026, 10, 29, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, (DEST/name).read_bytes())
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None or any(hashlib.sha256(z.read(prefix+n)).hexdigest() != v["sha256"] for n, v in manifest.items()):
            raise ValueError("Archive verification failed")
    write(PUBLIC/"football-predictive-v1.json", read(DEST/"summary.json"))
    write(PUBLIC/"football-predictive-v1-package.json", {"archive": archive.name, "bytes": archive.stat().st_size,
          "sha256": file_hash(archive), "files": len(names), "verified": True,
          "raw_data_required_for_replay": False,
          "scope": "Public IDSSE derived train features/labels, four models, validation features/emissions/mapping/labels; no raw tracking, target or logs"})
    return 0
