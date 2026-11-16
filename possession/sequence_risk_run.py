"""Pre-registered public train oracle -> risk training -> held-out validation."""
import hashlib
import json
from pathlib import Path
import platform
import time
import zipfile
import numpy as np
import torch

from possession import st_tcn_target as transfer
from possession.evidence_first_validation import FoldStore, load_models, assert_metrics
from possession.learning.study import ScoreContext
from possession.learning.temporal import fit_priors, TemporalDecoder
from possession.phase5.experiment import scenarios
from possession.phase8.evaluation import predict as predict_teacher
from possession.temporal_probe.windows import predict as predict_head, emissions
from possession.hsmm_pilot import save_context, load_context, score
from possession.night.common import atomic_json
from possession.external_data.schema import file_hash
from possession.transfer_validation import gates
from possession.sequence_risk import candidates, features, fit, choose, project, policies

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/"evaluation/reports/external/sequence-risk-v1"
PARENT = ROOT/"evaluation/reports/external/hsmm-pilot-v1"
PROTOCOL = ROOT/"configs/sequence_risk_v1.json"
PUBLIC = ROOT/"docs/public-results/2026-10-29"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sources():
    paths = list((ROOT/"possession").rglob("*.py"))
    paths += [ROOT/n for n in ("scripts/run_local.py", "scripts/sequence_risk.py",
        "tests/test_sequence_risk.py", "configs/sequence_risk_v1.json", "evaluation/possession_metrics.py")]
    return {x.relative_to(ROOT).as_posix(): hashlib.sha256(x.read_text(encoding="utf-8-sig").replace("\r\n", "\n").encode()).hexdigest() for x in paths}


class TrainStore(FoldStore):
    def __init__(self, records, split):
        super().__init__(records, split)
        self.allowed = set(split["train"])


def frozen_probabilities(view, models, heads, member, config, p, check):
    mean_z, alive = np.zeros(len(view.time)), np.zeros(len(view.time))
    for swap, reflect in p["views"]:
        check()
        transformed = transfer._transform_view(view, swap, reflect)
        primary, _, hidden = predict_teacher(models[0], transformed, embeddings=True)
        alternate, _, _ = predict_teacher(models[1], transformed)
        inputs = {"hidden": hidden, "time": view.time, "run": view.run}
        local, _, _ = predict_head(heads[0], inputs, config["core_frames"], config["batch_size"])
        multi, _, _ = predict_head(heads[1], inputs, config["core_frames"], config["batch_size"])
        z, a = transfer._fuse_member(primary, alternate, local, multi, transformed, member["params"], config)
        mean_z += .25*(-1 if swap else 1)*z
        alive += .25*a
    return emissions(mean_z, np.column_stack((alive, np.zeros(len(alive)), 1-alive)))


def base_decode(p, view, prior, spec):
    result = np.empty(len(p), np.int8)
    decoder = TemporalDecoder(spec, prior)
    for run in np.unique(view.run):
        ix = np.flatnonzero(view.run == run)
        result[ix] = decoder.decode(p[ix], view.time[ix])
    return result


def register():
    p = read(PROTOCOL)
    if (p["version"] != "sequence-risk-v1" or p["folds"] != [0, 6]
        or p["oracle_gain_min"] != .01 or p["optimizer_steps"] != 400
        or any(p[k] for k in ("test_access", "target_access", "parameter_search", "automatic_promotion"))):
        raise ValueError("Fixed risk pilot only")
    config, head_config, members, dependencies = transfer._transfer_spec()
    published = read(ROOT/"docs/public-results/2026-10-28/hsmm-pilot-v1.json")
    parent_record, seal = read(PARENT/"registration.json"), read(PARENT/"result-seal.json")
    if (file_hash(PARENT/"registration.json") != published["registration_sha256"]
            or seal["registration_sha256"] != published["registration_sha256"]
            or read(PARENT/"summary.json") != published):
        raise ValueError("Published reference identity changed")
    for name, digest in seal["outputs"].items():
        if file_hash(PARENT/name) != digest:
            raise ValueError("Reference capsule changed")
    partitions = {str(v["fold"]): {k: v[k] for k in ("train", "calibration", "validation", "test")}
                  for v in parent_record["partitions"]}
    records = read(ROOT/"evaluation/reports/external/folds/idsse-only.json")["artifacts"]
    for split in partitions.values():
        store = TrainStore(records, split)
        for key in store.allowed:
            path = Path(records[key]["path"])
            if file_hash(path) != records[key]["sha256"]:
                raise ValueError("Public training input changed")
            dependencies[str(path)] = file_hash(path)
    stamp = {"protocol": p, "sources": sources(), "splits": partitions,
        "dependencies": {Path(k).relative_to(ROOT).as_posix(): h for k, h in dependencies.items()},
        "reference_outputs": seal["outputs"], "reference_summary": published,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}}
    marker = DEST/"registration.json"
    if marker.exists() and read(marker) != stamp:
        raise ValueError("Immutable risk experiment changed")
    if not marker.exists():
        atomic_json(marker, stamp)
    return p, config, head_config, members, records, stamp


def run():
    DEST.mkdir(parents=True, exist_ok=True)
    lock = DEST/"running.lock"
    with lock.open("x") as stream:
        stream.write("Exclusive public sequence risk experiment\n")
    started = time.perf_counter()
    compute_path = DEST/"compute.json"
    previous = read(compute_path) if compute_path.exists() else {"seconds": 0., "sessions": []}
    status = "failed"
    try:
        p, config, hc, members, records, stamp = register()
        if (DEST/"result-seal.json").exists():
            return verify()
        torch.set_num_threads(p["cpu_threads"])
        torch.manual_seed(p["seed"])
        torch.use_deterministic_algorithms(True)
        def check():
            if previous["seconds"]+time.perf_counter()-started > p["max_seconds"]:
                raise TimeoutError("Risk pilot budget exhausted")
        folds, data = {}, {}
        for member in members:
            fold = member["fold"]
            split = stamp["splits"][str(fold)]
            store = TrainStore(records, split)
            prior = fit_priors([store.harmonized(k) for k in split["train"]])
            models, heads = load_models(member, split, hc)
            cases = []
            for match_index, key in enumerate(split["train"]):
                view = store.harmonized(key)
                context = ScoreContext(store.match(key), view)
                folder = DEST/f"train/fold-{fold}/match-{match_index}"
                folder.mkdir(parents=True, exist_ok=True)
                context_path = folder/"context.npz"
                if not context_path.exists():
                    save_context(context_path, context)
                for scenario_index, (scenario, (degraded, _)) in enumerate(scenarios(view, key, {}).items()):
                    check()
                    path = folder/f"{scenario}.npz"
                    if not path.exists():
                        probabilities = frozen_probabilities(degraded, models, heads, member, config, p, check)
                        base = base_decode(probabilities, view, prior, p["decoder"])
                        alternatives, keep = candidates(base, probabilities, view.run)
                        x = features(alternatives, probabilities, view.run)
                        rewards = []
                        for y in alternatives:
                            check()
                            hard = project(context, y)
                            rewards.append(context.pm.spell_scores(context.gt, hard, context.runs)["mean_f1"])
                        np.savez_compressed(path, probabilities=probabilities, hypotheses=alternatives,
                            keep=keep, features=x, rewards=np.array(rewards), time=view.time, run=view.run)
                    with np.load(path, allow_pickle=False) as z:
                        case = {k: z[k].copy() for k in ("features", "rewards", "keep")}
                    cases.append(dict(case, weight=p["weights"][scenario_index]/4,
                        scenario=scenario, match=key, path=path.relative_to(DEST).as_posix()))
                    atomic_json(DEST/"public-progress.json", {"stage": "train_pool", "fold": fold,
                        "match_index": match_index, "scenario": scenario, "cases_completed": len(cases)})
            weights = np.array([c["weight"] for c in cases])
            rewards = np.stack([c["rewards"] for c in cases])
            baseline = float(weights @ rewards[:, 0])
            oracle = float(weights @ rewards.max(1))
            fixed = int(np.argmax(weights @ rewards))
            folds[str(fold)] = {"train_baseline": baseline, "train_oracle": oracle,
                "train_oracle_gain": oracle-baseline, "train_gate": oracle >= baseline+p["oracle_gain_min"],
                "fixed_policy_index": fixed, "fixed_policy_name": policies()[fixed][0],
                "train_best_fixed": float(weights @ rewards[:, fixed]), "train_cases": len(cases)}
            data[str(fold)] = cases
            atomic_json(DEST/"train-gate-progress.json", folds)
        # Both finite-pool gates are evaluated before any selector or validation.
        gate = all(v["train_gate"] for v in folds.values())
        models_seal = {}
        if gate:
            for fold, cases in data.items():
                check()
                model = fit(np.stack([c["features"] for c in cases]),
                    1-np.stack([c["rewards"] for c in cases]), np.stack([c["keep"] for c in cases]),
                    np.array([c["weight"] for c in cases]), steps=p["optimizer_steps"])
                path = DEST/f"models/fold-{fold}.json"
                atomic_json(path, model)
                models_seal[fold] = file_hash(path)
                folds[fold]["train_selected"] = float(sum(c["weight"]*c["rewards"][choose(model, c["features"], c["keep"])] for c in cases))
            atomic_json(DEST/"model-seal.json", models_seal)
            for fold in p["folds"]:
                check()
                fold = str(fold)
                model = read(DEST/f"models/fold-{fold}.json")
                if file_hash(DEST/f"models/fold-{fold}.json") != models_seal[fold]:
                    raise ValueError("Ranking parameters changed before validation")
                src = PARENT/f"replay/fold-{fold}"
                root = DEST/f"validation/fold-{fold}"
                root.mkdir(parents=True, exist_ok=True)
                context = load_context(src/"context.npz")
                save_context(root/"context.npz", context)
                with np.load(src/"priors.npz", allow_pickle=False) as z:
                    prior = dict(z)
                metrics = {name: {} for name in (*p["references"], "ranker", "train_fixed")}
                selected = {}
                for scenario in p["scenarios"]:
                    check()
                    with np.load(src/f"{scenario}-emissions.npz", allow_pickle=False) as z:
                        probabilities = {name: z[name].copy() for name in p["references"]}
                    base = base_decode(probabilities["four_view_v2"], context.view, prior, p["decoder"])
                    alternatives, keep = candidates(base, probabilities["four_view_v2"], context.view.run)
                    x = features(alternatives, probabilities["four_view_v2"], context.view.run)
                    chosen = choose(model, x, keep)
                    selected[scenario] = policies()[chosen][0]
                    fixed = folds[fold]["fixed_policy_index"]
                    hard = {name: context.decoded(value, TemporalDecoder(p["decoder"], prior)) for name, value in probabilities.items()}
                    hard.update(ranker=project(context, alternatives[chosen]), train_fixed=project(context, alternatives[fixed]))
                    for name, y in hard.items():
                        metrics[name][scenario] = score(context, y)
                    np.savez_compressed(root/f"{scenario}.npz", **probabilities, hypotheses=alternatives,
                        keep=keep, features=x, time=context.view.time, run=context.view.run,
                        **{"hard__"+name: y for name, y in hard.items()})
                for ref in p["references"]:
                    assert_metrics(metrics[ref], stamp["reference_summary"]["folds"][fold]["metrics"][ref])
                comparisons = {ref: gates(metrics["ranker"], metrics[ref], p) for ref in (*p["references"], "train_fixed")}
                meaningful = comparisons["four_view_v2"]["candidate_objective"] >= comparisons["four_view_v2"]["baseline_objective"]+p["validation_gain_min"]
                folds[fold].update(validation=metrics, gates=comparisons, meaningful_gain=meaningful,
                    selected_policies=selected, passed=meaningful and all(v["passed"] for v in comparisons.values()))
                atomic_json(DEST/"public-progress.json", {"stage": "validation_complete", "fold": fold})
        if sources() != stamp["sources"]:
            raise ValueError("Sources changed during experiment")
        summary = {"folds": folds, "train_gate_passed": gate, "selector_trained": gate,
            "validation_accessed": gate, "passes_both": gate and all(v.get("passed", False) for v in folds.values()),
            "test_accessed": False, "target_accessed": False, "automatic_promotion": False,
            "seconds": previous["seconds"]+time.perf_counter()-started,
            "registration_sha256": file_hash(DEST/"registration.json")}
        atomic_json(DEST/"summary.json", summary)
        paths = [DEST/"summary.json", *sorted((DEST/"train").rglob("*.npz")),
                 *sorted((DEST/"models").glob("*.json")), *sorted((DEST/"validation").rglob("*.npz"))]
        if gate:
            paths.append(DEST/"model-seal.json")
        atomic_json(DEST/"result-seal.json", {"registration_sha256": summary["registration_sha256"],
            "outputs": {x.relative_to(DEST).as_posix(): file_hash(x) for x in paths}})
        verify()
        status = "complete"
        return 0
    finally:
        elapsed = time.perf_counter()-started
        previous["seconds"] += elapsed
        previous["sessions"].append({"status": status, "seconds": elapsed})
        atomic_json(compute_path, previous)
        lock.unlink()


def verify():
    stamp, seal, summary = read(DEST/"registration.json"), read(DEST/"result-seal.json"), read(DEST/"summary.json")
    if stamp["sources"] != sources() or seal["registration_sha256"] != file_hash(DEST/"registration.json"):
        raise ValueError("Source/registration changed")
    for name, digest in seal["outputs"].items():
        if file_hash(DEST/name) != digest:
            raise ValueError("Output changed")
    expected = {"summary.json"}
    for fold in stamp["protocol"]["folds"]:
        for match in range(4):
            expected.add(f"train/fold-{fold}/match-{match}/context.npz")
            expected.update(f"train/fold-{fold}/match-{match}/{s}.npz" for s in stamp["protocol"]["scenarios"])
    if summary["selector_trained"]:
        expected.add("model-seal.json")
        for fold in stamp["protocol"]["folds"]:
            expected.add(f"models/fold-{fold}.json")
            expected.add(f"validation/fold-{fold}/context.npz")
            expected.update(f"validation/fold-{fold}/{s}.npz" for s in stamp["protocol"]["scenarios"])
    if set(seal["outputs"]) != expected:
        raise ValueError("Incomplete risk capsule")
    if (summary["registration_sha256"] != seal["registration_sha256"]
        or set(summary["folds"]) != {"0", "6"} or any(summary[k] for k in ("test_accessed", "target_accessed", "automatic_promotion"))):
        raise ValueError("Scope changed")
    for fold, row in summary["folds"].items():
        if row["train_gate"] != (row["train_oracle_gain"] >= stamp["protocol"]["oracle_gain_min"]):
            raise ValueError("Train gate changed")
    if summary["train_gate_passed"] != all(v["train_gate"] for v in summary["folds"].values()):
        raise ValueError("Overall gate changed")
    if summary["selector_trained"] != summary["train_gate_passed"] or summary["validation_accessed"] != summary["train_gate_passed"]:
        raise ValueError("Stages violate train gate")
    return 0


def replay():
    verify()
    stamp, summary = read(DEST/"registration.json"), read(DEST/"summary.json")
    p = stamp["protocol"]
    for fold in p["folds"]:
        cases = []
        for match_index in range(4):
            root = DEST/f"train/fold-{fold}/match-{match_index}"
            context = load_context(root/"context.npz")
            for index, scenario in enumerate(p["scenarios"]):
                with np.load(root/f"{scenario}.npz", allow_pickle=False) as z:
                    alternatives, keep = candidates(z["hypotheses"][0], z["probabilities"], z["run"])
                    np.testing.assert_array_equal(alternatives, z["hypotheses"])
                    np.testing.assert_array_equal(keep, z["keep"])
                    np.testing.assert_array_equal(features(alternatives, z["probabilities"], z["run"]), z["features"])
                    rewards = np.array([context.pm.spell_scores(context.gt, project(context, y), context.runs)["mean_f1"] for y in alternatives])
                    np.testing.assert_allclose(rewards, z["rewards"], atol=1e-12, rtol=0)
                    cases.append((p["weights"][index]/4, rewards))
        row = summary["folds"][str(fold)]
        baseline = sum(w*r[0] for w, r in cases)
        oracle = sum(w*max(r) for w, r in cases)
        np.testing.assert_allclose([baseline, oracle], [row["train_baseline"], row["train_oracle"]], atol=1e-12, rtol=0)
        if summary["selector_trained"]:
            model = read(DEST/f"models/fold-{fold}.json")
            root = DEST/f"validation/fold-{fold}"
            context = load_context(root/"context.npz")
            for scenario in p["scenarios"]:
                with np.load(root/f"{scenario}.npz", allow_pickle=False) as z:
                    alternatives, keep = candidates(z["hypotheses"][0], z["four_view_v2"], z["run"])
                    x = features(alternatives, z["four_view_v2"], z["run"])
                    np.testing.assert_array_equal(project(context, alternatives[choose(model, x, keep)]), z["hard__ranker"])
                    for name in row["validation"]:
                        actual = score(context, z["hard__"+name])
                        assert_metrics({scenario: actual}, row["validation"][name])
    return 0


def export():
    verify()
    names = sorted([*read(DEST/"result-seal.json")["outputs"], "registration.json", "result-seal.json", "compute.json"])
    path = ROOT/"evaluation/reports/handoff/northmind-sequence-risk-v1.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {n: file_hash(DEST/n) for n in names}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in names:
            # Fixed archive metadata makes repeated exports byte-identical.
            info = zipfile.ZipInfo("evaluation/reports/external/sequence-risk-v1/"+name, (2026, 10, 29, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, (DEST/name).read_bytes())
        z.writestr(zipfile.ZipInfo("MANIFEST.json", (2026, 10, 29, 0, 0, 0)), json.dumps(manifest, sort_keys=True, indent=2))
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None or any(hashlib.sha256(z.read("evaluation/reports/external/sequence-risk-v1/"+n)).hexdigest() != h for n, h in manifest.items()):
            raise ValueError("Archive changed")
    atomic_json(PUBLIC/"sequence-risk-v1.json", read(DEST/"summary.json"))
    atomic_json(PUBLIC/"sequence-risk-v1-package.json", {"archive": path.name, "sha256": file_hash(path),
        "bytes": path.stat().st_size, "files": len(names), "verified": True,
        "scope": "Public train candidate pools and rewards, canonical labels/mapping; selector/validation only if train gate passed",
        "raw_data_required_for_replay": False})
    return 0
