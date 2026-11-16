"""Continue/train/select both folds, seal, then evaluate held-out public tests."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from possession.external_data.schema import file_hash
from possession.interval_probe.data import load
from possession.learning.study import ScoreContext
from possession.learning.temporal import fit_priors
from possession.night.common import atomic_json
from possession.temporal_probe.common import Store, Budget, read_json
from possession.temporal_probe.windows import predict, emissions
from possession.duration_balanced.training import init_head, paths
from .common import DEST, PARENT, ROOT, register, cap_path
from .training import continue_fold
from .selection import choose
from .metrics import measured, diagnose_components


def assert_sealed(spec, digest):
    seal = read_json(DEST / "selection-seal.json")
    if seal["registration_sha256"] != digest or seal["test_used"]:
        raise ValueError("Invalid selection seal")
    for fold in spec["pilots"]:
        if file_hash(DEST / f"selections/fold-{fold}.json") != seal["selections"][str(fold)]:
            raise ValueError("Both selections must be frozen before test inference")
    return seal


def seal_selections(spec, digest):
    selections = {str(fold): file_hash(DEST / f"selections/fold-{fold}.json") for fold in spec["pilots"]}
    value = {"registration_sha256": digest, "selections": selections, "test_used": False}
    path = DEST / "selection-seal.json"
    if path.exists() and read_json(path) != value:
        raise ValueError("Selection seal changed")
    if not path.exists():
        atomic_json(path, value)


def head(fold, cap, teacher, spec, digest, parent_digest):
    model = init_head(fold, teacher, spec)
    state = torch.load(cap_path(fold, cap), map_location="cpu", weights_only=False)
    if len(state["history"]) != cap or state["registration_sha256"] != (parent_digest if cap == 8 else digest):
        raise ValueError("Wrong frozen local-head budget")
    model.load_state_dict(state["model"])
    return model


def score_cap(store, fold, cap, teacher, spec, digest, parent_digest, part, prior, budget):
    if part == "test":
        assert_sealed(spec, digest)
    model = head(fold, cap, teacher, spec, digest, parent_digest)
    key = teacher["split"][part]
    context = ScoreContext(store.match(key), store.harmonized(key))
    results = {}; reference = {}; alive_checks = {}
    for scenario, path in paths(teacher, key, spec).items():
        budget.check()
        data = load(path)
        if not np.array_equal(data["native_index"], context.view.native_index):
            raise ValueError("Frozen embedding native clock mismatch")
        team, _, _ = predict(model, data, spec["core_frames"], spec["batch_size"])
        probabilities = emissions(team, data["probabilities"])
        error = float(np.max(np.abs(probabilities[:, 2]-data["probabilities"][:, 2])))
        if error > 2e-7 or not np.allclose(probabilities.sum(1), 1, atol=2e-6):
            raise ValueError("Frozen live/dead emission changed")
        if cap == 8:
            # Reproduce the existing unweighted head before interpreting extensions.
            old = PARENT / f"predictions/fold-{fold}/original/{part}/{scenario}.npz"
            with np.load(old, allow_pickle=False) as z:
                if not np.array_equal(probabilities, z["probabilities"]):
                    raise ValueError("Eight-epoch parent predictions not exactly reproduced")
        file = DEST / f"predictions/fold-{fold}/cap-{cap}/{part}/{scenario}.npz"
        if file.exists():
            with np.load(file, allow_pickle=False) as z:
                if not np.array_equal(probabilities, z["probabilities"]):
                    raise ValueError("Frozen prediction changed")
        else:
            file.parent.mkdir(parents=True, exist_ok=True)
            with file.with_suffix(".tmp").open("wb") as stream:
                np.savez_compressed(stream, probabilities=probabilities)
            file.with_suffix(".tmp").replace(file)
        results[scenario] = measured(context, probabilities, prior)
        # Stable conditional BCE is diagnostic; never selects caps by test.
        valid = (data["label"] >= 0) & (data["label"] < 2)
        if valid.any():
            x = team[valid].astype(np.float64); y = (data["label"][valid] == 0)
            results[scenario]["conditional_BCE"] = float(np.mean(np.logaddexp(0., x)-y*x))
        reference[scenario] = measured(context, data["probabilities"], prior)
        alive_checks[scenario] = error
    return results, reference, alive_checks


def select_fold(store, fold, teacher, spec, digest, parent_digest, budget):
    selected = DEST / f"selections/fold-{fold}.json"
    if selected.exists():
        return
    continue_fold(fold, teacher, spec, budget, digest, parent_digest)
    prior = fit_priors([store.harmonized(k) for k in teacher["split"]["train"]])
    trials, reference, unchanged = {}, None, {}
    for cap in spec["budgets"]:
        atomic_json(DEST / "progress.json", {"stage": "validation", "fold": fold, "cap": cap})
        trials[str(cap)], ref, unchanged[str(cap)] = score_cap(store, fold, cap, teacher, spec, digest,
                                                             parent_digest, "validation", prior, budget)
        if reference is None:
            reference = ref
    choice = choose(trials, reference, spec)
    atomic_json(selected, {**choice, "fold": fold, "trials": trials, "teacher": reference,
                           "checkpoint_hashes": {str(cap): file_hash(cap_path(fold, cap)) for cap in spec["budgets"]},
                           "alive_emission_max_errors": unchanged, "registration_sha256": digest})


def run(stage="all"):
    torch.use_deterministic_algorithms(True)
    record = register()
    spec = record["protocol"]; torch.set_num_threads(spec["cpu_threads"])
    if stage == "audit":
        return 0
    digest = file_hash(DEST / "registration.json")
    lock = DEST / "running.lock"
    with lock.open("x") as stream:
        stream.write("One CPU local-budget experiment at a time.\n")
    budget = Budget(spec["max_compute_seconds"], DEST)
    status = "failed"
    try:
        store = Store(record["records"])
        if stage in ("all", "select"):
            for fold in spec["pilots"]:
                select_fold(store, fold, record["teachers"][str(fold)], spec, digest,
                            record["parent_registration_sha256"], budget)
            seal_selections(spec, digest)
        if stage in ("all", "test"):
            assert_sealed(spec, digest)
            rows = []
            for fold in spec["pilots"]:
                teacher = record["teachers"][str(fold)]
                prior = fit_priors([store.harmonized(k) for k in teacher["split"]["train"]])
                for cap in spec["budgets"]:
                    atomic_json(DEST / "progress.json", {"stage": "test", "fold": fold, "cap": cap})
                    metrics, baseline, unchanged = score_cap(store, fold, cap, teacher, spec, digest,
                        record["parent_registration_sha256"], "test", prior, budget)
                    for scenario in spec["scenarios"]:
                        rows.append({"fold": fold, "cap": cap, "scenario": scenario, "metrics": metrics[scenario]})
                        if cap == 8:
                            rows.append({"fold": fold, "cap": "teacher", "scenario": scenario, "metrics": baseline[scenario]})
            atomic_json(DEST / "all_rows.json", rows)
        if stage in ("all", "test", "report"):
            assert_sealed(spec, digest)
            selections = {str(fold): read_json(DEST / f"selections/fold-{fold}.json") for fold in spec["pilots"]}
            common = [cap for cap in spec["budgets"][1:] if all(s["gates"][str(cap)]["passed"] for s in selections.values())]
            if not common and not (DEST / "D_diagnosis.json").exists():
                diagnosis = []
                for fold in spec["pilots"]:
                    teacher = record["teachers"][str(fold)]
                    prior = fit_priors([store.harmonized(k) for k in teacher["split"]["train"]])
                    for part in ("validation", "test"):
                        key = teacher["split"][part]
                        context = ScoreContext(store.match(key), store.harmonized(key))
                        data = load(paths(teacher, key, spec)["clean"])
                        for cap in sorted({8, selections[str(fold)]["winner"]}):
                            budget.check()
                            model = head(fold, cap, teacher, spec, digest, record["parent_registration_sha256"])
                            team, _, _ = predict(model, data, spec["core_frames"], spec["batch_size"])
                            p = emissions(team, data["probabilities"])
                            diagnosis.append({"fold": fold, "part": part, "cap": cap,
                                              "diagnosis": diagnose_components(context, p, data["probabilities"], prior)})
                atomic_json(DEST / "D_diagnosis.json", {"after_seal": True, "selection_used_oracles": False, "rows": diagnosis})
            atomic_json(DEST / "decision.json", {"common_budget_passes_both_validation_gates": common,
                        "close_8_12_16_line": not bool(common), "no_automatic_expansion": True})
        status = "complete"
    finally:
        budget.close(stage, status)
        lock.unlink()
    if stage in ("all", "test", "report"):
        from .report import report
        return report(record)
    return 0
