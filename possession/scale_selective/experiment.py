"""Validation-sealed evaluation of scale-selective temporal fusion on public IDSSE folds."""
import csv
import numpy as np
import torch

from possession.external_data.schema import file_hash
from possession.interval_probe.data import load
from possession.learning.study import ScoreContext
from possession.learning.temporal import fit_priors
from possession.local_head_budget.metrics import measured
from possession.night.common import atomic_json
from possession.phase5.experiment import scenarios
from possession.phase8.evaluation import predict as predict_teacher
from possession.temporal_probe.model import TemporalHead
from possession.temporal_probe.selection import objective, eligible, promotion
from possession.temporal_probe.teacher import load_teacher
from possession.temporal_probe.windows import predict as predict_head, emissions

from .common import DEST, PROBE_DIR, ROOT, Store, Budget, register, read_json, exclusive_run
from .fusion import (
    team_logit_from_probabilities,
    fuse_scale_selective,
    fused_emissions,
)


def load_frozen_probe_head(fold, arm, teacher, probe_config):
    state_teacher = torch.load(teacher["checkpoint"], map_location="cpu", weights_only=False)["model"]
    temporal = arm.startswith("multiscale")
    intervals = arm.endswith("intervals")
    model = TemporalHead(
        state_teacher["team_head.weight"],
        state_teacher["team_head.bias"],
        temporal=temporal,
        intervals=intervals,
        dilations=probe_config["dilations"],
        kernel_size=probe_config["kernel_size"],
        seed=probe_config["seed"],
    )
    ckpt_path = PROBE_DIR / f"training/fold-{fold}/{arm}/last.pt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def probe_export_paths(fold, key, config):
    folder = PROBE_DIR / f"embeddings/fold-{fold}" / key.replace(":", "_")
    result = {}
    for scenario in config["scenarios"]:
        path = folder / f"{scenario}.npz"
        stamp = path.with_suffix(".provenance.json")
        if not path.is_file() or not stamp.is_file():
            raise ValueError(f"Missing frozen temporal-probe export: {path}")
        prov = read_json(stamp)
        if prov["sha256"] != file_hash(path):
            raise ValueError(f"Provenance hash mismatch for {path}")
        result[scenario] = path
    return result


def get_alt_teacher_predictions(store, fold, alt_teacher, part, key, teacher_spec, config, budget, digest):
    if part == "test":
        assert_sealed(config, digest)
    view = store.harmonized(key)
    scen_map = scenarios(view, key, teacher_spec)
    model = None
    outputs = {}
    for scenario in config["scenarios"]:
        budget.check()
        cache_path = DEST / f"alt_teacher/fold-{fold}/{part}/{scenario}.npz"
        degraded, missing = scen_map[scenario]
        if cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as z:
                p = z["probabilities"]
        else:
            if model is None:
                model = load_teacher(alt_teacher)
            p, _, _ = predict_teacher(model, degraded, embeddings=False)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(".tmp")
            with tmp.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    probabilities=p,
                    actual_hidden_fraction=np.asarray(missing.mean(), dtype=np.float32),
                )
            tmp.replace(cache_path)
        outputs[scenario] = (degraded, p)
    return outputs


def prepare_part_inputs(store, record, fold, part, budget, digest):
    config = record["protocol"]
    probe_config = record["probe_protocol"]
    teacher_spec = record["teacher_protocol"]
    teacher = record["teachers"][str(fold)]
    alt_teacher = record["alt_teachers"][str(fold)]
    key = teacher["split"][part]
    if part == "test":
        assert_sealed(config, digest)

    view = store.harmonized(key)
    scen_map = scenarios(view, key, teacher_spec)
    export_paths = probe_export_paths(fold, key, config)
    local_head = load_frozen_probe_head(fold, "local_classification", teacher, probe_config)
    multi_head = load_frozen_probe_head(fold, "multiscale_classification", teacher, probe_config)
    alt_preds = get_alt_teacher_predictions(
        store, fold, alt_teacher, part, key, teacher_spec, config, budget, digest
    )

    packs = {}
    for scenario in config["scenarios"]:
        budget.check()
        data = load(export_paths[scenario])
        degraded, _ = scen_map[scenario]
        if not np.array_equal(data["native_index"], degraded.native_index):
            raise ValueError("Timeline mismatch between export and scenario view")
        z_local, _, _ = predict_head(local_head, data, config["core_frames"], config["batch_size"])
        z_multi, _, _ = predict_head(multi_head, data, config["core_frames"], config["batch_size"])
        z_teacher = team_logit_from_probabilities(data["probabilities"])
        _, alt_p = alt_preds[scenario]
        z_alt_teacher = team_logit_from_probabilities(alt_p)
        ball_available = degraded.global_features[:, 3] > 0.5
        packs[scenario] = {
            "z_local": z_local,
            "z_multi": z_multi,
            "z_teacher": z_teacher,
            "z_alt_teacher": z_alt_teacher,
            "teacher_p": data["probabilities"],
            "alt_teacher_p": alt_p,
            "ball_available": ball_available,
            "run": degraded.run,
        }
    return key, packs


def check_gates(metrics, baseline, local_metrics, config):
    gate = config["promotion"]
    checks = {
        "occlusion_floor": all(
            metrics[name]["spellF1"] >= baseline[name]["spellF1"] + gate["occlusion_floor_delta"]
            for name in config["scenarios"][1:]
        ),
        "long_recall_floor": (
            metrics["clean"]["long_recall"] >= baseline["clean"]["long_recall"] + gate["long_recall_floor_delta"]
        ),
        "robust_gain": objective(metrics, config) >= objective(baseline, config) + gate["robust_gain"],
        "short_recall_gain": (
            metrics["clean"]["short_recall"] >= baseline["clean"]["short_recall"] + gate["short_recall_gain"]
        ),
        "false_transitions_per_minute": all(
            metrics[name]["false_transitions_per_minute"]
            <= local_metrics[name]["false_transitions_per_minute"] + gate["false_transitions_per_minute_increase_max"]
            for name in config["scenarios"]
        ),
    }
    return {"passed": all(checks.values()), "checks": checks, "objective": objective(metrics, config)}


def choose_candidate(candidates, baseline, local_metrics, config):
    """Select best candidate on validation. Prefer candidates passing all gates; else eligible."""
    scored = []
    for idx, cand in enumerate(candidates):
        gate_info = check_gates(cand["metrics"], baseline, local_metrics, config)
        is_elig = eligible(cand["metrics"], baseline, config)
        # Composite score on validation: robust objective + 0.15 * clean short_recall
        val_rank_score = gate_info["objective"] + 0.15 * cand["metrics"]["clean"]["short_recall"]
        scored.append({
            "candidate_index": idx,
            "params": cand["params"],
            "metrics": cand["metrics"],
            "objective": gate_info["objective"],
            "val_rank_score": val_rank_score,
            "eligible": is_elig,
            "gate": gate_info,
        })
    passing = [c for c in scored if c["gate"]["passed"]]
    if passing:
        return max(passing, key=lambda c: (c["val_rank_score"], c["objective"]))
    elig = [c for c in scored if c["eligible"]]
    if elig:
        return max(elig, key=lambda c: (c["val_rank_score"], c["objective"]))
    return max(scored, key=lambda c: c["objective"])


def save_prediction(fold, arm, part, scenario, probabilities):
    if not (np.isfinite(probabilities).all() and (probabilities >= 0).all()
            and np.allclose(probabilities.sum(1), 1.0, atol=2e-6)):
        raise ValueError(f"Invalid probability distribution for {fold}/{arm}/{part}/{scenario}")
    path = DEST / f"predictions/fold-{fold}/{arm}/{part}/{scenario}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, probabilities=probabilities)
    tmp.replace(path)
    return path


def select_fold(store, record, fold, budget, digest):
    sel_path = DEST / f"selections/fold-{fold}.json"
    if sel_path.is_file():
        return read_json(sel_path)

    config = record["protocol"]
    teacher = record["teachers"][str(fold)]
    prior = fit_priors([store.harmonized(k) for k in teacher["split"]["train"]])
    key, packs = prepare_part_inputs(store, record, fold, "validation", budget, digest)
    context = ScoreContext(store.match(key), store.harmonized(key))

    base_arms = {}
    for arm in ("teacher", "local_classification", "multiscale_classification"):
        base_arms[arm] = {}
        for scenario, item in packs.items():
            budget.check()
            if arm == "teacher":
                p = item["teacher_p"]
            elif arm == "local_classification":
                p = emissions(item["z_local"], item["teacher_p"])
            else:
                p = emissions(item["z_multi"], item["teacher_p"])
            save_prediction(fold, arm, "validation", scenario, p)
            base_arms[arm][scenario] = measured(context, p, prior)

    baseline = base_arms["teacher"]
    local_metrics = base_arms["local_classification"]

    single_candidates = []
    multiseed_candidates = []
    for params in config["candidate_grid"]:
        budget.check()
        single_Params = dict(params, alt_seed_weight=0.0)
        single_m = {}
        multi_m = {}
        for scenario, item in packs.items():
            z_single = fuse_scale_selective(
                item["z_local"], item["z_multi"], item["ball_available"], item["run"],
                z_teacher=item["z_teacher"], z_alt_teacher=None, params=single_Params,
                obs_window=config["obs_window"], pulse_min_frames=config["pulse_min_frames"],
            )
            p_single = fused_emissions(z_single, item["teacher_p"])
            single_m[scenario] = measured(context, p_single, prior)

            z_ms = fuse_scale_selective(
                item["z_local"], item["z_multi"], item["ball_available"], item["run"],
                z_teacher=item["z_teacher"], z_alt_teacher=item["z_alt_teacher"], params=params,
                obs_window=config["obs_window"], pulse_min_frames=config["pulse_min_frames"],
            )
            p_ms = fused_emissions(
                z_ms, item["teacher_p"], item["alt_teacher_p"],
                alt_alive_weight=0.5 * float(params.get("alt_seed_weight", 0.0)),
            )
            multi_m[scenario] = measured(context, p_ms, prior)
        single_candidates.append({"params": single_Params, "metrics": single_m})
        multiseed_candidates.append({"params": dict(params), "metrics": multi_m})

    best_single = choose_candidate(single_candidates, baseline, local_metrics, config)
    best_multiseed = choose_candidate(multiseed_candidates, baseline, local_metrics, config)

    # Save validation predictions for the selected candidate of each new arm
    for scenario, item in packs.items():
        z_s = fuse_scale_selective(
            item["z_local"], item["z_multi"], item["ball_available"], item["run"],
            z_teacher=item["z_teacher"], z_alt_teacher=None, params=best_single["params"],
            obs_window=config["obs_window"], pulse_min_frames=config["pulse_min_frames"],
        )
        save_prediction(fold, "scale_selective", "validation", scenario, fused_emissions(z_s, item["teacher_p"]))

        z_m = fuse_scale_selective(
            item["z_local"], item["z_multi"], item["ball_available"], item["run"],
            z_teacher=item["z_teacher"], z_alt_teacher=item["z_alt_teacher"], params=best_multiseed["params"],
            obs_window=config["obs_window"], pulse_min_frames=config["pulse_min_frames"],
        )
        save_prediction(
            fold, "multiseed_scale_selective", "validation", scenario,
            fused_emissions(
                z_m, item["teacher_p"], item["alt_teacher_p"],
                alt_alive_weight=0.5 * float(best_multiseed["params"].get("alt_seed_weight", 0.0)),
            ),
        )

    arm_metrics = {
        "teacher": baseline,
        "local_classification": base_arms["local_classification"],
        "multiscale_classification": base_arms["multiscale_classification"],
        "scale_selective": best_single["metrics"],
        "multiseed_scale_selective": best_multiseed["metrics"],
    }
    arm_gates = {
        arm: check_gates(arm_metrics[arm], baseline, local_metrics, config)
        for arm in ("local_classification", "multiscale_classification", "scale_selective", "multiseed_scale_selective")
    }
    allowed_arms = ["teacher"] + [
        arm for arm in ("local_classification", "multiscale_classification", "scale_selective", "multiseed_scale_selective")
        if eligible(arm_metrics[arm], baseline, config)
    ]
    # Prefer arms that pass full promotion gates on own validation; break ties by objective
    passing_arms = [arm for arm in allowed_arms if arm != "teacher" and arm_gates[arm]["passed"]]
    winner_pool = passing_arms if passing_arms else allowed_arms
    winner_arm = max(winner_pool, key=lambda a: objective(arm_metrics[a], config))

    payload = {
        "fold": fold,
        "teacher": teacher,
        "alt_teacher": record["alt_teachers"][str(fold)],
        "winner": {
            "arm": winner_arm,
            "objective": objective(arm_metrics[winner_arm], config),
            "passed_all_gates": False if winner_arm == "teacher" else arm_gates[winner_arm]["passed"],
        },
        "selected_params": {
            "scale_selective": best_single["params"],
            "multiseed_scale_selective": best_multiseed["params"],
        },
        "arm_gates": arm_gates,
        "arm_objectives": {a: objective(m, config) for a, m in arm_metrics.items()},
        "validation_metrics": arm_metrics,
        "single_trials": single_candidates,
        "multiseed_trials": multiseed_candidates,
        "registration_sha256": digest,
        "test_used": False,
    }
    atomic_json(sel_path, payload)
    return payload


def seal_selections(config, digest):
    selections = {
        str(fold): file_hash(DEST / f"selections/fold-{fold}.json")
        for fold in config["pilots"]
    }
    record = {"selections": selections, "registration_sha256": digest, "test_used": False}
    seal_path = DEST / "selection-seal.json"
    if seal_path.is_file() and read_json(seal_path) != record:
        raise ValueError("Selection seal changed after sealing")
    if not seal_path.is_file():
        atomic_json(seal_path, record)
    return record


def assert_sealed(config, digest):
    seal_path = DEST / "selection-seal.json"
    if not seal_path.is_file():
        raise ValueError("Selections must be sealed before test inference")
    seal = read_json(seal_path)
    if seal["registration_sha256"] != digest or seal["test_used"]:
        raise ValueError("Invalid selection seal")
    for fold in config["pilots"]:
        if seal["selections"][str(fold)] != file_hash(DEST / f"selections/fold-{fold}.json"):
            raise ValueError(f"Selection for fold {fold} modified after seal")
    return seal


def evaluate_test_fold(store, record, fold, budget, digest):
    config = record["protocol"]
    assert_sealed(config, digest)
    row_path = DEST / f"rows/fold-{fold}.json"
    if row_path.is_file():
        return read_json(row_path)

    sel = read_json(DEST / f"selections/fold-{fold}.json")
    teacher = record["teachers"][str(fold)]
    prior = fit_priors([store.harmonized(k) for k in teacher["split"]["train"]])
    key, packs = prepare_part_inputs(store, record, fold, "test", budget, digest)
    context = ScoreContext(store.match(key), store.harmonized(key))

    rows = []
    for arm in config["arms"]:
        for scenario, item in packs.items():
            budget.check()
            if arm == "teacher":
                p = item["teacher_p"]
            elif arm == "local_classification":
                p = emissions(item["z_local"], item["teacher_p"])
            elif arm == "multiscale_classification":
                p = emissions(item["z_multi"], item["teacher_p"])
            elif arm == "scale_selective":
                params = sel["selected_params"]["scale_selective"]
                z_s = fuse_scale_selective(
                    item["z_local"], item["z_multi"], item["ball_available"], item["run"],
                    z_teacher=item["z_teacher"], z_alt_teacher=None, params=params,
                    obs_window=config["obs_window"], pulse_min_frames=config["pulse_min_frames"],
                )
                p = fused_emissions(z_s, item["teacher_p"])
            elif arm == "multiseed_scale_selective":
                params = sel["selected_params"]["multiseed_scale_selective"]
                z_m = fuse_scale_selective(
                    item["z_local"], item["z_multi"], item["ball_available"], item["run"],
                    z_teacher=item["z_teacher"], z_alt_teacher=item["z_alt_teacher"], params=params,
                    obs_window=config["obs_window"], pulse_min_frames=config["pulse_min_frames"],
                )
                p = fused_emissions(
                    z_m, item["teacher_p"], item["alt_teacher_p"],
                    alt_alive_weight=0.5 * float(params.get("alt_seed_weight", 0.0)),
                )
            else:
                raise ValueError(f"Unknown arm: {arm}")
            save_prediction(fold, arm, "test", scenario, p)
            rows.append({
                "fold": fold,
                "arm": arm,
                "scenario": scenario,
                "metrics": measured(context, p, prior),
            })
    atomic_json(row_path, rows)
    return rows


def verify_and_report(record, digest):
    config = record["protocol"]
    assert_sealed(config, digest)
    checks = []

    def add_check(cond, name):
        checks.append({"check": name, "passed": bool(cond)})

    for name, expected in record["sources"].items():
        add_check(file_hash(ROOT / name) == expected, f"source:{name}")
    for name, expected in record["references"].items():
        add_check(file_hash(ROOT / name) == expected, f"reference:{name}")

    rows = read_json(DEST / "all_rows.json")
    expected_rows = len(config["pilots"]) * len(config["arms"]) * len(config["scenarios"])
    add_check(len(rows) == expected_rows, "complete_test_rows")

    seal_time = (DEST / "selection-seal.json").stat().st_mtime_ns
    pred_files = list((DEST / "predictions").glob("**/*.npz"))
    test_pred_files = list((DEST / "predictions").glob("**/test/*.npz"))
    add_check(len(pred_files) == 2 * expected_rows, "all_prediction_files_present")
    add_check(
        bool(test_pred_files) and all(p.stat().st_mtime_ns >= seal_time for p in test_pred_files),
        "test_predictions_created_after_seal",
    )

    for path in pred_files:
        with np.load(path, allow_pickle=False) as z:
            p = z["probabilities"]
            valid_p = bool(np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1.0, atol=2e-6))
            add_check(valid_p, f"normalized:{path.relative_to(DEST)}")

    selections = {}
    for fold in config["pilots"]:
        sel = read_json(DEST / f"selections/fold-{fold}.json")
        selections[str(fold)] = sel
        add_check(not sel["test_used"], f"fold-{fold}_no_test_in_selection")
        add_check(sel["registration_sha256"] == digest, f"fold-{fold}_registration_hash")

    passed = all(item["passed"] for item in checks)
    atomic_json(DEST / "verification.json", {"passed": passed, "count": len(checks), "checks": checks})
    if not passed:
        raise ValueError("Scale-selective integrity verification failed")

    systems = {}
    for arm in config["arms"]:
        systems[arm] = {
            scen: float(np.mean([r["metrics"]["spellF1"] for r in rows if r["arm"] == arm and r["scenario"] == scen]))
            for scen in config["scenarios"]
        }
        systems[arm]["robust_objective"] = objective(
            {scen: {"spellF1": systems[arm][scen]} for scen in config["scenarios"]},
            config,
        )
        systems[arm]["short_recovered"] = int(sum(
            r["metrics"]["short_recovered"] for r in rows if r["arm"] == arm and r["scenario"] == "clean"
        ))
        systems[arm]["short_gt"] = int(sum(
            r["metrics"]["short_gt"] for r in rows if r["arm"] == arm and r["scenario"] == "clean"
        ))
        systems[arm]["clean_long_recall"] = float(np.mean([
            r["metrics"]["long_recall"] for r in rows if r["arm"] == arm and r["scenario"] == "clean"
        ]))
        systems[arm]["clean_macroF1"] = float(np.mean([
            r["metrics"]["macroF1"] for r in rows if r["arm"] == arm and r["scenario"] == "clean"
        ]))
        systems[arm]["clean_false_HA_per_min"] = float(np.mean([
            r["metrics"]["false_transitions_per_minute"] for r in rows if r["arm"] == arm and r["scenario"] == "clean"
        ]))

    selected_rows = []
    own_summary = {}
    for fold in config["pilots"]:
        sel = selections[str(fold)]
        w_arm = sel["winner"]["arm"]
        selected_rows.extend([r for r in rows if r["fold"] == fold and r["arm"] == w_arm])
        own_summary[str(fold)] = {
            "winner": sel["winner"],
            "selected_params": sel["selected_params"],
            "arm_objectives": sel["arm_objectives"],
            "arm_gates": {k: v["passed"] for k, v in sel["arm_gates"].items()},
        }

    systems["own_validation_selected"] = {
        scen: float(np.mean([r["metrics"]["spellF1"] for r in selected_rows if r["scenario"] == scen]))
        for scen in config["scenarios"]
    }
    systems["own_validation_selected"]["robust_objective"] = objective(
        {scen: {"spellF1": systems["own_validation_selected"][scen]} for scen in config["scenarios"]},
        config,
    )
    systems["own_validation_selected"]["short_recovered"] = int(sum(
        r["metrics"]["short_recovered"] for r in selected_rows if r["scenario"] == "clean"
    ))
    systems["own_validation_selected"]["short_gt"] = int(sum(
        r["metrics"]["short_gt"] for r in selected_rows if r["scenario"] == "clean"
    ))
    systems["own_validation_selected"]["clean_long_recall"] = float(np.mean([
        r["metrics"]["long_recall"] for r in selected_rows if r["scenario"] == "clean"
    ]))
    systems["own_validation_selected"]["clean_macroF1"] = float(np.mean([
        r["metrics"]["macroF1"] for r in selected_rows if r["scenario"] == "clean"
    ]))
    systems["own_validation_selected"]["clean_false_HA_per_min"] = float(np.mean([
        r["metrics"]["false_transitions_per_minute"] for r in selected_rows if r["scenario"] == "clean"
    ]))

    both_scale_selective_pass = all(
        selections[str(f)]["arm_gates"]["scale_selective"]["passed"] for f in config["pilots"]
    )
    both_multiseed_pass = all(
        selections[str(f)]["arm_gates"]["multiseed_scale_selective"]["passed"] for f in config["pilots"]
    )

    summary_payload = {
        "systems": systems,
        "own_selections": own_summary,
        "scale_selective_passes_both_validations": both_scale_selective_pass,
        "multiseed_passes_both_validations": both_multiseed_pass,
        "verification_passed": passed,
        "verification_count": len(checks),
        "folds": config["pilots"],
    }
    atomic_json(DEST / "summary.json", summary_payload)

    lines = [
        "# Scale-Selective & Multi-Seed Temporal Fusion — Public Pilot Results",
        "",
        "## Validation Selections (Sealed Before Test)",
        "",
        "| Fold | Arm | Robust Val Objective | Short Recall (Clean) | Long Recall (Clean) | Passes All Gates |",
        "|---|---|---:|---:|---:|---|",
    ]
    for fold in config["pilots"]:
        sel = selections[str(fold)]
        for arm in config["arms"]:
            vm = sel["validation_metrics"][arm]["clean"]
            obj = sel["arm_objectives"][arm]
            passed_gate = "baseline" if arm == "teacher" else str(sel["arm_gates"][arm]["passed"])
            lines.append(
                f"| {fold} | {arm} | {obj:.5f} | {vm['short_recovered']}/{vm['short_gt']} ({vm['short_recall']:.4f}) | {vm['long_recall']:.4f} | {passed_gate} |"
            )

    lines += [
        "",
        "## Held-Out Public Test Means (2 Pilot Folds)",
        "",
        "| System | Clean spellF1 | Missing40/5s | Missing40/15s | Robust Mean | Clean macroF1 | Short (<3s) | Long (>=10s) | False HA/min |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in systems.items():
        lines.append(
            f"| {name} | {s['clean']:.5f} | {s['missing40_5s']:.5f} | {s['missing40_15s']:.5f} | "
            f"{s['robust_objective']:.5f} | {s['clean_macroF1']:.5f} | "
            f"{s['short_recovered']}/{s['short_gt']} | {s['clean_long_recall']:.4f} | {s['clean_false_HA_per_min']:.3f} |"
        )
    (DEST / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with (DEST / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["fold", "arm", "scenario", "spellF1", "macroF1", "short_recall", "short_recovered", "short_gt", "long_recall", "false_transitions_per_minute"],
        )
        writer.writeheader()
        for r in rows:
            m = r["metrics"]
            writer.writerow({
                "fold": r["fold"],
                "arm": r["arm"],
                "scenario": r["scenario"],
                "spellF1": m["spellF1"],
                "macroF1": m["macroF1"],
                "short_recall": m["short_recall"],
                "short_recovered": m["short_recovered"],
                "short_gt": m["short_gt"],
                "long_recall": m["long_recall"],
                "false_transitions_per_minute": m["false_transitions_per_minute"],
            })
    return 0


def run(stage="all"):
    torch.use_deterministic_algorithms(True)
    with exclusive_run():
        record, records = register()
        config = record["protocol"]
        torch.set_num_threads(config["cpu_threads"])
        digest = file_hash(DEST / "registration.json")
        store = Store(records)
        budget = Budget(config["max_compute_seconds"], DEST)
        status = "failed"
        try:
            if stage in ("all", "select"):
                for fold in config["pilots"]:
                    select_fold(store, record, fold, budget, digest)
                seal_selections(config, digest)
            if stage in ("all", "test"):
                assert_sealed(config, digest)
                rows = []
                for fold in config["pilots"]:
                    rows.extend(evaluate_test_fold(store, record, fold, budget, digest))
                atomic_json(DEST / "all_rows.json", rows)
            if stage in ("all", "test", "report"):
                verify_and_report(record, digest)
            status = "complete"
        finally:
            budget.close(stage, status)
    return 0

