"""Two predeclared public validation arms; historical sources remain immutable."""
import hashlib
from pathlib import Path
import time
import numpy as np
import torch

from possession import evidence_first_validation as base
from possession import st_tcn_target as transfer
from possession.evidence_first import aligned_evidence, mean_evidence, fuse_evidence
from possession.pulse_consensus import candidates
from possession.external_data.schema import file_hash
from possession.learning.study import ScoreContext
from possession.learning.temporal import fit_priors
from possession.local_head_budget.metrics import measured
from possession.night.common import atomic_json
from possession.phase5.experiment import scenarios
from possession.phase8.evaluation import predict as predict_teacher
from possession.temporal_probe.windows import predict as predict_head, emissions
from possession.transfer_validation import gates, _compact

ROOT = base.ROOT
DEST = ROOT / "evaluation/reports/external/pulse-consensus-validation-v1"
PROTOCOL = ROOT / "configs/pulse_consensus_validation_v1.json"
ARMS = ("half_residual", "boundary_consensus")
TASK_NAMES = ("validate-pulse-consensus", "verify-pulse-consensus", "test-pulse-consensus")


def parent_integrity():
    marker = base.DEST / "registration.json"
    seal = transfer._json(base.DEST / "result-seal.json")
    if seal["registration_sha256"] != file_hash(marker):
        raise ValueError("Changed parent registration")
    for name, digest in seal["outputs"].items():
        if name not in ("summary.json", "fold-0.json", "fold-6.json"):
            raise ValueError("Unexpected parent result path")
        if file_hash(base.DEST / name) != digest:
            raise ValueError("Changed parent result")
    for name, digest in transfer._json(marker)["dependencies"].items():
        path = Path(name)
        if file_hash(path) == digest:
            continue
        if path != ROOT / "scripts/run_local.py":
            raise ValueError("Changed parent dependency")
        # Only remove our three exact task declarations, leaving the rest of
        # the launcher's bytes intact. Never rewrite the historical registry.
        content = path.read_bytes()
        lines = content.splitlines(keepends=True)
        kept = [line for line in lines if not any(
            line.lstrip().startswith(('"'+task+'":').encode()) for task in TASK_NAMES)]
        if len(lines)-len(kept) != 3 or hashlib.sha256(b"".join(kept)).hexdigest() != digest:
            raise ValueError("Parent launcher changed beyond new task declarations")


def setup():
    parent_integrity()
    parent, config, heads, members, records, stamp = base.setup()
    protocol = transfer._json(PROTOCOL)
    if (protocol["version"] != "pulse-consensus-validation-v1"
            or protocol["candidates"] != list(ARMS)
            or protocol["half_residual_weight"] != .5
            or protocol["boundary_support_min"] != 3 or protocol["overlap_min"] != .5
            or any(protocol[k] for k in ("train", "test_access", "target_access", "parameter_search"))
            or any(protocol[k] != parent[k] for k in ("folds", "scenarios", "weights", "references", "guardrails"))):
        raise ValueError("Fixed public-only protocol required")
    dependencies = stamp["dependencies"]
    for path in (PROTOCOL, Path(__file__), ROOT / "possession/pulse_consensus.py",
                 ROOT / "scripts/pulse_consensus_validation.py", ROOT / "tests/test_pulse_consensus.py",
                 base.DEST / "registration.json", base.DEST / "result-seal.json", base.DEST / "summary.json"):
        dependencies[str(path)] = file_hash(path)
    stamp["protocol"] = protocol
    return protocol, config, heads, members, records, stamp


def decisions(rows, protocol):
    return {arm: {ref: gates(rows[arm], rows[ref], protocol)
                  for ref in protocol["references"]} for arm in ARMS}


def run():
    DEST.mkdir(parents=True, exist_ok=True)
    lock = DEST / "running.lock"
    with lock.open("x") as stream:
        stream.write("Exclusive public validation\n")
    try:
        return _run()
    finally:
        lock.unlink()


def _run():
    protocol, config, head_config, members, records, stamp = setup()
    marker = DEST / "registration.json"
    if marker.exists() and transfer._json(marker) != stamp:
        raise ValueError("Immutable registration changed")
    if not marker.exists():
        atomic_json(marker, stamp)
    if (DEST / "result-seal.json").exists():
        return verify()
    fingerprint = file_hash(marker)
    parent = transfer._json(base.DEST / "summary.json")["folds"]
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    summaries = {}
    for member in members:
        fold = member["fold"]
        split = transfer._json(transfer.FUSION_DIR / f"selections/fold-{fold}.json")["teacher"]["split"]
        store = base.FoldStore(records, split)
        prior = fit_priors([store.harmonized(k) for k in split["train"]])
        view = store.harmonized(split["validation"])
        context = ScoreContext(store.match(split["validation"]), view)
        models, heads = base.load_models(member, split, head_config)
        metrics = {name: {} for name in (*protocol["references"], "evidence_first", *ARMS)}
        parity, diagnostics = {}, {}
        for scenario, (degraded, _) in scenarios(view, split["validation"], {}).items():
            with np.load(base.reference_path(fold, scenario), allow_pickle=False) as z:
                reference = z["probabilities"]
            old_z, alive = np.zeros(len(view.time)), np.zeros(len(view.time))
            views = []
            for swap, reflect in ((False, False), (False, True), (True, False), (True, True)):
                if time.perf_counter()-start > protocol["max_seconds"]:
                    raise TimeoutError("Budget exhausted")
                transformed = transfer._transform_view(degraded, swap, reflect)
                p, _, hidden = predict_teacher(models[0], transformed, embeddings=True)
                alt, _, _ = predict_teacher(models[1], transformed)
                inputs = {"hidden": hidden, "time": view.time, "run": view.run}
                local, _, _ = predict_head(heads[0], inputs, config["core_frames"], config["batch_size"])
                multi, _, _ = predict_head(heads[1], inputs, config["core_frames"], config["batch_size"])
                z, a = transfer._fuse_member(p, alt, local, multi, transformed, member["params"], config)
                if not swap and not reflect:
                    reconstructed = emissions(z, np.column_stack((a, np.zeros(len(a)), 1-a)))
                    parity[scenario] = bool(np.allclose(reconstructed, reference, atol=3e-6, rtol=3e-5))
                    if not parity[scenario]:
                        raise ValueError("Identity parity failed")
                old_z += .25*(-1 if swap else 1)*z
                alive += .25*a
                views.append(aligned_evidence(p, alt, local, multi, swap))
            obs = degraded.global_features[:, 3] > .5
            new_z = fuse_evidence(mean_evidence(views), obs, view.run, member["params"], config)
            signals, weight = candidates(old_z, new_z, views, obs, view.run, member["params"], config)
            alive_p = np.column_stack((alive, np.zeros(len(alive)), 1-alive))
            probabilities = {"registered_multiseed": reference, "four_view_v2": emissions(old_z, alive_p),
                             "evidence_first": emissions(new_z, alive_p),
                             **{arm: emissions(z, alive_p) for arm, z in signals.items()}}
            for name, p in probabilities.items():
                metrics[name][scenario] = _compact(measured(context, p, prior))
            diagnostics[scenario] = {"consensus_frames": int(weight.sum()),
                "short_changes_vs_four_view": {arm: base.short_changes(context,
                    probabilities["four_view_v2"], probabilities[arm], prior) for arm in ARMS}}
        for name in (*protocol["references"], "evidence_first"):
            base.assert_metrics(metrics[name], parent[str(fold)]["metrics"][name])
        summaries[str(fold)] = {"metrics": metrics, "gates": decisions(metrics, protocol),
            "diagnostics": diagnostics, "identity_parity": parity, "parent_metrics_parity": True}
        atomic_json(DEST / f"fold-{fold}.json", summaries[str(fold)])
    if setup()[-1] != stamp:
        raise ValueError("Inputs changed during inference")
    result = {"folds": summaries, "passes_both": {arm: all(
        all(g["passed"] for g in row["gates"][arm].values()) for row in summaries.values()) for arm in ARMS},
        "registration_sha256": fingerprint, "automatic_promotion": False,
        "test_accessed": False, "target_accessed": False, "seconds": time.perf_counter()-start}
    atomic_json(DEST / "summary.json", result)
    atomic_json(DEST / "result-seal.json", {"registration_sha256": fingerprint,
        "outputs": {name: file_hash(DEST / name) for name in ("summary.json", "fold-0.json", "fold-6.json")}})
    return verify()


def verify():
    protocol, _, _, _, _, stamp = setup()
    marker = DEST / "registration.json"
    seal = transfer._json(DEST / "result-seal.json")
    if transfer._json(marker) != stamp or seal["registration_sha256"] != file_hash(marker):
        raise ValueError("Changed registration")
    if set(seal["outputs"]) != {"summary.json", "fold-0.json", "fold-6.json"}:
        raise ValueError("Incomplete outputs")
    for name, digest in seal["outputs"].items():
        if file_hash(DEST / name) != digest:
            raise ValueError("Changed result")
    result = transfer._json(DEST / "summary.json")
    if (result["registration_sha256"] != file_hash(marker) or result["automatic_promotion"]
            or result["test_accessed"] or result["target_accessed"] or set(result["folds"]) != {"0", "6"}):
        raise ValueError("Invalid result scope")
    parent = transfer._json(base.DEST / "summary.json")["folds"]
    for fold, row in result["folds"].items():
        if row != transfer._json(DEST / f"fold-{fold}.json") or row["gates"] != decisions(row["metrics"], protocol):
            raise ValueError("Invalid decisions")
        if set(row["identity_parity"]) != set(protocol["scenarios"]) or not all(row["identity_parity"].values()):
            raise ValueError("Invalid parity")
        for ref in (*protocol["references"], "evidence_first"):
            base.assert_metrics(row["metrics"][ref], parent[fold]["metrics"][ref])
        for scenario in protocol["scenarios"]:
            for arm in ARMS:
                change = row["diagnostics"][scenario]["short_changes_vs_four_view"][arm]
                if (row["metrics"][arm][scenario]["short_recovered"] != change["retained"]+change["gained"]
                        or row["metrics"]["four_view_v2"][scenario]["short_recovered"] != change["retained"]+change["lost"]):
                    raise ValueError("Inconsistent short accounting")
    passed = {arm: all(all(g["passed"] for g in r["gates"][arm].values())
                       for r in result["folds"].values()) for arm in ARMS}
    if passed != result["passes_both"]:
        raise ValueError("Decision mismatch")
    return 0
