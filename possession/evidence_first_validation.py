"""Pre-registered, fixed public validation of pulse detection after view averaging.

No training, target loader, test prediction or calibration scoring is called.
Historical sources and artifacts remain read-only. All outputs are public IDSSE
aggregates under a separate ignored directory, never target-derived summaries.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from possession import st_tcn_target as transfer
from possession.evidence_first import aligned_evidence, mean_evidence, fuse_evidence, pulse_mask
from possession.external_data.schema import file_hash
from possession.learning.study import ScoreContext
from possession.learning.temporal import fit_priors
from possession.local_head_budget.metrics import measured
from possession.duration_balanced.metrics import matched
from possession.phase6.decoding import DirectionalDecoder
from possession.night.common import atomic_json
from possession.phase5.experiment import scenarios
from possession.phase7.model import SpatialE0
from possession.phase8.evaluation import predict as predict_teacher
from possession.temporal_probe.common import Store, splits_valid, public_path
from possession.temporal_probe.model import TemporalHead
from possession.temporal_probe.windows import predict as predict_head, emissions
from possession.transfer_validation import gates, _compact

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evidence_first_validation_v1.json"
DEST = ROOT / "evaluation/reports/external/evidence-first-validation-v1"
PREVIOUS = ROOT / "evaluation/reports/external/transfer-symmetry-validation-v1/summary.json"


class FoldStore(Store):
    """Enforce roles before opening any match, even if future callers change."""
    def __init__(self, records, split):
        if not splits_valid(split):
            raise ValueError("Invalid public split")
        self.allowed = set(split["train"] + [split["validation"]])
        super().__init__({k: records[k] for k in self.allowed})

    def match(self, key):
        if key not in self.allowed:
            raise ValueError("Only this fold's training and validation are accessible")
        return super().match(key)


def validate_protocol(protocol):
    if (protocol["version"] != "evidence-first-validation-v1"
            or protocol["part"] != "validation"
            or any(protocol[k] for k in ("train", "test_access", "target_access", "parameter_search"))
            or protocol["folds"] != [0, 6]
            or protocol["scenarios"] != ["clean", "missing40_5s", "missing40_15s"]
            or protocol["weights"] != [.5, .25, .25]
            or protocol["views"] != [[False, False], [False, True], [True, False], [True, True]]
            or protocol["references"] != ["registered_multiseed", "four_view_v2"]
            or protocol["guardrails"] != dict(robust_delta_min=0, clean_delta_min=0,
                short_recall_delta_min=0, long_recall_delta_min=-.01, macro_f1_delta_min=-.005)):
        raise ValueError("Unexpected validation-only fixed protocol")


def setup():
    protocol = transfer._json(PROTOCOL)
    validate_protocol(protocol)
    config, head_config, members, dependencies = transfer._transfer_spec()
    if [m["fold"] for m in members] != protocol["folds"]:
        raise ValueError("Incomplete folds")
    manifest = ROOT / "evaluation/reports/external/folds/idsse-only.json"
    records = transfer._json(manifest)["artifacts"]
    for record in records.values():
        public_path(record["path"])
    # Hash only public raw inputs used in train/validation roles. The Store
    # verifies their manifest hashes before loading, independently per fold.
    for member in members:
        selection = transfer._json(transfer.FUSION_DIR / f"selections/fold-{member['fold']}.json")
        split = selection["teacher"]["split"]
        if not splits_valid(split):
            raise ValueError("Overlapping split")
        for key in split["train"] + [split["validation"]]:
            path = public_path(records[key]["path"])
            if file_hash(path) != records[key]["sha256"]:
                raise ValueError("Public source changed")
            dependencies[str(path)] = records[key]["sha256"]
        for scenario in protocol["scenarios"]:
            path = reference_path(member["fold"], scenario)
            dependencies[str(path)] = file_hash(path)
    for path in (PROTOCOL, PREVIOUS, manifest, Path(__file__),
                 ROOT / "possession/evidence_first.py", ROOT / "possession/transfer_validation.py",
                 ROOT / "possession/st_tcn_target.py", ROOT / "scripts/evidence_first_validation.py",
                 ROOT / "scripts/run_local.py", ROOT / "tests/test_evidence_first.py"):
        dependencies[str(path)] = file_hash(path)
    stamp = {"protocol": protocol, "dependencies": dependencies,
             "environment": {"python": platform.python_version(),
                 "numpy": np.__version__, "torch": torch.__version__}}
    return protocol, config, head_config, members, records, stamp


def reference_path(fold, scenario):
    return transfer.FUSION_DIR / f"predictions/fold-{fold}/multiseed_scale_selective/validation/{scenario}.npz"


@contextmanager
def exclusive_run():
    DEST.mkdir(parents=True, exist_ok=True)
    lock = DEST / "running.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write("One evidence-first validation at a time.\n")
    try:
        yield
    finally:
        lock.unlink()


def assert_metrics(actual, expected):
    for scenario, row in actual.items():
        for name, value in row.items():
            if not np.isclose(value, expected[scenario][name], atol=1e-10, rtol=0):
                raise ValueError("Reconstructed reference metrics changed")


def short_changes(context, old_p, new_p, prior):
    decoder = DirectionalDecoder(context.view, prior, "hmm", .5)
    truth, _, old = matched(context, context.decoded(old_p, decoder))
    _, _, new = matched(context, context.decoded(new_p, decoder))
    short = np.array([(b-a)/25 < 3 for a, b, _ in truth])
    return {"gained": int((short & new & ~old).sum()),
            "lost": int((short & old & ~new).sum()),
            "retained": int((short & old & new).sum())}


def load_models(member, split, head_config):
    models, heads = [], []
    for name in ("teacher", "alternate"):
        state = torch.load(member[name], map_location="cpu", weights_only=False)
        if state["split"] != split:
            raise ValueError("Checkpoint split mismatch")
        model = SpatialE0("set_transformer", True)
        model.load_state_dict(state["model"])
        models.append(model.eval())
    for name in ("local", "multiscale"):
        model = TemporalHead(models[0].team_head.weight, models[0].team_head.bias,
            temporal=name == "multiscale", intervals=False,
            dilations=head_config["dilations"], kernel_size=head_config["kernel_size"],
            seed=head_config["seed"])
        state = torch.load(member[name], map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        heads.append(model.eval())
    return models, heads


def run():
    with exclusive_run():
        return _run()


def _run():
    protocol, config, head_config, members, records, stamp = setup()
    marker = DEST / "registration.json"
    if marker.exists() and transfer._json(marker) != stamp:
        raise ValueError("Immutable recipe changed; use a new version")
    if not marker.exists():
        atomic_json(marker, stamp)  # Written before inference or metric inspection.
    if (DEST / "result-seal.json").exists():
        return verify()
    fingerprint = file_hash(marker)
    previous = transfer._json(PREVIOUS)["folds"]
    registration = transfer._json(transfer.FUSION_DIR / "registration.json")
    torch.set_num_threads(protocol["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()

    def check_budget():
        if time.perf_counter() - start >= protocol["max_seconds"]:
            raise TimeoutError("Validation budget exhausted")

    summaries = {}
    for member in members:
        fold = member["fold"]
        selection = transfer._json(transfer.FUSION_DIR / f"selections/fold-{fold}.json")
        split = selection["teacher"]["split"]
        store = FoldStore(records, split)
        prior = fit_priors([store.harmonized(k) for k in split["train"]])
        view = store.harmonized(split["validation"])
        context = ScoreContext(store.match(split["validation"]), view)
        models, heads = load_models(member, split, head_config)
        metrics = {name: {} for name in ("registered_multiseed", "four_view_v2", "evidence_first")}
        parity, diagnostics = {}, {}
        for scenario, (degraded, _) in scenarios(view, split["validation"], registration["teacher_protocol"]).items():
            if scenario not in protocol["scenarios"]:
                raise ValueError("Unexpected scenario")
            with np.load(reference_path(fold, scenario), allow_pickle=False) as z:
                reference_p = z["probabilities"]
            # The historical file stores probabilities only. Its sealed hash,
            # shape and independently reconstructed identity establish parity.
            if reference_p.shape != (len(view.time), 3) or not np.isfinite(reference_p).all():
                raise ValueError("Invalid reference probabilities")
            mean_z, mean_alive = np.zeros(len(view.time)), np.zeros(len(view.time))
            views, masks = [], []
            observed = degraded.global_features[:, 3] > .5
            for swap, reflect in protocol["views"]:
                check_budget()
                transformed = transfer._transform_view(degraded, swap, reflect)
                p, _, hidden = predict_teacher(models[0], transformed, embeddings=True)
                alt_p, _, _ = predict_teacher(models[1], transformed)
                inputs = {"hidden": hidden, "time": view.time, "run": view.run}
                local, _, _ = predict_head(heads[0], inputs, config["core_frames"], config["batch_size"])
                multi, _, _ = predict_head(heads[1], inputs, config["core_frames"], config["batch_size"])
                z, alive = transfer._fuse_member(p, alt_p, local, multi, transformed, member["params"], config)
                if not swap and not reflect:
                    reconstructed = emissions(z, np.column_stack((alive, np.zeros(len(alive)), 1-alive)))
                    parity[scenario] = bool(np.allclose(reconstructed, reference_p, atol=3e-6, rtol=3e-5))
                    if not parity[scenario]:
                        raise ValueError("Identity parity failed")
                mean_z += .25 * (-1 if swap else 1) * z
                mean_alive += .25 * alive
                evidence = aligned_evidence(p, alt_p, local, multi, swap)
                views.append(evidence)
                masks.append(pulse_mask(evidence, observed, view.run, member["params"], config))
            combined = mean_evidence(views)
            new_z = fuse_evidence(combined, observed, view.run, member["params"], config)
            new_mask = pulse_mask(combined, observed, view.run, member["params"], config)
            alive_p = np.column_stack((mean_alive, np.zeros(len(mean_alive)), 1-mean_alive))
            old_p, new_p = emissions(mean_z, alive_p), emissions(new_z, alive_p)
            np.testing.assert_allclose(new_p[:, :2].sum(1), old_p[:, :2].sum(1), atol=2e-7, rtol=0)
            for name, probabilities in zip(metrics, (reference_p, old_p, new_p)):
                metrics[name][scenario] = _compact(measured(context, probabilities, prior))
            diagnostics[scenario] = {
                "view_pulse_frames": [int(m.sum()) for m in masks],
                "combined_pulse_frames": int(new_mask.sum()),
                "combined_only_frames": int((new_mask & ~np.any(masks, axis=0)).sum()),
                "view_only_frames": int((~new_mask & np.any(masks, axis=0)).sum()),
                "short_changes_vs_four_view": short_changes(context, old_p, new_p, prior),
            }
            check_budget()
            atomic_json(DEST / "progress.json", {"fold": fold, "scenario": scenario,
                "test_accessed": False, "target_accessed": False})
        assert_metrics(metrics["registered_multiseed"], previous[str(fold)]["baseline"])
        assert_metrics(metrics["four_view_v2"], previous[str(fold)]["four_view"])
        checks = {name: gates(metrics["evidence_first"], metrics[name], protocol)
                  for name in protocol["references"]}
        summary = {"metrics": metrics, "gates": checks, "identity_parity": parity,
            "historical_metrics_parity": True, "diagnostics": diagnostics,
            "passed": all(g["passed"] for g in checks.values()),
            "registration_sha256": fingerprint, "test_accessed": False, "target_accessed": False}
        atomic_json(DEST / f"fold-{fold}.json", summary)
        summaries[str(fold)] = summary
    if setup()[-1] != stamp:
        raise ValueError("Dependencies changed during inference")
    atomic_json(DEST / "summary.json", {"folds": summaries,
        "passes_both": all(s["passed"] for s in summaries.values()),
        "automatic_promotion": False, "test_accessed": False, "target_accessed": False,
        "registration_sha256": fingerprint, "seconds": time.perf_counter()-start})
    outputs = ["summary.json"] + [f"fold-{fold}.json" for fold in protocol["folds"]]
    atomic_json(DEST / "result-seal.json", {"registration_sha256": fingerprint,
        "outputs": {name: file_hash(DEST / name) for name in outputs}})
    return verify()


def verify():
    protocol, _, _, _, _, stamp = setup()
    marker = DEST / "registration.json"
    if transfer._json(marker) != stamp:
        raise ValueError("Registration or dependencies changed")
    fingerprint = file_hash(marker)
    seal = transfer._json(DEST / "result-seal.json")
    expected = {"summary.json", *(f"fold-{fold}.json" for fold in protocol["folds"])}
    if seal["registration_sha256"] != fingerprint or set(seal["outputs"]) != expected:
        raise ValueError("Incomplete result seal")
    if any(file_hash(DEST / name) != digest for name, digest in seal["outputs"].items()):
        raise ValueError("Changed result")
    summary = transfer._json(DEST / "summary.json")
    if (set(summary["folds"]) != {"0", "6"} or summary["automatic_promotion"]
            or summary["test_accessed"] or summary["target_accessed"]
            or summary["registration_sha256"] != fingerprint):
        raise ValueError("Invalid summary scope")
    previous = transfer._json(PREVIOUS)["folds"]
    for fold, row in summary["folds"].items():
        if transfer._json(DEST / f"fold-{fold}.json") != row:
            raise ValueError("Inconsistent fold result")
        if (row["test_accessed"] or row["target_accessed"]
                or row["registration_sha256"] != fingerprint
                or not row["historical_metrics_parity"]
                or set(row["identity_parity"]) != set(protocol["scenarios"])
                or not all(row["identity_parity"].values())):
            raise ValueError("Invalid fold scope/parity")
        assert_metrics(row["metrics"]["registered_multiseed"], previous[fold]["baseline"])
        assert_metrics(row["metrics"]["four_view_v2"], previous[fold]["four_view"])
        checks = {name: gates(row["metrics"]["evidence_first"], row["metrics"][name], protocol)
                  for name in protocol["references"]}
        if row["gates"] != checks or row["passed"] != all(g["passed"] for g in checks.values()):
            raise ValueError("Gate mismatch")
        for scenario in protocol["scenarios"]:
            change = row["diagnostics"][scenario]["short_changes_vs_four_view"]
            old = row["metrics"]["four_view_v2"][scenario]["short_recovered"]
            new = row["metrics"]["evidence_first"][scenario]["short_recovered"]
            if old != change["retained"] + change["lost"] or new != change["retained"] + change["gained"]:
                raise ValueError("Short-spell accounting mismatch")
    if summary["passes_both"] != all(r["passed"] for r in summary["folds"].values()):
        raise ValueError("Decision mismatch")
    return 0
