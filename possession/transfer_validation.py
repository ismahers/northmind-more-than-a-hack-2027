"""Fixed public validation of four-view symmetrization; never accesses target/test."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from possession.external_data.schema import file_hash
from possession.learning.study import ScoreContext
from possession.learning.temporal import fit_priors
from possession.local_head_budget.metrics import measured
from possession.night.common import atomic_json
from possession.phase5.experiment import scenarios
from possession.phase7.model import SpatialE0
from possession.phase8.evaluation import predict as predict_teacher
from possession.temporal_probe.common import Store, splits_valid, public_path
from possession.temporal_probe.model import TemporalHead
from possession.temporal_probe.windows import predict as predict_head, emissions
from possession import st_tcn_target as transfer

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/transfer_symmetry_validation.json"
DEST = ROOT / "evaluation/reports/external/transfer-symmetry-validation-v1"


def gates(candidate, baseline, protocol):
    guard = protocol["guardrails"]
    objective = lambda rows: sum(w * rows[s]["spellF1"]
                                for w, s in zip(protocol["weights"], protocol["scenarios"]))
    checks = {
        "robust": objective(candidate) >= objective(baseline) + guard["robust_delta_min"],
        "clean": candidate["clean"]["spellF1"] >= baseline["clean"]["spellF1"] + guard["clean_delta_min"],
        "short": candidate["clean"]["short_recall"] >= baseline["clean"]["short_recall"] + guard["short_recall_delta_min"],
        "long": candidate["clean"]["long_recall"] >= baseline["clean"]["long_recall"] + guard["long_recall_delta_min"],
        "macro": candidate["clean"]["macroF1"] >= baseline["clean"]["macroF1"] + guard["macro_f1_delta_min"],
    }
    return {"checks": checks, "passed": all(checks.values()),
            "baseline_objective": objective(baseline), "candidate_objective": objective(candidate)}


def _compact(metrics):
    return {k: metrics[k] for k in ("spellF1", "macroF1", "short_recall",
            "short_recovered", "short_gt", "long_recall", "false_transitions_per_minute")}


def run():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["part"] != "validation" or protocol["test_access"] or protocol["target_access"]:
        raise ValueError("Validation-only public protocol required")
    config, head_config, members, dependencies = transfer._transfer_spec()
    registration = transfer._json(transfer.FUSION_DIR / "registration.json")
    manifest_path = ROOT / "evaluation/reports/external/folds/idsse-only.json"
    records = transfer._json(manifest_path)["artifacts"]
    for record in records.values():
        public_path(record["path"])
    for member in members:
        for scenario in protocol["scenarios"]:
            path = transfer.FUSION_DIR / f"predictions/fold-{member['fold']}/multiseed_scale_selective/validation/{scenario}.npz"
            dependencies[str(path)] = file_hash(path)
    for path in (PROTOCOL, Path(__file__), ROOT / "scripts/transfer_symmetry_validation.py",
                 ROOT / "possession/st_tcn_target.py"):
        dependencies[str(path)] = file_hash(path)
    stamp = {"protocol": protocol, "dependencies": dependencies,
             "numpy": np.__version__, "torch": torch.__version__}
    DEST.mkdir(parents=True, exist_ok=True)
    marker = DEST / "registration.json"
    if marker.exists() and transfer._json(marker) != stamp:
        raise ValueError("Immutable validation recipe changed; use another version")
    if not marker.exists():
        atomic_json(marker, stamp)
    torch.set_num_threads(protocol["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    store = Store(records)
    start = time.perf_counter()
    def check_budget():
        if time.perf_counter() - start > protocol["max_seconds"]:
            raise TimeoutError("Validation budget exhausted")
    summaries = {}
    for member in members:
        fold = member["fold"]
        out = DEST / f"fold-{fold}.json"
        if out.exists():
            summaries[str(fold)] = transfer._json(out)
            continue
        selection = transfer._json(transfer.FUSION_DIR / f"selections/fold-{fold}.json")
        split = selection["teacher"]["split"]
        if not splits_valid(split):
            raise ValueError("Overlapping public split")
        # This fold loads only training for priors and its own validation for scoring.
        # Other folds' test roles do not enter its inference or parameter selection.
        prior = fit_priors([store.harmonized(k) for k in split["train"]])
        key = split["validation"]
        view = store.harmonized(key)
        context = ScoreContext(store.match(key), view)
        models = []
        for name in ("teacher", "alternate"):
            state = torch.load(member[name], map_location="cpu", weights_only=False)
            if state["split"] != split:
                raise ValueError("Checkpoint trained on another split")
            model = SpatialE0("set_transformer", True)
            model.load_state_dict(state["model"])
            models.append(model.eval())
        heads = []
        for name in ("local", "multiscale"):
            model = TemporalHead(models[0].team_head.weight, models[0].team_head.bias,
                temporal=name == "multiscale", intervals=False,
                dilations=head_config["dilations"], kernel_size=head_config["kernel_size"],
                seed=head_config["seed"])
            state = torch.load(member[name], map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"])
            heads.append(model.eval())
        baseline, candidate, parity = {}, {}, {}
        for scenario, (degraded, _) in scenarios(view, key, registration["teacher_protocol"]).items():
            if scenario not in protocol["scenarios"]:
                continue
            path = transfer.FUSION_DIR / f"predictions/fold-{fold}/multiseed_scale_selective/validation/{scenario}.npz"
            with np.load(path, allow_pickle=False) as z:
                reference_p = z["probabilities"]
            mean_z, mean_alive = np.zeros(len(view.time)), np.zeros(len(view.time))
            for swap in (False, True):
                for reflect in (False, True):
                    check_budget()
                    transformed = transfer._transform_view(degraded, swap, reflect)
                    p, _, hidden = predict_teacher(models[0], transformed, embeddings=True)
                    alt_p, _, _ = predict_teacher(models[1], transformed, embeddings=False)
                    inputs = {"hidden": hidden, "time": view.time, "run": view.run}
                    z_local, _, _ = predict_head(heads[0], inputs, config["core_frames"], config["batch_size"])
                    z_multi, _, _ = predict_head(heads[1], inputs, config["core_frames"], config["batch_size"])
                    z, alive = transfer._fuse_member(p, alt_p, z_local, z_multi,
                                                     transformed, member["params"], config)
                    if not swap and not reflect:
                        reconstructed = emissions(z, np.column_stack((alive, np.zeros(len(alive)), 1-alive)))
                        parity[scenario] = bool(np.allclose(reconstructed, reference_p, atol=3e-6, rtol=3e-5))
                        if not parity[scenario]:
                            raise ValueError("Identity inference does not reproduce registered probabilities")
                    mean_z += .25 * (-1 if swap else 1) * z
                    mean_alive += .25 * alive
            candidate_p = emissions(mean_z, np.column_stack((mean_alive, np.zeros(len(mean_alive)), 1-mean_alive)))
            baseline[scenario] = _compact(measured(context, reference_p, prior))
            candidate[scenario] = _compact(measured(context, candidate_p, prior))
            atomic_json(DEST / "progress.json", {"fold": fold, "scenario": scenario, "test_accessed": False})
        summary = {"baseline": baseline, "four_view": candidate, "identity_parity": parity,
                   "gate": gates(candidate, baseline, protocol),
                   "test_accessed": False, "target_accessed": False}
        atomic_json(out, summary)
        summaries[str(fold)] = summary
    # Immutable public inputs must still match after the run.
    if any(file_hash(Path(path)) != expected for path, expected in dependencies.items()):
        raise ValueError("Validation inputs changed during inference")
    atomic_json(DEST / "summary.json", {"folds": summaries,
        "passes_both": all(s["gate"]["passed"] for s in summaries.values()),
        "automatic_promotion": False, "test_accessed": False, "target_accessed": False})
    return 0
