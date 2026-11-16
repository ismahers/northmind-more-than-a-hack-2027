"""Explicit separately budgeted continuation; never reset the original budget."""
import torch

from possession.external_data.schema import file_hash
from possession.learning.study import ScoreContext
from possession.learning.temporal import fit_priors
from possession.night.common import atomic_json
from possession.phase5.experiment import scenarios
from possession.phase8.evaluation import predict
from possession.temporal_probe.common import ROOT, REPORTS, Store, Budget, read_json, exclusive_run
from possession.temporal_probe.experiment import metrics
from possession.temporal_probe.selection import objective
from possession.temporal_probe.teacher import TEACHERS, train_teacher, load_teacher

DEST = REPORTS / "temporal-teacher-continuation-v1"


def main():
    with exclusive_run():
        config = read_json(ROOT / "configs/temporal_teacher_continuation_protocol.json")
        parent_path = TEACHERS / "registration.json"
        parent = read_json(parent_path)
        digest = file_hash(parent_path)
        compute_path = TEACHERS / "compute.json"
        compute = read_json(compute_path)
        if compute["seconds"] < parent["bootstrap"]["bootstrap_max_compute_seconds"]:
            raise ValueError("Original bootstrap budget is not exhausted; continue original stage instead")
        for name, expected in parent["sources"].items():
            if file_hash(ROOT / name) != expected:
                raise ValueError("Original teacher algorithm changed; continuation rejected")
        manifest_path = REPORTS / "folds/idsse-only.json"
        if file_hash(manifest_path) != parent["manifest_sha256"]:
            raise ValueError("Original public manifest changed")
        marker = DEST / "registration.json"
        fixed = {"config": config, "parent_registration_sha256": digest,
                 "parent_compute_sha256": file_hash(compute_path),
                 "continuation_source_sha256": file_hash(__file__)}
        if marker.is_file():
            registration = read_json(marker)
            if any(registration[key] != value for key, value in fixed.items()):
                raise ValueError("Supplementary allocation changed")
        else:
            checkpoints = {}
            for fold in parent["bootstrap"]["pilots"]:
                for seed in parent["bootstrap"]["seeds"]:
                    for stage in ("base", "binary"):
                        path = TEACHERS / f"training/fold-{fold}/seed-{seed}/{stage}/last.pt"
                        if path.is_file():
                            checkpoints[str(path.relative_to(ROOT))] = file_hash(path)
            atomic_json(marker, dict(fixed, starting_checkpoints=checkpoints, test_used=False))
        store = Store(read_json(manifest_path)["artifacts"])
        spec, bootstrap = parent["protocol"], parent["bootstrap"]
        torch.set_num_threads(spec["cpu_threads"])
        budget, status = Budget(config["supplementary_max_compute_seconds"], DEST), "failed"
        try:
            for fold in bootstrap["pilots"]:
                selection = TEACHERS / f"selections/fold-{fold}.json"
                if selection.is_file():
                    continue
                parts = parent["parts"][str(fold)]
                prior = fit_priors([store.harmonized(key) for key in parts["train"]])
                key = parts["validation"]
                context = ScoreContext(store.match(key), store.harmonized(key))
                trials = []
                for seed in bootstrap["seeds"]:
                    train_teacher(store, parts, fold, seed, spec, budget, digest)
                    checkpoint = TEACHERS / f"training/fold-{fold}/seed-{seed}/binary/last.pt"
                    teacher = {"seed": seed, "split": parts, "checkpoint": str(checkpoint), "source": "fresh_public_pilots"}
                    model = load_teacher(teacher)
                    measured = {}
                    for scenario, (view, _) in scenarios(context.view, key, spec).items():
                        budget.check()
                        p, _, _ = predict(model, view)
                        measured[scenario] = metrics(context, p, prior)
                    trials.append({"teacher": teacher, "metrics": measured, "objective": objective(measured, spec),
                                   "checkpoint_sha256": file_hash(checkpoint)})
                winner = max(trials, key=lambda value: value["objective"])
                atomic_json(selection, {"teacher": winner["teacher"], "trials": trials, "test_used": False,
                                        "registration_sha256": digest,
                                        "continuation_registration_sha256": file_hash(marker)})
            status = "complete"
        finally:
            budget.close("continue_original_teachers", status)
        return 0
