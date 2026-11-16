"""New exports or verified read-only reuse of historical public embeddings."""
import numpy as np

from possession.external_data.schema import file_hash
from possession.interval_probe.data import native_targets, load
from possession.night.common import atomic_json
from possession.phase5.experiment import scenarios
from possession.phase8.evaluation import predict

from .common import DEST, REPORTS, read_json
from .teacher import load_teacher


def get_exports(store, key, fold, teacher, spec, budget):
    """Only export test after caller has sealed all pilot selections."""
    view = store.harmonized(key)
    result, model, targets = {}, None, None
    for name, (degraded, missing) in scenarios(view, key, spec).items():
        budget.check()
        path = DEST / f"embeddings/fold-{fold}" / key.replace(":", "_") / (name + ".npz")
        stamp = path.with_suffix(".provenance.json")
        expected = {"teacher_sha256": file_hash(teacher["checkpoint"]),
                    "public_data_sha256": store.records[key]["sha256"], "scenario": name}
        if path.exists():
            record = read_json(stamp)
            if any(record[k] != v for k, v in expected.items()) or record["sha256"] != file_hash(path):
                raise ValueError("Embedding export provenance mismatch")
        else:
            old_root = REPORTS / "phase8-interval-v1"
            old_registration = old_root / "registration.json"
            old = old_root / f"embeddings/fold-{fold}" / key.replace(":", "_") / (name + ".npz")
            reused = False
            if teacher.get("source") != "fresh_public_pilots" and old.is_file() and old_registration.is_file():
                old_record = read_json(old_registration)
                old_teacher = old_record["teachers"][str(fold)]
                if (old_teacher["seed"] == teacher["seed"] and old_teacher["split"] == teacher["split"]
                        and old_record["teacher_hashes"][str(fold)] == expected["teacher_sha256"]):
                    value = load(old)
                    reused = True
            if not reused:
                if model is None:
                    model = load_teacher(teacher)
                if targets is None:
                    targets = native_targets(store.match(key), view)
                p, q, hidden = predict(model, degraded, embeddings=True)
                value = dict(hidden=hidden, probabilities=p, transition=q, time=view.time,
                             run=view.run, native_index=view.native_index,
                             actual_hidden_fraction=np.asarray(missing.mean()), **targets)
            validate_export(value, view)
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".tmp")
            with temp.open("wb") as stream:
                np.savez_compressed(stream, **value)
            temp.replace(path)
            atomic_json(stamp, dict(expected, sha256=file_hash(path),
                                   reused=str(old) if reused else None))
        result[name] = path
    return result


def validate_export(value, view):
    n = len(view.time)
    if value["hidden"].shape != (n, 64) or value["probabilities"].shape != (n, 3):
        raise ValueError("Incorrect embedding/probability dimensions")
    if not np.isfinite(value["hidden"]).all() or not np.isfinite(value["probabilities"]).all():
        raise ValueError("Nonfinite public export")
    if (value["probabilities"] < 0).any() or not np.allclose(value["probabilities"].sum(1), 1, atol=2e-6):
        raise ValueError("Invalid teacher probabilities")
    for name in ("time", "run", "native_index"):
        if not np.array_equal(value[name], getattr(view, name)):
            raise ValueError("Export timeline mismatch")
    if value["label"].shape != (n,) or not np.isin(value["label"], (0, 1, 2)).all():
        raise ValueError("Complete native H/A/D labels required")
