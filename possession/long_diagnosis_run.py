"""Validation-only long-spell diagnosis. All detailed outputs stay local.

No target loader, test inference, training or historical run() is called. The
only files intended for human/agent reading are public-inventory.json and
public-summary.json: explicit aggregates of public IDSSE validation only.
"""
import hashlib
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from possession import evidence_first_validation as base
from possession import st_tcn_target as transfer
from possession.external_data.schema import file_hash
from possession.night.common import atomic_json
from possession.temporal_probe.common import public_path, splits_valid

ROOT = base.ROOT
DEST = ROOT / "evaluation/reports/external/long-diagnosis-v3"
PROTOCOL = ROOT / "configs/long_diagnosis_v3.json"
PULSE = ROOT / "evaluation/reports/external/pulse-consensus-validation-v1"
TASKS = ("freeze-long-diagnosis", "run-long-diagnosis", "verify-long-diagnosis", "test-long-diagnosis")


def validate_protocol(protocol):
    from possession.long_diagnosis import IOUS, REPAIRS
    if (protocol["version"] != "long-diagnosis-v3" or protocol["folds"] != [0, 6]
            or protocol["part"] != "validation" or protocol["fps"] != 25
            or protocol["long_seconds"] != 10 or protocol["short_seconds"] != 3
            or protocol["scenarios"] != ["clean", "missing40_5s", "missing40_15s"]
            or protocol["references"] != ["registered_multiseed", "four_view_v2"]
            or protocol["systems"] != ["registered_multiseed", "four_view_v2", "evidence_first", "half_residual", "boundary_consensus"]
            or protocol["ious"] != list(IOUS) or protocol["repairs"] != list(REPAIRS)
            or protocol["joint_groups"] != ["all_long", "missed_long_at_0.5"]
            or protocol["bootstrap_draws"] != 2000 or protocol["bootstrap_seed"] != 20261028
            or any(protocol[k] for k in ("train", "test_access", "target_access", "parameter_search"))):
        raise ValueError("Only the predeclared diagnostic protocol is allowed")


def historical_integrity(folder):
    """Verify historical bytes; allow only our exact added launcher task lines.

    Evidence-first predates the three pulse commands. No old registry is edited
    and no other source difference is excused by this extension.
    """
    marker = folder / "registration.json"
    record = transfer._json(marker)
    seal = transfer._json(folder / "result-seal.json")
    if seal["registration_sha256"] != file_hash(marker):
        raise ValueError("Historical registration mismatch")
    if set(seal["outputs"]) != {"summary.json", "fold-0.json", "fold-6.json"}:
        raise ValueError("Historical output scope mismatch")
    for name, digest in seal["outputs"].items():
        if file_hash(folder / name) != digest:
            raise ValueError("Historical output changed")
    for name, digest in record["dependencies"].items():
        path = Path(name)
        if file_hash(path) == digest:
            continue
        if path != ROOT / "scripts/run_local.py":
            atomic_json(DEST / "public-preflight-status.json", {"stage": "changed_dependency",
                        "dependency": str(path.relative_to(ROOT)), "scope": "public metadata only"})
            raise ValueError("Historical dependency changed")
        remove = TASKS + (("validate-pulse-consensus", "verify-pulse-consensus", "test-pulse-consensus")
                          if folder == base.DEST else ())
        lines = path.read_bytes().splitlines(keepends=True)
        kept = [line for line in lines if not any(
            line.lstrip().startswith(('"' + name + '":').encode()) for name in remove)]
        import subprocess
        allowed = {
            "49470a32e4454028829a023d69a3115cce601eeedfa37ec61e0dc89169421b32": "8eed019",
            "916eac5607ad16312abd3a96030b513f4a8121547fab1b525ef0919064d3b1a2": "0b33fa5",
        }
        if digest not in allowed or len(lines)-len(kept) != len(remove):
            raise ValueError("Unknown launcher ancestry")
        original = subprocess.check_output(["git", "show", allowed[digest]+":scripts/run_local.py"], cwd=ROOT)
        if b"".join(kept).replace(b"\r\n", b"\n") != original.replace(b"\r\n", b"\n"):
            raise ValueError("Launcher text differs from pinned historical commit")


def snapshot():
    # setup hashes public sources and metadata; it does not infer or touch target.
    def stage(name):
        atomic_json(DEST / "public-preflight-status.json", {"stage": name,
                    "scope": "public provenance checks only; no exception text"})
    stage("base_setup")
    _, config, head_config, members, records, stamp = base.setup()
    stage("evidence_history")
    historical_integrity(base.DEST)
    stage("pulse_history")
    historical_integrity(PULSE)
    dependencies = stamp["dependencies"]
    for folder in (base.DEST, PULSE):
        for name in ("registration.json", "result-seal.json", "summary.json", "fold-0.json", "fold-6.json"):
            path = folder / name
            dependencies[str(path)] = file_hash(path)
    dependencies[str(PROTOCOL)] = file_hash(PROTOCOL)
    probe_path = transfer.PROBE_DIR / "registration.json"
    probe = transfer._json(probe_path)
    teacher_path = transfer.TEACHERS_DIR / "registration.json"
    teacher_record = transfer._json(teacher_path)
    partitions = []
    for member in members:
        fold = member["fold"]
        stage(f"fold_{fold}_split")
        selection = transfer._json(transfer.FUSION_DIR / f"selections/fold-{fold}.json")
        split = selection["teacher"]["split"]
        if (not splits_valid(split) or selection["alt_teacher"]["split"] != split
                or probe["teachers"][str(fold)]["split"] != split
                or probe["teachers"][str(fold)]["seed"] != selection["teacher"]["seed"]
                or teacher_record["parts"][str(fold)] != split or teacher_record["test_used"]):
            raise ValueError("Incompatible fold ancestry")
        for name in ("teacher", "alternate"):
            stage(f"fold_{fold}_{name}_provenance")
            state = torch.load(member[name], map_location="cpu", weights_only=False)
            seed = selection["teacher" if name == "teacher" else "alt_teacher"]["seed"]
            reference_key = str(member[name].relative_to(ROOT))
            if (state["split"] != split or state["seed"] != seed
                    or state["registration_sha256"] != file_hash(teacher_path)
                    or probe["references"].get(reference_key) != file_hash(member[name])):
                raise ValueError("Teacher used protected match")
        for name in ("local", "multiscale"):
            stage(f"fold_{fold}_{name}_provenance")
            state = torch.load(member[name], map_location="cpu", weights_only=False)
            if state["registration_sha256"] != file_hash(probe_path):
                raise ValueError("Head ancestry mismatch")
        partitions.append({"fold": fold, **split,
            "teacher_seed": selection["teacher"]["seed"], "alternate_seed": selection["alt_teacher"]["seed"],
            "head_seed": head_config["seed"], "params": member["params"],
            "checkpoints": {key: str(member[key].relative_to(ROOT)) for key in ("teacher", "alternate", "local", "multiscale")}})
    stage("verified")
    return config, head_config, members, records, dependencies, partitions


def freeze():
    DEST.mkdir(parents=True, exist_ok=True)
    validate_protocol(transfer._json(PROTOCOL))
    # Explicit expected paths only; no inventory or availability check of target.
    expected = [base.PROTOCOL, base.PREVIOUS, PROTOCOL,
                ROOT / "evaluation/reports/external/folds/idsse-only.json",
                transfer.FUSION_DIR / "registration.json", transfer.FUSION_DIR / "selection-seal.json",
                transfer.PROBE_DIR / "registration.json", transfer.TEACHERS_DIR / "registration.json"]
    for folder in (base.DEST, PULSE):
        expected.extend(folder / name for name in ("registration.json", "result-seal.json", "summary.json", "fold-0.json", "fold-6.json"))
    for fold in (0, 6):
        expected.append(transfer.FUSION_DIR / f"selections/fold-{fold}.json")
        expected.extend(transfer.TEACHERS_DIR / f"training/fold-{fold}/seed-{seed}/binary/last.pt" for seed in (2027, 2028))
        expected.extend(transfer.PROBE_DIR / f"training/fold-{fold}/{arm}_classification/last.pt" for arm in ("local", "multiscale"))
        expected.extend(base.reference_path(fold, s) for s in ("clean", "missing40_5s", "missing40_15s"))
    missing = [str(p.relative_to(ROOT)) for p in expected if not p.is_file()]
    inventory = {"scope": "public IDSSE metadata only", "missing_required": missing,
                 "test_accessed": False, "target_accessed": False, "training": False}
    manifest_path = ROOT / "evaluation/reports/external/folds/idsse-only.json"
    if manifest_path.is_file():
        from possession.phase6.experiment import split
        manifest = transfer._json(manifest_path)
        if len(manifest["artifacts"]) != 7 or not all(k.startswith("idsse:") for k in manifest["artifacts"]):
            raise ValueError("Unexpected public manifest")
        inventory["provisional_partitions_from_manifest"] = [{"fold": fold, **split(manifest, fold)} for fold in (0, 6)]
        inventory["partition_status"] = "provisional until checked against sealed selections and checkpoints"
    external = ROOT / "evaluation/reports/external"
    inventory["available_public_experiment_directories"] = sorted(p.name for p in external.iterdir() if p.is_dir())
    sought = {"temporal-teachers-v1", "temporal-probe-v1", "scale-selective-v1",
              "transfer-symmetry-validation-v1", "evidence-first-validation-v1", "pulse-consensus-validation-v1"}
    copies = []
    # Names only, within the user's projects folder; prune assets and environments.
    for folder, directories, files in os.walk(ROOT.parent, followlinks=False):
        directories[:] = [d for d in directories if d not in {"data", ".git", ".venv", "venv", "node_modules", "__pycache__", ".next"}]
        for name in directories:
            if name in sought:
                copies.append(str(Path(folder) / name))
        directories[:] = [d for d in directories if d not in sought]
    inventory["possible_copies_by_directory_name"] = sorted(copies)
    inventory["copy_search_scope"] = str(ROOT.parent)
    atomic_json(DEST / "public-inventory.json", inventory)
    if missing:
        return 1
    config, heads, members, records, dependencies, partitions = snapshot()
    # These historical writers saved aggregate JSON, not emissions. Enumerate only
    # those three public validation directories, never historical test exports.
    cache_files = {str(folder.relative_to(ROOT)): [str(p.relative_to(ROOT)) for p in sorted(folder.rglob("*.npz"))]
                   for folder in (base.PREVIOUS.parent, base.DEST, PULSE)}
    inventory.update(partitions=partitions, public_validation_cache_files=cache_files,
        reused="six sealed multiseed probability arrays; eight checkpoints; selections; historical aggregate metrics",
        missing_diagnostic_artifacts=["four_view_v2 emissions", "evidence_first emissions", "half_residual emissions",
                                      "boundary_consensus emissions", "per-view logits/pulse masks"],
        regeneration="inference only from same-fold sealed checkpoints; historical metric parity required; no training",
        dependency_hashes={str(Path(p).relative_to(ROOT)): h for p, h in dependencies.items()},
        historical_failures_preserved=True)
    frozen = DEST / "frozen-inputs.json"
    value = {"dependencies": dependencies, "partitions": partitions}
    if frozen.exists() and transfer._json(frozen) != value:
        raise ValueError("Frozen input changed")
    if not frozen.exists():
        atomic_json(frozen, value)
    atomic_json(DEST / "public-inventory.json", inventory)
    return 0


def canonical(context, values):
    """Same nearest mapping as ScoreContext, with no interpolation across runs."""
    values = np.asarray(values)
    out = np.zeros((len(context.gt),) + values.shape[1:], values.dtype)
    for selected, mapping, output in context.parts:
        out[output] = values[selected][mapping]
    return out


def infer(member, view, models, heads, config, scenario):
    """Reconstruct only the already frozen systems, including diagnostic evidence."""
    from possession.evidence_first import aligned_evidence, mean_evidence, fuse_evidence, pulse_mask
    from possession.pulse_consensus import candidates
    from possession.phase8.evaluation import predict as predict_teacher
    from possession.temporal_probe.windows import predict as predict_head, emissions
    with np.load(base.reference_path(member["fold"], scenario), allow_pickle=False) as z:
        reference = z["probabilities"].copy()
    if reference.shape != (len(view.time), 3) or not np.isfinite(reference).all():
        raise ValueError("Invalid frozen reference")
    old_z, alive = np.zeros(len(view.time)), np.zeros(len(view.time))
    views, pulses, logits = [], [], []
    observed = view.global_features[:, 3] > .5
    for swap, reflect in ((False, False), (False, True), (True, False), (True, True)):
        transformed = transfer._transform_view(view, swap, reflect)
        p, _, hidden = predict_teacher(models[0], transformed, embeddings=True)
        alt, _, _ = predict_teacher(models[1], transformed)
        inputs = {"hidden": hidden, "time": view.time, "run": view.run}
        local, _, _ = predict_head(heads[0], inputs, config["core_frames"], config["batch_size"])
        multi, _, _ = predict_head(heads[1], inputs, config["core_frames"], config["batch_size"])
        z, a = transfer._fuse_member(p, alt, local, multi, transformed, member["params"], config)
        if not swap and not reflect:
            reconstructed = emissions(z, np.column_stack((a, np.zeros(len(a)), 1-a)))
            if not np.allclose(reconstructed, reference, atol=3e-6, rtol=3e-5):
                raise ValueError("Registered identity parity failed")
        old_z += .25*(-1 if swap else 1)*z
        alive += .25*a
        aligned = aligned_evidence(p, alt, local, multi, swap)
        views.append(aligned)
        pulses.append(pulse_mask(aligned, observed, view.run, member["params"], config))
        logits.append((-1 if swap else 1)*z)
    mean = mean_evidence(views)
    new_z = fuse_evidence(mean, observed, view.run, member["params"], config)
    arms, gate = candidates(old_z, new_z, views, observed, view.run, member["params"], config)
    alive_p = np.column_stack((alive, np.zeros(len(alive)), 1-alive))
    result = {"p__registered_multiseed": reference, "p__four_view_v2": emissions(old_z, alive_p),
              "p__evidence_first": emissions(new_z, alive_p),
              **{"p__"+key: emissions(value, alive_p) for key, value in arms.items()},
              "time": view.time, "run": view.run, "observed": observed,
              "view_logits": np.stack(logits, axis=1), "view_pulses": np.stack(pulses, axis=1),
              "mean_pulse": pulse_mask(mean, observed, view.run, member["params"], config), "consensus_gate": gate}
    return result


def source_hashes():
    names = ("possession/long_diagnosis.py", "possession/long_diagnosis_run.py", "scripts/long_diagnosis.py",
             "scripts/run_local.py", "tests/test_long_diagnosis.py", "docs/LONG_POSSESSION_DIAGNOSTIC_PROTOCOL.md",
             "possession/pulse_consensus.py", "evaluation/possession_metrics.py", "scripts/public_handoff.py")
    return {name: file_hash(ROOT / name) for name in names}


def run():
    DEST.mkdir(parents=True, exist_ok=True)
    lock = DEST / "running.lock"
    with lock.open("x") as stream:
        stream.write("Exclusive public diagnostic\n")
    try:
        return _run()
    finally:
        lock.unlink()


def _run():
    from possession.duration_balanced.metrics import hard_head
    from possession.learning.study import ScoreContext
    from possession.learning.temporal import fit_priors, TemporalDecoder
    from possession.local_head_budget.metrics import measured
    from possession.long_diagnosis import GlobalMatcher, analyze, paired_uncertainty
    from possession.phase5.experiment import scenarios
    from possession.phase6.decoding import DirectionalDecoder
    from possession.transfer_validation import _compact
    if freeze():
        return 1  # Inventory first; no fallback model, training or scoring.
    protocol = transfer._json(PROTOCOL)
    config, head_config, members, records, dependencies, partitions = snapshot()
    registration = {"protocol": protocol, "dependencies": dependencies, "partitions": partitions,
                    "sources": source_hashes(), "environment": {"python": platform.python_version(),
                    "numpy": np.__version__, "torch": torch.__version__}}
    marker = DEST / "registration.json"
    if marker.exists() and transfer._json(marker) != registration:
        raise ValueError("Immutable diagnostic changed; use a new version")
    if not marker.exists():
        atomic_json(marker, registration)
    if (DEST / "result-seal.json").exists():
        return verify()
    fingerprint = file_hash(marker)
    torch.set_num_threads(protocol["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    previous = transfer._json(PULSE / "summary.json")
    start = time.monotonic()
    rows, outputs = [], {}
    for member in members:
        fold = member["fold"]
        partition = next(p for p in partitions if p["fold"] == fold)
        split = {key: partition[key] for key in ("train", "calibration", "validation", "test")}
        store = base.FoldStore(records, split)
        prior = fit_priors([store.harmonized(key) for key in split["train"]])
        view = store.harmonized(split["validation"])
        context = ScoreContext(store.match(split["validation"]), view)
        matcher = GlobalMatcher(context.pm, context.gt)
        models = heads = None
        for scenario, (degraded, missing) in scenarios(view, split["validation"], {}).items():
            if time.monotonic()-start > protocol["max_seconds"]:
                raise TimeoutError("Fixed diagnostic budget exhausted")
            stem = f"fold-{fold}/{scenario}"
            cache = DEST / f"cache/{stem}.npz"
            seal = cache.with_suffix(".json")
            if cache.exists() and seal.exists():
                cached = transfer._json(seal)
                if cached != {"registration_sha256": fingerprint, "sha256": file_hash(cache)}:
                    raise ValueError("Diagnostic cache changed")
                with np.load(cache, allow_pickle=False) as z:
                    arrays = {key: z[key].copy() for key in z.files}
            else:
                if models is None:
                    models, heads = base.load_models(member, split, head_config)
                arrays = infer(member, degraded, models, heads, config, scenario)
                arrays["missing"] = missing
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache, **arrays)
                atomic_json(seal, {"registration_sha256": fingerprint, "sha256": file_hash(cache)})
            if (not np.array_equal(arrays["time"], view.time) or not np.array_equal(arrays["run"], view.run)
                    or not np.array_equal(arrays["missing"], missing)
                    or not np.array_equal(arrays["observed"], degraded.global_features[:, 3] > .5)):
                raise ValueError("Timeline or registered occlusion mask mismatch")
            systems, evidence = {}, {}
            decoder = DirectionalDecoder(view, prior, "hmm", .5)
            for name in protocol["systems"]:
                p = arrays["p__"+name]
                actual = _compact(measured(context, p, prior))
                expected = previous["folds"][str(fold)]["metrics"][name][scenario]
                base.assert_metrics({scenario: actual}, {scenario: expected})
                conditional = (p[:, 1] > p[:, 0]).astype(np.int8)
                systems[name] = {"conditional": context.decoded(np.eye(3)[conditional], TemporalDecoder({"kind": "argmax"}, None)),
                                 "raw": hard_head(context, p), "hmm": context.decoded(p, decoder)}
                pulse = arrays["view_pulses"][:, 0] if name == "registered_multiseed" else arrays["mean_pulse"] if name == "evidence_first" else arrays["view_pulses"].mean(1)
                evidence[name] = {"home_margin": canonical(context, np.log(np.clip(p[:, 0], 1e-7, 1)/np.clip(p[:, 1], 1e-7, 1))),
                    "alive": canonical(context, p[:, :2].sum(1)), "ball_observed": canonical(context, arrays["observed"]),
                    "pulse_support": canonical(context, pulse), "consensus_gate": canonical(context, arrays["consensus_gate"]),
                    "view_agreement": canonical(context, np.abs(np.sign(arrays["view_logits"]).mean(1)))}
            metadata = {"fold": fold, "match": split["validation"], "scenario": scenario}
            local, summary = analyze(matcher, systems, evidence, metadata, protocol["references"])
            local_file = DEST / f"cases/{stem}.json"
            atomic_json(local_file, local)
            # Both original and effective sequences retained locally, including X.
            sequence_file = DEST / f"cases/{stem}.npz"
            np.savez_compressed(sequence_file, **{f"{n}__{s}": a for n, stages in systems.items() for s, a in stages.items()},
                **{f"{n}__effective": matcher.score(stages["hmm"]).effective for n, stages in systems.items()})
            row = {"fold": fold, "scenario": scenario, "systems": summary, "historical_metrics_parity": True,
                   "mask_sha256": hashlib.sha256(np.asarray(missing, np.uint8).tobytes()).hexdigest()}
            rows.append(row)
            for path in (local_file, sequence_file, cache, seal):
                outputs[str(path.relative_to(DEST))] = file_hash(path)
    if source_hashes() != registration["sources"] or snapshot()[4:] != (dependencies, partitions):
        raise ValueError("Sources or references changed during diagnosis")
    public = {"scope": "public IDSSE own validation aggregates only", "rows": rows,
              "paired_uncertainty": paired_uncertainty(rows, protocol["bootstrap_draws"], protocol["bootstrap_seed"]),
              "test_accessed": False, "target_accessed": False, "training": False, "new_fusions": False,
              "automatic_promotion": False, "historical_failures_preserved": previous["passes_both"],
              "registration_sha256": fingerprint,
              "interpretation": "GT-assisted diagnostic repairs; not deployable or guaranteed achievable performance; category counts overlap"}
    atomic_json(DEST / "public-summary.json", public)
    outputs["public-summary.json"] = file_hash(DEST / "public-summary.json")
    atomic_json(DEST / "result-seal.json", {"registration_sha256": fingerprint, "outputs": outputs})
    return verify()


def verify():
    from possession.long_diagnosis import paired_uncertainty
    record = transfer._json(DEST / "registration.json")
    validate_protocol(record["protocol"])
    if source_hashes() != record["sources"] or snapshot()[4:] != (record["dependencies"], record["partitions"]):
        raise ValueError("Changed diagnostic inputs")
    seal = transfer._json(DEST / "result-seal.json")
    fingerprint = file_hash(DEST / "registration.json")
    if seal["registration_sha256"] != fingerprint:
        raise ValueError("Changed registration")
    expected = {"public-summary.json"}
    for fold in (0, 6):
        for scenario in record["protocol"]["scenarios"]:
            expected.update(f"{kind}/fold-{fold}/{scenario}.{ext}" for kind, ext in
                            (("cache", "npz"), ("cache", "json"), ("cases", "json"), ("cases", "npz")))
    if set(seal["outputs"]) != expected:
        raise ValueError("Incomplete output seal")
    for name, digest in seal["outputs"].items():
        if file_hash(DEST / name) != digest:
            raise ValueError("Changed output")
    result = transfer._json(DEST / "public-summary.json")
    if (any(result[k] for k in ("test_accessed", "target_accessed", "training", "new_fusions", "automatic_promotion"))
            or result["registration_sha256"] != fingerprint or not all(r["historical_metrics_parity"] for r in result["rows"])
            or result["paired_uncertainty"] != paired_uncertainty(result["rows"])):
        raise ValueError("Invalid public result scope or accounting")
    return 0
