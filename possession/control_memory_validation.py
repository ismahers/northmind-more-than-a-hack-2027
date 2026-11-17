"""Frozen external validation of a geometry-only control-memory ablation.

This is a neutral-tagging ablation on previously seen public validation folds,
not a replay of the complete target recipe and never an automatic promotion.
Only run through the reviewed launcher, which captures stdout/stderr at birth.
No target cache, tactical labels, test or calibration inputs are loaded here.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/control_memory_validation_v1.json"
DEST = ROOT / "evaluation/reports/external/control-memory-validation-v1"
PUBLIC = ROOT / "docs/public-results/2026-11-17/control-memory-v1.json"
TEST_RECEIPT = ROOT / "docs/public-results/2026-11-17/control-memory-tests.json"
STATUS = ROOT / "docs/public-results/2026-11-17/control-memory-status.json"
_PHASE = "setup"
SCENARIOS = ("clean", "missing40_5s", "missing40_15s")
METRICS = ("spellF1", "macroF1", "short_recall", "long_recall",
           "false_transitions_per_minute")
GUARDS = dict(robust_delta_min=.005, clean_delta_min=0,
              short_recall_delta_min=0, long_recall_delta_min=-.01,
              macro_f1_delta_min=-.005, false_transitions_delta_max=0)


def validate_protocol(p):
    from possession.control_memory import ControlMemoryConfig
    expected = dict(version="control-memory-validation-v1", part="validation",
        train=False, test_access=False, target_access=False, parameter_search=False,
        folds=[0, 6], scenarios=list(SCENARIOS), weights=[.5, .25, .25],
        views=[[False, False], [False, True], [True, False], [True, True]],
        cpu_threads=4, max_seconds=3600, memory_weight=.15,
        weak_strength_threshold=.25, neutral_tagging=True, guardrails=GUARDS,
        fps=25.0, memory=asdict(ControlMemoryConfig()),
        memory_smoothing="within_predicted_live_spans")
    if any(p.get(k) != v for k, v in expected.items()):
        raise ValueError("Unexpected fixed validation protocol")
    # JSON booleans are required for access barriers, not integer lookalikes.
    for k in ("train", "test_access", "target_access", "parameter_search"):
        if p[k] is not False:
            raise ValueError("Invalid access barrier")


@dataclass(frozen=True)
class NativeGeometry:
    time: np.ndarray
    run: np.ndarray
    native_index: np.ndarray
    anchor_index: np.ndarray
    ball: np.ndarray
    distances: np.ndarray


def build_native_geometry(match, view, hidden):
    """25Hz per-segment geometry; hidden is the EXACT hide_ball 10Hz mask.

    Only coordinates, availability and clocks are read from match; supervision
    is deliberately outside this adapter. Coordinates become corner-origin
    metres to reproduce the target outside-pitch rule on a generic pitch.
    """
    from possession.external_data.audit import segments
    from possession.external_data.features import nearest_indices
    hidden = np.asarray(hidden)
    if hidden.shape != (len(view.time),) or hidden.dtype != np.bool_:
        raise ValueError("Invalid feature corruption mask")
    if match.metadata["native_fps"] != 25:
        raise ValueError("This fixed adapter requires native 25Hz")
    pitch = np.asarray(match.metadata["pitch_m"], float)
    if pitch.shape != (2,) or not np.isfinite(pitch).all() or np.any(pitch <= 0):
        raise ValueError("Invalid pitch")
    times, ids, natives, anchors = [], [], [], []
    spans = list(segments(match))
    if set(np.unique(view.run)) != set(range(len(spans))):
        raise ValueError("Model/native segment mismatch")
    for run_id, (a, b) in enumerate(spans):
        grid = match.time[a] + np.arange(round((match.time[b-1]-match.time[a])*25)+1)/25
        selected = np.flatnonzero(view.run == run_id)
        times.append(grid)
        ids.append(np.full(len(grid), run_id, dtype=np.int32))
        natives.append(a + nearest_indices(match.time[a:b], grid))
        anchors.append(selected[nearest_indices(view.time[selected], grid)])
    native, anchor = np.concatenate(natives), np.concatenate(anchors)
    ball = np.asarray(match.ball[native, :2], float).copy()
    visible = np.asarray(match.ball_available[native], bool) & ~hidden[anchor]
    visible &= np.isfinite(ball).all(1)
    ball[~visible] = np.nan
    players = np.asarray(match.players[native], float)
    available = np.asarray(match.player_available[native], bool)
    available &= np.isfinite(players).all(-1) & visible[:, None, None]
    delta = np.nan_to_num(players, nan=0, posinf=0, neginf=0) - np.nan_to_num(
        ball, nan=0)[:, None, None, :]
    distances = np.where(available, np.linalg.norm(delta, axis=-1), np.inf).min(2)
    return NativeGeometry(np.concatenate(times), np.concatenate(ids), native,
                          anchor, ball + pitch/2, distances)


def geometric_dead(distances, ball, pitch, config):
    """Same outside/still hypotheses as inference, with no tagging restart mask."""
    import pandas as pd
    from possession.inference import minimum_runs
    nearest = distances.min(1)
    visible = np.isfinite(ball).all(1)
    margin = config.outside_margin_m
    outside = visible & ((ball[:, 0] < -margin) | (ball[:, 0] > pitch[0]+margin)
                         | (ball[:, 1] < -margin) | (ball[:, 1] > pitch[1]+margin))
    outside = minimum_runs(outside, config.outside_min_frames)
    window = config.stationary_window_frames
    support = int(np.ceil(window*config.stationary_min_support))
    rolling = pd.DataFrame(ball).rolling(window, center=True, min_periods=support)
    extent = rolling.max()-rolling.min()
    diameter = np.hypot(extent[0].to_numpy(), extent[1].to_numpy())
    unattended = pd.Series(np.where(np.isfinite(nearest), nearest, np.nan)).rolling(
        window, center=True, min_periods=support).min().to_numpy()
    stationary = visible & (diameter <= config.stationary_max_range_m) & (
        unattended >= config.unattended_min_distance_m)
    return outside | stationary


def geometric_reference(distances, ball, pitch, config):
    """Nearest codes + official smooth50 + weighted vote + independent D."""
    from starter.baseline import smooth
    from possession.inference import team_strength, window_sum
    observed = np.isfinite(distances.min(1))
    codes = np.where(observed, np.argmin(distances, axis=1), -1)
    labels = smooth(codes, config.half_window_frames)
    strength = team_strength(distances, config, None)
    vote = window_sum(strength, config.half_window_frames)
    support = window_sum(np.abs(strength), config.half_window_frames)
    use = (support > 1e-10) & (np.abs(vote) > 1e-10)
    labels[use] = np.where(vote[use] > 0, "H", "A")
    labels[geometric_dead(distances, ball, pitch, config)] = "D"
    return labels


def smooth_live_delta(delta, reference, window):
    """Smooth the intervention separately, with no memory crossing D/X."""
    from possession.sequence_decoding import runs, _smooth_strength
    result = np.zeros(len(delta),float)
    for a,b in runs(np.isin(reference,("H","A"))):
        result[a:b] = _smooth_strength(delta[a:b],window)
    return result


def decode_pair(native, neural, pitch):
    """One shared decoder; candidate differs only by gated +0.15 memory.

    Preserve target base smoothing (including smoothing across D inside each
    segment). Smooth only the added delta within predicted live spans, so the
    intervention, memory and DP reset at D/X and segment edges.
    The plain control decoder is used: unsupported D bridging is not tested.
    """
    from possession.inference import Config, team_strength
    from possession.control_memory import ControlMemoryConfig, temporal_control_memory
    from possession.sequence_decoding import (SequenceConfig, runs, _smooth_strength,
        _ball_kinematics_weights, _adaptive_switch_costs, decode_control,
        _filter_oscillations, _align_transitions_to_geometry)
    neural = np.asarray(neural, float)
    if neural.shape != native.time.shape or not np.isfinite(neural).all() or np.any(abs(neural)>1):
        raise ValueError("Invalid neural signal")
    cfg, seq = Config(), SequenceConfig()
    reference = np.full(len(neural), "X", dtype="<U1")
    raw_geom, base = np.zeros(len(neural)), np.zeros(len(neural))
    for identity in np.unique(native.run):
        ix = np.flatnonzero(native.run == identity)
        d, b = native.distances[ix], native.ball[ix]
        reference[ix] = geometric_reference(d, b, pitch, cfg)
        raw_geom[ix] = np.clip(team_strength(d, cfg, None) * _ball_kinematics_weights(
            b, d, pass_speed_threshold_ms=seq.pass_speed_threshold_ms), -1, 1)
        w = np.where(np.isfinite(d.min(1)), seq.neural_weight_observed,
                     seq.neural_weight_occluded)
        base[ix] = np.clip((1-.5*w)*raw_geom[ix] + w*neural[ix], -1, 1)
    live = np.isin(reference, ("H", "A"))
    memory = temporal_control_memory(native.distances, native.ball, native.time,
        native.run, live, ControlMemoryConfig(fps=25.0))
    delta = .15*memory.memory_signal*(np.abs(base)<.25)
    outputs = [reference.copy(), reference.copy()]
    for identity in np.unique(native.run):
        ix = np.flatnonzero(native.run == identity)
        values = reference[ix]
        geom_strength = _smooth_strength(raw_geom[ix], seq.strength_smooth_frames)
        smoothed_base = _smooth_strength(base[ix], seq.strength_smooth_frames)
        candidate = np.clip(smoothed_base + smooth_live_delta(
            delta[ix],values,seq.strength_smooth_frames),-1,1)
        alignment = values.copy()
        for a, b in runs(np.isin(values, ("H", "A"))):
            alignment[a:b] = decode_control(geom_strength[a:b],
                _adaptive_switch_costs(geom_strength[a:b], seq), fallback=values[a])
        alignment = _filter_oscillations(alignment)
        for output, smoothed in zip(outputs, (smoothed_base, candidate)):
            labels = values.copy()
            for a, b in runs(np.isin(values, ("H", "A"))):
                labels[a:b] = decode_control(smoothed[a:b],
                    _adaptive_switch_costs(smoothed[a:b], seq), fallback=values[a])
            labels = _align_transitions_to_geometry(_filter_oscillations(labels), alignment, 20)
            output[ix] = labels
    for output in outputs:
        if not np.array_equal(output[~live], reference[~live]):
            raise ValueError("Decoder changed D/X barrier")
    return outputs[0], outputs[1]


def expand_hard(context, labels):
    codes = np.full(len(labels), 3, dtype=np.int8)
    for index, name in enumerate(("H", "A", "D")):
        codes[np.asarray(labels) == name] = index
    output = np.full(len(context.gt), 3, dtype=np.int8)
    for selected, mapping, span in context.parts:
        output[span] = codes[selected][mapping]
    return output


def gates(candidate, baseline):
    objective = lambda rows: sum(w*rows[s]["spellF1"] for w, s in zip((.5,.25,.25), SCENARIOS))
    old, new = baseline["clean"], candidate["clean"]
    checks = dict(robust=objective(candidate) >= objective(baseline)+.005,
        clean=new["spellF1"] >= old["spellF1"], short=new["short_recall"] >= old["short_recall"],
        long=new["long_recall"] >= old["long_recall"]-.01,
        macro=new["macroF1"] >= old["macroF1"]-.005,
        false_HA=new["false_transitions_per_minute"] <= old["false_transitions_per_minute"])
    return {"checks": {k: bool(v) for k,v in checks.items()},
            "passed": bool(all(checks.values())),
            "robust_delta": float(objective(candidate)-objective(baseline))}


def _write_new(path, value):
    """Never replace registration, completed metrics, seals or public export."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)


def setup():
    from possession import st_tcn_target as transfer
    from possession.evidence_first_validation import reference_path
    from possession.external_data.schema import file_hash
    from possession.temporal_probe.common import public_path, splits_valid
    import torch
    protocol = transfer._json(PROTOCOL)
    validate_protocol(protocol)
    _validate_test_receipt(transfer._json(TEST_RECEIPT))
    config, heads, members, dependencies = transfer._transfer_spec()
    if [m["fold"] for m in members] != [0,6]:
        raise ValueError("Wrong folds")
    manifest = ROOT / "evaluation/reports/external/folds/idsse-only.json"
    records = transfer._json(manifest)["artifacts"]
    inputs = {}
    for member in members:
        selection = transfer._json(transfer.FUSION_DIR / f"selections/fold-{member['fold']}.json")
        split = selection["teacher"]["split"]
        if not splits_valid(split):
            raise ValueError("Invalid split")
        inputs[str(member["fold"])] = {}
        for key in split["train"]+[split["validation"]]:
            path = public_path(records[key]["path"])
            digest = file_hash(path)
            if digest != records[key]["sha256"]:
                raise ValueError("Public source changed")
            inputs[str(member["fold"])][key] = digest
        for scenario in SCENARIOS:
            path = reference_path(member["fold"], scenario)
            dependencies[str(path)] = file_hash(path)
    # Freeze the complete own-source import closure conservatively. This reads
    # source files only; raw inputs above are exclusively train/validation.
    sources = {}
    paths = [*sorted((ROOT/"possession").rglob("*.py")),
             ROOT/"games.py", ROOT/"starter/baseline.py", ROOT/"evaluation/possession_metrics.py",
             ROOT/"scripts/run_local.py", ROOT/"scripts/control_memory.py",
             ROOT/"tests/test_control_memory.py", ROOT/"tests/test_control_memory_validation.py", PROTOCOL]
    for path in paths:
        sources[str(path.relative_to(ROOT))] = file_hash(path)
    dependencies[str(manifest)] = file_hash(manifest)
    dependencies[str(TEST_RECEIPT)] = file_hash(TEST_RECEIPT)
    stamp = {"protocol": protocol, "sources": sources, "dependencies": dependencies,
        "fold_train_validation_sha256": inputs,
        "environment": dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__)}
    return protocol, config, heads, members, records, stamp


def _neural(view, degraded, member, models, heads, config, reference, check_budget):
    global _PHASE
    _PHASE = "parity"
    from possession import st_tcn_target as transfer
    from possession.phase8.evaluation import predict as predict_teacher
    from possession.temporal_probe.windows import predict as predict_head, emissions
    mean = np.zeros(len(view.time))
    for swap, reflect in ((False,False),(False,True),(True,False),(True,True)):
        check_budget()
        transformed = transfer._transform_view(degraded, swap, reflect)
        p, _, hidden = predict_teacher(models[0], transformed, embeddings=True)
        alt, _, _ = predict_teacher(models[1], transformed)
        inputs = dict(hidden=hidden, time=view.time, run=view.run)
        local, _, _ = predict_head(heads[0], inputs, config["core_frames"], config["batch_size"])
        multi, _, _ = predict_head(heads[1], inputs, config["core_frames"], config["batch_size"])
        z, alive = transfer._fuse_member(p, alt, local, multi, transformed, member["params"], config)
        if not swap and not reflect:
            actual = emissions(z, np.column_stack((alive, np.zeros(len(alive)), 1-alive)))
            if actual.shape != reference.shape or not np.allclose(actual, reference, atol=3e-6, rtol=3e-5):
                raise ValueError("Frozen identity parity failed")
        mean += .25*(-1 if swap else 1)*z
    # Exact native target conversion, without calling any target loader/cache.
    return np.tanh(mean/2.2)


@contextmanager
def _exclusive():
    DEST.mkdir(parents=True, exist_ok=True)
    lock = DEST/"running.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write("control-memory-validation-v1\n")
    try:
        yield
    finally:
        lock.unlink()


def _validate_test_receipt(receipt):
    import hashlib
    names = ("possession/control_memory.py", "possession/control_memory_validation.py",
        "tests/test_control_memory.py", "tests/test_control_memory_validation.py",
        "scripts/control_memory.py", "configs/control_memory_validation_v1.json")
    if (receipt.get("synthetic_only") is not True or receipt.get("success") is not True
            or receipt.get("tests", 0) < 20 or receipt.get("failures") != 0
            or receipt.get("errors") != 0 or receipt.get("skipped") != 0
            or set(receipt.get("sources_sha256_lf",{})) != set(names)):
        raise ValueError("Synthetic review receipt required")
    for name in names:
        data = (ROOT/name).read_text(encoding="utf-8-sig").replace("\r\n","\n").encode("utf-8")
        if receipt["sources_sha256_lf"][name] != hashlib.sha256(data).hexdigest():
            raise ValueError("Synthetic source changed after tests")


def _status(state, exception=None):
    """Only closed enums may cross the public diagnostic boundary."""
    phases = {"setup", "parity", "adapt", "score", "seal", "complete"}
    classes = {"ValueError", "TypeError", "KeyError", "IndexError", "OSError",
        "FileNotFoundError", "FileExistsError", "PermissionError", "TimeoutError",
        "RuntimeError", "AssertionError", "ImportError", "ModuleNotFoundError", "MemoryError"}
    if state not in {"running","failed","complete"} or _PHASE not in phases:
        raise ValueError("Invalid diagnostic enum")
    name = None if exception is None else type(exception).__name__
    record = {"state":state,"phase":_PHASE,
              "exception_class":name if name in classes else ("Exception" if name else None)}
    STATUS.parent.mkdir(parents=True,exist_ok=True)
    STATUS.write_text(json.dumps(record,sort_keys=True)+"\n",encoding="utf-8")


def run():
    global _PHASE
    _PHASE = "setup"
    _status("running")
    try:
        result = _run()
    except Exception as error:
        _status("failed",error)
        raise
    _PHASE = "complete"
    _status("complete")
    return result


def _run():
    global _PHASE
    from possession import st_tcn_target as transfer
    from possession.evidence_first_validation import FoldStore, load_models, reference_path
    from possession.external_data.schema import file_hash
    from possession.learning.study import ScoreContext
    from possession.phase5.experiment import scenarios
    from possession.hsmm_pilot import score
    import torch
    start = time.perf_counter()
    with _exclusive():
        protocol, config, head_config, members, records, stamp = setup()
        marker = DEST/"registration.json"
        if marker.exists():
            if transfer._json(marker) != stamp:
                raise ValueError("Immutable registration changed")
        else:
            _write_new(marker, stamp)  # BEFORE loading validation labels or computing metrics.
        if (DEST/"result-seal.json").exists():
            return verify()
        if (DEST/"summary.json").exists():
            raise ValueError("Unsealed output requires review")
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        def check_budget():
            if time.perf_counter()-start >= 3600:
                raise TimeoutError("Fixed compute budget exhausted")
        parent = transfer._json(transfer.FUSION_DIR/"registration.json")
        summaries = {}
        for member in members:
            check_budget()
            fold = member["fold"]
            split = transfer._json(transfer.FUSION_DIR/f"selections/fold-{fold}.json")["teacher"]["split"]
            store = FoldStore(records, split)
            # No train-label prior is needed by this neutral-tag ablation.
            match = store.match(split["validation"])
            view = store.harmonized(split["validation"])
            models, heads = load_models(member, split, head_config)
            rows = {"baseline": {}, "memory": {}}
            parity = {}
            for scenario, (degraded, hidden) in scenarios(view, split["validation"], parent["teacher_protocol"]).items():
                if scenario not in SCENARIOS:
                    raise ValueError("Unexpected scenario")
                with np.load(reference_path(fold,scenario), allow_pickle=False) as z:
                    reference = z["probabilities"].copy()
                if reference.shape != (len(view.time),3) or not np.isfinite(reference).all():
                    raise ValueError("Invalid frozen reference")
                signal = _neural(view, degraded, member, models, heads, config, reference, check_budget)
                parity[scenario] = True
                _PHASE = "adapt"
                native = build_native_geometry(match, view, hidden)
                baseline, candidate = decode_pair(native, signal[native.anchor_index], match.metadata["pitch_m"])
                _PHASE = "score"
                context = ScoreContext(match, SimpleNamespace(time=native.time,run=native.run,
                                                              native_index=native.native_index))
                for name, labels in (("baseline",baseline),("memory",candidate)):
                    metrics = score(context, expand_hard(context, labels))
                    rows[name][scenario] = {k: float(metrics[k]) for k in METRICS}
                check_budget()
            summaries[str(fold)] = {"metrics": rows,
                "gates": gates(rows["memory"],rows["baseline"]), "identity_parity": parity}
        _PHASE = "seal"
        if setup()[-1] != stamp:
            raise ValueError("Dependencies changed during run")
        check_budget()
        summary = {"folds": summaries, "registration_sha256": file_hash(marker),
            "seconds": time.perf_counter()-start, "test_accessed": False,
            "target_accessed": False, "calibration_accessed": False,
            "automatic_promotion": False}
        safe_public_summary(summary)  # Reject malformed metrics before committing any outputs.
        _write_new(DEST/"summary.json", summary)
        _write_new(DEST/"result-seal.json", {"registration_sha256": file_hash(marker),
            "outputs": {"summary.json": file_hash(DEST/"summary.json")}})
    return verify()


def safe_public_summary(summary):
    """Whitelist writer: no paths, labels, arrays, free input text or errors."""
    import re
    if (any(summary[k] is not False for k in ("test_accessed", "target_accessed",
                "calibration_accessed", "automatic_promotion"))
            or set(summary["folds"]) != {"0","6"}
            or not isinstance(summary["registration_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}",summary["registration_sha256"]) is None):
        raise ValueError("Invalid summary scope")
    result = dict(schema_version=1, scope="external_seen_validation_neutral_tag_ablation",
        full_target_recipe=False, automatic_promotion=False,
        registration_sha256=summary["registration_sha256"], folds={})
    for fold in ("0","6"):
        source = summary["folds"][fold]
        if (set(source["identity_parity"]) != set(SCENARIOS)
                or any(v is not True for v in source["identity_parity"].values())):
            raise ValueError("Missing identity parity")
        rows = {}
        if set(source["metrics"]) != {"baseline","memory"}:
            raise ValueError("Unexpected variant")
        for variant in ("baseline","memory"):
            if set(source["metrics"][variant]) != set(SCENARIOS):
                raise ValueError("Incomplete scenarios")
            rows[variant] = {}
            for scenario in SCENARIOS:
                metric = source["metrics"][variant][scenario]
                if set(metric) != set(METRICS):
                    raise ValueError("Unexpected metric schema")
                checked = {}
                for key in METRICS:
                    value = metric[key]
                    if (isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float))
                            or not np.isfinite(value) or value < 0
                            or (key != "false_transitions_per_minute" and value > 1)):
                        raise ValueError("Invalid aggregate metric")
                    checked[key] = float(value)
                rows[variant][scenario] = checked
        checks = gates(rows["memory"], rows["baseline"])
        if source["gates"] != checks:
            raise ValueError("Gate mismatch")
        result["folds"][fold] = {"metrics": rows, "gates": checks, "identity_parity": True}
    result["passes_both"] = all(row["gates"]["passed"] for row in result["folds"].values())
    return result


def verify():
    from possession import st_tcn_target as transfer
    from possession.external_data.schema import file_hash
    stamp = setup()[-1]
    marker = DEST/"registration.json"
    if transfer._json(marker) != stamp:
        raise ValueError("Registration changed")
    seal = transfer._json(DEST/"result-seal.json")
    if (seal["registration_sha256"] != file_hash(marker)
            or set(seal["outputs"]) != {"summary.json"}
            or seal["outputs"]["summary.json"] != file_hash(DEST/"summary.json")):
        raise ValueError("Result seal mismatch")
    summary = transfer._json(DEST/"summary.json")
    if summary["registration_sha256"] != file_hash(marker):
        raise ValueError("Summary provenance mismatch")
    safe_public_summary(summary)
    return 0


def export():
    from possession import st_tcn_target as transfer
    verify()
    public = safe_public_summary(transfer._json(DEST/"summary.json"))
    if PUBLIC.exists():
        if transfer._json(PUBLIC) != public:
            raise ValueError("Immutable public output changed")
    else:
        _write_new(PUBLIC, public)
    return 0


def test():
    """Pure fixture suite: no raw inputs, model checkpoints or result caches."""
    import unittest
    import importlib.util
    spec = importlib.util.spec_from_file_location("control_memory_validation_tests",
        ROOT/"tests/test_control_memory_validation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromModule(module))
    return 0 if result.wasSuccessful() else 1
