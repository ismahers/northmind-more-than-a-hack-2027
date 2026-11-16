"""Transfer the sealed public fusion recipe to the target, without target tuning.

Call only through scripts.run_local: input hashes, caches and errors stay local.
The cache reader deliberately needs NumPy, not PyTorch.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import uuid

import numpy as np

from games import GAME, asset_paths, source_frames
from possession.external_data.schema import file_hash

ROOT = Path(__file__).resolve().parents[1]
# Preserve the historical cache for local comparison; never accept it as v2.
CACHE_PATH = ROOT / "evaluation/reports/st_tcn_target_v2_cache.npz"
TEACHERS_DIR = ROOT / "evaluation/reports/external/temporal-teachers-v1"
PROBE_DIR = ROOT / "evaluation/reports/external/temporal-probe-v1"
FUSION_DIR = ROOT / "evaluation/reports/external/scale-selective-v1"
CACHE_VERSION = 2
PUBLIC_REFERENCE_DIRS = frozenset({
    "temporal-teachers-v1", "temporal-probe-v1", "scale-selective-v1", "folds",
    "transfer-symmetry-validation-v1", "evidence-first-validation-v1",
    "pulse-consensus-validation-v1", "long-diagnosis-v1", "long-diagnosis-v3",
})
REGISTERED_SOURCE_DIRS = frozenset({
    "scale_selective", "temporal_probe", "local_head_budget", "duration_balanced",
    "phase8", "phase7", "phase6", "phase5", "learning", "external_data", "night",
})


def _registered_path(name, *, source=False):
    """Relocate a sealed name in memory; never alter registered bytes or hashes."""
    if not isinstance(name, str) or not name or "\x00" in name:
        raise ValueError("Invalid registered dependency name")
    portable = name.replace("\\", "/")
    if ".." in portable.split("/") or "//" in portable:
        raise ValueError("Unsafe registered dependency name")
    absolute = portable.startswith("/") or re.match(r"^[A-Za-z]:/", portable)
    if absolute:
        markers = ("/possession/", "/scripts/", "/evaluation/") if source else ("/evaluation/reports/",)
        matches = [(portable.count(marker), marker) for marker in markers if marker in portable]
        if len(matches) != 1 or matches[0][0] != 1:
            raise ValueError("Ambiguous registered dependency root")
        marker = matches[0][1]
        portable = marker[1:] + portable.split(marker, 1)[1]
    parts = portable.split("/")
    if any(part in ("", ".") or ":" in part for part in parts):
        raise ValueError("Unsafe registered dependency components")
    if source:
        allowed = ((len(parts) == 3 and parts[0] == "possession"
                    and parts[1] in REGISTERED_SOURCE_DIRS and parts[2].endswith(".py"))
                   or portable in {"scripts/scale_selective.py", "evaluation/possession_metrics.py"})
    else:
        allowed = (len(parts) >= 5 and parts[:3] == ["evaluation", "reports", "external"]
                   and parts[3] in PUBLIC_REFERENCE_DIRS
                   and Path(parts[-1]).suffix in {".json", ".npz", ".pt"}
                   and "test" not in parts)
    if not allowed:
        raise ValueError("Registered dependency outside public allowlist")
    path = (ROOT / portable).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("Registered dependency escapes repository")
    return path


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _fingerprint(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode("utf-8")).hexdigest()


def _registered_source_matches(path, expected):
    """Accept only a reconstruction of the exact sealed bytes: UTF-8 BOM/EOL differences."""
    content = path.read_bytes().removeprefix(b"\xef\xbb\xbf").replace(b"\r\n", b"\n")
    body = content.rstrip(b"\n")
    for newline in (b"\n", b"\r\n"):
        normalized = body.replace(b"\n", newline)
        for trailing in range(4):
            for bom in (b"", b"\xef\xbb\xbf"):
                if hashlib.sha256(bom + normalized + newline * trailing).hexdigest() == expected:
                    return True
    return False


def _transfer_spec():
    """Resolve every required member from validation, verifying the original seal."""
    registration_path = FUSION_DIR / "registration.json"
    seal_path = FUSION_DIR / "selection-seal.json"
    registration, seal = _json(registration_path), _json(seal_path)
    digest = file_hash(registration_path)
    if seal["test_used"] or seal["registration_sha256"] != digest:
        raise ValueError("Invalid public fusion seal")
    config = registration["protocol"]
    if config["pilots"] != [0, 6]:
        raise ValueError("Expected the two registered transfer folds")
    dependencies = {str(registration_path.resolve()): digest, str(seal_path.resolve()): file_hash(seal_path)}
    resolved = set()
    for group in ("sources", "references"):
        for name, expected in registration[group].items():
            path = _registered_path(name, source=group == "sources")
            key = str(path).casefold()
            if key in resolved:
                raise ValueError("Colliding registered dependencies")
            resolved.add(key)
            actual = file_hash(path)
            if actual != expected and not (group == "sources" and _registered_source_matches(path, expected)):
                raise ValueError("Registered public fusion dependency changed")
            dependencies[str(path)] = actual
    members = []
    for fold in config["pilots"]:
        selection_path = FUSION_DIR / f"selections/fold-{fold}.json"
        selection_hash = file_hash(selection_path)
        if selection_hash != seal["selections"][str(fold)]:
            raise ValueError("Public validation selection changed after sealing")
        dependencies[str(selection_path)] = selection_hash
        selection = _json(selection_path)
        if selection["test_used"] or selection["registration_sha256"] != digest:
            raise ValueError("Invalid public validation selection provenance")
        if not selection["arm_gates"]["multiseed_scale_selective"]["passed"]:
            raise ValueError("Transfer arm did not pass validation promotion gates")
        seed = int(selection["teacher"]["seed"])
        alt_seed = int(selection["alt_teacher"]["seed"])
        if {seed, alt_seed} != {2027, 2028}:
            raise ValueError("Incomplete transfer ensemble")
        member = {
            "fold": fold,
            "params": selection["selected_params"]["multiseed_scale_selective"],
            "teacher": TEACHERS_DIR / f"training/fold-{fold}/seed-{seed}/binary/last.pt",
            "alternate": TEACHERS_DIR / f"training/fold-{fold}/seed-{alt_seed}/binary/last.pt",
            "local": PROBE_DIR / f"training/fold-{fold}/local_classification/last.pt",
            "multiscale": PROBE_DIR / f"training/fold-{fold}/multiscale_classification/last.pt",
        }
        if any(str(member[key].resolve()) not in dependencies for key in ("teacher", "alternate", "local", "multiscale")):
            raise ValueError("Unregistered transfer checkpoint")
        members.append(member)
    return config, registration["probe_protocol"], members, dependencies


def _cache_provenance(dependencies):
    dependencies = dict(dependencies)
    for name in ("possession/st_tcn_target.py", "possession/target_domain.py", "games.py",
                 "starter/baseline.py", "possession/external_data/audit.py"):
        path = ROOT / name
        dependencies[str(path)] = file_hash(path)
    paths = asset_paths()
    for name in ("atd_meta", "atd_raw"):
        path = Path(paths[name])
        dependencies[str(path)] = file_hash(path)
    return {"version": CACHE_VERSION, "game": GAME, "frames": source_frames(),
            "dependencies": dependencies,
            "symmetries": ["identity", "team_swap", "xy_reflection", "swap_and_xy"],
            "fold_weights": [0.5, 0.5], "signal_divisor": 2.2}


def _array_digest(signal, alive):
    digest = hashlib.sha256()
    for value in (signal, alive):
        digest.update(np.asarray(value, dtype="<f4").tobytes())
    return digest.hexdigest()


def _read_cache(expected_fingerprint, n_frames):
    try:
        with np.load(CACHE_PATH, allow_pickle=False) as z:
            signal, alive = z["team_signal"], z["alive_prob"]
            if (str(z["fingerprint"]) != expected_fingerprint
                    or signal.shape != (n_frames,) or alive.shape != (n_frames,)
                    or not np.isfinite(signal).all() or not np.isfinite(alive).all()
                    or np.any(np.abs(signal) > 1) or np.any((alive < 0) | (alive > 1))
                    or str(z["array_sha256"]) != _array_digest(signal, alive)):
                return None
            return {"team_signal": signal, "alive_prob": alive,
                    "fingerprint": expected_fingerprint}
    except (OSError, ValueError, KeyError, EOFError):
        return None


def load_cached_st_tcn_target(n_frames=None):
    """An absent v2 cache is optional; an invalid existing cache is an error."""
    if not CACHE_PATH.is_file():
        return None
    _, _, _, dependencies = _transfer_spec()
    provenance = _cache_provenance(dependencies)
    expected_n = source_frames() if n_frames is None else int(n_frames)
    result = _read_cache(_fingerprint(provenance), expected_n)
    if result is None:
        raise ValueError("Stale or invalid neural cache; run compute-target-st-tcn")
    return result


def _transform_view(view, swap_teams=False, reflect_xy=False):
    nodes, mask, g = view.nodes.copy(), view.node_mask.copy(), view.global_features.copy()
    if reflect_xy:
        nodes[..., :4] *= -1.0
        g[:, :2] *= -1.0
    if swap_teams:
        nodes, mask = nodes[:, [1, 0], :, :], mask[:, [1, 0], :]
    return replace(view, nodes=nodes, node_mask=mask, global_features=g)


def _fuse_member(primary_p, alternate_p, z_local, z_multi, view, params, config):
    """Exactly the per-fold multiseed emission recipe used in the public study."""
    from possession.scale_selective.fusion import (
        team_logit_from_probabilities, fuse_scale_selective, fused_emissions,
    )
    z = fuse_scale_selective(
        z_local, z_multi, view.global_features[:, 3] > 0.5, view.run,
        z_teacher=team_logit_from_probabilities(primary_p),
        z_alt_teacher=team_logit_from_probabilities(alternate_p), params=params,
        obs_window=config["obs_window"], pulse_min_frames=config["pulse_min_frames"],
    )
    return z, fused_emissions(z, primary_p, alternate_p,
                             alt_alive_weight=0.5 * params["alt_seed_weight"])[:, :2].sum(1)


def _native_signals(match, view, signal, alive, n_frames):
    """Nearest 10 Hz anchors within each contiguous segment, including clock gaps."""
    from possession.external_data.audit import segments
    from possession.external_data.features import nearest_indices
    result = [np.zeros(n_frames, np.float32), np.zeros(n_frames, np.float32)]
    spans = list(segments(match))
    if set(np.unique(view.run)) != set(range(len(spans))):
        raise ValueError("Native/model segment mismatch")
    for run_id, (start, stop) in enumerate(spans):
        ix = np.flatnonzero(view.run == run_id)
        frames = match.frame[start:stop]
        if np.any((frames < 0) | (frames >= n_frames)):
            raise ValueError("Native frame outside submission timeline")
        nearest = nearest_indices(view.time[ix], match.time[start:stop])
        for output, values in zip(result, (signal, alive)):
            output[frames] = values[ix[nearest]]
    return result


def compute_and_cache_st_tcn_target():
    """Require the complete sealed ensemble; rebuild stale caches atomically."""
    config, probe_config, members, dependencies = _transfer_spec()
    provenance = _cache_provenance(dependencies)
    fingerprint = _fingerprint(provenance)
    if _read_cache(fingerprint, source_frames()) is not None:
        return 0

    import torch
    from possession.target_domain import load_target
    from possession.external_data.features import build_features
    from possession.phase7.features import harmonize
    from possession.phase7.model import SpatialE0
    from possession.phase8.evaluation import predict as predict_teacher
    from possession.temporal_probe.model import TemporalHead
    from possession.temporal_probe.windows import predict as predict_head

    match = load_target()
    if match is None:
        raise ValueError("Target inputs unavailable")
    torch.set_num_threads(config["cpu_threads"])
    view = harmonize(build_features(match))
    z_mean = np.zeros(len(view.time), np.float64)
    alive_mean = np.zeros(len(view.time), np.float64)
    for member in members:
        models = []
        for key in ("teacher", "alternate"):
            state = torch.load(member[key], map_location="cpu", weights_only=False)["model"]
            model = SpatialE0("set_transformer", True)
            model.load_state_dict(state)
            models.append(model.eval())
        heads = []
        for key in ("local", "multiscale"):
            model = TemporalHead(
                models[0].team_head.weight, models[0].team_head.bias,
                temporal=key == "multiscale", intervals=False,
                dilations=probe_config["dilations"], kernel_size=probe_config["kernel_size"],
                seed=probe_config["seed"],
            )
            state = torch.load(member[key], map_location="cpu", weights_only=False)["model"]
            model.load_state_dict(state)
            heads.append(model.eval())
        for swap in (False, True):
            for reflect in (False, True):
                transformed = _transform_view(view, swap, reflect)
                primary_p, _, hidden = predict_teacher(models[0], transformed, embeddings=True)
                alternate_p, _, _ = predict_teacher(models[1], transformed, embeddings=False)
                inputs = {"hidden": hidden, "time": view.time, "run": view.run}
                local_z, _, _ = predict_head(heads[0], inputs, config["core_frames"], config["batch_size"])
                multi_z, _, _ = predict_head(heads[1], inputs, config["core_frames"], config["batch_size"])
                z, alive = _fuse_member(primary_p, alternate_p, local_z, multi_z,
                                        transformed, member["params"], config)
                weight = 1.0 / (4 * len(members))
                z_mean += (-1 if swap else 1) * weight * z
                alive_mean += weight * alive
    signal, alive = _native_signals(match, view, np.tanh(z_mean / 2.2),
                                    np.clip(alive_mean, 0, 1), source_frames())
    # Do not publish an output if a dependency changed while inference ran.
    _, _, _, current_dependencies = _transfer_spec()
    if _fingerprint(_cache_provenance(current_dependencies)) != fingerprint:
        raise ValueError("Transfer inputs changed during inference")
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_name(CACHE_PATH.stem + "-" + uuid.uuid4().hex + ".tmp.npz")
    try:
        with tmp.open("xb") as stream:
            np.savez_compressed(stream, team_signal=signal, alive_prob=alive,
                                fingerprint=np.asarray(fingerprint),
                                provenance=np.asarray(json.dumps(provenance, sort_keys=True)),
                                array_sha256=np.asarray(_array_digest(signal, alive)))
        if not np.isfinite(signal).all() or not np.isfinite(alive).all():
            raise ValueError("Nonfinite neural transfer")
        tmp.replace(CACHE_PATH)
    finally:
        tmp.unlink(missing_ok=True)
    if load_cached_st_tcn_target() is None:
        raise ValueError("Neural transfer cache was not created")
    return 0
