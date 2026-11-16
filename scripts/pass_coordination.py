"""Reviewed public-train-only entries. Child output must be captured by run_local."""
import hashlib
import json
from pathlib import Path
import platform
import time
from types import SimpleNamespace
import unittest
import zipfile
import numpy as np
from possession.pass_coordination import candidates, event_features, evaluate, summarize
from possession.external_data.audit import segments
from possession.external_data.features import nearest_indices
from possession.external_data.schema import file_hash

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/"evaluation/reports/external/pass-coordination-v1"
PUBLIC = ROOT/"docs/public-results/2026-10-30"
PARENT = ROOT/"evaluation/reports/external/conditional-fusion-v1"
CONFIG = ROOT/"configs/pass_coordination_v1.json"
KEYS = ["idsse:DFL-MAT-"+k for k in ("J03WOH", "J03WOY", "J03WPY")]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


def sources():
    names = ["possession/pass_coordination.py", "scripts/pass_coordination.py", "scripts/run_local.py",
             "tests/test_pass_coordination.py", "configs/pass_coordination_v1.json",
             "possession/football_control.py", "possession/external_data/features.py",
             "possession/external_data/schema.py", "possession/external_data/audit.py",
             "evaluation/possession_metrics.py"]
    return {n: hashlib.sha256((ROOT/n).read_text(encoding="utf-8-sig").replace("\r\n", "\n").encode()).hexdigest() for n in names}


def tests():
    suite = unittest.defaultTestLoader.discover(str(ROOT/"tests"), pattern="test_pass_coordination.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    write(DEST/"synthetic-status.json", {"tests": result.testsRun, "success": result.wasSuccessful(), "sources": sources()})
    return 0 if result.wasSuccessful() else 1


def register(check):
    spec = read(CONFIG)
    if (spec["matches"] != KEYS or spec["oof_outer_fold"] != 0 or spec["scenario"] != "clean"
        or any(spec[k] for k in ("parameter_fitting", "new_neural_inference", "validation_access",
                                 "calibration_access", "test_access", "target_access", "video_or_pose_access"))):
        raise ValueError("Fixed public train diagnostic only")
    tested = read(DEST/"synthetic-status.json")
    if not tested["success"] or tested["sources"] != sources():
        raise ValueError("Reviewed synthetic tests required")
    public_parent = read(ROOT/"docs/public-results/2026-10-29/conditional-fusion-v1.json")
    if file_hash(PARENT/"registration.json") != public_parent["registration_sha256"]:
        raise ValueError("Parent provenance failed")
    parent, seal = read(PARENT/"registration.json"), read(PARENT/"result-seal.json")
    if seal["registration_sha256"] != public_parent["registration_sha256"]:
        raise ValueError("Parent seal failed")
    inputs, oof = {}, {}
    for key in KEYS:
        check(); item = parent["inputs"][key]; path = (ROOT/item["path"]).resolve()
        if not path.is_relative_to((ROOT/"data/external/processed").resolve()) or path.suffix != ".npz" or file_hash(path) != item["sha256"]:
            raise ValueError("Public allowlisted input changed")
        inputs[key] = item
        units = [u for u in parent["units"] if u["outer_fold"] == 0 and key in u["heldout"]]
        if len(units) != 1:
            raise ValueError("Unique heldout role required")
        u = units[0]; index = u["heldout"].index(key)
        name = f"oof/fold-0/inner-{u['inner_fold']}/match-{index}-clean.npz"
        provenance = str(Path(name).with_suffix(".provenance.json")).replace("\\", "/")
        if file_hash(PARENT/name) != seal["outputs"][name] or file_hash(PARENT/provenance) != seal["outputs"][provenance]:
            raise ValueError("OOF output seal changed")
        prov = read(PARENT/provenance)
        if (prov["key"] != key or prov["sha256"] != seal["outputs"][name]
            or set(prov["fit_keys"]) != set(u["train"])
            or (set(prov["fit_keys"]+prov["selection_keys"]) & set(u["heldout"]+u["excluded"]))):
            raise ValueError("OOF fit/selection exclusion failed")
        oof[key] = {"file": name, "sha256": seal["outputs"][name], "provenance": prov, "unit": u}
    stamp = {"protocol": spec, "sources": sources(), "inputs": inputs, "oof": oof,
             "parent_registration_sha256": public_parent["registration_sha256"],
             "environment": {"python": platform.python_version(), "numpy": np.__version__}}
    marker = DEST/"registration.json"
    if marker.exists() and read(marker) != stamp:
        raise ValueError("Immutable diagnostic changed")
    write(marker, stamp)
    return stamp


def run():
    started = time.perf_counter()
    def check():
        if time.perf_counter()-started > 600:
            raise TimeoutError("Fixed compute budget exhausted")
    stamp = register(check); spec = stamp["protocol"]; reports = {}
    for match_index, key in enumerate(KEYS):
        check(); path = ROOT/stamp["inputs"][key]["path"]
        # Geometry-only phase: labels are not even loaded until selection is frozen.
        with np.load(path, allow_pickle=False) as z:
            metadata = json.loads(str(z["metadata"]))
            m = SimpleNamespace(metadata=metadata, **{k: z[k] for k in ("time", "period", "frame", "players", "player_available", "ball_available")}, ball=z["ball"][:, :2])
        if metadata["provider"]+":"+metadata["match_id"] != key or metadata["native_fps"] != spec["native_hz"]:
            raise ValueError("Unexpected public geometry identity")
        native_run = np.empty(len(m.time), int)
        for r, (a, b) in enumerate(segments(m)): native_run[a:b] = r
        chosen = candidates(m, native_run, spec, check)
        gs, cs, counts = [], [], []
        for i, cosine, outgoing in chosen:
            check(); g, c, n = event_features(m, native_run, i, cosine, outgoing, spec)
            gs.append(g); cs.append(c); counts.append(n)
        indices = np.array([e[0] for e in chosen], int)
        with np.load(path, allow_pickle=False) as z:
            alive, team, conf = z["alive"][indices], z["team"][indices], z["confidence"][indices]
        truth = np.full(len(indices), 3, np.int8)
        truth[(alive == 0) & (conf[:, 0] > 0)] = 2
        labeled = (alive == 1) & (conf[:, 0] > 0) & (conf[:, 1] > 0) & np.isin(team, [0, 1])
        truth[labeled] = team[labeled]
        with np.load(PARENT/stamp["oof"][key]["file"], allow_pickle=False) as z:
            p, oof_time, oof_run = z["base"], z["time"], z["run"]
        if p.shape != (len(oof_time), 3) or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(-1), 1, atol=1e-5):
            raise ValueError("Frozen probabilities malformed")
        home = np.full(len(indices), np.nan)
        for r in np.unique(native_run[indices]):
            selected = np.flatnonzero(native_run[indices] == r); oi = np.flatnonzero(oof_run == r)
            if not len(oi): raise ValueError("Missing OOF run")
            mapped = oi[nearest_indices(oof_time[oi], m.time[indices[selected]])]
            if np.any(np.abs(oof_time[mapped]-m.time[indices[selected]]) > .061):
                raise ValueError("OOF clock mismatch")
            a = p[mapped, :2].sum(-1)
            home[selected] = np.divide(p[mapped, 0], a, out=np.full(len(a), .5), where=a > 1e-12)
        data = {"g": np.array(gs).reshape((-1, 2)), "c": np.array(cs).reshape((-1, 4, 2)),
                "counts": np.array(counts, int).reshape((-1, 2)), "truth": truth, "base_home": home}
        # No coordinates, actual frame indices or video are required for aggregate replay.
        np.savez_compressed(DEST/f"match-{match_index}.npz", **data)
        reports[key] = evaluate(data, spec)
        if file_hash(path) != stamp["inputs"][key]["sha256"] or file_hash(PARENT/stamp["oof"][key]["file"]) != stamp["oof"][key]["sha256"]:
            raise ValueError("Input changed during computation")
    summary = summarize(reports, spec)
    summary.update(seconds=time.perf_counter()-started, registration_sha256=file_hash(DEST/"registration.json"))
    if sources() != stamp["sources"]: raise ValueError("Diagnostic sources changed")
    write(DEST/"summary.json", summary); check()
    names = ["summary.json", "synthetic-status.json", *[f"match-{i}.npz" for i in range(3)]]
    write(DEST/"result-seal.json", {"registration_sha256": summary["registration_sha256"],
          "files": {n: file_hash(DEST/n) for n in names}})
    return verify()


def verify():
    stamp, seal = read(DEST/"registration.json"), read(DEST/"result-seal.json")
    if stamp["sources"] != sources() or seal["registration_sha256"] != file_hash(DEST/"registration.json"):
        raise ValueError("Diagnostic provenance changed")
    for name, digest in seal["files"].items():
        if file_hash(DEST/name) != digest: raise ValueError("Diagnostic output changed")
    return 0


def replay():
    verify(); spec = read(DEST/"registration.json")["protocol"]; reports = {}
    for i, key in enumerate(KEYS):
        with np.load(DEST/f"match-{i}.npz", allow_pickle=False) as z: data = dict(z)
        reports[key] = evaluate(data, spec)
    computed = summarize(reports, spec); saved = read(DEST/"summary.json")
    if any(saved[k] != value for k, value in computed.items()):
        raise ValueError("Replay differs")
    return 0


def export():
    verify()
    names = sorted([*read(DEST/"result-seal.json")["files"], "registration.json", "result-seal.json"])
    manifest = {n: {"sha256": file_hash(DEST/n), "bytes": (DEST/n).stat().st_size} for n in names}
    archive = ROOT/"evaluation/reports/handoff/northmind-pass-coordination-v1.zip"
    archive.parent.mkdir(parents=True, exist_ok=True); prefix = "evaluation/reports/external/pass-coordination-v1/"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names: z.write(DEST/n, prefix+n)
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() or any(hashlib.sha256(z.read(prefix+n)).hexdigest() != meta["sha256"] for n, meta in manifest.items()):
            raise ValueError("Capsule verification failed")
    write(PUBLIC/"pass-coordination-v1.json", read(DEST/"summary.json"))
    write(PUBLIC/"pass-coordination-v1-package.json", {"archive": archive.name,
          "sha256": file_hash(archive), "bytes": archive.stat().st_size, "files": len(names), "verified": True,
          "replay_scope": "event decisions and aggregates from derived scores; does not rerun geometry extraction"})
    return 0
