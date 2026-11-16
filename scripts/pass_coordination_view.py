"""Build an offline explanation of the original first pair. No new evaluation."""
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
import zipfile
import numpy as np
from possession.pass_coordination_view import explain, viewer
from possession.pass_coordination import candidates
from possession.football_control import velocity
from possession.external_data.audit import segments
from possession.external_data.features import nearest_indices
from possession.external_data.schema import file_hash

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/"evaluation/reports/external/pass-coordination-view-v1"
REVIEW = ROOT/"evaluation/reports/external/oof-case-review-v1"
FROZEN = ROOT/"evaluation/reports/external/pass-coordination-v1"
PUBLIC = ROOT/"docs/public-results/2026-10-30"
CONFIG = ROOT/"configs/pass_coordination_v1.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


def source_hash(path):
    return hashlib.sha256(Path(path).read_text(encoding="utf-8-sig").replace("\r\n", "\n").encode()).hexdigest()


def sources():
    names = ["possession/pass_coordination_view.py", "scripts/pass_coordination_view.py",
             "tests/test_pass_coordination_view.py", "scripts/run_local.py",
             "possession/pass_coordination.py", "possession/football_control.py",
             "possession/external_data/features.py", "possession/external_data/audit.py",
             "possession/external_data/schema.py", "configs/pass_coordination_v1.json"]
    return {n: source_hash(ROOT/n) for n in names}


def tests():
    suite = unittest.defaultTestLoader.discover(str(ROOT/"tests"), pattern="test_pass_coordination_view.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    write(DEST/"synthetic-status.json", {"tests": result.testsRun, "success": result.wasSuccessful(), "sources": sources()})
    return 0 if result.wasSuccessful() else 1


def build():
    started = time.perf_counter()
    def check():
        if time.perf_counter()-started > 120: raise TimeoutError("Bounded explanation build expired")
    tested = read(DEST/"synthetic-status.json")
    if not tested["success"] or tested["sources"] != sources(): raise ValueError("Synthetic UI tests required")
    if file_hash(REVIEW/"registration.json") != read(ROOT/"docs/public-results/2026-10-29/oof-case-review-v1.json")["registration_sha256"]:
        raise ValueError("Original review identity changed")
    if file_hash(FROZEN/"registration.json") != read(PUBLIC/"pass-coordination-v1.json")["registration_sha256"]:
        raise ValueError("Frozen diagnostic identity changed")
    original, frozen = read(REVIEW/"registration.json"), read(FROZEN/"registration.json")
    seal = read(REVIEW/"result-seal.json")
    if seal["registration_sha256"] != file_hash(REVIEW/"registration.json"):
        raise ValueError("Original review seal changed")
    # The runner evolved; the actual frozen geometry implementation/configuration must not.
    frozen_names = ["possession/pass_coordination.py", "possession/football_control.py",
                    "possession/external_data/features.py", "possession/external_data/audit.py",
                    "possession/external_data/schema.py", "configs/pass_coordination_v1.json"]
    if any(source_hash(ROOT/n) != frozen["sources"][n] for n in frozen_names):
        raise ValueError("Do not change the closed formula during explanation")
    dependencies = {}
    def reviewed(name):
        if file_hash(REVIEW/name) != seal["files"][name]: raise ValueError("Review dependency changed")
        dependencies[name] = seal["files"][name]
        return REVIEW/name
    # Real clips/catalogue are consumed only inside this captured process, never by the agent.
    pair = read(reviewed("review-clips.json"))["pairs"][0]
    catalog = {r["id"]: r for r in read(reviewed("case-catalog.json"))}
    if pair["fold"] != "0" or len(pair["cases"]) != 2: raise ValueError("Original first pair only")
    selected = [catalog[c["id"]] for c in pair["cases"]]
    inputs = {}
    for row in selected:
        key = row["key"]
        unit = next(u for u in original["units"] if u["outer_fold"] == 0 and key in u["heldout"])
        expected = f"fold-0/inner-{unit['inner_fold']}/match-{unit['heldout'].index(key)}"
        if row["identity"] != expected or set(unit["train"]) & set(unit["heldout"]+unit["excluded"]):
            raise ValueError("Review role mismatch")
        item = original["inputs"][key]; path = (ROOT/item["path"]).resolve()
        if not path.is_relative_to((ROOT/"data/external/processed").resolve()) or path.suffix != ".npz" or file_hash(path) != item["sha256"]:
            raise ValueError("Review geometry identity mismatch")
        inputs[key] = item
        reviewed("cases/"+row["identity"]+".context.npz")
    stamp = {"version": "pass-coordination-view-v1", "sources": sources(), "inputs": inputs,
             "review_dependencies": dependencies, "case_order": [r["id"] for r in selected],
             "review_registration_sha256": file_hash(REVIEW/"registration.json"),
             "frozen_registration_sha256": file_hash(FROZEN/"registration.json"),
             "protocol": frozen["protocol"], "max_build_seconds": 120,
             "scope": "Original first pair, same order, geometry-only human review. No possession metrics or new labels."}
    marker = DEST/"registration.json"
    if marker.exists() and read(marker) != stamp: raise ValueError("Immutable view build changed")
    write(marker, stamp)
    cache = {}; payload = {"cases": []}; scope_count = frames_count = 0
    for row in selected:
        check(); key = row["key"]; spec = stamp["protocol"]
        if key not in cache:
            with np.load(ROOT/inputs[key]["path"], allow_pickle=False) as z:
                metadata = json.loads(str(z["metadata"]))
                m = SimpleNamespace(metadata=metadata, ball=z["ball"][:, :2],
                    **{k: z[k] for k in ("frame", "time", "period", "players", "player_available", "ball_available")})
            if metadata["provider"]+":"+metadata["match_id"] != key or metadata["native_fps"] != 25:
                raise ValueError("Unexpected geometry")
            spans = segments(m); run = np.empty(len(m.time), int)
            for r, (a, b) in enumerate(spans): run[a:b] = r
            vb, ok = velocity(m.ball, m.ball_available, m.time, run, 3, spec["max_ball_speed_mps"], 1.5/25)
            in_scope = key in spec["matches"]
            selected_frames = {r[0] for r in candidates(m, run, spec, check)} if in_scope else set()
            cache[key] = m, spans, run, vb, ok, in_scope, selected_frames
        m, spans, run, vb, ok, in_scope, selected_frames = cache[key]
        with np.load(REVIEW/("cases/"+row["identity"]+".context.npz"), allow_pickle=False) as z:
            parts = [tuple(map(int, z[f"output_{i}"])) for i in range(int(z["parts"]))]
            part = next(i for i, (a, b) in enumerate(parts) if a <= row["start"] and row["stop"] <= b)
            origin = float(z["time"][z[f"selected_{part}"][0]])
        output_a, output_b = parts[part]; a, b = spans[part]
        if abs(origin-m.time[a]) > .061: raise ValueError("Review clock mismatch")
        center_start = origin+(row["start"]-output_a)/25
        center_end = origin+(row["stop"]-output_a)/25
        first = max(origin, center_start-2); last = min(origin+(output_b-output_a-1)/25, center_end+2)
        grid = m.time[a]+np.arange(int(np.floor((m.time[b-1]-m.time[a])*10))+1)/10
        grid = grid[(grid >= first-1e-6) & (grid <= last+1e-6)]
        indices = a+nearest_indices(m.time[a:b], grid); frames = []
        for i in indices:
            check()
            e = explain(m, run, i, vb[i], vb[i+3], spec) if i+3 < b and ok[i] and ok[i+3] else {
                "valid": False, "reason": "Falta un tramo continuo del balón de ±0,12 s; sin proyección fiable."}
            players = [[float(x), float(y), side, slot] for side in (0, 1)
                       for slot, (x, y) in enumerate(m.players[i, side]) if m.player_available[i, side, slot] and np.isfinite([x, y]).all()]
            ball = m.ball[i].tolist() if m.ball_available[i] and np.isfinite(m.ball[i]).all() else None
            frames.append({"t": float(m.time[i]-first), "players": players, "ball": ball,
                           "explanation": e, "selected": int(i) in selected_frames})
        if not frames: raise ValueError("Empty selected clip")
        payload["cases"].append({"id": row["id"], "pitch": m.metadata["pitch_m"],
                                  "original_diagnostic_scope": in_scope, "frames": frames})
        scope_count += int(in_scope); frames_count += len(frames)
    write(DEST/"review-payload.json", payload)
    (DEST/"review.html").write_text(viewer(payload), encoding="utf-8", newline="\n")
    summary = {"cases": 2, "pair": 1, "order_preserved": True, "displayed_frames": frames_count,
               "cases_in_closed_diagnostic_scope": scope_count, "geometry_formula_changed": False,
               "possession_labels_read_from_original_npz": False, "new_evaluation": False,
               "training": False, "target_accessed": False, "seconds": time.perf_counter()-started,
               "registration_sha256": file_hash(marker), "interpretation": "Human-only visual explanation, not predictive evidence."}
    write(DEST/"summary.json", summary); check()
    if sources() != stamp["sources"]: raise ValueError("Explanation source changed")
    for key, item in inputs.items():
        if file_hash(ROOT/item["path"]) != item["sha256"]: raise ValueError("Geometry changed during build")
    write(DEST/"result-seal.json", {"registration_sha256": file_hash(marker),
          "files": {n: file_hash(DEST/n) for n in ("summary.json", "review.html", "review-payload.json", "synthetic-status.json")}})
    return verify()


def verify():
    stamp, seal = read(DEST/"registration.json"), read(DEST/"result-seal.json")
    if stamp["sources"] != sources() or seal["registration_sha256"] != file_hash(DEST/"registration.json"):
        raise ValueError("Explanation provenance changed")
    for name, digest in seal["files"].items():
        if file_hash(DEST/name) != digest: raise ValueError("Explanation file changed")
    return 0


def replay():
    verify()
    if viewer(read(DEST/"review-payload.json")) != (DEST/"review.html").read_text(encoding="utf-8"):
        raise ValueError("HTML replay differs")
    return 0


def export():
    verify(); names = sorted([*read(DEST/"result-seal.json")["files"], "registration.json", "result-seal.json"])
    archive = ROOT/"evaluation/reports/handoff/northmind-pass-coordination-view-v1.zip"
    archive.parent.mkdir(parents=True, exist_ok=True); prefix = "evaluation/reports/external/pass-coordination-view-v1/"
    manifest = {n: {"sha256": file_hash(DEST/n), "bytes": (DEST/n).stat().st_size} for n in names}
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names: z.write(DEST/n, prefix+n)
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() or any(hashlib.sha256(z.read(prefix+n)).hexdigest() != meta["sha256"] for n, meta in manifest.items()):
            raise ValueError("Explanation archive failed integrity check")
    write(PUBLIC/"pass-coordination-view-v1.json", read(DEST/"summary.json"))
    write(PUBLIC/"pass-coordination-view-v1-package.json", {"archive": archive.name, "bytes": archive.stat().st_size,
          "sha256": file_hash(archive), "files": len(names), "verified": True,
          "html_sha256": file_hash(DEST/"review.html"), "html_bytes": (DEST/"review.html").stat().st_size,
          "scope": "Public IDSSE-derived first-pair human viewer; no target, GT labels, logs or complete matches"})
    return 0
