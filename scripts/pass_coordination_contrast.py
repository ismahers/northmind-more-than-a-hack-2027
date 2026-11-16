"""Human-only review of errors/corrections from the sealed diagnostic. No refitting."""
import json
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
import zipfile
import hashlib
import numpy as np
from scripts.pass_coordination_view import read, write, source_hash
from possession.pass_coordination_contrast import review_records, choose_pair, viewer
from possession.pass_coordination_view import explain
from possession.pass_coordination import candidates, event_features
from possession.football_control import velocity
from possession.external_data.audit import segments
from possession.external_data.features import nearest_indices
from possession.external_data.schema import file_hash

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/"evaluation/reports/external/pass-coordination-contrast-v1"
PARENT = ROOT/"evaluation/reports/external/pass-coordination-v1"
PUBLIC = ROOT/"docs/public-results/2026-10-30"


def sources():
    names = ["possession/pass_coordination_contrast.py", "scripts/pass_coordination_contrast.py",
             "possession/pass_coordination_view.py", "scripts/pass_coordination_view.py",
             "tests/test_pass_coordination_contrast.py", "scripts/run_local.py",
             "possession/pass_coordination.py", "possession/football_control.py",
             "possession/external_data/features.py", "possession/external_data/audit.py",
             "possession/external_data/schema.py", "configs/pass_coordination_v1.json"]
    return {n: source_hash(ROOT/n) for n in names}


def tests():
    suite = unittest.defaultTestLoader.discover(str(ROOT/"tests"), pattern="test_pass_coordination_contrast.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    status = {"tests": result.testsRun, "success": result.wasSuccessful(), "sources": sources(),
              "failed_test_names": [case.id() for case, _ in result.failures],
              "error_test_names": [case.id() for case, _ in result.errors]}
    write(DEST/"synthetic-status.json", status)
    # Synthetic-only test IDs; no exception strings, traces, cases or data in this aggregate.
    write(PUBLIC/"pass-coordination-contrast-synthetic.json", status)
    return 0 if result.wasSuccessful() else 1


def build():
    started = time.perf_counter()
    def check():
        if time.perf_counter()-started > 120: raise TimeoutError("Bounded contrast build expired")
    tested = read(DEST/"synthetic-status.json")
    if not tested["success"] or tested["sources"] != sources(): raise ValueError("Synthetic tests required")
    published = read(PUBLIC/"pass-coordination-v1.json")
    if file_hash(PARENT/"registration.json") != published["registration_sha256"]:
        raise ValueError("Closed diagnostic identity mismatch")
    parent, seal = read(PARENT/"registration.json"), read(PARENT/"result-seal.json")
    if seal["registration_sha256"] != published["registration_sha256"]:
        raise ValueError("Closed diagnostic seal mismatch")
    spec = parent["protocol"]
    frozen_names = ["possession/pass_coordination.py", "possession/football_control.py",
                    "possession/external_data/features.py", "possession/external_data/audit.py",
                    "possession/external_data/schema.py", "configs/pass_coordination_v1.json"]
    if any(source_hash(ROOT/n) != parent["sources"][n] for n in frozen_names):
        raise ValueError("Closed formula must remain unchanged")
    cache = {}; records = []; dependencies = {}
    for index, key in enumerate(spec["matches"]):
        check(); name = f"match-{index}.npz"
        if file_hash(PARENT/name) != seal["files"][name]: raise ValueError("Closed context changed")
        dependencies[name] = seal["files"][name]
        with np.load(PARENT/name, allow_pickle=False) as z: data = dict(z)
        cache[key] = data; records.extend(review_records(data, key, spec))
    harmed = sum(r["role"] == "harm" for r in records); corrected = len(records)-harmed
    if harmed != published["total_new_errors"] or corrected != published["total_corrections"]:
        raise ValueError("Frozen event decisions do not reproduce known counts")
    pair = choose_pair(records)
    if len(pair) != 2: raise ValueError("Both roles required; do not invent a contrast")
    inputs = {r["key"]: parent["inputs"][r["key"]] for r in pair}
    for item in inputs.values():
        path = (ROOT/item["path"]).resolve()
        if not path.is_relative_to((ROOT/"data/external/processed").resolve()) or path.suffix != ".npz" or file_hash(path) != item["sha256"]:
            raise ValueError("Allowlisted public geometry mismatch")
    stamp = {"version": "pass-coordination-contrast-v1", "sources": sources(), "protocol": spec,
             "parent_registration_sha256": published["registration_sha256"], "dependencies": dependencies,
             "inputs": inputs, "case_selection": pair,
             "selection": "Median harm by absolute logit shift; match correction by proposed side, match, then signal/base-margin distance; hash order.",
             "max_build_seconds": 120, "new_fitting_or_evaluation": False}
    marker = DEST/"registration.json"
    if marker.exists() and read(marker) != stamp: raise ValueError("Immutable contrast changed")
    write(marker, stamp)
    geometry = {}; payload = {"cases": []}; frame_count = 0
    for row in pair:
        check(); key = row["key"]
        if key not in geometry:
            with np.load(ROOT/inputs[key]["path"], allow_pickle=False) as z:
                meta = json.loads(str(z["metadata"]))
                m = SimpleNamespace(metadata=meta, ball=z["ball"][:, :2],
                     **{k: z[k] for k in ("frame", "time", "period", "players", "player_available", "ball_available")})
            if meta["provider"]+":"+meta["match_id"] != key or meta["native_fps"] != 25:
                raise ValueError("Unexpected geometry identity")
            spans = segments(m); run = np.empty(len(m.time), int)
            for r, (a, b) in enumerate(spans): run[a:b] = r
            chosen = candidates(m, run, spec, check); saved = cache[key]
            if len(chosen) != len(saved["truth"]): raise ValueError("Frozen candidate count changed")
            # Replay the geometry, not predictions/labels, to recover exact original centres.
            for j, (i, cosine, outgoing) in enumerate(chosen):
                check(); g, c, n = event_features(m, run, i, cosine, outgoing, spec)
                if not (np.allclose(g, saved["g"][j], atol=1e-12, rtol=1e-12, equal_nan=True)
                        and np.allclose(c, saved["c"][j], atol=1e-12, rtol=1e-12, equal_nan=True)
                        and np.array_equal(n, saved["counts"][j])):
                    raise ValueError("Frozen event geometry changed")
            vb, ok = velocity(m.ball, m.ball_available, m.time, run, 3, spec["max_ball_speed_mps"], 1.5/25)
            geometry[key] = m, spans, run, chosen, vb, ok
        m, spans, run, chosen, vb, ok = geometry[key]
        focus, _, _ = chosen[row["event_index"]]; a, b = spans[int(run[focus])]
        first, last = max(m.time[a], m.time[focus]-2), min(m.time[b-1], m.time[focus]+2)
        grid = m.time[a]+np.arange(int(np.floor((m.time[b-1]-m.time[a])*10))+1)/10
        grid = grid[(grid >= first-1e-6) & (grid <= last+1e-6)]
        indices = a+nearest_indices(m.time[a:b], grid)
        focus_display = np.flatnonzero(indices == focus)
        if len(focus_display) != 1: raise ValueError("Exact focus frame required")
        selected = {x[0] for x in chosen}; frames = []
        for i in indices:
            check()
            e = explain(m, run, i, vb[i], vb[i+3], spec) if i+3 < b and ok[i] and ok[i+3] else {
                "valid": False, "reason": "Sin tramo continuo del balón de ±0,12 s."}
            players = [[float(x), float(y), side, slot] for side in (0, 1)
                       for slot, (x, y) in enumerate(m.players[i, side]) if m.player_available[i, side, slot] and np.isfinite([x, y]).all()]
            ball = m.ball[i].tolist() if m.ball_available[i] and np.isfinite(m.ball[i]).all() else None
            frames.append({"t": float(m.time[i]-first), "players": players, "ball": ball,
                           "explanation": e, "selected": int(i) in selected})
        names = "HA"
        outcome = ("Resultado ya registrado en el instante analizado (H/A condicional, no posesión completa):\n"
                   f"Base: {names[row['base']]} · fórmula: {names[row['full']]} · etiqueta pública IDSSE: {names[row['truth']]}\n"
                   +("El cambio creó un error respecto a la base." if row["role"] == "harm" else "El cambio corrigió un error de la base.")
                   +"\nEsto no identifica un rebote o disputa ni permite modificar la etiqueta.")
        payload["cases"].append({"id": row["id"], "pitch": m.metadata["pitch_m"], "frames": frames,
                                  "original_diagnostic_scope": True, "inspection_side": row["inspection_side"],
                                  "focus_frame": int(focus_display[0]), "outcome": outcome})
        frame_count += len(frames)
    write(DEST/"review-payload.json", payload)
    (DEST/"review.html").write_text(viewer(payload), encoding="utf-8", newline="\n")
    write(DEST/"summary.json", {"existing_harms": harmed, "existing_corrections": corrected,
          "review_pairs": 1, "cases": 2, "displayed_frames": frame_count,
          "geometry_replay_verified": True, "formula_changed": False, "new_fitting_or_evaluation": False,
          "native_npz_possession_labels_read": False, "historical_public_train_labels_used_for_selection": True,
          "target_accessed": False, "seconds": time.perf_counter()-started,
          "registration_sha256": file_hash(marker), "interpretation": "Selected human review; harms are not annotated rebounds or disputes."})
    check()
    if sources() != stamp["sources"]: raise ValueError("Contrast sources changed")
    for item in inputs.values():
        if file_hash(ROOT/item["path"]) != item["sha256"]: raise ValueError("Geometry input changed")
    write(DEST/"result-seal.json", {"registration_sha256": file_hash(marker),
          "files": {n: file_hash(DEST/n) for n in ("summary.json", "review.html", "review-payload.json", "synthetic-status.json")}})
    return verify()


def verify():
    stamp, seal = read(DEST/"registration.json"), read(DEST/"result-seal.json")
    if stamp["sources"] != sources() or seal["registration_sha256"] != file_hash(DEST/"registration.json"):
        raise ValueError("Contrast provenance changed")
    for n, digest in seal["files"].items():
        if file_hash(DEST/n) != digest: raise ValueError("Contrast output changed")
    return 0


def replay():
    verify()
    if viewer(read(DEST/"review-payload.json")) != (DEST/"review.html").read_text(encoding="utf-8"):
        raise ValueError("Contrast HTML replay differs")
    return 0


def export():
    verify(); names = sorted([*read(DEST/"result-seal.json")["files"], "registration.json", "result-seal.json"])
    prefix = "evaluation/reports/external/pass-coordination-contrast-v1/"
    archive = ROOT/"evaluation/reports/handoff/northmind-pass-coordination-contrast-v1.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    manifest = {n: {"sha256": file_hash(DEST/n), "bytes": (DEST/n).stat().st_size} for n in names}
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names: z.write(DEST/n, prefix+n)
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() or any(hashlib.sha256(z.read(prefix+n)).hexdigest() != m["sha256"] for n, m in manifest.items()):
            raise ValueError("Contrast archive integrity failed")
    write(PUBLIC/"pass-coordination-contrast-v1.json", read(DEST/"summary.json"))
    write(PUBLIC/"pass-coordination-contrast-v1-package.json", {"archive": archive.name, "bytes": archive.stat().st_size,
          "sha256": file_hash(archive), "files": len(names), "verified": True,
          "html_sha256": file_hash(DEST/"review.html"), "html_bytes": (DEST/"review.html").stat().st_size,
          "scope": "Two selected public IDSSE-derived clips for human review, no target, complete matches or logs"})
    return 0
