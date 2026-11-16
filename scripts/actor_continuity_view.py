"""Reuse three sealed public clips for a fixed-slot distance display. No data loader."""
import hashlib
import json
from pathlib import Path
import time
import unittest
import zipfile
from scripts.pass_coordination_view import read, write, source_hash
from possession.actor_continuity_view import viewer
from possession.external_data.schema import file_hash

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/"evaluation/reports/external/actor-continuity-view-v1"
PUBLIC = ROOT/"docs/public-results/2026-10-30"


def sources():
    names = ["possession/actor_continuity_view.py", "scripts/actor_continuity_view.py",
             "tests/test_actor_continuity_view.py", "possession/pass_coordination_view.py",
             "possession/pass_coordination_contrast.py", "scripts/pass_coordination_view.py", "scripts/run_local.py",
             "possession/external_data/schema.py"]
    return {n: source_hash(ROOT/n) for n in names}


def tests():
    suite = unittest.defaultTestLoader.discover(str(ROOT/"tests"), pattern="test_actor_continuity_view.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    write(DEST/"synthetic-status.json", {"tests": result.testsRun, "success": result.wasSuccessful(), "sources": sources()})
    return 0 if result.wasSuccessful() else 1


def build():
    started = time.perf_counter(); tested = read(DEST/"synthetic-status.json")
    if not tested["success"] or tested["sources"] != sources(): raise ValueError("Synthetic measurement/UI tests required")
    inputs = {}; selected = []
    for name in ("pass-coordination-view-v1", "pass-coordination-contrast-v1"):
        parent = ROOT/"evaluation/reports/external"/name
        public = read(PUBLIC/(name+".json")); seal = read(parent/"result-seal.json")
        if file_hash(parent/"registration.json") != public["registration_sha256"] or seal["registration_sha256"] != public["registration_sha256"]:
            raise ValueError("Existing clip identity mismatch")
        for file in ("review-payload.json",):
            if file_hash(parent/file) != seal["files"][file]: raise ValueError("Existing clip changed")
        inputs[name] = {"registration_sha256": public["registration_sha256"], "payload_sha256": seal["files"]["review-payload.json"]}
        payload = read(parent/"review-payload.json")
        cases = payload["cases"][:1] if name == "pass-coordination-view-v1" else payload["cases"]
        if len(cases) != (1 if name == "pass-coordination-view-v1" else 2): raise ValueError("Three original clips only")
        for i, case in enumerate(cases):
            if name == "pass-coordination-view-v1":
                initial = next((j for j, f in enumerate(case["frames"]) if f["selected"] and f["explanation"].get("valid")), 0)
                case.update(focus_frame=initial, inspection_side=1, label="Ejemplo anterior",
                            outcome="Ejemplo anterior: aquí no se añade una clasificación de posesión.")
            else:
                case["label"] = "Contraste "+str(i+1)
            case["outcome"] = "Esta vista de medición no muestra resultados de posesión."
            selected.append(case)
    stamp = {"version": "actor-continuity-view-v1", "sources": sources(), "inputs": inputs,
             "case_order": [c["id"] for c in selected], "scope": "Three existing public clips; browser-only fixed-slot distances, no classifier or thresholds."}
    marker = DEST/"registration.json"
    if marker.exists() and read(marker) != stamp: raise ValueError("Immutable display changed")
    write(marker, stamp); payload = {"cases": selected}
    write(DEST/"review-payload.json", payload)
    (DEST/"review.html").write_text(viewer(payload), encoding="utf-8", newline="\n")
    write(DEST/"summary.json", {"cases": len(selected), "cached_clips_only": True,
          "original_npz_accessed": False, "new_evaluation_or_training": False,
          "pose_or_possession_labels_created": False, "threshold_selected": False,
          "seconds": time.perf_counter()-started, "registration_sha256": file_hash(marker),
          "interpretation": "Qualitative measurement display; reference selection never changes classifier/GT."})
    if time.perf_counter()-started > 30 or sources() != stamp["sources"]: raise ValueError("Display build bound/source failed")
    write(DEST/"result-seal.json", {"registration_sha256": file_hash(marker),
          "files": {n: file_hash(DEST/n) for n in ("review-payload.json", "review.html", "summary.json", "synthetic-status.json")}})
    return verify()


def verify():
    stamp, seal = read(DEST/"registration.json"), read(DEST/"result-seal.json")
    if stamp["sources"] != sources() or seal["registration_sha256"] != file_hash(DEST/"registration.json"):
        raise ValueError("Display provenance changed")
    for n, digest in seal["files"].items():
        if file_hash(DEST/n) != digest: raise ValueError("Display artifact changed")
    return 0


def replay():
    verify()
    if viewer(read(DEST/"review-payload.json")) != (DEST/"review.html").read_text(encoding="utf-8"):
        raise ValueError("Display replay differs")
    return 0


def export():
    verify(); names = sorted([*read(DEST/"result-seal.json")["files"], "registration.json", "result-seal.json"])
    archive = ROOT/"evaluation/reports/handoff/northmind-actor-continuity-view-v1.zip"; archive.parent.mkdir(parents=True, exist_ok=True)
    prefix = "evaluation/reports/external/actor-continuity-view-v1/"
    manifest = {n: {"sha256": file_hash(DEST/n), "bytes": (DEST/n).stat().st_size} for n in names}
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names: z.write(DEST/n, prefix+n)
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() or any(hashlib.sha256(z.read(prefix+n)).hexdigest() != m["sha256"] for n, m in manifest.items()):
            raise ValueError("Display archive integrity failed")
    write(PUBLIC/"actor-continuity-view-v1.json", read(DEST/"summary.json"))
    write(PUBLIC/"actor-continuity-view-v1-package.json", {"archive": archive.name, "bytes": archive.stat().st_size,
          "sha256": file_hash(archive), "files": len(names), "verified": True,
          "html_sha256": file_hash(DEST/"review.html"), "html_bytes": (DEST/"review.html").stat().st_size,
          "scope": "Same three public IDSSE-derived clips, no raw originals, new predictions, target or logs"})
    return 0
