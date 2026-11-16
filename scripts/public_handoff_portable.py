"""Publishable archive of explicit PUBLIC experiment dependencies, never target.

Called only by the captured verify-long-diagnosis task after verification.
Copies bytes without inspecting model arrays, per-case tables or child logs.
"""
import json
from pathlib import Path
import zipfile

from possession.external_data.schema import file_hash
from possession.long_diagnosis_run import ROOT, DEST, snapshot
from possession.night.common import atomic_json


def export():
    dependencies = json.loads((DEST / "registration.json").read_text())["dependencies"]
    external = ROOT / "evaluation/reports/external"
    files = {Path(p) for p in dependencies if Path(p).is_relative_to(external)}
    for folder in (external / "evidence-first-validation-v1", external / "pulse-consensus-validation-v1"):
        files.update(folder / name for name in ("registration.json", "summary.json", "fold-0.json", "fold-6.json", "result-seal.json"))
    # All these files are produced solely by the public IDSSE diagnostic.
    files.update(p for p in DEST.rglob("*") if p.is_file() and p.suffix in (".json", ".npz")
                 and p.name != "running.lock")
    files = sorted(files)
    if any(not p.resolve().is_relative_to(external.resolve()) or p.is_symlink()
           or "test" in p.relative_to(external).parts or p.suffix not in (".json", ".npz", ".pt") for p in files):
        raise ValueError("Public handoff path outside allowlist")
    rows = [{"path": p.relative_to(ROOT).as_posix(), "sha256": file_hash(p), "bytes": p.stat().st_size} for p in files]
    manifest = {"version": "northmind-public-handoff-v1", "diagnostic": DEST.name,
        "scope": "public IDSSE experiment dependencies and diagnostic artifacts only",
        "files": rows, "excluded": ["data/", "target assets/results", "logs", "private GT", "Python environment"],
        "legacy_path_root": str(ROOT), "restore": "extract at repository root; preserve original registrations and hashes",
        "portability": "historical metadata contains original Windows absolute paths; original bytes preserved; see TEAM_HANDOFF.md"}
    out = ROOT / "evaluation/reports/handoff"
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "northmind-public-experiments-2026-10-28.zip"
    with zipfile.ZipFile(archive.with_suffix(".tmp"), "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p, row in zip(files, rows):
            z.write(p, row["path"])
        z.writestr("ARTIFACT-MANIFEST.json", json.dumps(manifest, indent=2))
    archive.with_suffix(".tmp").replace(archive)
    # Check member integrity and original byte hashes without opening arrays.
    import hashlib
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("Archive CRC mismatch")
        for row in rows:
            if hashlib.sha256(z.read(row["path"])).hexdigest() != row["sha256"]:
                raise ValueError("Archive member hash mismatch")
    atomic_json(out / "public-package-status.json", {"archive": archive.name,
        "sha256": file_hash(archive), "bytes": archive.stat().st_size, "file_count": len(rows),
        "scope": manifest["scope"], "verified": True})
    # Public aggregates are copied verbatim; inventory paths are intentionally
    # omitted from the small Git-tracked report to avoid personal path listings.
    report = ROOT / "docs/public-results/2026-10-28"
    report.mkdir(parents=True, exist_ok=True)
    for source, name in ((DEST / "public-summary.json", "long-diagnosis-v3.json"),
                         (external / "evidence-first-validation-v1/summary.json", "evidence-first-v1.json"),
                         (external / "pulse-consensus-validation-v1/summary.json", "pulse-consensus-v1.json")):
        # The source writers have been reviewed: aggregate metrics only.
        record = json.loads(source.read_text())
        forbidden = {"stable_id", "gt_index", "interval", "edits", "gained_gt_ids", "lost_gt_ids", "match"}
        def check(value):
            if isinstance(value, dict):
                if set(value) & forbidden:
                    raise ValueError("Per-case content in public report")
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)
        check(record)
        atomic_json(report / name, record)
    atomic_json(report / "package.json", {k: v for k, v in json.loads((out / "public-package-status.json").read_text()).items()})
    atomic_json(report / "audit.json", json.loads((DEST / "public-audit-v1.json").read_text()))
    return 0
