"""Existence-only audit: never open assets, archives, GT or generated case data.

Run only through scripts.run_local. Public output contains booleans and counts,
no local absolute paths or asset contents. Directory traversal never follows links.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from games import asset_paths, DATA_DIR

ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg", ".ts"}
POSE_NAMES = {"1925299.jsonl.zip", "1996435.jsonl.zip", "1925299.jsonl", "1996435.jsonl",
              "sample_1925299_phase406.jsonl.gz"}


def inventory(root):
    counts = Counter()
    pose_candidates = 0
    videos = []
    known_pose_names = 0
    errors = []
    def failed(_error):
        errors.append(True)  # Do not publish paths or exception contents.
    if not root.is_dir():
        return {"directory_present": False, "video_files_by_extension": {},
                "video_candidates": [], "pose_named_candidates": 0,
                "official_pose_filename_candidates": 0, "walk_errors": 0}
    for current, dirs, files in os.walk(root, followlinks=False, onerror=failed):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "node_modules"}
                   and not (Path(current)/d).is_symlink()]
        for name in files:
            extension = Path(name).suffix.lower()
            if extension in VIDEO_EXTENSIONS:
                counts[extension] += 1
                item = Path(current)/name
                videos.append({"relative_path": item.relative_to(root).as_posix(),
                               "bytes": item.stat().st_size})
            if "bodypose" in name.lower() or "body_pose" in name.lower():
                pose_candidates += 1
            if name in POSE_NAMES:
                known_pose_names += 1
    return {"directory_present": True, "video_files_by_extension": dict(sorted(counts.items())),
            "video_candidates": sorted(videos, key=lambda v: v["relative_path"]),
            "pose_named_candidates": pose_candidates, "official_pose_filename_candidates": known_pose_names,
            "walk_errors": len(errors)}


def main():
    # Only exact expected names are checked for the gated target. No file is read.
    target = {source: {key: Path(path).is_file() for key, path in asset_paths(source=source).items()}
              for source in ("tactical", "broadcast")}
    external = ROOT/"data/external"
    names = ("scripts/video_availability.py", "games.py", "configs/external_sources.json",
             "CHALLENGE.md", "DATA.md", "RULES.md")
    hashes = {name: hashlib.sha256((ROOT/name).read_text(encoding="utf-8-sig")
              .replace("\r\n", "\n").encode()).hexdigest() for name in names}
    result = {"audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "existence_only": True, "asset_contents_opened": False,
        "training_or_inference": False, "ground_truth_opened": False,
        "data_root_environment_override": bool(os.environ.get("MTAH_DATA_DIR")),
        "expected_target_assets": target,
        "configured_data_root": inventory(Path(DATA_DIR)),
        "repository_external_data": inventory(external),
        "repository_root_videos": sum(p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
                                       for p in ROOT.iterdir()),
        "scope": "Configured data root, repository data/external and repository root only; no disks, Downloads, archive members or asset contents scanned.",
        "limitations": "Existence does not establish validity or synchronisation. Pose filename heuristic is not a content/schema audit. No exhaustive machine-wide search.",
        "sources_sha256_lf": hashes}
    text = json.dumps(result, indent=2, allow_nan=False)+"\n"
    local = ROOT/"evaluation/reports/infrastructure/video-availability-2026-10-30.json"
    public = ROOT/"docs/public-results/2026-10-30/video-availability.json"
    for dest in (local, public):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8", newline="\n")
    return 0
