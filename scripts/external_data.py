"""Reviewed public-data commands; execute through scripts.run_local."""

import argparse
import json
from pathlib import Path

from possession.external_data.audit import audit
from possession.external_data.download import ROOT, SOURCES, download_public
from possession.external_data.idsse import discover_idsse, load_idsse
from possession.external_data.schema import Match
from possession.external_data.skillcorner import load_skillcorner
from possession.learning.protocol import inventory, lomo

EXTERNAL = ROOT / "data" / "external"
REPORTS = ROOT / "evaluation" / "reports" / "external"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def download_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("idsse", "skillcorner"), required=True)
    parser.add_argument("--match", action="append", help="Repeat for selected matches; default all pinned matches")
    args = parser.parse_args()
    config = json.loads(SOURCES.read_text(encoding="utf-8"))[args.provider]
    matches = args.match or list(config["matches"])
    download_public(args.provider, matches, EXTERNAL / "raw" / args.provider)
    return 0


def prepare_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("idsse", "skillcorner"), required=True)
    args = parser.parse_args()
    raw = EXTERNAL / "raw" / args.provider
    if args.provider == "idsse":
        pairs = discover_idsse(raw)
        loader = load_idsse
    else:
        pairs = [(p, p.with_name(p.name.replace("_match.json", "_tracking_extrapolated.jsonl")))
                 for p in sorted(raw.glob("*_match.json"))]
        loader = load_skillcorner
    if not pairs:
        raise ValueError("Download public matches first")
    failed = False
    for meta, tracking in pairs:
        match = loader(meta, tracking)
        report = audit(match)
        write_json(REPORTS / "audits" / (match.key.replace(":", "_") + ".json"), report)
        temporal_issues = report["adapter_issues"]
        if (report["temporal_error"] or temporal_issues.get("inconsistent_object_timestamps", 0)
                or temporal_issues.get("invalid_timestamp", 0)):
            failed = True
            continue
        target = EXTERNAL / "processed" / (match.key.replace(":", "_") + ".npz")
        match.save(target)
    if failed:
        raise ValueError("Quarantined matches with invalid clocks; inspect external audits locally")
    return 0


def folds_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-skillcorner", action="store_true")
    parser.add_argument("--name", default="idsse-only")
    args = parser.parse_args()
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.name):
        raise ValueError("Invalid manifest name")
    pattern = "*.npz" if args.include_skillcorner else "idsse_*.npz"
    paths = sorted((EXTERNAL / "processed").glob(pattern))
    records = inventory(paths)
    if args.include_skillcorner and not any(key.startswith("skillcorner:") for key in records):
        raise ValueError("Condition B requires at least one prepared SkillCorner match")
    folds = lomo(records)
    target = REPORTS / "folds" / (args.name + ".json")
    document = {"protocol": "idsse-lomo-v1; rotating inner validation; no test selection",
                "artifacts": records, "folds": [f.to_dict() for f in folds]}
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != document:
        raise ValueError("Manifest is frozen; choose a new name for changed data")
    write_json(target, document)
    return 0
