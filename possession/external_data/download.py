"""Explicit downloads of pinned public sources. Never uploads or scans competition data."""

import json
from pathlib import Path
import shutil
import ssl
from urllib.request import urlopen

import certifi

from .schema import file_hash

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "configs" / "external_sources.json"


def download_plan(provider, match_id):
    config = json.loads(SOURCES.read_text(encoding="utf-8"))[provider]
    revision, repo = config["revision"], config["repo"]
    if provider == "idsse":
        if match_id not in config["matches"]:
            raise ValueError("Unknown IDSSE match")
        suffix = "DFL-COM-" + config["matches"][match_id] + "_DFL-MAT-" + match_id + ".xml"
        names = ["DFL_02_01_matchinformation_" + suffix, "DFL_04_03_positions_raw_observed_" + suffix]
        return [(name, "https://huggingface.co/datasets/" + repo + "/resolve/" + revision + "/" + name)
                for name in names]
    if provider != "skillcorner" or int(match_id) not in config["matches"]:
        raise ValueError("Unknown SkillCorner match")
    prefix = "data/matches/" + str(match_id) + "/"
    metadata, tracking = str(match_id) + "_match.json", str(match_id) + "_tracking_extrapolated.jsonl"
    return [(metadata, "https://raw.githubusercontent.com/" + repo + "/" + revision + "/" + prefix + metadata),
            (tracking, "https://media.githubusercontent.com/media/" + repo + "/" + revision + "/" + prefix + tracking)]


def download_public(provider, match_ids, destination):
    config = json.loads(SOURCES.read_text(encoding="utf-8"))[provider]
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for match_id in match_ids:
        for name, url in download_plan(provider, str(match_id)):
            target = destination / name
            receipt_path = destination / (name + ".receipt.json")
            if target.exists():
                if not receipt_path.exists():
                    raise ValueError("Existing source lacks pinned download receipt; verify before reuse")
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if receipt.get("url") != url or receipt.get("sha256") != file_hash(target):
                    raise ValueError("Existing source differs from receipt")
            else:
                temporary = destination / (name + ".part")
                context = ssl.create_default_context(cafile=certifi.where())
                with urlopen(url, timeout=120, context=context) as source, temporary.open("wb") as stream:
                    shutil.copyfileobj(source, stream, length=1024 * 1024)
                with temporary.open("rb") as stream:
                    head = stream.read(128)
                if not head or head.startswith(b"version https://git-lfs"):
                    raise ValueError("Empty download or unresolved Git LFS pointer")
                receipt = {"url": url, "revision": config["revision"], "license": config["license"],
                           "sha256": file_hash(temporary), "bytes": temporary.stat().st_size}
                temporary.replace(target)
                receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
            records.append(receipt)
    return records
