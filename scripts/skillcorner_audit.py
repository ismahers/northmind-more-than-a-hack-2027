"""Review public annotations independently of tracking/model/target loaders.

Only explicit pinned SkillCorner URLs. Child output is captured by run_local.
Inputs stay ignored; publish schema/coverage aggregates and content hashes only.
"""
import csv
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import ssl
import time
from urllib.request import Request, urlopen
import zipfile

import certifi

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "evaluation/reports/external/skillcorner-audit-v1"
PROTOCOL = ROOT / "configs/skillcorner_audit_v1.json"


def sha(value):
    return hashlib.sha256(value).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def csv_summary(content):
    text = content.decode("utf-8-sig")
    dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t")
    rows = list(csv.DictReader(io.StringIO(text), dialect=dialect))
    fields = list(rows[0]) if rows else []
    summary = {"rows": len(rows), "columns": fields, "null_counts": {}, "categories": {}, "numeric_ranges": {}}
    for field in fields:
        values = [r[field] for r in rows]
        nonempty = [v for v in values if v not in (None, "", "null", "None", "nan")]
        summary["null_counts"][field] = len(values)-len(nonempty)
        if any(s in field.lower() for s in ("phase", "possession", "team", "period", "type")):
            counts = Counter(nonempty)
            if len(counts) <= 35:
                summary["categories"][field] = dict(sorted(counts.items()))
        if any(s in field.lower() for s in ("frame", "time", "duration")):
            try:
                numeric = [float(v) for v in nonempty]
                if numeric:
                    summary["numeric_ranges"][field] = {"min": min(numeric), "max": max(numeric)}
            except ValueError:
                pass
    return summary


def run():
    DEST.mkdir(parents=True, exist_ok=True)
    p = read(PROTOCOL)
    old = read(ROOT / p["reference_config"])["skillcorner"]
    start = time.monotonic()
    context = ssl.create_default_context(cafile=certifi.where())
    receipts = {}

    def fetch(url, name):
        if not url.startswith(("https://api.github.com/repos/SkillCorner/opendata/",
                               "https://raw.githubusercontent.com/SkillCorner/opendata/",
                               "https://media.githubusercontent.com/media/SkillCorner/opendata/")):
            raise ValueError("Public allowlist only")
        if time.monotonic()-start > p["max_seconds"]:
            raise TimeoutError("Audit budget")
        path = DEST / "inputs" / name
        receipt = path.with_suffix(path.suffix+".receipt.json")
        if path.exists():
            data = path.read_bytes()
            saved = read(receipt)
            if saved["url"] != url or saved["sha256"] != sha(data):
                raise ValueError("Cached public input changed")
        else:
            request = Request(url, headers={"User-Agent": "Northmind-public-compatibility-audit", "Accept": "application/vnd.github+json"})
            with urlopen(request, context=context, timeout=45) as response:
                data = response.read(p["max_file_bytes"]+1)
            if len(data) > p["max_file_bytes"]:
                raise ValueError("Public file exceeds audit scope")
            if data.startswith(b"version https://git-lfs"):
                raise ValueError("LFS pointer must use reviewed media URL")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            write(receipt, {"url": url, "sha256": sha(data), "bytes": len(data)})
        receipts[name] = read(receipt)
        return data

    marker = DEST/"revision.json"
    if marker.exists():
        current = read(marker)["revision"]
    else:
        commit = json.loads(fetch("https://api.github.com/repos/SkillCorner/opendata/commits/master", "discovery.json"))
        current = commit["sha"]
        if not re.fullmatch("[0-9a-f]{40}", current):
            raise ValueError("Invalid revision")
        write(marker, {"revision": current, "resolved_before_annotation_audit": True})
    report = {"protocol": p, "old_revision": old["revision"], "current_revision": current,
              "configured_old_ids": sorted(old["matches"]), "revisions": {}}
    trees = {}
    for kind, revision in (("old", old["revision"]), ("current", current)):
        tree = json.loads(fetch(f"https://api.github.com/repos/SkillCorner/opendata/git/trees/{revision}?recursive=1", f"{kind}/tree.json"))
        if tree["truncated"]:
            raise ValueError("Incomplete public tree")
        entries = {e["path"]: e for e in tree["tree"] if e["type"] == "blob"}
        trees[kind] = entries
        matches = sorted({int(m.group(1)) for name in entries if (m := re.fullmatch(r"data/matches/(\d+)/(\d+)_match.json", name)) and m[1] == m[2]})
        row = {"match_ids": matches, "matches": {}, "documents": {}}
        for name in ("README.md", "LICENSE", "data/matches.json"):
            if name in entries:
                content = fetch(f"https://raw.githubusercontent.com/SkillCorner/opendata/{revision}/{name}", f"{kind}/{name}")
                row["documents"][name] = {"sha256": sha(content), "bytes": len(content)}
        for match in matches:
            prefix = f"data/matches/{match}/{match}"
            item = {"files": {name: {"git_blob": e["sha"], "bytes_in_git": e.get("size")}
                              for name, e in entries.items() if name.startswith(prefix)}}
            meta = json.loads(fetch(f"https://raw.githubusercontent.com/SkillCorner/opendata/{revision}/{prefix}_match.json", f"{kind}/{match}/metadata.json"))
            # Avoid personal/row content: report shape, teams and pitch availability only.
            item["metadata"] = {"keys": sorted(meta), "id": meta.get("id"),
                "pitch_length": meta.get("pitch_length"), "pitch_width": meta.get("pitch_width"),
                "roster_count": len(meta.get("players", []))}
            phases = prefix+"_phases_of_play.csv"
            if phases in entries:
                base = "https://media.githubusercontent.com/media" if entries[phases].get("size", 0) < 200 else "https://raw.githubusercontent.com"
                content = fetch(f"{base}/SkillCorner/opendata/{revision}/{phases}", f"{kind}/{match}/phases.csv")
                item["phases"] = csv_summary(content)
            row["matches"][str(match)] = item
        report["revisions"][kind] = row
        write(DEST/"public-progress.json", {"stage": "inventory", "revision_role": kind, "matches": len(matches)})
    old_ids, new_ids = (set(report["revisions"][k]["match_ids"]) for k in ("old", "current"))
    report["comparison"] = {"added_ids": sorted(new_ids-old_ids), "removed_ids": sorted(old_ids-new_ids),
        "current_ids_not_in_config": sorted(new_ids-set(old["matches"])),
        "changed_common_paths": [name for name in sorted(trees["current"].keys() & trees["old"].keys())
            if name.startswith("data/matches/") and trees["current"][name]["sha"] != trees["old"][name]["sha"]]}
    report.update(training_run=False, test_accessed=False, target_accessed=False,
        tracking_downloaded=False, full_HAD_labels_validated=False,
        seconds=time.monotonic()-start)
    write(DEST/"summary.json", report)
    write(DEST/"inputs-manifest.json", receipts)
    sources = [Path(__file__), PROTOCOL, ROOT/"configs/external_sources.json"]
    write(DEST/"result-seal.json", {"sources": {x.relative_to(ROOT).as_posix(): sha(x.read_bytes()) for x in sources},
        "outputs": {name: sha((DEST/name).read_bytes()) for name in ("summary.json", "inputs-manifest.json", "revision.json")}})
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
