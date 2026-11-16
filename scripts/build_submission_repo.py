"""Build a source-only review directory. Never initialize or publish a repository.

Run through scripts.run_local. Inputs are exclusively allowlisted source files;
no target assets, generated reports, predictions, model weights or logs are read.
"""
import fnmatch
import hashlib
import json
from pathlib import Path, PurePosixPath
import uuid


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "_pub"
STATUS = ROOT / "docs/public-results/2026-11-16/submission-source-status.json"
ALLOWLIST = ROOT / "configs/submission_source_allowlist.json"
TEXT_SUFFIXES = frozenset({".py", ".json", ".txt", ".md"})
GENERATED = frozenset({".gitignore", ".gitattributes", "SOURCE-MANIFEST.json",
                       "PUBLIC-MODEL-MANIFEST.json"})


def _relative(value):
    if not isinstance(value, str) or "\\" in value or "\x00" in value:
        raise ValueError("Invalid source-relative path")
    path = PurePosixPath(value)
    if (path.is_absolute() or any(p in {"", ".", ".."} or ":" in p
                                  for p in value.split("/"))):
        raise ValueError("Source-relative path escapes root")
    return path


def _confined(root, path):
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Path escapes source root")
    for node in (path, *path.parents):
        if node == root:
            break
        if node.is_symlink() or (hasattr(node, "is_junction") and node.is_junction()):
            raise ValueError("Links are not allowed in source export")


def _excluded(relative, policy):
    lower = relative.casefold()
    for prefix in policy["excluded_prefixes"]:
        prefix = prefix.casefold().rstrip("/")
        if lower == prefix or lower.startswith(prefix + "/"):
            return True
    return any(fnmatch.fnmatchcase(lower, pattern.casefold())
               or (pattern.startswith("**/") and
                   fnmatch.fnmatchcase(lower, pattern[3:].casefold()))
               for pattern in policy["excluded_globs"])


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def build(root=ROOT, destination=DESTINATION, status=STATUS, *, refresh=False):
    root, destination, status = Path(root), Path(destination), Path(status)
    _confined(root, destination)
    _confined(root, status)
    if destination.resolve() == root.resolve():
        raise ValueError("Review directory must be separate from source root")
    if destination.exists() and (not destination.is_dir() or
                                 (not refresh and any(destination.iterdir()))):
        raise ValueError("Review directory must be absent or empty")
    policy_path = root / "configs/submission_source_allowlist.json"
    _confined(root, policy_path)
    policy = json.loads(policy_path.read_text(encoding="utf-8-sig"))
    if policy["version"] != 1:
        raise ValueError("Unsupported source allowlist version")
    selected = {}

    def select(source_name, target_name=None, *, optional=False):
        source_rel = _relative(source_name).as_posix()
        target_rel = _relative(target_name or source_name).as_posix()
        if _excluded(source_rel, policy) or _excluded(target_rel, policy):
            raise ValueError("Excluded path selected for source export")
        source = root / source_rel
        _confined(root, source)
        if optional and not source.exists():
            return
        if not source.is_file():
            raise ValueError("Required source file is missing")
        if source.name != "LICENSE" and source.suffix.lower() not in TEXT_SUFFIXES:
            raise ValueError("Unsupported source type")
        if target_rel.casefold() in {name.casefold() for name in GENERATED}:
            raise ValueError("Source collides with generated export metadata")
        key = target_rel.casefold()
        if key in selected:
            if selected[key] != (source_rel, target_rel):
                raise ValueError("Colliding destination paths")
            return
        selected[key] = (source_rel, target_rel)

    for name in policy["files"]:
        select(name)
    for name in policy.get("optional_files", []):
        select(name, optional=True)
    for pattern in policy["globs"]:
        _relative(pattern)
        # Reject broad globs before enumeration; only reviewed source roots.
        if pattern.split("/")[0] not in {"starter", "possession", "scripts", "configs"}:
            raise ValueError("Source glob outside reviewed roots")
        for path in sorted(root.glob(pattern)):
            _confined(root, path)
            if path.is_file():
                select(path.relative_to(root).as_posix())
    for source, target in policy["document_mappings"].items():
        select(source, target)

    # Read and hash only the reviewed source set before creating any output.
    contents = []
    rows = []
    for source_rel, target_rel in sorted(selected.values(), key=lambda row: row[1]):
        content = (root / source_rel).read_bytes().replace(b"\r\n", b"\n")
        contents.append((target_rel, content))
        rows.append({"path": target_rel, "source": source_rel, "bytes": len(content),
                     "sha256": hashlib.sha256(content).hexdigest()})
    models = {
        "archive": "northmind-public-experiments-2026-10-28.zip",
        "bytes": 35910601,
        "sha256": "f09c894ad962364a2a16357749d9de304552532db58bb047f6fd2f9a1ca25162",
        "file_count": 66,
        "url": "https://github.com/ismahers/northmind-submission-2027/releases/download/submission-v1/northmind-public-experiments-2026-10-28.zip",
        "scope": "Public IDSSE model and experiment dependencies; no target assets or predictions",
        "attribution": {"dataset": "pysport/idsse-data", "license": "CC-BY-4.0",
                        "revision": "a715a38dfbaf5f58e431727c2b78d174101a703c",
                        "url": "https://huggingface.co/datasets/pysport/idsse-data"},
        "access_status": "Public Release access must be verified before submission",
    }
    contents.extend([
        (".gitignore", b"data/\nevaluation/reports/\ndocs/public-results/\n*.csv\n*.npz\n*.pt\n*.log\n*logs/\n.venv/\nvenv/\n__pycache__/\n*.py[cod]\n"),
        (".gitattributes", b"* text=auto eol=lf\n"),
        ("SOURCE-MANIFEST.json", _json_bytes({"version": 1, "files": rows,
                                             "hash_scope": "LF normalized source bytes"})),
        ("PUBLIC-MODEL-MANIFEST.json", _json_bytes(models)),
    ])
    # Preflight only the known destinations. Do not enumerate, read or remove
    # any existing caches, predictions, logs or unrelated files during refresh.
    for relative, _ in contents:
        target = destination / relative
        _confined(destination, target)
        if target.exists() and not target.is_file():
            raise ValueError("Source destination is not a regular file")
        if not refresh and target.exists():
            raise ValueError("Source destination already exists")
        for parent in target.parents:
            if parent == destination:
                break
            if parent.exists() and not parent.is_dir():
                raise ValueError("Source destination parent is not a directory")
    destination.mkdir(parents=True, exist_ok=True)
    for relative, content in contents:
        target = destination / relative
        _confined(destination, target)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(content)
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()
    safe = {"version": 1, "status": "source-review-copy-built",
            "files_count": len(contents), "bytes": sum(len(c) for _, c in contents),
            "excluded_paths_checked": True, "files": sorted(name for name, _ in contents),
            "publication_performed": False, "target_data_read": False,
            "refreshed": bool(refresh)}
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_bytes(_json_bytes(safe))
    return safe


def main():
    build()
    return 0


def refresh_review_copy():
    """Update only allowlisted sources and generated metadata, preserving outputs."""
    build(refresh=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
