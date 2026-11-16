"""Restore only the pinned public handoff, preserving every original sealed byte.

Run through scripts.run_local; no network, target data, or model deserialization.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "evaluation/reports/public-assets/northmind-public-experiments-2026-10-28.zip"
ARCHIVE_SHA256 = "f09c894ad962364a2a16357749d9de304552532db58bb047f6fd2f9a1ca25162"
ARCHIVE_BYTES = 35910601
FILE_COUNT = 66
STATUS = ROOT / "docs/public-results/2026-11-16/submission-assets-status.json"
ALLOWED_DIRS = frozenset({
    "temporal-teachers-v1", "temporal-probe-v1", "scale-selective-v1", "folds",
    "transfer-symmetry-validation-v1", "evidence-first-validation-v1",
    "pulse-consensus-validation-v1", "long-diagnosis-v1", "long-diagnosis-v3",
})


def _digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _io_path(path):
    """Use Windows extended paths for I/O without changing archive member names."""
    path = Path(path)
    if os.name != "nt":
        return path.absolute()
    name = os.path.abspath(path)
    if name.startswith("\\\\?\\"):
        return Path(name)
    if name.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + name[2:])
    return Path("\\\\?\\" + name)


def _destination(name, root):
    root = _io_path(root)
    if not isinstance(name, str) or "\\" in name or "\x00" in name:
        raise ValueError("Invalid public archive member")
    parts = name.split("/")
    if (len(parts) < 5 or parts[:3] != ["evaluation", "reports", "external"]
            or parts[3] not in ALLOWED_DIRS or "test" in parts
            or any(part in ("", ".", "..") or ":" in part for part in parts)
            or PurePosixPath(name).suffix not in {".json", ".npz", ".pt"}):
        raise ValueError("Public archive member outside allowlist")
    path = root.joinpath(*parts)
    if any(parent.is_symlink() for parent in (path, *path.parents) if parent != root and parent.is_relative_to(root)):
        raise ValueError("Symlink in public archive destination")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Public archive member escapes repository")
    return path


def restore(archive=ARCHIVE, root=ROOT, *, expected_sha256=ARCHIVE_SHA256,
            expected_bytes=ARCHIVE_BYTES, expected_count=FILE_COUNT):
    """Validate the complete archive before writing any of its public members."""
    archive, root = _io_path(archive), _io_path(root)
    if archive.stat().st_size != expected_bytes or _digest(archive) != expected_sha256:
        raise ValueError("Public archive identity mismatch")
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(name.casefold() for name in names)):
            raise ValueError("Colliding public archive members")
        if any(info.is_dir() or stat.S_IFMT(info.external_attr >> 16) not in (0, stat.S_IFREG) for info in infos):
            raise ValueError("Nonregular public archive member")
        manifest = json.loads(bundle.read("ARTIFACT-MANIFEST.json"))
        if manifest["version"] != "northmind-public-handoff-v1":
            raise ValueError("Unknown public manifest version")
        rows = manifest["files"]
        if len(rows) != expected_count:
            raise ValueError("Incomplete public archive manifest")
        declared = [row["path"] for row in rows]
        if len(declared) != len(set(name.casefold() for name in declared)):
            raise ValueError("Colliding public manifest members")
        if set(names) != set(declared) | {"ARTIFACT-MANIFEST.json"}:
            raise ValueError("Public archive and manifest disagree")
        pending = []
        valid = 0
        # Hash every member and check every existing destination before writing.
        for row in rows:
            path = _destination(row["path"], root)
            content = bundle.read(row["path"])
            if len(content) != row["bytes"] or hashlib.sha256(content).hexdigest() != row["sha256"]:
                raise ValueError("Public member identity mismatch")
            if path.exists():
                if not path.is_file() or path.is_symlink() or _digest(path) != row["sha256"]:
                    raise ValueError("Existing public artifact differs; preserve it")
                valid += 1
            else:
                pending.append((path, row))
        for path, row in pending:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Recheck after directory creation to reject a redirected parent.
            _destination(row["path"], root)
            temp = path.with_name(path.name + ".restore-" + uuid.uuid4().hex)
            try:
                with temp.open("xb") as stream:
                    stream.write(bundle.read(row["path"]))
                if path.exists():
                    raise ValueError("Public destination appeared during restore")
                if os.name == "nt":
                    temp.rename(path)  # Windows rename fails if destination exists.
                else:
                    os.link(temp, path)  # Atomic exclusive publication on POSIX.
            finally:
                temp.unlink(missing_ok=True)
    return {"status": "valid", "phase": "restore", "valid": valid + len(pending),
            "restored": len(pending), "missing": 0}


def _status(record):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temp = STATUS.with_suffix(".tmp")
    temp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temp.replace(STATUS)


def audit_headers():
    """Export names only from the authenticated public archive, never payloads."""
    if ARCHIVE.stat().st_size != ARCHIVE_BYTES or _digest(ARCHIVE) != ARCHIVE_SHA256:
        raise ValueError("Public archive identity mismatch")
    with zipfile.ZipFile(ARCHIVE) as bundle:
        manifest = json.loads(bundle.read("ARTIFACT-MANIFEST.json"))
        families = sorted({row["path"].split("/")[3] for row in manifest["files"]
                           if row["path"].startswith("evaluation/reports/external/")})
        rejected = []
        for row in manifest["files"]:
            try:
                _destination(row["path"], ROOT)
            except ValueError:
                rejected.append(row["path"])
    return {"public_families": families, "rejected_public_members": rejected}


def main():
    if not ARCHIVE.is_file():
        _status({"status": "missing", "phase": "archive", "valid": 0,
                 "restored": 0, "missing": FILE_COUNT})
        return 1
    headers = audit_headers()
    try:
        result = restore()
    except Exception as error:
        messages = {"Public archive identity mismatch": "zip-sha",
                    "Existing public artifact differs; preserve it": "existing-collision",
                    "Public member identity mismatch": "member",
                    "Public archive member outside allowlist": "path"}
        phase = messages.get(str(error), "manifest-or-io")
        _status({"status": "failed", "phase": phase, "exception_type": type(error).__name__,
                 "valid": 0, "restored": 0, "missing": FILE_COUNT, **headers})
        raise
    result.update(headers)
    _status(result)
    return 0


def tests():
    """Export only synthetic test IDs and exception types, never tracebacks."""
    import unittest
    class PublicResult(unittest.TestResult):
        def __init__(self):
            super().__init__()
            self.public_errors = []
        def addError(self, test, err):
            self.errors.append((test, ""))
            self.public_errors.append({"id": test.id(), "exception_type": err[0].__name__})
        def addFailure(self, test, err):
            self.failures.append((test, ""))
            self.public_errors.append({"id": test.id(), "exception_type": err[0].__name__})
        def addSubTest(self, test, subtest, err):
            if err is not None:
                self.addError(test, err)
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_submission_portability.py")
    result = PublicResult()
    suite.run(result)
    report = STATUS.with_name("submission-portability-tests.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"status": "valid" if result.wasSuccessful() else "failed",
                                 "tests": result.testsRun, "failures": result.public_errors}, indent=2) + "\n")
    return 0 if result.wasSuccessful() else 1
