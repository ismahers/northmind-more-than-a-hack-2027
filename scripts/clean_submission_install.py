"""Cold install into a new isolated venv, capturing all child output at creation.

No target inputs, models, previous environments, captured logs or pip reports
are inspected. Public exports contain status and package names/versions only.
"""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = ROOT / "evaluation/reports/final-clean-venv-v2"
if os.name == "nt":
    ENVIRONMENT = Path("\\\\?\\" + str(ENVIRONMENT.resolve()))
LOCAL = ROOT / "evaluation/reports/final-delivery"
STATUS = ROOT / "docs/public-results/2026-11-16/submission-clean-install.json"

IMPORT_CHECK = r'''
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import site
import sys

destination = Path(sys.argv[1])
record = {"status": "failed", "phase": "imports", "cold_install": True}
try:
    assert sys.prefix != sys.base_prefix and site.ENABLE_USER_SITE is False
    import numpy as np
    import pandas as pd
    import torch
    import numba
    import llvmlite
    import certifi
    assert np.array([1, 2]).sum().item() == 3
    assert pd.DataFrame({"value": [1, 2]}).sum().item() == 3
    assert torch.tensor([1., 2.]).sum().item() == 3
    direct = {name: metadata.version(name) for name in
              ("numpy", "pandas", "torch", "python-dateutil", "six", "numba", "llvmlite", "certifi")}
    full = {}
    for distribution in metadata.distributions():
        name, version = distribution.metadata["Name"], distribution.version
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
            raise ValueError("Invalid public distribution name")
        if not re.fullmatch(r"[A-Za-z0-9.+!_-]+", version):
            raise ValueError("Invalid public distribution version")
        full[name] = version
    record.update({"status": "valid", "phase": "complete", "returncode": 0,
                   "isolated_venv": True, "system_site_packages": False,
                   "python": sys.version.split()[0], "direct_packages": direct,
                   "all_packages": dict(sorted(full.items())),
                   "checks": ["numpy-synthetic-sum", "pandas-synthetic-sum", "torch-cpu-synthetic-sum",
                              "numba-import", "llvmlite-import", "certifi-import"]})
except Exception as error:
    record.update({"returncode": 1, "exception_type": type(error).__name__})
destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
raise SystemExit(record["returncode"])
'''


def _write(record):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def _run(command, log_name, timeout):
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME"):
        environment.pop(key, None)
    environment["PYTHONNOUSERSITE"] = "1"
    with (LOCAL / log_name).open("xb") as log:
        process = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=log, timeout=timeout, check=False,
                                 env=environment,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return process.returncode


def main():
    LOCAL.mkdir(parents=True, exist_ok=True)
    if ENVIRONMENT.exists():
        _write({"status": "failed", "phase": "environment-already-exists", "cold_install": False,
                "returncode": 2})
        return 2
    phase = "create"
    try:
        _write({"status": "running", "phase": phase, "cold_install": True})
        code = _run([sys.executable, "-m", "venv", str(ENVIRONMENT)], "install-create-v2.log", 180)
        if code:
            _write({"status": "failed", "phase": phase, "returncode": code, "cold_install": True})
            return code
        python = ENVIRONMENT / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        phase = "install"
        _write({"status": "running", "phase": phase, "cold_install": True})
        code = _run([str(python), "-m", "pip", "--isolated", "install", "--disable-pip-version-check",
                     "--index-url", "https://pypi.org/simple", "--no-input",
                     "-r", str(ROOT / "requirements-submission.txt")],
                    "install-pip-v2.log", 900)
        if code:
            _write({"status": "failed", "phase": phase, "returncode": code, "cold_install": True})
            return code
        phase = "imports"
        _write({"status": "running", "phase": phase, "cold_install": True})
        return _run([str(python), "-c", IMPORT_CHECK, str(STATUS)], "install-imports-v2.log", 180)
    except Exception as error:
        _write({"status": "failed", "phase": phase, "returncode": 1, "cold_install": True,
                "exception_type": type(error).__name__})
        return 1


def complete():
    """Complete only the independent v2 environment; leave the old runtime intact."""
    LOCAL.mkdir(parents=True, exist_ok=True)
    python = ENVIRONMENT / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file() or ENVIRONMENT.is_symlink():
        _write({"status": "failed", "phase": "independent-environment-missing", "cold_install": False,
                "returncode": 2})
        return 2
    phase = "complete-install"
    try:
        _write({"status": "running", "phase": phase, "cold_install": True})
        code = _run([str(python), "-m", "pip", "--isolated", "install", "--disable-pip-version-check",
                     "--index-url", "https://pypi.org/simple", "--no-input",
                     "-r", str(ROOT / "requirements-submission.txt")],
                    "install-complete-v2.log", 900)
        if code:
            _write({"status": "failed", "phase": phase, "returncode": code, "cold_install": True})
            return code
        phase = "complete-imports"
        _write({"status": "running", "phase": phase, "cold_install": True})
        return _run([str(python), "-c", IMPORT_CHECK, str(STATUS)], "install-complete-imports-v2.log", 180)
    except Exception as error:
        _write({"status": "failed", "phase": phase, "returncode": 1, "cold_install": True,
                "exception_type": type(error).__name__})
        return 1


def _complete_public_wheels(packages, wheel_suffix, log_tag):
    """Install exact public wheels with authenticated PyPI byte hashes."""
    phase = "public-wheel-download"
    try:
        python = ENVIRONMENT / "Scripts/python.exe"
        if os.name != "nt" or not python.is_file() or ENVIRONMENT.is_symlink():
            raise ValueError("Expected independent Windows environment")
        wheel_root = Path("\\\\?\\" + str((LOCAL / "public-install-wheels").resolve()))
        wheel_root.mkdir(parents=True, exist_ok=True)
        paths = []
        _write({"status": "running", "phase": phase, "cold_install": True})
        for name, version in packages:
            with urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=45) as response:
                record = json.load(response)
            candidates = [row for row in record["urls"] if not row.get("yanked", False)
                          and row["filename"].endswith(wheel_suffix)]
            if len(candidates) != 1:
                raise ValueError("Ambiguous public wheel")
            row = candidates[0]
            if (Path(row["filename"]).name != row["filename"]
                    or not row["filename"].startswith(f"{name}-{version}-")
                    or not row["url"].startswith("https://files.pythonhosted.org/")):
                raise ValueError("Unexpected public wheel source")
            path = wheel_root / row["filename"]
            part = path.with_suffix(".download")
            if path.exists():
                raise ValueError("Public wheel staging already exists")
            digest = hashlib.sha256()
            try:
                with urlopen(row["url"], timeout=120) as response, part.open("xb") as destination:
                    while chunk := response.read(1024 * 1024):
                        digest.update(chunk)
                        destination.write(chunk)
                if digest.hexdigest() != row["digests"]["sha256"] or part.stat().st_size != row["size"]:
                    raise ValueError("Public wheel digest mismatch")
                part.rename(path)
            finally:
                part.unlink(missing_ok=True)
            paths.append(str(path))
        phase = "public-wheel-install"
        _write({"status": "running", "phase": phase, "cold_install": True})
        code = _run([str(python), "-m", "pip", "--isolated", "install", "--no-index", "--no-deps",
                     "--disable-pip-version-check", *paths], f"install-{log_tag}-v2.log", 300)
        if code:
            _write({"status": "failed", "phase": phase, "returncode": code, "cold_install": True})
            return code
        phase = "dependency-check"
        code = _run([str(python), "-m", "pip", "--isolated", "check"], f"install-{log_tag}-dependency-check-v2.log", 60)
        if code:
            _write({"status": "failed", "phase": phase, "returncode": code, "cold_install": True})
            return code
        phase = "public-wheel-imports"
        _write({"status": "running", "phase": phase, "cold_install": True})
        return _run([str(python), "-c", IMPORT_CHECK, str(STATUS)], f"install-{log_tag}-imports-v2.log", 180)
    except Exception as error:
        _write({"status": "failed", "phase": phase, "returncode": 1, "cold_install": True,
                "exception_type": type(error).__name__})
        return 1


def complete_public_wheels():
    return _complete_public_wheels((("numba", "0.61.2"), ("llvmlite", "0.44.0")),
                                   "-cp313-cp313-win_amd64.whl", "public-wheels")


def complete_certifi_public_wheel():
    return _complete_public_wheels((("certifi", "2025.10.5"),), "-py3-none-any.whl", "certifi-public-wheel")
