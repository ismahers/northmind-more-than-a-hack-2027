"""Cold install into a new isolated venv, capturing all child output at creation.

No target inputs, models, previous environments, captured logs or pip reports
are inspected. Public exports contain status and package names/versions only.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = ROOT / "evaluation/reports/final-clean-venv"
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
    assert np.array([1, 2]).sum().item() == 3
    assert pd.DataFrame({"value": [1, 2]}).sum().item() == 3
    assert torch.tensor([1., 2.]).sum().item() == 3
    direct = {name: metadata.version(name) for name in
              ("numpy", "pandas", "torch", "python-dateutil", "six")}
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
                   "checks": ["numpy-synthetic-sum", "pandas-synthetic-sum", "torch-cpu-synthetic-sum"]})
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
        code = _run([sys.executable, "-m", "venv", str(ENVIRONMENT)], "install-create.log", 180)
        if code:
            _write({"status": "failed", "phase": phase, "returncode": code, "cold_install": True})
            return code
        python = ENVIRONMENT / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        phase = "install"
        _write({"status": "running", "phase": phase, "cold_install": True})
        code = _run([str(python), "-m", "pip", "install", "--disable-pip-version-check",
                     "--no-input", "-r", str(ROOT / "requirements-submission.txt")],
                    "install-pip.log", 900)
        if code:
            _write({"status": "failed", "phase": phase, "returncode": code, "cold_install": True})
            return code
        phase = "imports"
        _write({"status": "running", "phase": phase, "cold_install": True})
        return _run([str(python), "-c", IMPORT_CHECK, str(STATUS)], "install-imports.log", 180)
    except Exception as error:
        _write({"status": "failed", "phase": phase, "returncode": 1, "cold_install": True,
                "exception_type": type(error).__name__})
        return 1
