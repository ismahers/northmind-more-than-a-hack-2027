"""Install a separate compatible environment in ignored local reports."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "evaluation/reports"


def main():
    uv = REPORTS / "tooling/bin/uv.exe"
    if not uv.is_file():
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                        "--target", str(REPORTS / "tooling"), "uv"], cwd=ROOT, check=True)
    env = dict(os.environ, UV_PYTHON_INSTALL_DIR=str(REPORTS / "python"),
               UV_CACHE_DIR=str(REPORTS / "uv-cache"))
    dest = REPORTS / "temporal-venv"
    python = dest / "Scripts/python.exe"
    if not python.is_file():
        subprocess.run([str(uv), "venv", "--python", "3.13", "--managed-python", str(dest)],
                       cwd=ROOT, env=env, check=True)
    subprocess.run([str(uv), "pip", "install", "--python", str(python),
                    "--index-url", "https://download.pytorch.org/whl/cpu", "torch==2.10.0"],
                   cwd=ROOT, env=env, check=True)
    subprocess.run([str(uv), "pip", "install", "--python", str(python),
                    "-r", "requirements-night.txt"], cwd=ROOT, env=env, check=True)
    return 0
