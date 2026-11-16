"""Export package versions only, with no data or environment variable access."""
import importlib.metadata
import json
from pathlib import Path
import platform


def main():
    versions = {name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "torch", "python-dateutil", "six", "numba", "llvmlite", "certifi")}
    destination = Path(__file__).resolve().parents[1] / "docs/public-results/2026-11-16/submission-environment.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"python": platform.python_version(), "packages": versions}, indent=2) + "\n", encoding="utf-8")
    return 0
