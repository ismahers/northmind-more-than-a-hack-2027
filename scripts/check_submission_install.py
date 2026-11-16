"""Check public wheel metadata only; no installation, target I/O or log reading."""
import json
from pathlib import Path
import re
from urllib.parse import unquote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "docs/public-results/2026-11-16/submission-install-status.json"


def _fetch(url):
    with urlopen(Request(url, headers={"User-Agent": "Northmind-public-install-check/1"}), timeout=45) as response:
        return response.read().decode("utf-8")


def main():
    rows = []
    for line in (ROOT / "requirements-submission.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "--")):
            continue
        name, version = line.split("==")
        row = {"package": name, "version": version, "status": "failed", "phase": "public-metadata"}
        try:
            if name == "torch":
                source = "https://download.pytorch.org/whl/cpu/torch/"
                text = unquote(_fetch(source))
                filenames = re.findall(r"torch-[^\"<>/ ]+\.whl", text)
                filenames = [value for value in filenames if value.startswith("torch-" + version + "-")]
            else:
                source = f"https://pypi.org/pypi/{name}/{version}/json"
                metadata = json.loads(_fetch(source))
                filenames = [value["filename"] for value in metadata["urls"]
                             if value["packagetype"] == "bdist_wheel" and not value.get("yanked", False)]
            universal = any("-py2.py3-none-any.whl" in value or "-py3-none-any.whl" in value for value in filenames)
            windows = universal or any("-cp313-cp313-win_amd64.whl" in value for value in filenames)
            linux = universal or any("-cp313-cp313-manylinux" in value and "x86_64.whl" in value for value in filenames)
            row.update({"status": "valid" if windows else "missing-compatible-wheel",
                        "source": source, "python313_windows_x86_64_wheel": windows,
                        "python313_linux_x86_64_wheel": linux,
                        "public_wheel_count": len(set(filenames))})
        except Exception as error:
            row["exception_type"] = type(error).__name__
        rows.append(row)
    valid = bool(rows) and all(row["status"] == "valid" for row in rows)
    record = {"status": "valid" if valid else "failed", "phase": "public-wheel-metadata",
              "installed": False, "scope": "Public release metadata; no dependency resolution or clean installation performed",
              "packages": rows}
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return 0 if valid else 1
