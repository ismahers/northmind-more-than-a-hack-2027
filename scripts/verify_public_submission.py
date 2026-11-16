"""Verify anonymous public access; no credentials or target data are sent."""
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://api.github.com/repos/ismahers/northmind-submission-2027"
MODEL = "https://github.com/ismahers/northmind-submission-2027/releases/download/submission-v1/northmind-public-experiments-2026-10-28.zip"
EXPECTED = "f09c894ad962364a2a16357749d9de304552532db58bb047f6fd2f9a1ca25162"


def fetch(url):
    return urlopen(Request(url, headers={"User-Agent": "Northmind-access-verification"}), timeout=90)


def main():
    record = {"repository_public": False, "anonymous_source_access": False,
              "anonymous_model_access": False, "model_sha256_valid": False}
    try:
        with fetch(REPO) as response:
            repository = json.load(response)
        record["repository_public"] = repository["private"] is False
        with fetch("https://raw.githubusercontent.com/ismahers/northmind-submission-2027/main/README.md") as response:
            record["anonymous_source_access"] = response.status == 200 and len(response.read()) > 0
        h = hashlib.sha256()
        with fetch(MODEL) as response:
            record["anonymous_model_access"] = response.status == 200
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                h.update(chunk)
        record["model_sha256_valid"] = h.hexdigest() == EXPECTED
    except Exception as error:
        record["exception_type"] = type(error).__name__
    record["verified"] = all(record.get(key) is True for key in
                             ("repository_public", "anonymous_source_access", "anonymous_model_access", "model_sha256_valid"))
    destination = ROOT / "docs/public-results/2026-11-16/submission-public-access.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return 0 if record["verified"] else 1
