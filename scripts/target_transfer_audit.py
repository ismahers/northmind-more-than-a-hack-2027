"""Read-only audit of public sealed transfer inputs; never imports target loaders."""
import json
from pathlib import Path
import platform

import numpy as np
import torch

from possession.external_data.schema import file_hash
from possession import st_tcn_target as transfer

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evaluation/reports/external/target-transfer-v2"


def main():
    registration_path = transfer.FUSION_DIR / "registration.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    checks = []
    for kind in ("sources", "references"):
        for name, expected in registration[kind].items():
            path = (ROOT / name).resolve()
            allowed = ((ROOT / "possession").resolve(), (ROOT / "scripts").resolve(),
                       (ROOT / "evaluation/reports/external").resolve())
            if not (any(path.is_relative_to(base) for base in allowed)
                    or path == (ROOT / "evaluation/possession_metrics.py").resolve()):
                raise ValueError("Public audit path outside explicit allowlist")
            actual = file_hash(path)
            status = "exact" if actual == expected else "changed"
            if status == "changed" and kind == "sources" and transfer._registered_source_matches(path, expected):
                status = "text_format_only"
            checks.append({"kind": kind, "path": name, "status": status})
    report = {"checks": checks, "registered_environment": registration["environment"],
              "current_environment": {"python": platform.python_version(),
                  "numpy": np.__version__, "torch": torch.__version__},
              "changed": [c for c in checks if c["status"] == "changed"],
              "text_format_only": [c["path"] for c in checks if c["status"] == "text_format_only"],
              "passed": all(c["status"] != "changed" for c in checks),
              "target_assets_accessed": False}
    if report["passed"]:
        transfer._transfer_spec()  # Also require both sealed selections and all members.
        report["selection_seal_verified"] = True
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "public-integrity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
