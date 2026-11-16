"""Check only explicit public experiment prerequisites; never inspect target assets."""
import importlib.util
import json
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "evaluation/reports/external"


def main():
    packages = {name: importlib.util.find_spec(name) is not None
                for name in ("numpy", "pandas", "torch", "numba")}
    fresh = ["folds/idsse-only.json", "temporal-teachers-v1/registration.json"]
    fresh += [f"temporal-teachers-v1/selections/fold-{i}.json" for i in (0, 6)]
    historical = ["folds/idsse-only.json", "phase8-v1/registration.json"]
    historical += [f"phase8-v1/selections/IDSSE/fold-{i}/seed-{seed}/set_transformer.json"
                   for i in (0, 6) for seed in (2027, 2028)]
    available = {name: (REPORTS / name).is_file() for name in fresh+historical}
    result = {"python": platform.python_version(), "packages": packages,
              "public_prerequisites": available,
              "ready": all(packages.values()) and all(available[name] for name in fresh),
              "historical_phase8_ready": all(available[name] for name in historical),
              "scope": "Explicit public paths only; no target asset checks or log reads."}
    dest = REPORTS / "temporal-probe-readiness.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0
