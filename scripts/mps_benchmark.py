"""Reviewed entry point. Run only through scripts.run_local."""
import argparse
import json
import os
import platform
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "evaluation/reports/external/mps-benchmark-v1"


def environment():
    import numpy as np
    import torch
    from possession.external_data.schema import file_hash
    def command(args):
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"python": platform.python_version(), "executable": sys.executable,
            "torch": str(torch.__version__), "numpy": np.__version__,
            "macos": platform.mac_ver()[0], "machine": platform.machine(),
            "chip": command(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]),
            "physical_memory_bytes": command(["/usr/sbin/sysctl", "-n", "hw.memsize"]),
            "power": command(["/usr/bin/pmset", "-g", "batt"]),
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "public_manifest_available": (ROOT / "evaluation/reports/external/folds/idsse-only.json").is_file(),
            "mps_environment": {k: v for k, v in os.environ.items()
                                if k.startswith("PYTORCH_MPS_") or k == "PYTORCH_ENABLE_MPS_FALLBACK"},
            "git_head": command(["git", "rev-parse", "HEAD"]),
            "protected_hashes": {str(p.relative_to(ROOT)): file_hash(p)
                                  for d in ("starter", "evaluation", "possession/external_data", "possession/learning",
                                            "possession/night", "possession/phase5", "possession/phase6",
                                            "possession/phase7", "possession/phase8", "possession/temporal_probe")
                                  for p in sorted((ROOT / d).glob("*.py"))} |
                                {"configs/phase8_protocol.json": file_hash(ROOT / "configs/phase8_protocol.json")}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("environment", "compatibility", "diagnose", "benchmark", "report", "all"), default="all")
    parser.add_argument("--run", default="m3-first")
    parser.add_argument("--config", choices=("configs/mps_benchmark_v1.json", "configs/mps_benchmark_v2.json"), default="configs/mps_benchmark_v1.json")
    args = parser.parse_args()
    if not args.run or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.run):
        raise ValueError("Invalid isolated run name")
    folder = DEST / args.run
    folder.mkdir(parents=True, exist_ok=True)
    if args.stage == "environment":
        path = folder / "environment.json"
        if path.exists():
            raise ValueError("Use a new run name; environment already recorded")
        path.write_text(json.dumps(environment(), indent=2))
        return 0
    if args.stage == "diagnose":
        from possession.mps_benchmark.diagnosis import diagnose
        return diagnose(folder, environment())
    from possession.mps_benchmark.experiment import run
    return run(folder, args.stage, environment, args.config)
