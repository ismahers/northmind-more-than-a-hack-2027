"""Record public experiment environment and available CUDA memory, without opening match assets."""

import platform
import os
import shutil

import certifi

import numpy as np
import pandas as pd
import torch

from scripts.external_data import REPORTS, ROOT, write_json


def main():
    report = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
              "torch": str(torch.__version__), "certifi": certifi.__version__,
              "cuda_available": torch.cuda.is_available(), "logical_cpus": os.cpu_count(),
              "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
              "platform": platform.platform(), "machine": platform.machine(),
              "disk_free_bytes": shutil.disk_usage(ROOT).free}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        report.update(gpu=torch.cuda.get_device_name(), free_bytes=free, total_bytes=total)
    write_json(REPORTS / "environment.json", report)
    return 0
