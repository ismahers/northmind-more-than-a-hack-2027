"""Bounded end-to-end smoke run on synthetic public-format fixtures only."""

from dataclasses import replace
from pathlib import Path
import tempfile

import torch

from possession.external_data.audit import audit
from possession.external_data.idsse import IDSSE_IDS, load_idsse
from possession.external_data.synthetic import write_idsse_fixture
from possession.learning.protocol import lomo
from possession.learning.training import TrainConfig, evaluate_checkpoint, train
from scripts.external_data import REPORTS, write_json


def main():
    torch.set_num_threads(1)
    root = REPORTS / "synthetic-smoke"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fixtures-", dir=root) as temporary:
        directory = Path(temporary)
        paths, audits = [], []
        for mid in IDSSE_IDS:
            match = load_idsse(*write_idsse_fixture(directory / mid, "DFL-MAT-" + mid))
            path = directory / (mid + ".npz")
            match.save(path)
            paths.append(path)
            audits.append(audit(match))
        fold = lomo(["idsse:DFL-MAT-" + mid for mid in IDSSE_IDS])[0]
        config = TrainConfig(epochs=1, core_frames=32, batch_size=2, spatial_dim=8,
                             channels=8, dilations=(1, 2), device="cpu")
        output = root / "checkpoint"
        train(paths, fold, config, output)
        train(paths, fold, replace(config, epochs=2), output, output / "last.pt")
        validation = evaluate_checkpoint(paths, output / "best.pt", "validation", root / "validation")
        # This test score verifies plumbing on synthetic fixtures; it is not competitive evidence.
        test = evaluate_checkpoint(paths, output / "best.pt", "test", root / "test")
        write_json(root / "smoke.json", {"synthetic_only": True, "epochs": 2,
                                        "audits": audits, "validation": validation, "test": test})
    return 0
