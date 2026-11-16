"""Compare independent real-data E0 reruns without reading competition assets."""

import json

import numpy as np
import torch

from scripts.external_data import REPORTS, write_json


def main():
    roots = [REPORTS / "experiments" / run / "fold-0"
             for run in ("e0-idsse-cpu-first", "e0-idsse-cpu-first-repeat")]
    states = [torch.load(root / "last.pt", map_location="cpu", weights_only=False) for root in roots]
    if states[0]["provenance"] != states[1]["provenance"] or states[0]["config"] != states[1]["config"]:
        raise ValueError("Reproduction inputs/configuration differ")
    if states[0]["history"] != states[1]["history"]:
        raise ValueError("Nonidentical deterministic training history")
    for name in states[0]["model"]:
        torch.testing.assert_close(states[0]["model"][name], states[1]["model"][name], rtol=0, atol=0)
    arrays = [np.load(next((root / "test").glob("*_probabilities.npz")), allow_pickle=False) for root in roots]
    try:
        for name in ("probabilities", "native_probabilities", "native_frame", "native_period", "native_time"):
            np.testing.assert_array_equal(arrays[0][name], arrays[1][name])
    finally:
        for array in arrays:
            array.close()
    metrics = [next(iter(json.loads((root / "test/metrics.json").read_text(encoding="utf-8")).values())) for root in roots]
    for name in ("spell_f1_mean", "macro_f1", "spell_f1", "n_gt_spells", "n_pred_spells"):
        if metrics[0][name] != metrics[1][name]:
            raise ValueError("Reproduced metrics differ")
    write_json(REPORTS / "reproduction.json", {"source": "real public IDSSE", "fold": 0,
        "independent_training_runs": 2, "weights_exact": True, "probabilities_exact": True,
        "training_loss_history_exact": True, "spellF1": metrics[0]["spell_f1_mean"],
        "macroF1": metrics[0]["macro_f1"], "device": "cpu", "seed": states[0]["config"]["seed"]})
    return 0
