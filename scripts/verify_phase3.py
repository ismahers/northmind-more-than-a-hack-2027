"""Cross-artifact verification of the completed public-external study only."""

import json
from pathlib import Path
import platform
import sys

import llvmlite
import numba
import numpy as np
import torch

from possession.external_data.schema import file_hash
from possession.learning.metrics import SCORER_PATH
from possession.learning.training import implementation_hashes
from scripts.phase3 import DEST, directory, environment
from scripts.train_e0 import manifest
from possession.learning.study import save_json


def main():
    protocol, data = environment()
    combined, _ = manifest("idsse-skillcorner-phase3")
    from scripts.audit_external_identities import main as audit_identities
    audit_identities()
    identities = json.loads((DEST / "source_identities.json").read_text())
    assert identities["crossprovider_match_overlap_excluded_by_date"]
    assert len(identities["idsse"]) == 7 and len(identities["skillcorner"]) == 20
    all_rows = json.loads((DEST / "all_rows.json").read_text())
    assert len(all_rows) == 910
    scorer_hash = file_hash(SCORER_PATH)
    frozen_implementation = implementation_hashes()
    checked_probabilities = 0
    for i in range(7):
        fold = data["folds"][i]
        assert combined["folds"][i]["test"] == fold["test"]
        assert combined["folds"][i]["validation"] == fold["validation"]
        groups = [set(fold[k]) for k in ("train", "validation", "test")]
        assert all(not groups[a] & groups[b] for a in range(3) for b in range(a))
        selection = json.loads((DEST / "selections" / (f"fold-{i}.json")).read_text())
        assert selection["outer_test_used"] is False
        best_objective = max(trial["objective"] for result in selection["conditions"].values()
                             for trial in result["trials"])
        assert selection["winner"]["objective"] == best_objective
        for result in selection["conditions"].values():
            for trial in result["trials"]:
                scores = trial["scores"]
                assert trial["objective"] == .5 * scores["clean"] + .25 * scores["missing40_5s"] + .25 * scores["missing40_15s"]
        reference_updates = json.loads((directory(i, "A_clean") / "training.json").read_text())["updates_per_epoch"]
        for condition in protocol["conditions"]:
            training = json.loads((directory(i, condition) / "training.json").read_text())
            assert training["validation_key"] == fold["validation"][0]
            assert not set(training["train_keys"]) & (groups[1] | groups[2])
            assert sum(k.startswith("idsse:") for k in training["train_keys"]) == 5
            assert sum(k.startswith("skillcorner:") for k in training["train_keys"]) == (20 if condition.startswith("B") else 0)
            assert training["fingerprint"]["protocol"] == protocol
            assert training["fingerprint"]["implementation"] == frozen_implementation
            assert training["updates_per_epoch"] == reference_updates
            assert training["chosen"]["validation_spellF1"] == max(c["validation_spellF1"] for c in training["caps"])
            for epoch in training["history"]:
                assert np.isfinite(epoch["train"]["loss"]) and np.isfinite(epoch["validation"]["loss"])
                assert np.isfinite(epoch["train"]["gradient_norm_max"])
                assert epoch["train"]["optimizer_steps"] == reference_updates
            for cap in training["caps"]:
                losses = [e["validation"]["loss"] for e in training["history"][:cap["epochs_executed"]]]
                assert cap["validation_loss"] == min(losses)
            result = selection["conditions"][condition]
            for family, chosen in result["by_family"].items():
                assert chosen["objective"] == max(t["objective"] for t in result["trials"]
                                                  if t["specification"]["kind"] == family)
        baseline = next(iter(json.loads((DEST / "baseline" / f"fold-{i}" / "test/metrics.json").read_text()).values()))
        cap2 = json.loads((DEST / "durations" / f"fold-{i}" / "2.json").read_text())
        assert baseline["spell_f1_mean"] == cap2["spell_f1_mean"]
        assert baseline["macro_f1"] == cap2["macro_f1"]
        assert cap2["evaluation_contract"]["scorer_sha256"] == scorer_hash
        selected = [r for r in all_rows if r["fold"] == i and r["validation_selected_pipeline"]]
        assert len(selected) == 21
        for r in selected:
            assert r["condition"] == selection["winner"]["condition"]
            assert r["decoder"] == selection["winner"]["family"]
            assert r["gt_spells"] == baseline["n_gt_spells"]
        original = [r for r in all_rows if r["fold"] == i and r["seed"] == 2027]
        assert len(original) == 116
        for row in original:
            assert row["gt_spells"] == baseline["n_gt_spells"]
        for path in (DEST / "test").glob(f"seed-*/fold-{i}/*/*/probabilities.npz"):
            with np.load(path, allow_pickle=False) as archive:
                p, times, runs = archive["probabilities"], archive["time"], archive["run"]
                assert np.isfinite(p).all() and (p >= 0).all()
                np.testing.assert_allclose(p.sum(-1), 1, atol=1e-6)
                assert len(p) == len(times) == len(runs)
                assert all(np.all(np.diff(times[runs == run]) > 0) for run in np.unique(runs))
                checked_probabilities += 1
        cap = json.loads((directory(i, "A_clean") / "training.json").read_text())["chosen"]["budget"]
        with np.load(DEST / "durations" / f"fold-{i}" / f"{cap}-probabilities.npz") as duration_p:
            with np.load(DEST / "test/seed-2027" / f"fold-{i}" / "A_clean/clean/probabilities.npz") as clean_p:
                np.testing.assert_array_equal(duration_p["probabilities"], clean_p["probabilities"])
    assert checked_probabilities == 294  # 4*7*7 original conditions + 2*7*7 additional seeds.
    save_json(DEST / "verification.json", {"status": "OK", "rows": len(all_rows),
        "probability_archives_checked": checked_probabilities, "folds": 7,
        "scorer_sha256": scorer_hash, "baseline_equals_cap2": True,
        "protocol_and_partitions_consistent": True, "validation_choices_recomputed": True,
        "source": "public external artifacts only",
        "runtime": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                    "torch": torch.__version__, "numba": numba.__version__, "llvmlite": llvmlite.__version__,
                    "torch_threads": torch.get_num_threads(), "device": "cpu"},
        "source_hashes": {**frozen_implementation, **{str(p): file_hash(p) for p in (
            Path("scripts/phase3.py"), Path("scripts/phase3_report.py"), Path("scripts/verify_phase3.py"),
            Path("scripts/audit_external_identities.py"),
            Path("configs/phase3_protocol.json"), Path("requirements-phase3.txt"))}}})
    return 0
