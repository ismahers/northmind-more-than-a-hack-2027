"""Public aggregate report; selections stay fixed, no retrospective test promotion."""
import csv
import numpy as np
import torch

from possession.external_data.schema import file_hash
from possession.night.common import atomic_json

from .common import ROOT, DEST, register, read_json
from .experiment import freeze


def verify():
    registration, _ = register()
    config = registration["protocol"]
    digest = file_hash(DEST / "registration.json")
    freeze(config, digest)
    checked = []

    def check(condition, name):
        checked.append({"check": name, "passed": bool(condition)})

    for name, expected in registration["sources"].items():
        check(file_hash(ROOT / name) == expected, "source:"+name)
    for name, expected in registration["references"].items():
        check(file_hash(ROOT / name) == expected, "reference:"+name)
    rows = read_json(DEST / "all_rows.json")
    check(len(rows) == 21*len(config["pilots"]), "complete evaluation rows")
    seal = (DEST / "selection-seal.json").stat().st_mtime_ns
    for fold in config["pilots"]:
        selected = read_json(DEST / f"selections/fold-{fold}.json")
        check(not selected["test_used"], f"fold-{fold} own selection without test")
        check(selected["registration_sha256"] == digest, f"fold-{fold} registration")
        rng, steps = [], []
        for arm in config["arms"]:
            folder = DEST / f"training/fold-{fold}" / arm
            complete = read_json(folder / "complete.json")
            check(len(complete["history"]) == config["epochs"], f"fold-{fold}/{arm} epochs")
            check(file_hash(folder / "last.pt") == selected["checkpoints"][arm], f"fold-{fold}/{arm} frozen checkpoint")
            state = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
            check(state["registration_sha256"] == digest, f"fold-{fold}/{arm} checkpoint provenance")
            rng.append(complete["loader_rng"])
            steps.append([epoch["steps"] for epoch in complete["history"]])
        check(all(value == rng[0] for value in rng), f"fold-{fold} identical sampler RNG")
        check(all(value == steps[0] for value in steps), f"fold-{fold} identical optimizer steps")
        key = selected["teacher"]["split"]["test"].replace(":", "_")
        for scenario in config["scenarios"]:
            path = DEST / f"embeddings/fold-{fold}" / key / (scenario+".npz")
            check(path.stat().st_mtime_ns >= seal, f"fold-{fold}/{scenario} test exported after selection seal")
        for row in (r for r in rows if r["fold"] == fold and r["system"] == "proposal" and r["strength"] == 0.):
            base = next(r for r in rows if r["fold"] == fold and r["arm"] == row["arm"]
                        and r["scenario"] == row["scenario"] and r["system"] == "hmm")
            check(base["metrics"] == row["metrics"], f"fold-{fold}/{row['arm']}/{row['scenario']} zero bonus equals HMM")
    for path in sorted((DEST / "embeddings").rglob("*.provenance.json")):
        stamp = read_json(path)
        actual = path.with_name(path.name.replace(".provenance.json", ".npz"))
        check(file_hash(actual) == stamp["sha256"], "export integrity:"+str(actual.relative_to(DEST)))
    passed = all(item["passed"] for item in checked)
    atomic_json(DEST / "verification.json", {"passed": passed, "checks": checked})
    if not passed:
        raise ValueError("Temporal probe integrity checks failed")
    return rows, registration


def report():
    rows, registration = verify()
    config = registration["protocol"]
    summary = {}
    systems = [("teacher", "hmm")] + [(arm, system) for arm in config["arms"]
               for system in (("hmm", "proposal") if arm.endswith("intervals") else ("hmm",))]
    for arm, system in systems:
        summary[arm+"/"+system] = {name: float(np.mean([r["metrics"]["spellF1"] for r in rows
                    if r["arm"] == arm and r["system"] == system and r["scenario"] == name])) for name in config["scenarios"]}
    selected_rows = []
    gates = {}
    for fold in config["pilots"]:
        own = read_json(DEST / f"selections/fold-{fold}.json")
        winner = own["winner"]
        system = "proposal" if winner["strength"] else "hmm"
        selected_rows.extend([row for row in rows if row["fold"] == fold and row["arm"] == winner["arm"] and row["system"] == system])
        gates[str(fold)] = {"winner": {key: winner[key] for key in ("arm", "strength", "objective")},
                           "own_fold_promotion": own["own_fold_promotion"]}
    summary["own_validation_selected"] = {name: float(np.mean([r["metrics"]["spellF1"] for r in selected_rows if r["scenario"] == name])) for name in config["scenarios"]}
    contrasts = {}
    for a, b in (("multiscale_classification/hmm", "local_classification/hmm"),
                 ("local_intervals/hmm", "local_classification/hmm"),
                 ("multiscale_intervals/hmm", "multiscale_classification/hmm"),
                 ("multiscale_intervals/proposal", "multiscale_intervals/hmm")):
        contrasts[a+" minus "+b] = {name: summary[a][name]-summary[b][name] for name in config["scenarios"]}
    atomic_json(DEST / "summary.json", {"systems": summary, "paired_contrasts": contrasts,
                                        "own_selections": gates, "folds": config["pilots"],
                                        "limitations": "Two previously researched IDSSE matches. Fresh teacher reference; no historical Phase8 identity claim. Extra temporal capacity is not parameter matched. No automatic expansion or test-based promotion."})
    lines = ["# Frozen temporal probe — public pilot results", "",
             "Two exploratory pilots; fresh teacher reference, not exact historical Phase8 reproduction.", "",
             "| System | Clean spellF1 | Missing40/5s | Missing40/15s |", "|---|---:|---:|---:|"]
    for name, measured in summary.items():
        lines.append("| "+name+" | "+" | ".join(f"{measured[n]:.5f}" for n in config["scenarios"])+" |")
    lines += ["", "Selections and promotion gates use each fold's own validation only.",
              "Means describe two matches. They are not seven-fold results or target-domain scores.",
              "Temporal heads add capacity; the comparison does not isolate context from parameter count.",
              "Teacher alive/dead emissions stay frozen; joint decoding can still change hard D decisions.",
              "All integrity checks passed. See summary.json for paired contrasts and own-fold gates."]
    (DEST / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    with (DEST / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["fold", "arm", "scenario", "system", "strength", "spellF1", "short_recall", "long_recall"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: row[k] for k in ("fold", "arm", "scenario", "system", "strength")},
                             **{k: row["metrics"][k] for k in ("spellF1", "short_recall", "long_recall")}})
