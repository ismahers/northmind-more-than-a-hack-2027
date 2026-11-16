"""Plot fixed-HMM public pilot contrasts, without selecting systems from test."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    dest = Path(__file__).resolve().parents[1] / "evaluation/reports/external/temporal-probe-v1"
    verification = json.loads((dest / "verification.json").read_text(encoding="utf-8"))
    if not verification["passed"]:
        raise ValueError("Verify public results before plotting")
    rows = json.loads((dest / "all_rows.json").read_text(encoding="utf-8"))
    arms = ["teacher", "local_classification", "local_intervals",
            "multiscale_classification", "multiscale_intervals"]
    labels = ["Teacher", "Local\nclassification", "Local\n+ intervals",
              "Multiscale\nclassification", "Multiscale\n+ intervals"]
    scenarios = ["clean", "missing40_5s", "missing40_15s"]
    colors = ["#2368a2", "#d88c22", "#64994b"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    x = np.arange(len(arms))
    for j, (scenario, color) in enumerate(zip(scenarios, colors)):
        values = [np.mean([r["metrics"]["spellF1"] for r in rows
                          if r["arm"] == arm and r["system"] == "hmm" and r["scenario"] == scenario]) for arm in arms]
        axes[0].bar(x+(j-1)*.24, values, width=.23, color=color, label=scenario)
    axes[0].set(title="Possession segmentation — fixed HMM", ylabel="spellF1", ylim=(0, 1), xticks=x, xticklabels=labels)
    axes[0].legend(frameon=False, fontsize=8)
    for fold, marker in ((0, "o"), (6, "s")):
        values = [next(r["metrics"]["short_recall"] for r in rows
                       if r["fold"] == fold and r["arm"] == arm and r["system"] == "hmm" and r["scenario"] == "clean") for arm in arms]
        axes[1].plot(x, values, marker=marker, label=f"fold {fold}")
    axes[1].set(title="Short possession recall — clean, IoU 0.5", ylabel="Recall below 3 seconds", ylim=(0, 1), xticks=x, xticklabels=labels)
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Two exploratory IDSSE pilots — newly trained frozen teachers", fontsize=13)
    folder = dest / "figures"
    folder.mkdir(exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(folder / ("paired_comparisons."+suffix), dpi=180)
    plt.close(fig)
    return 0
