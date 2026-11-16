"""Export aggregate public compute progress only; no console-log or target reads."""
import json
from pathlib import Path
import torch


def main():
    root = Path(__file__).resolve().parents[1] / "evaluation/reports/external"
    items = []
    for fold in (0, 6):
        for seed in (2027, 2028):
            for stage, total in (("base", 12), ("binary", 8)):
                path = root / f"temporal-teachers-v1/training/fold-{fold}/seed-{seed}/{stage}/last.pt"
                if not path.is_file():
                    continue
                state = torch.load(path, map_location="cpu", weights_only=False)
                history = state["history"]
                items.append({"fold": fold, "seed": seed, "stage": stage, "epochs": len(history),
                              "planned_epochs": total,
                              "seconds": sum(epoch["seconds"] for epoch in history)})
    result = {"public_only": True, "teacher_training": items,
              "completed_teacher_epochs": sum(item["epochs"] for item in items), "planned_teacher_epochs": 80}
    heads = []
    for fold in (0, 6):
        for arm in ("local_classification", "local_intervals", "multiscale_classification", "multiscale_intervals"):
            path = root / f"temporal-probe-v1/training/fold-{fold}/{arm}/last.pt"
            if path.is_file():
                state = torch.load(path, map_location="cpu", weights_only=False)
                heads.append({"fold": fold, "arm": arm, "epochs": len(state["history"]),
                              "seconds": sum(epoch["seconds"] for epoch in state["history"])})
    result.update(head_training=heads, completed_head_epochs=sum(item["epochs"] for item in heads),
                  planned_head_epochs=64)
    (root / "temporal-probe-status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0
