"""Existence-only inventory. This task never opens match assets or ground truth."""

import json
from pathlib import Path
import os

from games import asset_paths


def main():
    inventory = {}
    for source in ("tactical", "broadcast"):
        inventory[source] = {name: {"expected_filename": Path(path).name, "available": Path(path).is_file()}
                             for name, path in asset_paths(source=source).items()}
    # Filename-only search for misplaced bundles; never read their contents.
    names={item["expected_filename"] for row in inventory.values() for item in row.values()}
    root=Path(__file__).resolve().parents[1]
    found=[]
    for start in (root,Path.home()/"Downloads"):
        if not start.is_dir():
            continue
        for current,dirs,files in os.walk(start):
            relative=Path(current).relative_to(start)
            dirs[:]=[d for d in dirs if d not in {".git",".venv","node_modules","external","reports"} and len(relative.parts)<5]
            found.extend(str(Path(current)/name) for name in files if name in names)
    output = Path(__file__).resolve().parents[1] / "evaluation/reports/phase6-target/availability.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"existence_only": True, "assets": inventory,"alternate_matches":found}, indent=2))
    return 0
