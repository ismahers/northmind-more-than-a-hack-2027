#!/usr/bin/env python3
"""
Sweep switch_cost values for the sequence decoder and compare spellF1.

Usage:
    .venv/bin/python scripts/sweep_switch_cost.py
    .venv/bin/python scripts/sweep_switch_cost.py --costs 10 15 20 25 30 40 50
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--costs", type=float, nargs="+",
        default=[10, 15, 20, 25, 30, 40, 50],
        help="switch_cost values to test"
    )
    args = parser.parse_args()

    print(f"Testing switch_cost values: {args.costs}")
    print("=" * 80)

    # We need to monkey-patch the SequenceConfig.switch_cost before each run
    # Since the module is imported by run_experiments, we'll create a temporary
    # wrapper that sets the value before importing.

    results = []

    for cost in args.costs:
        print(f"\n>>> Testing switch_cost = {cost}")

        # Create a temporary config file approach: we'll modify the module attribute
        # by running a subprocess that sets it before importing run_experiments
        wrapper_code = f"""
import sys
sys.path.insert(0, '{ROOT}')
from possession.sequence_decoding import SequenceConfig
SequenceConfig.switch_cost = {cost}
from scripts.run_experiments import run
run(sequence=True)
"""

        result = subprocess.run(
            [sys.executable, "-c", wrapper_code],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            print(f"  FAILED: {result.stderr[:200]}")
            results.append({"cost": cost, "error": result.stderr[:500]})
            continue

        # Read the comparison.json
        comp_path = ROOT / "evaluation" / "reports" / "sequence-model-v1" / "comparison.json"
        if not comp_path.exists():
            print(f"  No comparison.json found")
            results.append({"cost": cost, "error": "No comparison.json"})
            continue

        with open(comp_path) as f:
            comp = json.load(f)

        for name in ("northmind_sequence", "northmind_sequence_bridged"):
            d = comp["candidates"][name]
            print(f"  {name:30s} | spellF1={d['spell_f1_mean']:.4f} "
                  f"| delta_vs_ref={d['spell_delta_vs_reference']:+.4f} "
                  f"| delta_vs_s50={d['spell_delta_vs_smooth50']:+.4f} "
                  f"| n_spells={d['n_pred_spells']}")
            results.append({
                "cost": cost,
                "variant": name,
                "spell_f1": d["spell_f1_mean"],
                "delta_vs_ref": d["spell_delta_vs_reference"],
                "delta_vs_smooth50": d["spell_delta_vs_smooth50"],
                "n_pred_spells": d["n_pred_spells"],
            })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for variant in ("northmind_sequence", "northmind_sequence_bridged"):
        print(f"\n{variant}:")
        variant_results = [r for r in results if r.get("variant") == variant]
        for r in variant_results:
            if "error" in r:
                print(f"  cost={r['cost']:>3} | ERROR: {r['error'][:80]}")
            else:
                print(f"  cost={r['cost']:>3} | spellF1={r['spell_f1']:.4f} "
                      f"| vs_ref={r['delta_vs_ref']:+.4f} | vs_s50={r['delta_vs_smooth50']:+.4f} "
                      f"| spells={r['n_pred_spells']}")

    # Best
    for variant in ("northmind_sequence", "northmind_sequence_bridged"):
        variant_results = [r for r in results if r.get("variant") == variant and "error" not in r]
        if variant_results:
            best = max(variant_results, key=lambda x: x["spell_f1"])
            print(f"\n  BEST {variant}: cost={best['cost']} | spellF1={best['spell_f1']:.4f}")


if __name__ == "__main__":
    main()