#!/usr/bin/env python3
"""Summarize pilot results by model and by model x item family.

Requires outputs/pilot_results.csv and data/pilot_items.csv.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: str):
    try:
        return float(value)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True, help="Path to outputs CSV")
    parser.add_argument(
        "--items",
        default="data/pilot_items.csv",
        help="Path to pilot items CSV for task/regime metadata",
    )
    args = parser.parse_args()

    results = read_csv(Path(args.infile))
    items = read_csv(Path(args.items))
    item_meta = {row["item_id"]: row for row in items}

    by_model: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    by_model_regime: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for row in results:
        model = row["model"]
        item_id = row["item_id"]
        regime = item_meta.get(item_id, {}).get("regime", "UNKNOWN")

        for metric in ["TaskCorrect", "LangCorrect", "CodeSwitch", "JointSuccess"]:
            val = safe_float(row.get(metric, ""))
            if val is not None:
                by_model[model][metric].append(val)
                by_model_regime[(model, regime)][metric].append(val)

    print("=== Summary by model ===")
    for model, metrics in by_model.items():
        print(f"\nModel: {model}")
        for metric in ["TaskCorrect", "LangCorrect", "CodeSwitch", "JointSuccess"]:
            vals = metrics.get(metric, [])
            if vals:
                print(f"  {metric}: {mean(vals):.3f} (n={len(vals)})")
            else:
                print(f"  {metric}: N/A")

    print("\n=== Summary by model x regime ===")
    for (model, regime), metrics in sorted(by_model_regime.items()):
        print(f"\nModel: {model} | Regime: {regime}")
        for metric in ["TaskCorrect", "LangCorrect", "CodeSwitch", "JointSuccess"]:
            vals = metrics.get(metric, [])
            if vals:
                print(f"  {metric}: {mean(vals):.3f} (n={len(vals)})")
            else:
                print(f"  {metric}: N/A")


if __name__ == "__main__":
    main()
