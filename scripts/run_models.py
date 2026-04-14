#!/usr/bin/env python3
"""Run models on benchmark items.

This is a scaffold script. By default it creates placeholder outputs in dry-run mode.
Replace `generate_output` with your actual model/API call.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def read_items(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def generate_output(item: Dict[str, str], model_name: str, dry_run: bool = True) -> str:
    if dry_run:
        return (
            f"[DRY RUN] Model={model_name}; item={item['item_id']}; "
            f"response_language={item['response_language']}"
        )
    raise NotImplementedError(
        "Replace generate_output() with your actual model or API call."
    )


def append_results(out_path: Path, rows: List[Dict[str, str]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = out_path.exists()
    fieldnames = [
        "model",
        "item_id",
        "output_text",
        "TaskCorrect",
        "LangCorrect",
        "CodeSwitch",
        "JointSuccess",
        "evaluator_notes",
    ]
    with out_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists or out_path.stat().st_size == 0:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True, help="Path to pilot_items.csv")
    parser.add_argument("--model", required=True, help="Model name label")
    parser.add_argument("--out", required=True, help="Path to output results CSV")
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Disable dry-run mode and use the real generation function",
    )
    args = parser.parse_args()

    items = read_items(Path(args.items))
    rows: List[Dict[str, str]] = []
    for item in items:
        output = generate_output(item, args.model, dry_run=not args.no_dry_run)
        rows.append(
            {
                "model": args.model,
                "item_id": item["item_id"],
                "output_text": output,
                "TaskCorrect": "",
                "LangCorrect": "",
                "CodeSwitch": "",
                "JointSuccess": "",
                "evaluator_notes": "",
            }
        )

    append_results(Path(args.out), rows)
    print(f"Wrote {len(rows)} outputs for model '{args.model}' to {args.out}")


if __name__ == "__main__":
    main()
