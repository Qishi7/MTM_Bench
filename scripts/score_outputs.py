#!/usr/bin/env python3
"""Compute JointSuccess from manual labels and report missing annotations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


FIELDS = [
    "model",
    "item_id",
    "output_text",
    "TaskCorrect",
    "LangCorrect",
    "CodeSwitch",
    "JointSuccess",
    "evaluator_notes",
]


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def safe_int(value: str):
    try:
        return int(value)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True)
    args = parser.parse_args()

    path = Path(args.infile)
    rows = read_rows(path)

    missing = 0
    updated = 0
    for row in rows:
        tc = safe_int(row.get("TaskCorrect", ""))
        lc = safe_int(row.get("LangCorrect", ""))
        if tc is None or lc is None:
            missing += 1
            continue
        row["JointSuccess"] = "1" if (tc == 2 and lc == 2) else "0"
        updated += 1

    write_rows(path, rows)
    print(f"Updated JointSuccess for {updated} rows.")
    if missing:
        print(f"Skipped {missing} rows due to missing TaskCorrect/LangCorrect labels.")


if __name__ == "__main__":
    main()
