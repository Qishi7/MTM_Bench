#!/usr/bin/env python3
"""
Analysis pipeline for the expanded triplet benchmark.

Design assumption (override via --items): each item has
    item_id, category, instruction_language, content_language, response_language
Full triplet coverage means 3^3 = 27 (L_instr, L_content, L_resp) combinations.
With 3 task categories x 30 base items x 27 triplets = 2430 instances per model.

The script produces one LONG-FORMAT csv (one row per slice) so downstream
pivot/plot logic stays simple. For backward compatibility we also keep the
wide-format "one row per model" csv as an optional second output.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence


METRICS = [
    "SemanticCorrect",
    "LangCorrect",
    "ConstraintCorrect",
    "CodeSwitchRatio",
    "JointSuccess",
]

LANGS = ("en", "es", "zh")


# --------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------
def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(x: Optional[float]) -> str:
    return "" if x is None else f"{x:.4f}"


def mean_metric(rows: Sequence[Dict], metric: str) -> Optional[float]:
    vals = [safe_float(r.get(metric, "")) for r in rows]
    vals = [v for v in vals if v is not None]
    return mean(vals) if vals else None


# --------------------------------------------------------------------
# Triplet labelling
# --------------------------------------------------------------------
def triplet_tag(instr: str, content: str, resp: str) -> str:
    """Compact label like (en,en,zh)."""
    return f"({instr},{content},{resp})"


def mismatch_class(instr: str, content: str, resp: str) -> str:
    """Structured mismatch taxonomy.

    With 3 languages and 3 slots, the 27 triplets fall into exactly 5 classes
    based on which pair(s) of slots agree:

        same            (3 triplets)  : instr = content = resp
        instr_only      (6)           : content = resp, instr differs
        resp_only       (6)           : instr = content, resp differs
        content_only    (6)           : instr = resp, content differs
        full_mismatch   (6)           : all three differ

    These are mutually exclusive and exhaustive (3 + 6*3 + 6 = 27).
    """
    if instr == content == resp:
        return "same"
    if content == resp and instr != content:
        return "instr_only"
    if instr == content and content != resp:
        return "resp_only"
    if instr == resp and instr != content:
        return "content_only"
    # the remaining case must have all three different
    return "full_mismatch"


def script_pair(content: str, resp: str) -> str:
    """Latin vs non-Latin script axis for the content->response direction."""
    latin = {"en", "es"}
    c = "latin" if content in latin else "nonlatin"
    r = "latin" if resp in latin else "nonlatin"
    return f"{c}->{r}"


# --------------------------------------------------------------------
# Core aggregation
# --------------------------------------------------------------------
def slice_metrics(rows: Sequence[Dict], metric_list: Sequence[str] = METRICS) -> Dict[str, Optional[float]]:
    n = len(rows)
    out = {"n": n}
    for m in metric_list:
        out[m] = mean_metric(rows, m)
    return out


def group_by(rows: Sequence[Dict], key: str) -> Dict[str, List[Dict]]:
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        buckets[r.get(key, "UNKNOWN")].append(r)
    return buckets


def emit_long_rows(
    model: str,
    group_name: str,
    bucket_label: str,
    rows: Sequence[Dict],
) -> List[Dict]:
    """One row per (model, slice_type, slice_value), wide across metrics."""
    agg = slice_metrics(rows)
    return [{
        "model": model,
        "slice_type": group_name,
        "slice_value": bucket_label,
        "n": agg["n"],
        **{m: fmt(agg[m]) for m in METRICS},
    }]


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate triplet-benchmark model outputs into long-format slices."
    )
    parser.add_argument("--infile", required=True,
                        help="Per-instance model outputs with item_id + metric columns.")
    parser.add_argument("--items", default="data/tri_main.csv",
                        help="Item metadata (item_id, category, *_language).")
    parser.add_argument("--long_out", required=True,
                        help="Long-format output CSV (one row per slice).")
    parser.add_argument("--wide_out", default=None,
                        help="Optional wide-format CSV (one row per model).")
    parser.add_argument("--categories", default="A,B,C",
                        help="Comma-separated list of task categories in the data.")
    args = parser.parse_args()

    infile = Path(args.infile)
    items = read_csv(Path(args.items))
    results = read_csv(infile)

    if not results:
        raise ValueError(f"No rows found in {infile}")

    item_meta = {row["item_id"]: row for row in items}
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    model = results[0].get("model", infile.stem).strip()

    # --- join results with item metadata and derive triplet labels ----
    merged: List[Dict] = []
    missing_meta = 0
    for row in results:
        item_id = row.get("item_id", "").strip()
        meta = item_meta.get(item_id)
        if meta is None:
            missing_meta += 1
            continue
        instr = meta.get("instruction_language", "UNKNOWN")
        content = meta.get("content_language", "UNKNOWN")
        resp = meta.get("response_language", "UNKNOWN")
        merged.append({
            **row,
            **meta,
            "triplet": triplet_tag(instr, content, resp),
            "mismatch_class": mismatch_class(instr, content, resp),
            "script_pair": script_pair(content, resp),
        })

    if missing_meta:
        print(f"[warn] {missing_meta} rows in {infile.name} had no matching item metadata")

    if not merged:
        raise ValueError("All rows failed to match item metadata; check --items path.")

    # ----------------------------------------------------------------
    # LONG-FORMAT output: one row per (model, slice_type, slice_value)
    # ----------------------------------------------------------------
    long_rows: List[Dict] = []

    # overall
    long_rows += emit_long_rows(model, "overall", "all", merged)

    # by category
    for cat, sub in group_by(merged, "category").items():
        long_rows += emit_long_rows(model, "category", cat, sub)

    # by single language axis
    for axis in ("instruction_language", "content_language", "response_language"):
        for val, sub in group_by(merged, axis).items():
            long_rows += emit_long_rows(model, axis, val, sub)

    # by full triplet (up to 27)
    for trip, sub in group_by(merged, "triplet").items():
        long_rows += emit_long_rows(model, "triplet", trip, sub)

    # by mismatch class
    for cls, sub in group_by(merged, "mismatch_class").items():
        long_rows += emit_long_rows(model, "mismatch_class", cls, sub)

    # by script pair (content -> response)
    for sp, sub in group_by(merged, "script_pair").items():
        long_rows += emit_long_rows(model, "script_pair", sp, sub)

    # interaction: category x response_language
    for cat in categories:
        for lang in LANGS:
            sub = [r for r in merged
                   if r.get("category") == cat and r.get("response_language") == lang]
            if sub:
                long_rows += emit_long_rows(
                    model, "category_x_response", f"{cat}|{lang}", sub
                )

    # interaction: category x triplet (small but informative for top/bottom triplets)
    for cat in categories:
        for trip, sub in group_by([r for r in merged if r.get("category") == cat],
                                  "triplet").items():
            long_rows += emit_long_rows(
                model, "category_x_triplet", f"{cat}|{trip}", sub
            )

    # ---- write long CSV (append mode, matching the old script's behaviour) ----
    long_path = Path(args.long_out)
    long_path.parent.mkdir(parents=True, exist_ok=True)
    long_fields = ["model", "slice_type", "slice_value", "n"] + METRICS
    write_header = not long_path.exists() or long_path.stat().st_size == 0
    with long_path.open("a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=long_fields)
        if write_header:
            w.writeheader()
        w.writerows(long_rows)
    print(f"[long ] appended {len(long_rows)} rows to {long_path}")

    # ----------------------------------------------------------------
    # Optional WIDE-FORMAT output for quick side-by-side comparison
    # ----------------------------------------------------------------
    if args.wide_out:
        wide: Dict[str, str] = {
            "file": infile.name,
            "model": model,
            "n": str(len(merged)),
        }
        # overall
        for m in METRICS:
            wide[f"overall_{m}"] = fmt(mean_metric(merged, m))
        # category
        for cat in categories:
            sub = [r for r in merged if r.get("category") == cat]
            for m in METRICS:
                wide[f"{cat}_{m}"] = fmt(mean_metric(sub, m))
        # language axes
        for axis_short, axis_full in (("instr", "instruction_language"),
                                      ("cont", "content_language"),
                                      ("resp", "response_language")):
            for lang in LANGS:
                sub = [r for r in merged if r.get(axis_full) == lang]
                for m in METRICS:
                    wide[f"{axis_short}_{lang}_{m}"] = fmt(mean_metric(sub, m))
        # mismatch class
        for cls in ("same", "instr_only", "resp_only", "content_only",
                    "full_mismatch"):
            sub = [r for r in merged if r.get("mismatch_class") == cls]
            for m in METRICS:
                wide[f"mm_{cls}_{m}"] = fmt(mean_metric(sub, m))

        wide_path = Path(args.wide_out)
        wide_path.parent.mkdir(parents=True, exist_ok=True)
        write_header_w = not wide_path.exists() or wide_path.stat().st_size == 0
        with wide_path.open("a", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(wide.keys()))
            if write_header_w:
                w.writeheader()
            w.writerow(wide)
        print(f"[wide ] appended 1 row  to {wide_path}")


if __name__ == "__main__":
    main()
