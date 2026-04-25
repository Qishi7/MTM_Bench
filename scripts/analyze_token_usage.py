#!/usr/bin/env python3
"""Analyze average input / output / thinking tokens per model.

Scans all `*_lan_main.csv` files in an outputs directory and aggregates
per-model averages. Thinking tokens are computed as
    thinking = max(0, total_tokens - input_tokens - output_tokens)

For providers that bundle reasoning into `output_tokens` (e.g. DeepSeek-R1,
MiniMax-M2.x, QwQ via OpenAI-compatible APIs), `thinking` will be ~0 and the
long reasoning will instead inflate the `output` column. That matches the
convention in the paper-style table (R1 / MiniMax shown in a separate group).
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict
from statistics import mean
from typing import Dict, List, Optional


# Display order matches the paper-style table.
# Each entry is (csv_model_name, friendly_name).
# Use None as csv_model_name for rows you want as placeholders (model not yet run).
# Use the special tuple ("__SEP__", "") to insert a midrule before the reasoning group.
DEFAULT_ORDER = [
    ("deepseek/deepseek-v3.2", "DeepSeek-V3.2"),
    ("deepseek/deepseek-chat-v3.1", "DeepSeek-V3.1"),
    ("qwen3-max", "Qwen3-Max"),
    ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
    ("gemini-2.5-pro", "Gemini 2.5 Pro"),
    ("gemini-2.5-flash", "Gemini 2.5 Flash"),
    ("qwen3.5-flash", "Qwen3.5-Flash"),
    ("gpt-4.1", "GPT-4.1"),
    ("qwen3.5-plus", "Qwen3.5-Plus"),
    ("gpt-5.4", "GPT-5.4"),
    ("gpt-5.4-mini", "GPT-5.4-mini"),
    ("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
    ("claude-opus-4-6", "Claude Opus 4.6"),
    ("llama3.1:8b", "LLaMA-3.1-8B"),
    ("llama-3.3-70b-versatile", "LLaMA-3.3-70B"),
    ("gpt-5.3-codex", "GPT-5.3-Codex"),
    ("llama3:latest", "LLaMA-3"),
    ("__SEP__", ""),
    ("minimax-m2.7", "MiniMax-M2.7"),
    ("minimax-m2.5", "MiniMax-M2.5"),
    ("deepseek/deepseek-r1", "DeepSeek-R1$^\\dagger$"),
]


def safe_int(value: str) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def aggregate(
    token_rows: List[Dict[str, str]],
    score_rows: List[Dict[str, str]],
) -> Dict[str, Dict[str, List]]:
    by_model: Dict[str, Dict[str, List]] = defaultdict(
        lambda: {"input": [], "output": [], "thinking": [], "total": [], "js": []}
    )
    for row in token_rows:
        model = (row.get("model") or "").strip()
        if not model:
            continue
        i = safe_int(row.get("input_tokens", ""))
        o = safe_int(row.get("output_tokens", ""))
        t = safe_int(row.get("total_tokens", ""))
        if i is None or o is None or t is None:
            continue
        thinking = max(0, t - i - o)
        by_model[model]["input"].append(i)
        by_model[model]["output"].append(o)
        by_model[model]["thinking"].append(thinking)
        by_model[model]["total"].append(t)
    for row in score_rows:
        model = (row.get("model") or "").strip()
        if not model:
            continue
        js = safe_float(row.get("JointSuccess", ""))
        if js is not None:
            by_model[model]["js"].append(js)
    return by_model


def fmt(x: Optional[float]) -> str:
    return "--" if x is None else f"{x:.1f}"


def fmt_thinking(x: Optional[float]) -> str:
    """Like fmt, but render 0 as blank (those models bundle reasoning into output)."""
    if x is None:
        return "--"
    if x == 0:
        return ""
    return f"{x:.1f}"


def fmt_js(x: Optional[float]) -> str:
    """Render JS as 3-decimal value without leading zero (e.g. .923)."""
    if x is None:
        return "--"
    s = f"{x:.3f}"
    if s.startswith("0."):
        s = s[1:]
    return s


def _empty_row(friendly: str) -> Dict:
    return {
        "Model": friendly,
        "N": 0,
        "Input (avg)": None,
        "Output (avg)": None,
        "Thinking (avg)": None,
        "Total (avg)": None,
        "JS": None,
    }


def build_rows_in_order(
    by_model: Dict[str, Dict[str, List]],
    order: List,
    skip_missing: bool,
) -> List[Dict]:
    rows: List[Dict] = []
    for csv_name, friendly in order:
        if csv_name == "__SEP__":
            rows.append({"sep": True})
            continue
        if csv_name is None:
            if skip_missing:
                continue
            rows.append(_empty_row(friendly))
            continue
        m = by_model.get(csv_name)
        if not m or not m["input"]:
            if skip_missing:
                continue
            rows.append(_empty_row(friendly))
            continue
        rows.append(
            {
                "Model": friendly,
                "N": len(m["input"]),
                "Input (avg)": mean(m["input"]),
                "Output (avg)": mean(m["output"]),
                "Thinking (avg)": mean(m["thinking"]),
                "Total (avg)": mean(m["total"]),
                "JS": mean(m["js"]) if m["js"] else None,
            }
        )
    return rows


def build_rows_sorted(
    by_model: Dict[str, Dict[str, List]],
    sort_by: str,
    descending: bool,
) -> List[Dict]:
    rows = []
    for model, m in by_model.items():
        if not m["input"]:
            continue
        rows.append(
            {
                "Model": model,
                "N": len(m["input"]),
                "Input (avg)": mean(m["input"]),
                "Output (avg)": mean(m["output"]),
                "Thinking (avg)": mean(m["thinking"]),
                "Total (avg)": mean(m["total"]),
                "JS": mean(m["js"]) if m["js"] else None,
            }
        )
    key_map = {
        "model": "Model",
        "input": "Input (avg)",
        "output": "Output (avg)",
        "thinking": "Thinking (avg)",
        "total": "Total (avg)",
        "js": "JS",
    }
    sort_key = key_map.get(sort_by, "JS")

    def sort_value(r):
        v = r.get(sort_key)
        if sort_key == "Model":
            return r["Model"].lower()
        return v if v is not None else (float("-inf") if descending else float("inf"))

    rows.sort(key=sort_value, reverse=descending)
    return rows


def print_text_table(rows: List[Dict], include_js: bool) -> None:
    headers = ["Model", "N", "Input (avg)", "Output (avg)", "Thinking (avg)", "Total (avg)"]
    if include_js:
        headers.append("JS")

    str_rows: List[List[str]] = []
    for r in rows:
        if r.get("sep"):
            str_rows.append(["__SEP__"])
            continue
        cells = [
            r["Model"],
            str(r["N"]) if r["N"] else "--",
            fmt(r["Input (avg)"]),
            fmt(r["Output (avg)"]),
            fmt_thinking(r["Thinking (avg)"]),
            fmt(r["Total (avg)"]),
        ]
        if include_js:
            cells.append(fmt_js(r.get("JS")))
        str_rows.append(cells)

    data_rows = [r for r in str_rows if r != ["__SEP__"]]
    widths = [max(len(h), *(len(row[i]) for row in data_rows)) for i, h in enumerate(headers)]

    def line(cells: List[str]) -> str:
        return "  ".join(
            cells[i].ljust(widths[i]) if i == 0 else cells[i].rjust(widths[i])
            for i in range(len(cells))
        )

    print(line(headers))
    print("  ".join("-" * w for w in widths))
    for row in str_rows:
        if row == ["__SEP__"]:
            print("  ".join("-" * w for w in widths))
        else:
            print(line(row))


def print_latex_table(rows: List[Dict], include_js: bool) -> None:
    col_spec = "lrrrrr" + ("r" if include_js else "")
    headers = [
        r"\textbf{Model}",
        r"\textbf{Input (avg)}",
        r"\textbf{Output (avg)}",
        r"\textbf{Thinking (avg)}",
        r"\textbf{Total (avg)}",
    ]
    if include_js:
        headers.append(r"\textbf{JS}")

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\small")
    print(r"\setlength{\tabcolsep}{4pt}")
    print(r"\renewcommand{\arraystretch}{1.05}")
    print(r"\begin{tabular}{" + col_spec + "}")
    print(r"\toprule")
    print(" & ".join(headers) + r" \\")
    print(r"\midrule")
    for r in rows:
        if r.get("sep"):
            print(r"\midrule")
            continue
        cells = [
            r["Model"],
            fmt(r["Input (avg)"]),
            fmt(r["Output (avg)"]),
            fmt_thinking(r["Thinking (avg)"]),
            fmt(r["Total (avg)"]),
        ]
        if include_js:
            cells.append(fmt_js(r.get("JS")))
        print(" & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    js_note = " JS = Joint Success rate." if include_js else ""
    print(r"\caption{Mean input, output, thinking, and total tokens per instance "
          r"(168 instances total). $^\dagger$ denotes a reasoning model whose reasoning "
          r"tokens are bundled into the output count; MiniMax and DeepSeek-R1 are grouped "
          r"separately due to substantially higher output lengths. Blank cells in the "
          r"Thinking column indicate that the provider does not report thinking tokens "
          r"separately." + js_note + r"}")
    print(r"\label{tab:token-usage}")
    print(r"\end{table}")


def write_csv(by_model: Dict[str, Dict[str, List[int]]], out_path: str) -> None:
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "n", "input_avg", "output_avg", "thinking_avg", "total_avg"])
        for model, m in by_model.items():
            if not m["input"]:
                continue
            w.writerow(
                [
                    model,
                    len(m["input"]),
                    f"{mean(m['input']):.2f}",
                    f"{mean(m['output']):.2f}",
                    f"{mean(m['thinking']):.2f}",
                    f"{mean(m['total']):.2f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-dir",
        default="MTM_Bench/outputs",
        help="Directory containing *_lan_main.csv files",
    )
    parser.add_argument(
        "--pattern",
        default="*_lan_main.csv",
        help="Glob pattern within outputs-dir",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Explicit list of CSV files to analyze (overrides --pattern)",
    )
    parser.add_argument(
        "--order",
        default="paper",
        choices=["paper", "sort"],
        help="Row order: 'paper' uses the built-in display order (matches the table image); "
             "'sort' sorts by --sort-by",
    )
    parser.add_argument(
        "--sort-by",
        default="thinking",
        choices=["model", "input", "output", "thinking", "total", "js"],
        help="Column to sort by (only used when --order sort)",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort ascending instead of descending",
    )
    parser.add_argument(
        "--format",
        default="text",
        choices=["text", "latex", "both"],
        help="Output format",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip rows in the paper order whose CSV is not found (otherwise show '--')",
    )
    parser.add_argument(
        "--scored-dir",
        default="MTM_Bench/outputs_scored",
        help="Directory containing *_scored.csv files for JointSuccess (set to '' to skip)",
    )
    parser.add_argument(
        "--scored-pattern",
        default="*_lan_main_scored.csv",
        help="Glob pattern within scored-dir",
    )
    parser.add_argument(
        "--no-js",
        action="store_true",
        help="Do not include the JS column even if scored CSVs are available",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Optional path to also write the aggregated table as CSV",
    )
    args = parser.parse_args()

    if args.files:
        files = args.files
    else:
        files = sorted(glob.glob(os.path.join(args.outputs_dir, args.pattern)))

    if not files:
        print(f"No CSV files found (dir={args.outputs_dir}, pattern={args.pattern}).")
        return

    print(f"Scanning {len(files)} file(s):")
    for p in files:
        print(f"  - {p}")
    print()

    token_rows: List[Dict[str, str]] = []
    for p in files:
        token_rows.extend(read_csv(p))

    score_rows: List[Dict[str, str]] = []
    scored_files: List[str] = []
    if args.scored_dir and not args.no_js:
        scored_files = sorted(glob.glob(os.path.join(args.scored_dir, args.scored_pattern)))
        if scored_files:
            print(f"Also reading {len(scored_files)} scored file(s) for JS:")
            for p in scored_files:
                print(f"  - {p}")
            print()
            for p in scored_files:
                score_rows.extend(read_csv(p))

    include_js = bool(scored_files) and not args.no_js

    by_model = aggregate(token_rows, score_rows)
    if not by_model:
        print("No valid rows with input/output/total token counts found.")
        return

    if args.order == "paper":
        rows = build_rows_in_order(by_model, DEFAULT_ORDER, skip_missing=args.skip_missing)
    else:
        rows = build_rows_sorted(by_model, sort_by=args.sort_by, descending=not args.ascending)

    if args.format in ("text", "both"):
        print_text_table(rows, include_js=include_js)

    if args.format == "both":
        print()

    if args.format in ("latex", "both"):
        print_latex_table(rows, include_js=include_js)

    if args.out_csv:
        write_csv(by_model, args.out_csv)
        print(f"\nWrote aggregated CSV to {args.out_csv}")


if __name__ == "__main__":
    main()
