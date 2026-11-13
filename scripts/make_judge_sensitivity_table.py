#!/usr/bin/env python3
"""Generate the GPT second-judge sensitivity table from released scored CSVs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

AXES = {
    "Instruction": "instruction_language",
    "Content": "content_language",
    "Response": "response_language",
}


def load_scored(scored_dir: Path) -> pd.DataFrame:
    files = sorted(scored_dir.glob("*_tri_main_scored.csv"))
    if not files:
        raise FileNotFoundError(f"No *_tri_main_scored.csv files found in {scored_dir}")
    frames = []
    for path in files:
        df = pd.read_csv(path)
        if "model" not in df.columns:
            df["model"] = path.stem.replace("_tri_main_scored", "")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    required = ["core_id", "instruction_language", "content_language", "response_language", "JointSuccess"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns in {scored_dir}: {missing}")
    out["JointSuccess"] = pd.to_numeric(out["JointSuccess"], errors="coerce")
    return out.dropna(subset=["JointSuccess", "core_id"]).copy()


def axis_ranges(df: pd.DataFrame) -> Dict[str, float]:
    ranges = {}
    for axis_name, col in AXES.items():
        means = df.groupby(col)["JointSuccess"].mean()
        ranges[axis_name] = float(means.max() - means.min())
    return ranges


def bootstrap_resp_minus_content(df: pd.DataFrame, n_boot: int, seed: int) -> tuple[float, float, float]:
    observed = axis_ranges(df)
    observed_delta = observed["Response"] - observed["Content"]

    rng = np.random.default_rng(seed)
    core_ids = np.array(sorted(df["core_id"].dropna().unique()))
    n_core = len(core_ids)
    core_index = {cid: i for i, cid in enumerate(core_ids)}

    stats = {}
    for axis_name, col in AXES.items():
        stats[axis_name] = {}
        for lang, sub in df.groupby(col):
            sums = np.zeros(n_core, dtype=float)
            counts = np.zeros(n_core, dtype=float)
            by_core = sub.groupby("core_id")["JointSuccess"].agg(["sum", "count"])
            for cid, row in by_core.iterrows():
                j = core_index[cid]
                sums[j] = row["sum"]
                counts[j] = row["count"]
            stats[axis_name][lang] = (sums, counts)

    samples: List[float] = []
    probs = np.full(n_core, 1.0 / n_core)
    for _ in range(n_boot):
        weights = rng.multinomial(n_core, probs).astype(float)
        boot_ranges = {}
        for axis_name, lang_stats in stats.items():
            means = []
            for sums, counts in lang_stats.values():
                denom = float(weights @ counts)
                means.append(float(weights @ sums) / denom if denom else np.nan)
            boot_ranges[axis_name] = float(np.nanmax(means) - np.nanmin(means))
        samples.append(boot_ranges["Response"] - boot_ranges["Content"])

    vals = np.asarray(samples, dtype=float)
    return observed_delta, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate judge-sensitivity table from Haiku and GPT judged CSVs.")
    parser.add_argument("--haiku_dir", default="outputs_scored/haiku judge")
    parser.add_argument("--gpt_dir", default="outputs_scored/gpt judge")
    parser.add_argument("--out", default="results/table8_judge_sensitivity.csv")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rows = []
    for judge, scored_dir in [("Haiku", args.haiku_dir), ("GPT-4.1", args.gpt_dir)]:
        df = load_scored(Path(scored_dir))
        ranges = axis_ranges(df)
        delta, low, high = bootstrap_resp_minus_content(df, args.n_boot, args.seed)
        rows.append({
            "Judge": judge,
            "Instruction": ranges["Instruction"],
            "Content": ranges["Content"],
            "Response": ranges["Response"],
            "Resp_minus_Content": delta,
            "CI_lower": low,
            "CI_upper": high,
            "n_boot": args.n_boot,
            "cluster": "core_id",
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
