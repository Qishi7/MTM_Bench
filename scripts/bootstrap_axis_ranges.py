#!/usr/bin/env python3
"""Clustered bootstrap intervals for language-axis ranges.

The bootstrap samples base items, identified by ``core_id``, with replacement.
For each sampled base item, all associated triplet instances and model outputs
are retained. This matches the paper's item-clustered uncertainty analysis.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

METRIC = "JointSuccess"
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
    for f in files:
        df = pd.read_csv(f)
        if "model" not in df.columns:
            df["model"] = f.stem.replace("_tri_main_scored", "")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    required = ["core_id", "instruction_language", "content_language", "response_language", METRIC]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    out[METRIC] = pd.to_numeric(out[METRIC], errors="coerce")
    return out.dropna(subset=[METRIC]).copy()


def axis_ranges(df: pd.DataFrame) -> Dict[str, float]:
    ranges = {}
    for axis_name, col in AXES.items():
        means = df.groupby(col)[METRIC].mean()
        ranges[axis_name] = float(means.max() - means.min())
    return ranges


def bootstrap(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Run a fast cluster bootstrap using core-level sufficient statistics."""
    rng = np.random.default_rng(seed)
    core_ids = np.array(sorted(df["core_id"].dropna().unique()))
    n_core = len(core_ids)
    core_index = {cid: i for i, cid in enumerate(core_ids)}

    observed = axis_ranges(df)
    samples: Dict[str, List[float]] = {axis: [] for axis in AXES}

    # Precompute core-level sums and counts for every axis-language cell.
    stats = {}
    for axis_name, col in AXES.items():
        stats[axis_name] = {}
        for lang, sub in df.groupby(col):
            sums = np.zeros(n_core, dtype=float)
            counts = np.zeros(n_core, dtype=float)
            by_core = sub.groupby("core_id")[METRIC].agg(["sum", "count"])
            for cid, row in by_core.iterrows():
                j = core_index[cid]
                sums[j] = row["sum"]
                counts[j] = row["count"]
            stats[axis_name][lang] = (sums, counts)

    probs = np.full(n_core, 1.0 / n_core)
    for _ in range(n_boot):
        # Multinomial counts are equivalent to sampling core_ids with replacement.
        weights = rng.multinomial(n_core, probs).astype(float)
        for axis_name, lang_stats in stats.items():
            means = []
            for sums, counts in lang_stats.values():
                denom = float(weights @ counts)
                means.append(float(weights @ sums) / denom if denom else np.nan)
            samples[axis_name].append(float(np.nanmax(means) - np.nanmin(means)))

    rows = []
    for axis in AXES:
        vals = np.asarray(samples[axis], dtype=float)
        rows.append({
            "Axis": axis,
            "Range": observed[axis],
            "CI_lower": float(np.percentile(vals, 2.5)),
            "CI_upper": float(np.percentile(vals, 97.5)),
            "n_boot": n_boot,
            "cluster": "core_id",
        })
    return pd.DataFrame(rows)

def main() -> None:
    parser = argparse.ArgumentParser(description="Clustered bootstrap for language-axis JS ranges.")
    parser.add_argument("--scored_dir", default="outputs_scored/haiku judge", help="Directory containing scored output CSVs.")
    parser.add_argument("--out", default="results/bootstrap_axis_ranges.csv", help="Output CSV path.")
    parser.add_argument("--n_boot", type=int, default=2000, help="Number of bootstrap replicates.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    args = parser.parse_args()

    df = load_scored(Path(args.scored_dir))
    res = bootstrap(df, n_boot=args.n_boot, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.round(4).to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
