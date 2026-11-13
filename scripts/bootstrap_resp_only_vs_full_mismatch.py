#!/usr/bin/env python3
"""Clustered bootstrap for response-only versus full-mismatch triplets.

This script supports the mismatch-class analysis in Section 7.3. It estimates
the aggregate JOINTSUCCESS difference

    JS(full mismatch) - JS(response only)

using a clustered bootstrap over base items. Base items are identified by
``core_id``. For each sampled base item, all associated language triplets and
model outputs are retained, matching the item-clustered uncertainty analysis
used elsewhere in the paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METRIC = "JointSuccess"

LANGUAGE_COLUMNS = [
    "instruction_language",
    "content_language",
    "response_language",
]

REQUIRED_COLUMNS = [
    "core_id",
    "instruction_language",
    "content_language",
    "response_language",
    METRIC,
]


def load_scored(scored_dir: Path) -> pd.DataFrame:
    """Load scored model outputs from *_tri_main_scored.csv files."""
    files = sorted(scored_dir.glob("*_tri_main_scored.csv"))
    if not files:
        raise FileNotFoundError(
            f"No *_tri_main_scored.csv files found in {scored_dir}"
        )

    frames = []
    for path in files:
        df = pd.read_csv(path)
        if "model" not in df.columns:
            df["model"] = path.stem.replace("_tri_main_scored", "")
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    missing = [col for col in REQUIRED_COLUMNS if col not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out[METRIC] = pd.to_numeric(out[METRIC], errors="coerce")
    out = out.dropna(subset=[METRIC]).copy()
    return out


def add_mismatch_class(df: pd.DataFrame) -> pd.DataFrame:
    """Assign each triplet to a mismatch class."""
    out = df.copy()

    instr = out["instruction_language"]
    content = out["content_language"]
    response = out["response_language"]

    conditions = [
        (instr == content) & (content == response),
        (content == response) & (instr != content),
        (instr == response) & (content != instr),
        (instr == content) & (response != instr),
        (instr != content) & (instr != response) & (content != response),
    ]

    choices = [
        "same",
        "instr_only",
        "content_only",
        "resp_only",
        "full_mismatch",
    ]

    out["mismatch_class"] = np.select(conditions, choices, default="other")
    return out


def check_design(df: pd.DataFrame, expected_triplets_per_core: int = 27) -> None:
    """Check that each model and base item has the expected triplet expansion."""
    if "model" not in df.columns:
        raise ValueError("Expected a model column after loading scored outputs.")

    per_model_core = df.groupby(["model", "core_id"]).size()
    bad_rows = per_model_core[per_model_core != expected_triplets_per_core]
    if not bad_rows.empty:
        example = bad_rows.head().to_dict()
        raise ValueError(
            "Unexpected number of rows per model-core_id. "
            f"Expected {expected_triplets_per_core}. Examples: {example}"
        )

    per_model_core_triplets = (
        df.drop_duplicates(["model", "core_id"] + LANGUAGE_COLUMNS)
          .groupby(["model", "core_id"])
          .size()
    )
    bad_triplets = per_model_core_triplets[
        per_model_core_triplets != expected_triplets_per_core
    ]
    if not bad_triplets.empty:
        example = bad_triplets.head().to_dict()
        raise ValueError(
            "Unexpected number of unique language triplets per model-core_id. "
            f"Expected {expected_triplets_per_core}. Examples: {example}"
        )


def observed_stats(df: pd.DataFrame) -> dict[str, float]:
    """Compute observed response-only and full-mismatch means."""
    means = df.groupby("mismatch_class")[METRIC].mean()

    needed = ["resp_only", "full_mismatch"]
    missing = [cls for cls in needed if cls not in means.index]
    if missing:
        raise ValueError(f"Missing mismatch classes: {missing}")

    resp = float(means["resp_only"])
    full = float(means["full_mismatch"])

    return {
        "JS_resp_only": resp,
        "JS_full_mismatch": full,
        "diff_full_minus_resp": full - resp,
    }


def bootstrap(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Run clustered bootstrap over core_id."""
    rng = np.random.default_rng(seed)

    df = add_mismatch_class(df)
    keep = df[df["mismatch_class"].isin(["resp_only", "full_mismatch"])].copy()

    core_ids = np.array(sorted(keep["core_id"].dropna().unique()))
    n_core = len(core_ids)
    n_models = int(keep["model"].nunique()) if "model" in keep.columns else np.nan

    core_index = {cid: idx for idx, cid in enumerate(core_ids)}

    obs = observed_stats(keep)

    class_stats = {}
    for cls in ["resp_only", "full_mismatch"]:
        sub = keep[keep["mismatch_class"] == cls]
        sums = np.zeros(n_core, dtype=float)
        counts = np.zeros(n_core, dtype=float)

        by_core = sub.groupby("core_id")[METRIC].agg(["sum", "count"])
        for cid, row in by_core.iterrows():
            idx = core_index[cid]
            sums[idx] = row["sum"]
            counts[idx] = row["count"]

        class_stats[cls] = (sums, counts)

    probs = np.full(n_core, 1.0 / n_core)

    resp_samples = []
    full_samples = []
    diff_samples = []

    for _ in range(n_boot):
        weights = rng.multinomial(n_core, probs).astype(float)

        boot_means = {}
        for cls, (sums, counts) in class_stats.items():
            denom = float(weights @ counts)
            if denom == 0:
                boot_means[cls] = np.nan
            else:
                boot_means[cls] = float(weights @ sums) / denom

        resp = boot_means["resp_only"]
        full = boot_means["full_mismatch"]

        resp_samples.append(resp)
        full_samples.append(full)
        diff_samples.append(full - resp)

    rows = [
        {
            "quantity": "JS_resp_only",
            "observed": obs["JS_resp_only"],
            "ci_lower": float(np.percentile(resp_samples, 2.5)),
            "ci_upper": float(np.percentile(resp_samples, 97.5)),
        },
        {
            "quantity": "JS_full_mismatch",
            "observed": obs["JS_full_mismatch"],
            "ci_lower": float(np.percentile(full_samples, 2.5)),
            "ci_upper": float(np.percentile(full_samples, 97.5)),
        },
        {
            "quantity": "diff_full_minus_resp",
            "observed": obs["diff_full_minus_resp"],
            "ci_lower": float(np.percentile(diff_samples, 2.5)),
            "ci_upper": float(np.percentile(diff_samples, 97.5)),
        },
    ]

    result = pd.DataFrame(rows)
    result["metric"] = METRIC
    result["comparison"] = "full_mismatch_minus_resp_only"
    result["n_boot"] = n_boot
    result["seed"] = seed
    result["cluster"] = "core_id"
    result["n_core"] = n_core
    result["n_models"] = n_models
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clustered bootstrap for the JOINTSUCCESS difference between "
            "full-mismatch and response-only triplets."
        )
    )
    parser.add_argument(
        "--scored_dir",
        default="outputs_scored/haiku judge",
        help="Directory containing *_tri_main_scored.csv files.",
    )
    parser.add_argument(
        "--out",
        default="results/bootstrap_resp_only_vs_full_mismatch.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--n_boot",
        type=int,
        default=5000,
        help="Number of bootstrap replicates.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed.",
    )
    parser.add_argument(
        "--expected_triplets_per_core",
        type=int,
        default=27,
        help="Expected number of triplet rows for each model-core_id pair.",
    )
    parser.add_argument(
        "--skip_design_check",
        action="store_true",
        help="Skip the model-core_id triplet expansion check.",
    )
    args = parser.parse_args()

    df = load_scored(Path(args.scored_dir))

    if not args.skip_design_check:
        check_design(
            df,
            expected_triplets_per_core=args.expected_triplets_per_core,
        )

    result = bootstrap(df, n_boot=args.n_boot, seed=args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.round(4).to_csv(out, index=False)

    print(result.round(4).to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
