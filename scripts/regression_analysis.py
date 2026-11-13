#!/usr/bin/env python3
"""Fixed-effect regression analysis for MTM-Bench.

Fits a linear probability model for JointSuccess with language-role predictors,
task-family fixed effects, model fixed effects, and base-item fixed effects.
Standard errors are clustered by base item.

This implementation uses an explicit design matrix and a cluster-robust sandwich
covariance estimator to keep the script lightweight and reproducible.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

LANG_ORDER = ["en", "es", "zh"]
PREDICTORS: List[Tuple[str, str]] = [
    ("Instruction: Spanish vs. English", "instr_es"),
    ("Instruction: Chinese vs. English", "instr_zh"),
    ("Content: Spanish vs. English", "content_es"),
    ("Content: Chinese vs. English", "content_zh"),
    ("Response: Spanish vs. English", "resp_es"),
    ("Response: Chinese vs. English", "resp_zh"),
]


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
    required = [
        "JointSuccess", "instruction_language", "content_language", "response_language",
        "category", "model", "core_id",
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    out = out[required].copy()
    out["JointSuccess"] = pd.to_numeric(out["JointSuccess"], errors="coerce")
    return out.dropna(subset=["JointSuccess", "core_id"]).reset_index(drop=True)


def add_indicator_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["instr_es"] = (out["instruction_language"] == "es").astype(float)
    out["instr_zh"] = (out["instruction_language"] == "zh").astype(float)
    out["content_es"] = (out["content_language"] == "es").astype(float)
    out["content_zh"] = (out["content_language"] == "zh").astype(float)
    out["resp_es"] = (out["response_language"] == "es").astype(float)
    out["resp_zh"] = (out["response_language"] == "zh").astype(float)
    return out


def design_matrix(df: pd.DataFrame):
    df = add_indicator_columns(df)
    parts = []
    names = []

    parts.append(np.ones((len(df), 1), dtype=float)); names.append("Intercept")

    main_cols = [name for _, name in PREDICTORS]
    parts.append(df[main_cols].to_numpy(dtype=float)); names.extend(main_cols)

    # Fixed effects. Drop the first category for each set to avoid collinearity.
    for col, prefix in [("category", "family"), ("model", "model"), ("core_id", "core")]:
        dummies = pd.get_dummies(df[col].astype(str), prefix=prefix, drop_first=True, dtype=float)
        if dummies.shape[1] > 0:
            parts.append(dummies.to_numpy(dtype=float)); names.extend(dummies.columns.tolist())

    X = np.column_stack(parts)
    y = df["JointSuccess"].to_numpy(dtype=float)
    return X, y, names


def fit_ols_cluster(df: pd.DataFrame):
    X, y, names = design_matrix(df)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape

    xtx_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((k, k), dtype=float)
    clusters = df["core_id"].astype(str).to_numpy()
    unique_clusters = np.unique(clusters)
    for g in unique_clusters:
        idx = clusters == g
        Xg = X[idx, :]
        ug = resid[idx]
        score = Xg.T @ ug
        meat += np.outer(score, score)

    G = len(unique_clusters)
    # Small-sample correction commonly used for cluster-robust covariance.
    correction = (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 and n > k else 1.0
    cov = correction * (xtx_inv @ meat @ xtx_inv)
    se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    return beta, se, names, G


def extract_table(beta, se, names, n_clusters: int) -> pd.DataFrame:
    name_to_idx = {name: i for i, name in enumerate(names)}
    rows = []
    z = 1.959963984540054
    for label, term in PREDICTORS:
        idx = name_to_idx[term]
        coef = float(beta[idx])
        s = float(se[idx])
        p = float(2 * (1 - stats.norm.cdf(abs(coef / s)))) if s > 0 else float("nan")
        rows.append({
            "Predictor": label,
            "Coef.": coef,
            "SE": s,
            "95% CI lower": coef - z * s,
            "95% CI upper": coef + z * s,
            "p_value": p,
            "n_clusters": n_clusters,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-effect regression for JointSuccess.")
    parser.add_argument("--scored_dir", default="outputs_scored/haiku judge", help="Directory containing scored output CSVs.")
    parser.add_argument("--out", default="results/regression_table.csv", help="Output CSV path.")
    args = parser.parse_args()

    df = load_scored(Path(args.scored_dir))
    beta, se, names, n_clusters = fit_ols_cluster(df)
    table = extract_table(beta, se, names, n_clusters)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
