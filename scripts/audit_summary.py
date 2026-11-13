#!/usr/bin/env python
"""
Summarize human audit labels.

Main paper scores are computed from the automatic scoring pipeline.
Human audit labels are summarized separately to assess scoring reliability.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def get_label_columns(df: pd.DataFrame) -> dict[str, str]:
    colmap = {}

    candidates = {
        "auto_sc": [
            "auto_sc", "auto_SC", "auto_semantic_correct",
            "SemanticCorrect_auto", "semantic_correct_auto",
            "sc_auto", "SC_auto",
            "SemanticCorrect", "semantic_correct",
        ],
        "auto_lc": [
            "auto_lc", "auto_LC", "auto_lang_correct",
            "LangCorrect_auto", "lang_correct_auto",
            "lc_auto", "LC_auto",
            "LangCorrect", "lang_correct",
        ],
        "auto_cc": [
            "auto_cc", "auto_CC", "auto_constraint_correct",
            "ConstraintCorrect_auto", "constraint_correct_auto",
            "cc_auto", "CC_auto",
            "ConstraintCorrect", "constraint_correct",
        ],
        "human_sc": [
            "human_sc", "human_SC", "human_semantic_correct",
            "SemanticCorrect_human", "semantic_correct_human",
            "sc_human", "SC_human",
            "Human_SC", "HumanCorrect_SC",
        ],
        "human_lc": [
            "human_lc", "human_LC", "human_lang_correct",
            "LangCorrect_human", "lang_correct_human",
            "lc_human", "LC_human",
            "Human_LC",
        ],
        "human_cc": [
            "human_cc", "human_CC", "human_constraint_correct",
            "ConstraintCorrect_human", "constraint_correct_human",
            "cc_human", "CC_human",
            "Human_CC",
        ],
    }

    missing = []
    for key, cands in candidates.items():
        col = find_col(df, cands)
        if col is None:
            missing.append(key)
        else:
            colmap[key] = col

    if missing:
        raise ValueError(
            "Missing required audit label columns: "
            + ", ".join(missing)
            + "\nAvailable columns: "
            + ", ".join(df.columns)
        )

    return colmap


def normalize_binary(series: pd.Series) -> pd.Series:
    def convert(x):
        if pd.isna(x):
            return pd.NA
        if isinstance(x, str):
            x = x.strip()
            if x == "":
                return pd.NA
            if x.lower() in {"true", "yes", "y"}:
                return 1
            if x.lower() in {"false", "no", "n"}:
                return 0
        try:
            return int(float(x))
        except Exception:
            return pd.NA

    return series.map(convert).astype("Int64")


def summarize_file(path: Path) -> dict[str, object]:
    df = pd.read_csv(path)
    col = get_label_columns(df)

    auto_sc = normalize_binary(df[col["auto_sc"]])
    auto_lc = normalize_binary(df[col["auto_lc"]])
    auto_cc = normalize_binary(df[col["auto_cc"]])

    human_sc_raw = normalize_binary(df[col["human_sc"]])
    human_lc_raw = normalize_binary(df[col["human_lc"]])
    human_cc_raw = normalize_binary(df[col["human_cc"]])

    # Blank human labels are treated as agreement with the automatic label.
    human_sc = human_sc_raw.fillna(auto_sc)
    human_lc = human_lc_raw.fillna(auto_lc)
    human_cc = human_cc_raw.fillna(auto_cc)

    valid = (
        auto_sc.notna() & auto_lc.notna() & auto_cc.notna()
        & human_sc.notna() & human_lc.notna() & human_cc.notna()
    )

    auto_sc = auto_sc[valid].astype(int)
    auto_lc = auto_lc[valid].astype(int)
    auto_cc = auto_cc[valid].astype(int)
    human_sc = human_sc[valid].astype(int)
    human_lc = human_lc[valid].astype(int)
    human_cc = human_cc[valid].astype(int)

    n = int(valid.sum())

    component_changed = (
        (auto_sc != human_sc) | (auto_lc != human_lc) | (auto_cc != human_cc)
    )

    auto_js = auto_sc * auto_lc * auto_cc
    human_js = human_sc * human_lc * human_cc
    js_changed = auto_js != human_js

    subset_name = "Targeted audit set" if "targeted" in path.name.lower() else "Unflagged check"

    return {
        "subset": subset_name,
        "n": n,
        "component_overrides": int(component_changed.sum()),
        "component_override_rate": component_changed.mean() if n else 0.0,
        "js_overrides": int(js_changed.sum()),
        "js_override_rate": js_changed.mean() if n else 0.0,
        "sc_agreement": (auto_sc == human_sc).mean() if n else 0.0,
        "lc_agreement": (auto_lc == human_lc).mean() if n else 0.0,
        "cc_agreement": (auto_cc == human_cc).mean() if n else 0.0,
        "overall_agreement": (~component_changed).mean() if n else 0.0,
        "source_file": str(path),
    }


def summarize_by_response_language(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        col = get_label_columns(df)
        resp_col = find_col(df, ["response_language", "target_language", "response_lang"])
        if resp_col is None:
            continue

        auto_sc = normalize_binary(df[col["auto_sc"]])
        auto_lc = normalize_binary(df[col["auto_lc"]])
        auto_cc = normalize_binary(df[col["auto_cc"]])
        human_sc = normalize_binary(df[col["human_sc"]]).fillna(auto_sc)
        human_lc = normalize_binary(df[col["human_lc"]]).fillna(auto_lc)
        human_cc = normalize_binary(df[col["human_cc"]]).fillna(auto_cc)

        valid = (
            auto_sc.notna() & auto_lc.notna() & auto_cc.notna()
            & human_sc.notna() & human_lc.notna() & human_cc.notna()
            & df[resp_col].notna()
        )
        tmp = pd.DataFrame({
            "response_language": df.loc[valid, resp_col].astype(str),
            "auto_sc": auto_sc[valid].astype(int),
            "auto_lc": auto_lc[valid].astype(int),
            "auto_cc": auto_cc[valid].astype(int),
            "human_sc": human_sc[valid].astype(int),
            "human_lc": human_lc[valid].astype(int),
            "human_cc": human_cc[valid].astype(int),
        })
        tmp["source_file"] = str(path)
        rows.append(tmp)

    if not rows:
        return pd.DataFrame()

    all_rows = pd.concat(rows, ignore_index=True)
    all_rows["auto_js"] = all_rows["auto_sc"] * all_rows["auto_lc"] * all_rows["auto_cc"]
    all_rows["human_js"] = all_rows["human_sc"] * all_rows["human_lc"] * all_rows["human_cc"]
    all_rows["component_agree"] = (
        (all_rows["auto_sc"] == all_rows["human_sc"])
        & (all_rows["auto_lc"] == all_rows["human_lc"])
        & (all_rows["auto_cc"] == all_rows["human_cc"])
    )

    out_rows = []
    for lang, sub in all_rows.groupby("response_language", sort=True):
        out_rows.append({
            "response_language": lang,
            "n": len(sub),
            "sc_agreement": (sub["auto_sc"] == sub["human_sc"]).mean(),
            "lc_agreement": (sub["auto_lc"] == sub["human_lc"]).mean(),
            "cc_agreement": (sub["auto_cc"] == sub["human_cc"]).mean(),
            "js_agreement": (sub["auto_js"] == sub["human_js"]).mean(),
            "overall_component_agreement": sub["component_agree"].mean(),
        })
    return pd.DataFrame(out_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit_files",
        nargs="+",
        default=[
            "human_validation/targeted_audit_cases.csv",
            "human_validation/unflagged_sanity_check.csv",
        ],
        help="Audit CSV files to summarize.",
    )
    parser.add_argument("--out", default="results/audit_summary.csv")
    parser.add_argument("--by_response_out", default="results/audit_by_response_language.csv")
    args = parser.parse_args()

    rows = []
    audit_paths = [Path(f) for f in args.audit_files]
    for path in audit_paths:
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue
        rows.append(summarize_file(path))

    if not rows:
        raise FileNotFoundError("No audit files were found.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    result = pd.DataFrame(rows)
    result.to_csv(out, index=False)
    print(f"Wrote {out}")
    print(result.to_string(index=False))

    by_response = summarize_by_response_language(audit_paths)
    if not by_response.empty:
        by_response_out = Path(args.by_response_out)
        by_response_out.parent.mkdir(parents=True, exist_ok=True)
        by_response.to_csv(by_response_out, index=False)
        print(f"Wrote {by_response_out}")


if __name__ == "__main__":
    main()
