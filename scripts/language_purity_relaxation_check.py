#!/usr/bin/env python3
"""Sensitivity check for strict language-purity scoring in Family C.

The default slice uses all available models, all 30 Family-C base items, and the
three clean response-language conditions (en,en,en), (en,en,es), and (en,en,zh).
The original 6-model targeted slice remains available with --model_set targeted.

The relaxed rule changes only language correctness:
LangCorrect_relaxed = 1 when the original LangCorrect is 1, or when the output
is otherwise semantically and constraint correct and CodeSwitchRatio <= 0.05.
SemanticCorrect and ConstraintCorrect are unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TARGETED_MODEL_FILES = {
    "high": [
        "qwen3_max_tri_main_scored.csv",
        "deepseek_r1_tri_main_scored.csv",
    ],
    "middle": [
        "gemini_2.5_flash_lite_tri_main_scored.csv",
        "gpt_4.1_tri_main_scored.csv",
    ],
    "low": [
        "minimax_m2.5_tri_main_scored.csv",
        "llama3.1_8b_tri_main_scored.csv",
    ],
}


def selected_files(scored_dir: Path, model_set: str) -> list[tuple[str, str, Path]]:
    if model_set == "targeted":
        selected = []
        for stratum, files in TARGETED_MODEL_FILES.items():
            for filename in files:
                selected.append((stratum, filename, scored_dir / filename))
        return selected

    if model_set == "all":
        return [("all", path.name, path) for path in sorted(scored_dir.glob("*_tri_main_scored.csv"))]

    raise ValueError(f"Unknown model_set: {model_set}")


def load_slice(scored_dir: Path, model_set: str) -> pd.DataFrame:
    frames = []
    selected = selected_files(scored_dir, model_set)
    if not selected:
        raise FileNotFoundError(f"No *_tri_main_scored.csv files found in {scored_dir}")

    for stratum, filename, path in selected:
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df["selection_stratum"] = stratum
        df["source_file"] = filename
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    required = [
        "model", "item_id", "core_id", "category", "instruction_language",
        "content_language", "response_language", "SemanticCorrect",
        "LangCorrect", "ConstraintCorrect", "CodeSwitchRatio", "JointSuccess",
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in ["SemanticCorrect", "LangCorrect", "ConstraintCorrect", "JointSuccess"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    out["CodeSwitchRatio"] = pd.to_numeric(out["CodeSwitchRatio"], errors="coerce").fillna(0.0)

    keep = (
        (out["category"] == "C")
        & (out["instruction_language"] == "en")
        & (out["content_language"] == "en")
        & (out["response_language"].isin(["en", "es", "zh"]))
    )
    out = out.loc[keep].copy()

    n_models = out["model"].nunique()
    n_core = out["core_id"].nunique()
    expected = n_models * n_core * 3
    expected_models = 6 if model_set == "targeted" else len(selected)
    if len(out) != expected or n_models != expected_models or n_core != 30:
        raise ValueError(
            f"Unexpected slice size: rows={len(out)}, models={n_models}, "
            f"core_ids={n_core}, expected={expected}"
        )
    return out


def apply_relaxed_rule(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    rescue = (
        (out["LangCorrect"] == 0)
        & (out["SemanticCorrect"] == 1)
        & (out["ConstraintCorrect"] == 1)
        & (out["CodeSwitchRatio"] <= threshold)
    )
    out["LangCorrect_relaxed"] = out["LangCorrect"].where(~rescue, 1).astype(int)
    out["JointSuccess_relaxed"] = (
        (out["SemanticCorrect"] == 1)
        & (out["LangCorrect_relaxed"] == 1)
        & (out["ConstraintCorrect"] == 1)
    ).astype(int)
    out["relaxed_rescue"] = rescue.astype(int)
    return out


def summarize(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for lang, sub in df.groupby("response_language", sort=True):
        original_js = sub["JointSuccess"].mean()
        relaxed_js = sub["JointSuccess_relaxed"].mean()
        rows.append({
            "response_language": lang,
            "n": len(sub),
            "original_JS": original_js,
            "relaxed_JS": relaxed_js,
            "delta_JS": relaxed_js - original_js,
            "original_LC": sub["LangCorrect"].mean(),
            "relaxed_LC": sub["LangCorrect_relaxed"].mean(),
            "rescued_rows": int(sub["relaxed_rescue"].sum()),
        })
    by_lang = pd.DataFrame(rows)

    wide = by_lang.set_index("response_language")
    en_orig = float(wide.loc["en", "original_JS"])
    en_relaxed = float(wide.loc["en", "relaxed_JS"])
    contrasts = pd.DataFrame([
        {
            "contrast": "ES_vs_EN_penalty",
            "original_penalty": float(wide.loc["es", "original_JS"] - en_orig),
            "relaxed_penalty": float(wide.loc["es", "relaxed_JS"] - en_relaxed),
            "penalty_change": float((wide.loc["es", "relaxed_JS"] - en_relaxed) - (wide.loc["es", "original_JS"] - en_orig)),
        },
        {
            "contrast": "ZH_vs_EN_penalty",
            "original_penalty": float(wide.loc["zh", "original_JS"] - en_orig),
            "relaxed_penalty": float(wide.loc["zh", "relaxed_JS"] - en_relaxed),
            "penalty_change": float((wide.loc["zh", "relaxed_JS"] - en_relaxed) - (wide.loc["zh", "original_JS"] - en_orig)),
        },
    ])

    model_rows = []
    for (stratum, model, lang), sub in df.groupby(["selection_stratum", "model", "response_language"], sort=True):
        model_rows.append({
            "selection_stratum": stratum,
            "model": model,
            "response_language": lang,
            "n": len(sub),
            "original_JS": sub["JointSuccess"].mean(),
            "relaxed_JS": sub["JointSuccess_relaxed"].mean(),
            "delta_JS": sub["JointSuccess_relaxed"].mean() - sub["JointSuccess"].mean(),
            "rescued_rows": int(sub["relaxed_rescue"].sum()),
        })
    by_model = pd.DataFrame(model_rows)

    by_lang["threshold"] = threshold
    contrasts["threshold"] = threshold
    by_model["threshold"] = threshold
    return by_lang, contrasts, by_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict-vs-relaxed Family-C language-purity check.")
    parser.add_argument("--scored_dir", default="outputs_scored/haiku judge")
    parser.add_argument("--out_prefix", default="results/language_purity_relaxation")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--model_set", choices=["all", "targeted"], default="all")
    args = parser.parse_args()

    df = load_slice(Path(args.scored_dir), args.model_set)
    df = apply_relaxed_rule(df, args.threshold)
    by_lang, contrasts, by_model = summarize(df, args.threshold)

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    by_lang.round(4).to_csv(prefix.with_name(prefix.name + "_by_response.csv"), index=False)
    contrasts.round(4).to_csv(prefix.with_name(prefix.name + "_contrasts.csv"), index=False)
    by_model.round(4).to_csv(prefix.with_name(prefix.name + "_by_model.csv"), index=False)
    df.to_csv(prefix.with_name(prefix.name + "_rows.csv"), index=False)

    print("Family C strict-vs-relaxed language-purity check")
    print(f"Rows: {len(df)}")
    print(by_lang.round(4).to_string(index=False))
    print(contrasts.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
