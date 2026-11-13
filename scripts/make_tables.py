#!/usr/bin/env python3
"""Generate paper tables from scored MTM-Bench outputs.

This script reads the per-model scored output files in ``outputs_scored/`` and
writes CSV files corresponding to the main paper and appendix tables:

  table1_main_results.csv
  table2_mismatch_task.csv
  table3_axis_marginals.csv
  table4_model_family_tokens.csv

Main paper results are computed from the automatic scoring pipeline over the full
output set. Human audit labels are not merged into these aggregate scores.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

METRICS = ["SemanticCorrect", "LangCorrect", "ConstraintCorrect", "CodeSwitchRatio", "JointSuccess"]
LANGS = ["en", "es", "zh"]
AXES = {
    "Instruction": "instruction_language",
    "Content": "content_language",
    "Response": "response_language",
}
MISMATCH_ORDER = ["same", "instr_only", "content_only", "resp_only", "full_mismatch"]
FAMILY_ORDER = ["A", "B", "C"]
FAMILY_LABELS = {
    "A": "A: Sem. rev.",
    "B": "B: Final-state",
    "C": "C: Lang. purity",
}

MODEL_DISPLAY = {
    "qwen3-max": "Qwen3-Max",
    "deepseek-r1": "DeepSeek-R1",
    "qwen3-5-plus": "Qwen3.5-Plus",
    "gpt-5-4-mini": "GPT-5.4-mini",
    "gpt-5-4": "GPT-5.4",
    "deepseek-v3-2": "DeepSeek-V3.2",
    "deepseek-v3-1": "DeepSeek-V3.1",
    "gpt-5-3-codex": "GPT-5.3-Codex",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemini-2-5-flash-lite": "Gemini 2.5 Flash-Lite",
    "qwen3-5-flash": "Qwen3.5-Flash",
    "gpt-4-1": "GPT-4.1",
    "claude-opus-4-1": "Claude Opus 4.1",
    "llama3-3-70b": "LLaMA-3.3-70B",
    "gemini-2-5-flash": "Gemini 2.5 Flash",
    "gemini-2-5-flash-no-think": "Gemini 2.5 Flash",
    "minimax-m2-7": "MiniMax-M2.7",
    "minimax-m2-5": "MiniMax-M2.5",
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "llama3": "LLaMA-3",
    "llama3-1-8b": "LLaMA-3.1-8B",
}


def normalize_model_name(name: str) -> str:
    s = str(name).strip()
    s = s.split("/")[-1]  # drop provider prefixes such as deepseek/ or accounts/.../models/
    s = s.replace("_", "-").replace(".", "-").replace(":", "-")
    s = re.sub(r"-tri-main-scored$", "", s)
    s = re.sub(r"-fireworks$", "", s)
    s = re.sub(r"-instruct$", "", s)
    if s == "deepseek-chat-v3-1":
        s = "deepseek-v3-1"
    if s == "llama-v3p3-70b":
        s = "llama3-3-70b"
    if s == "llama3-latest":
        s = "llama3"
    return s.lower()


def display_model(name: str) -> str:
    key = normalize_model_name(name)
    return MODEL_DISPLAY.get(key, str(name).strip())


def mismatch_class(instr: str, content: str, resp: str) -> str:
    if instr == content == resp:
        return "same"
    if content == resp and instr != content:
        return "instr_only"
    if instr == resp and instr != content:
        return "content_only"
    if instr == content and content != resp:
        return "resp_only"
    return "full_mismatch"


def load_scored(scored_dir: Path) -> pd.DataFrame:
    files = sorted(scored_dir.glob("*_tri_main_scored.csv"))
    if not files:
        raise FileNotFoundError(f"No *_tri_main_scored.csv files found in {scored_dir}")
    frames: List[pd.DataFrame] = []
    for path in files:
        df = pd.read_csv(path)
        if "model" not in df.columns:
            df["model"] = path.stem.replace("_tri_main_scored", "")
        df["source_file"] = path.name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)

    required = ["model", "category", "core_id", "instruction_language", "content_language", "response_language"] + METRICS
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns in scored outputs: {missing}")

    for m in METRICS:
        out[m] = pd.to_numeric(out[m], errors="coerce")
    out["model_display"] = out["model"].map(display_model)
    out["mismatch_class"] = [
        mismatch_class(i, c, r)
        for i, c, r in zip(out["instruction_language"], out["content_language"], out["response_language"])
    ]
    return out


def metric_means(df: pd.DataFrame) -> Dict[str, float]:
    return {m: float(df[m].mean()) for m in METRICS}


def rounded_frame(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(digits)
    return out


def make_table1(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, sub in df.groupby("model_display", sort=False):
        row = {"Model": model, **metric_means(sub)}
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab = tab.sort_values("JointSuccess", ascending=False).reset_index(drop=True)
    mean_row = {"Model": "Mean"}
    for m in METRICS:
        mean_row[m] = tab[m].mean()
    tab = pd.concat([tab, pd.DataFrame([mean_row])], ignore_index=True)
    return rounded_frame(tab.rename(columns={"CodeSwitchRatio": "CTR"}), 3)


def make_table2(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_models = df["model_display"].nunique()

    for cls in MISMATCH_ORDER:
        sub = df[df["mismatch_class"] == cls]
        row = {"Block": "By mismatch class", "Class": cls, "n": int(round(len(sub) / n_models))}
        row.update(metric_means(sub))
        rows.append(row)

    for cat in FAMILY_ORDER:
        sub = df[df["category"] == cat]
        row = {"Block": "By task family", "Class": FAMILY_LABELS[cat], "n": int(round(len(sub) / n_models))}
        row.update(metric_means(sub))
        rows.append(row)

    return rounded_frame(pd.DataFrame(rows).rename(columns={"CodeSwitchRatio": "CTR"}), 3)


def make_table3(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for axis_name, col in AXES.items():
        js_by_lang = df.groupby(col)["JointSuccess"].mean()
        delta = float(js_by_lang.max() - js_by_lang.min())
        for lang in LANGS:
            sub = df[df[col] == lang]
            row = {"Axis": axis_name, "Delta_JS": delta, "Lang": lang}
            row.update(metric_means(sub))
            rows.append(row)
    return rounded_frame(pd.DataFrame(rows).rename(columns={"CodeSwitchRatio": "CTR"}), 3)


def make_table4(df: pd.DataFrame, token_file: Optional[Path]) -> pd.DataFrame:
    """Generate appendix model-family and token-usage table.

    Token usage is stored with raw model identifiers, while the table uses
    display names. We therefore normalize model names before merging.
    """

    fam = df.groupby(["model_display", "category"])["JointSuccess"].mean().unstack()
    fam = fam.rename(columns={"A": "A_JS", "B": "B_JS", "C": "C_JS"})

    overall = df.groupby("model_display")["JointSuccess"].mean().rename("JS")
    tab = pd.concat([fam, overall], axis=1).reset_index().rename(columns={"model_display": "Model"})

    if token_file is not None and token_file.exists():
        tok = pd.read_csv(token_file)

        if "model" in tok.columns:
            tok["Model"] = tok["model"].map(display_model)
        elif "Model" not in tok.columns:
            raise ValueError("Token usage file must contain either 'model' or 'Model'.")

        tok = tok.rename(
            columns={
                "mean_input_tokens": "In",
                "mean_output_tokens": "Out",
                "input_avg": "In",
                "output_avg": "Out",
                "Input_tokens": "In",
                "Output_tokens": "Out",
            }
        )

        required = {"Model", "In", "Out"}
        missing = required - set(tok.columns)
        if missing:
            raise ValueError(f"Token usage file is missing columns: {sorted(missing)}")

        tok = tok[["Model", "In", "Out"]].copy()

        tab = tab.drop(columns=[c for c in ["In", "Out"] if c in tab.columns])
        tab = tab.merge(tok, on="Model", how="left")

        if tab["In"].isna().any() or tab["Out"].isna().any():
            missing_models = tab.loc[tab["In"].isna() | tab["Out"].isna(), "Model"].tolist()
            raise ValueError(f"Missing token usage for models: {missing_models}")

    tab = tab.sort_values("JS", ascending=False).reset_index(drop=True)

    ordered_cols = ["Model", "A_JS", "B_JS", "C_JS", "In", "Out", "JS"]
    tab = tab[ordered_cols]

    mean_row = {"Model": "Mean"}
    for c in ["A_JS", "B_JS", "C_JS", "In", "Out", "JS"]:
        mean_row[c] = tab[c].mean()

    tab = pd.concat([tab, pd.DataFrame([mean_row])], ignore_index=True)
    return rounded_frame(tab, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MTM-Bench paper tables from scored outputs.")
    parser.add_argument("--scored_dir", default="outputs_scored/haiku judge", help="Directory containing *_tri_main_scored.csv files.")
    parser.add_argument("--token_file", default="outputs_scored/haiku judge/token_usage_tri_main.csv", help="Token usage CSV.")
    parser.add_argument("--out_dir", default="results", help="Output directory for table CSVs.")
    args = parser.parse_args()

    scored_dir = Path(args.scored_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_scored(scored_dir)

    make_table1(df).to_csv(out_dir / "table1_main_results.csv", index=False)
    make_table2(df).to_csv(out_dir / "table2_mismatch_task.csv", index=False)
    make_table3(df).to_csv(out_dir / "table3_axis_marginals.csv", index=False)
    token_path = Path(args.token_file) if args.token_file else None
    make_table4(df, token_path).to_csv(out_dir / "table4_model_family_tokens.csv", index=False)

    print(f"Wrote table CSVs to {out_dir}")


if __name__ == "__main__":
    main()
