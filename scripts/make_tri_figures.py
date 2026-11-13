from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


# =========================================================
# Global style
# =========================================================
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "legend.frameon": False,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
})


# =========================================================
# Colors
# =========================================================
C_BLUE = "#2D5596"
C_RED = "#BE4141"
C_GREEN = "#3C8246"
C_GRAY = "#6E6E6E"
C_ORANGE = "#D98B3A"
C_PURPLE = "#7158A8"
C_TEAL = "#1F9FA8"

LANG_COLORS = {
    "en": C_BLUE,
    "es": C_GREEN,
    "zh": C_RED,
}

ROLE_COLOR = {
    "Instruction": C_BLUE,
    "Content": C_ORANGE,
    "Response": C_GREEN,
}

FAMILY_COLOR = {
    "DeepSeek": C_BLUE,
    "Qwen": C_ORANGE,
    "Anthropic": C_PURPLE,
    "Google": C_GREEN,
    "OpenAI": C_RED,
    "MiniMax": C_TEAL,
    "Meta": C_GRAY,
}

FAILURE_COLORS = {
    "Semantic": "#4C78A8",
    "Language": "#F58518",
    "Constraint": "#54A24B",
}


# =========================================================
# Model mappings
# =========================================================
NAME_MAP = {
    "deepseek/deepseek-v3.2": "DeepSeek-V3.2",
    "deepseek/deepseek-chat-v3.1": "DeepSeek-V3.1",
    "deepseek/deepseek-r1": "DeepSeek-R1",

    "qwen3-max": "Qwen3-Max",
    "qwen3.5-flash": "Qwen3.5-Flash",
    "qwen3.5-plus": "Qwen3.5-Plus",

    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "claude-opus-4-1": "Claude Opus 4.1",

    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite",

    "gpt-4.1": "GPT-4.1",
    "gpt-5.4": "GPT-5.4",
    "gpt-5.4-mini": "GPT-5.4-mini",
    "gpt-5.3-codex": "GPT-5.3-Codex",

    "llama3:latest": "LLaMA-3",
    "llama3.1:8b": "LLaMA-3.1-8B",
    "accounts/fireworks/models/llama-v3p3-70b-instruct": "LLaMA-3.3-70B",

    "minimax-m2.5": "MiniMax-M2.5",
    "minimax-m2.7": "MiniMax-M2.7",
}

FAMILY_OF = {
    "DeepSeek-V3.2": "DeepSeek",
    "DeepSeek-V3.1": "DeepSeek",
    "DeepSeek-R1": "DeepSeek",

    "Qwen3-Max": "Qwen",
    "Qwen3.5-Flash": "Qwen",
    "Qwen3.5-Plus": "Qwen",

    "Claude Sonnet 4.6": "Anthropic",
    "Claude Haiku 4.5": "Anthropic",
    "Claude Opus 4.1": "Anthropic",

    "Gemini 2.5 Flash": "Google",
    "Gemini 2.5 Flash-Lite": "Google",

    "GPT-4.1": "OpenAI",
    "GPT-5.4": "OpenAI",
    "GPT-5.4-mini": "OpenAI",
    "GPT-5.3-Codex": "OpenAI",

    "MiniMax-M2.5": "MiniMax",
    "MiniMax-M2.7": "MiniMax",

    "LLaMA-3": "Meta",
    "LLaMA-3.1-8B": "Meta",
    "LLaMA-3.3-70B": "Meta",
}


# =========================================================
# Triplet helpers
# =========================================================
MISMATCH_ORDER = [
    "same",
    "instr_only",
    "content_only",
    "resp_only",
    "full_mismatch",
]

MISMATCH_LABEL = {
    "same": "Same",
    "instr_only": "Instr\nonly",
    "content_only": "Content\nonly",
    "resp_only": "Resp\nonly",
    "full_mismatch": "Full\nmismatch",
}


def triplet_to_class(triplet: str) -> str:
    instr, content, resp = triplet.strip("()").split(",")

    if instr == content == resp:
        return "same"
    if content == resp and instr != content:
        return "instr_only"
    if instr == content and content != resp:
        return "resp_only"
    if instr == resp and instr != content:
        return "content_only"

    return "full_mismatch"


def resp_lang_of(triplet: str) -> str:
    return triplet.strip("()").split(",")[-1]


# =========================================================
# Data loading
# =========================================================
def load_long_data(long_path: Path) -> pd.DataFrame:
    df = pd.read_csv(long_path, encoding="utf-8-sig")
    df.columns = [c.lstrip("\ufeff") for c in df.columns]

    if "model" not in df.columns:
        raise ValueError("Missing required column: model")

    df["model"] = df["model"].map(lambda m: NAME_MAP.get(m, m))

    for c in [
        "SemanticCorrect",
        "LangCorrect",
        "ConstraintCorrect",
        "CodeSwitchRatio",
        "JointSuccess",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def save_fig(path: Path) -> None:
    png_path = path.with_suffix(".png")
    pdf_path = path.with_suffix(".pdf")
    plt.savefig(png_path, facecolor="white")
    plt.savefig(pdf_path, facecolor="white")
    plt.close()
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


# =========================================================
# Main text figure: language axis effects
# =========================================================
def make_axes_effects(outdir: Path) -> None:
    outpath = outdir / "axes_effects.png"

    data = {
        "Instruction": {
            "delta": 0.018,
            "en": {"SC": 0.896, "LC": 0.882, "JS": 0.744},
            "es": {"SC": 0.894, "LC": 0.908, "JS": 0.755},
            "zh": {"SC": 0.888, "LC": 0.925, "JS": 0.737},
        },
        "Content": {
            "delta": 0.029,
            "en": {"SC": 0.893, "LC": 0.889, "JS": 0.729},
            "es": {"SC": 0.888, "LC": 0.928, "JS": 0.758},
            "zh": {"SC": 0.897, "LC": 0.897, "JS": 0.750},
        },
        "Response": {
            "delta": 0.113,
            "en": {"SC": 0.942, "LC": 0.965, "JS": 0.819},
            "es": {"SC": 0.885, "LC": 0.890, "JS": 0.711},
            "zh": {"SC": 0.851, "LC": 0.860, "JS": 0.706},
        },
    }

    langs = ["en", "es", "zh"]
    x = np.arange(len(langs))

    metrics = [
        ("SC", "SemanticCorrect", "#3F6FB5"),
        ("LC", "LangCorrect", "#C85C5C"),
        ("JS", "JointSuccess", "#4F9A63"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.75), sharey=True)

    for ax, (axis_name, axis_data) in zip(axes, data.items()):
        if axis_name == "Response":
            ax.axhspan(0.68, 1.0, color="#F7FBF7", zorder=0)

        for metric, label, color in metrics:
            vals = np.array([axis_data[lang][metric] for lang in langs])

            lw = 2.4 if metric == "JS" else 1.8
            alpha = 1.0 if metric == "JS" else 0.82
            marker_size = 6.5 if metric == "JS" else 5.5

            ax.plot(
                x,
                vals,
                marker="o",
                markersize=marker_size,
                linewidth=lw,
                color=color,
                alpha=alpha,
                label=label,
                zorder=3,
            )

            if metric == "JS":
                for xi, v in zip(x, vals):
                    ax.text(
                        xi,
                        v + 0.010,
                        f"{v:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        color=color,
                        fontweight="bold",
                    )

        ax.set_title(
            f"{axis_name} language",
            fontweight="bold" if axis_name == "Response" else "normal",
            pad=8,
        )

        ax.text(
            0.96,
            0.93,
            rf"$\Delta_{{JS}}={axis_data['delta']:.3f}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10.5,
            color="#222222",
            bbox=dict(
                boxstyle="round,pad=0.23",
                facecolor="white",
                edgecolor="#B8B8B8",
                linewidth=0.7,
                alpha=0.95,
            ),
        )

        ax.set_xticks(x)
        ax.set_xticklabels(langs)

        for tick, lang in zip(ax.get_xticklabels(), langs):
            tick.set_color(LANG_COLORS[lang])
            tick.set_fontweight("bold")

        ax.set_xlim(-0.25, 2.25)
        ax.set_ylim(0.68, 0.985)
        ax.set_yticks(np.arange(0.70, 1.00, 0.05))

        ax.grid(axis="y", linestyle=":", linewidth=0.8, color="#D9D9D9", zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Score")
    axes[1].set_xlabel("Language value")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.8,
    )

    plt.tight_layout(rect=[0.02, 0.00, 0.98, 0.92])
    save_fig(outpath)


# =========================================================
# Main text figure: task family by response language
# =========================================================
def make_task_response(outdir: Path) -> None:
    outpath = outdir / "task_response.png"

    values = np.array([
        [0.880, 0.762, 0.717],
        [0.743, 0.694, 0.746],
        [0.835, 0.676, 0.655],
    ])

    row_labels = [
        "A: Reversal",
        "B: Final-state",
        "C: Purity",
    ]

    col_labels = ["en", "es", "zh"]

    cmap = LinearSegmentedColormap.from_list(
        "soft_green",
        ["#F7FAF7", "#E5EFE4", "#C5DCC3", "#8FBE8E", "#3E8D4B"],
    )

    fig, ax = plt.subplots(figsize=(4.05, 2.45))
    ax.imshow(values, cmap=cmap, vmin=0.64, vmax=0.89, aspect="auto")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            text_color = "white" if v >= 0.82 else "#1F1F1F"
            weight = "bold" if (v >= 0.83 or (i == 1 and j == 2)) else "normal"

            ax.text(
                j,
                i,
                f"{v:.3f}",
                ha="center",
                va="center",
                fontsize=10.5,
                color=text_color,
                fontweight=weight,
            )

    ax.add_patch(
        Rectangle(
            (2 - 0.5, 1 - 0.5),
            1,
            1,
            fill=False,
            edgecolor="#2F7D3F",
            linewidth=1.4,
        )
    )

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=3)

    for tick, lang in zip(ax.get_xticklabels(), col_labels):
        tick.set_color(LANG_COLORS[lang])
        tick.set_fontweight("bold")

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.tick_params(axis="y", length=0, pad=4)

    ax.set_xticks(np.arange(-0.5, values.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, values.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(
        0.5,
        1.10,
        "Response language",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        color="#333333",
    )

    plt.tight_layout(pad=0.25)
    save_fig(outpath)


# =========================================================
# Main text figure: component failure profile
# =========================================================
def make_failure_profile(long_df: pd.DataFrame, outdir: Path) -> None:
    outpath = outdir / "failure_profile.png"

    df = long_df[long_df["slice_type"] == "category_x_response"].copy()
    if df.empty:
        raise ValueError("No rows found where slice_type == 'category_x_response'.")

    df[["Task", "Resp"]] = df["slice_value"].str.split("|", expand=True)

    agg = (
        df.groupby(["Task", "Resp"], as_index=False)[
            ["SemanticCorrect", "LangCorrect", "ConstraintCorrect"]
        ]
        .mean()
    )

    task_order = ["A", "B", "C"]
    resp_order = ["en", "es", "zh"]

    task_names = {
        "A": "Semantic reversal",
        "B": "Final-state extraction",
        "C": "Language purity",
    }

    vals = {}
    for _, r in agg.iterrows():
        vals[(r["Task"], r["Resp"])] = {
            "Semantic": 1 - r["SemanticCorrect"],
            "Language": 1 - r["LangCorrect"],
            "Constraint": 1 - r["ConstraintCorrect"],
        }

    maxval = max(v for cell in vals.values() for v in cell.values())
    xmax = max(0.28, maxval + 0.035)

    fig, axes = plt.subplots(3, 3, figsize=(9.8, 6.6), sharex=True)

    for i, task in enumerate(task_order):
        for j, resp in enumerate(resp_order):
            ax = axes[i, j]

            cell = vals[(task, resp)]
            names = ["Semantic", "Language", "Constraint"]
            data = [cell[n] for n in names]
            y = np.array([2, 1, 0])

            ax.barh(
                y,
                data,
                height=0.46,
                color=[FAILURE_COLORS[n] for n in names],
                edgecolor="none",
            )

            ax.set_xlim(0, xmax)
            ax.set_ylim(-0.7, 2.7)
            ax.set_xticks([0.0, 0.1, 0.2])
            ax.grid(axis="x", alpha=0.20)

            if j == 0:
                ax.set_yticks(y)
                ax.set_yticklabels(names, fontsize=10.5)
                ax.set_ylabel(task_names[task], fontsize=12)
            else:
                ax.set_yticks(y)
                ax.set_yticklabels([])

            if i == 2:
                ax.set_xlabel("Failure rate", fontsize=11)

            if i == 0:
                ax.set_title(
                    f"Response: {resp}",
                    fontsize=12.5,
                    color=LANG_COLORS[resp],
                    fontweight="bold",
                    pad=8,
                )

            for yy, vv in zip(y, data):
                ax.text(vv + 0.006, yy, f"{vv:.2f}", va="center", fontsize=9.5)

            dom_idx = int(np.argmax(data))
            ax.add_patch(
                Rectangle(
                    (0, y[dom_idx] - 0.27),
                    data[dom_idx],
                    0.54,
                    fill=False,
                    linewidth=1.15,
                    edgecolor="black",
                )
            )

            ax.spines["top"].set_visible(True)
            ax.spines["top"].set_linewidth(1.4)
            ax.spines["top"].set_color(LANG_COLORS[resp])
            ax.spines["right"].set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=FAILURE_COLORS["Semantic"],
            markersize=8,
            label="Semantic failure",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=FAILURE_COLORS["Language"],
            markersize=8,
            label="Language failure",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=FAILURE_COLORS["Constraint"],
            markersize=8,
            label="Constraint failure",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=False,
        fontsize=11.5,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    save_fig(outpath)


# =========================================================
# Appendix figure: full 27 triplet heatmap
# =========================================================
def make_triplet_heatmap(long_df: pd.DataFrame, outdir: Path) -> None:
    outpath = outdir / "triplet_heatmap.png"

    trip = long_df[long_df["slice_type"] == "triplet"].copy()
    if trip.empty:
        raise ValueError("No rows found where slice_type == 'triplet'.")

    trip["mm_class"] = trip["slice_value"].apply(triplet_to_class)
    trip["resp"] = trip["slice_value"].apply(resp_lang_of)

    mean_js = trip.groupby("slice_value")["JointSuccess"].mean()
    trip["mean_js"] = trip["slice_value"].map(mean_js)

    order_df = trip[["slice_value", "mm_class", "mean_js"]].drop_duplicates("slice_value")
    order_df["mm_rank"] = order_df["mm_class"].map(
        {k: i for i, k in enumerate(MISMATCH_ORDER)}
    )

    order_df = order_df.sort_values(["mm_rank", "mean_js"], ascending=[True, False])
    triplet_order = order_df["slice_value"].tolist()

    overall = (
        long_df[long_df["slice_type"] == "overall"]
        .set_index("model")["JointSuccess"]
    )

    model_order = overall.sort_values(ascending=False).index.tolist()

    mat = trip.pivot(index="model", columns="slice_value", values="JointSuccess")
    mat = mat.reindex(index=model_order, columns=triplet_order)

    n_m, n_t = mat.shape

    fig, ax = plt.subplots(figsize=(max(13, n_t * 0.50), max(7.2, n_m * 0.39)))

    cmap = LinearSegmentedColormap.from_list(
        "white_to_green",
        ["#FFFFFF", "#E8F4EA", "#B9DFC1", "#7DC08A", "#3C8246", "#1B5228"],
    )

    vmin = max(0.2, float(np.nanmin(mat.values)) - 0.05)

    im = ax.imshow(mat.values, cmap=cmap, aspect="auto", vmin=vmin, vmax=1.0)

    for i in range(n_m):
        for j in range(n_t):
            v = mat.values[i, j]
            if np.isnan(v):
                continue

            ax.text(
                j,
                i,
                f"{v:.2f}".lstrip("0"),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if v > 0.75 else "#202020",
                weight="bold" if v >= 0.9 else "normal",
            )

    ax.set_xticks(range(n_t))
    ax.set_xticklabels(triplet_order, rotation=55, ha="right", fontsize=9.5)

    for tick, triplet in zip(ax.get_xticklabels(), triplet_order):
        tick.set_color(LANG_COLORS.get(resp_lang_of(triplet), "#333333"))
        tick.set_fontweight("bold")

    ax.set_yticks(range(n_m))
    ax.set_yticklabels(model_order, fontsize=10)

    for tick, model in zip(ax.get_yticklabels(), model_order):
        fam = FAMILY_OF.get(model, "Meta")
        tick.set_color(FAMILY_COLOR.get(fam, "#333333"))

    blocks = []
    cur = 0

    for cls in MISMATCH_ORDER:
        members = [t for t in triplet_order if triplet_to_class(t) == cls]
        if members:
            blocks.append((cls, cur, cur + len(members) - 1))
            cur += len(members)

    for _, _, end in blocks[:-1]:
        ax.axvline(end + 0.5, color="white", lw=2.5)
        ax.axvline(end + 0.5, color="#333333", lw=0.6, alpha=0.9)

    y_top = -0.8

    for cls, a, b in blocks:
        mid = (a + b) / 2

        ax.annotate(
            "",
            xy=(a - 0.35, y_top),
            xytext=(b + 0.35, y_top),
            arrowprops=dict(arrowstyle="-", lw=1.0, color="#333333"),
            annotation_clip=False,
        )

        ax.text(
            mid,
            y_top - 0.5,
            MISMATCH_LABEL[cls].replace("\n", " "),
            ha="center",
            va="top",
            fontsize=10.5,
            style="italic",
            clip_on=False,
        )

    ax.set_xlabel("")
    ax.set_title("JointSuccess across 27 triplets", fontsize=13.5, pad=30)

    cbar = plt.colorbar(im, ax=ax, shrink=0.55, pad=0.02)
    cbar.set_label("JointSuccess", fontsize=11.5)

    save_fig(outpath)


# =========================================================
# Appendix figure: model role sensitivity
# =========================================================
def make_model_role_sensitivity(long_df: pd.DataFrame, outdir: Path) -> None:
    outpath = outdir / "model_role_sensitivity.png"

    role_map = {
        "instruction_language": "Instruction",
        "content_language": "Content",
        "response_language": "Response",
    }

    rows = []

    for role_key, role_label in role_map.items():
        sub = long_df[long_df["slice_type"] == role_key].copy()
        if sub.empty:
            raise ValueError(f"No rows found for slice_type == {role_key}")

        for model, g in sub.groupby("model"):
            rows.append({
                "model": model,
                "role": role_label,
                "range_js": g["JointSuccess"].max() - g["JointSuccess"].min(),
            })

    ranges = pd.DataFrame(rows)

    order = (
        ranges[ranges["role"] == "Response"]
        .sort_values("range_js", ascending=False)["model"]
        .tolist()
    )

    pivot = ranges.pivot(index="model", columns="role", values="range_js").reindex(order)

    fig, ax = plt.subplots(figsize=(7.9, 7.2))
    y = np.arange(len(pivot))

    offsets = {
        "Instruction": -0.20,
        "Content": 0.0,
        "Response": 0.20,
    }

    markers = {
        "Instruction": "o",
        "Content": "s",
        "Response": "^",
    }

    for role in ["Instruction", "Content", "Response"]:
        vals = pivot[role].values
        yy = y + offsets[role]

        ax.scatter(
            vals,
            yy,
            s=35,
            marker=markers[role],
            color=ROLE_COLOR[role],
            label=role,
            zorder=3,
        )

        for yi, val in zip(yy, vals):
            ax.plot(
                [0, val],
                [yi, yi],
                color=ROLE_COLOR[role],
                linewidth=0.8,
                alpha=0.35,
                zorder=1,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index, fontsize=9.8)

    for tick, model in zip(ax.get_yticklabels(), pivot.index):
        tick.set_color(FAMILY_COLOR.get(FAMILY_OF.get(model, "Meta"), "#333333"))

    ax.invert_yaxis()

    ax.set_xlabel("Range in JS across language values")
    ax.set_xlim(0, max(0.22, float(np.nanmax(pivot.values)) + 0.025))

    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="lower right", frameon=False, fontsize=10)

    plt.tight_layout()
    save_fig(outpath)


# =========================================================
# Appendix figure: Chinese response LC rank
# =========================================================
def make_zh_lang_rank(long_df: pd.DataFrame, outdir: Path) -> None:
    outpath = outdir / "zh_lang_rank.png"

    sub = long_df[
        (long_df["slice_type"] == "response_language")
        & (long_df["slice_value"] == "zh")
    ].copy()

    if sub.empty:
        raise ValueError(
            "No rows found where slice_type == 'response_language' and slice_value == 'zh'."
        )

    sub = sub.sort_values("LangCorrect", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.4, 6.3))
    y = np.arange(len(sub))

    bar_colors = [
        FAMILY_COLOR.get(FAMILY_OF.get(model, "Meta"), "#888888")
        for model in sub["model"]
    ]

    ax.barh(
        y,
        sub["LangCorrect"],
        color=bar_colors,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.92,
    )

    for yi, v in zip(y, sub["LangCorrect"]):
        ax.text(v + 0.008, yi, f"{v:.2f}", va="center", fontsize=9.5, color="#222222")

    mean_val = sub["LangCorrect"].mean()
    ax.axvline(mean_val, linestyle="--", color="#555555", lw=1)

    ax.text(
        mean_val,
        len(sub) + 0.2,
        f"mean ({mean_val:.2f})",
        fontsize=9.5,
        color="#555555",
        ha="center",
        style="italic",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(sub["model"], fontsize=9.8)

    for tick, model in zip(ax.get_yticklabels(), sub["model"]):
        tick.set_color(FAMILY_COLOR.get(FAMILY_OF.get(model, "Meta"), "#333333"))

    ax.set_xlim(max(0, float(sub["LangCorrect"].min()) - 0.05), 1.02)
    ax.set_xlabel("LC when response language = zh")

    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_fig(outpath)


# =========================================================
# Main
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--long",
        default="tri_main_analysis_long.csv",
        help="Path to tri_main_analysis_long.csv",
    )

    parser.add_argument(
        "--outdir",
        default="pic",
        help="Output directory",
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    long_df = load_long_data(Path(args.long))

    make_axes_effects(outdir)
    make_task_response(outdir)
    make_failure_profile(long_df, outdir)
    make_triplet_heatmap(long_df, outdir)
    make_model_role_sensitivity(long_df, outdir)
    make_zh_lang_rank(long_df, outdir)

    print("All figures saved.")


if __name__ == "__main__":
    main()
