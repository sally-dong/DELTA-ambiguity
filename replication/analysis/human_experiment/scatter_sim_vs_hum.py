"""Combined scatter of human-rater vs. LLM-rater DELTA score, one point per policy.

Writes a single PDF with two side-by-side panels:
  - left  — raw scores, Pearson r
  - right — ranks,      Spearman ρ
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import FIG_WIDTH, set_style

HERE = Path(__file__).resolve().parent
SCORES_PATH = HERE / "all_scores.csv"
OUT_PATH = HERE / "scatter_sim_vs_hum_delta.pdf"

X_COL, Y_COL = "sim_delta", "hum_delta"
RAW_LIMITS = (0.40, 1.00)
RAW_TICK_STEP = 0.2
POINT_COLOR = "tab:blue"


def draw_panel(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    stat_name: str,
    stat_value: float,
    xlabel: str,
    ylabel: str,
    subtitle: str,
    limits: tuple[float, float],
    ticks: list[float],
) -> None:
    ax.plot(limits, limits, color="0.7", lw=0.7, ls=(0, (4, 3)), zorder=0)
    ax.scatter(x, y, s=14, color=POINT_COLOR, edgecolor="none", zorder=2)

    ax.text(
        0.96, 0.05,
        f"{stat_name} = {stat_value:+.2f}",
        transform=ax.transAxes, va="bottom", ha="right",
        fontsize=5.5,
        bbox=dict(facecolor="white", edgecolor="none", pad=2.0, alpha=0.85),
    )

    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(axis="both", length=3, pad=2)

    ax.set_xlabel(xlabel, labelpad=3)
    ax.set_ylabel(ylabel, labelpad=3)
    ax.set_title(subtitle, pad=2, fontsize=plt.rcParams["axes.labelsize"])

    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_linewidth(0.8)


def main():
    df = pd.read_csv(SCORES_PATH).sort_values("policy").reset_index(drop=True)
    x = df[X_COL].to_numpy()
    y = df[Y_COL].to_numpy()
    n = len(df)

    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    print(f"n={n}")
    print(f"  Pearson  r = {pearson.statistic:+.3f}  (p = {pearson.pvalue:.3f})")
    print(f"  Spearman ρ = {spearman.statistic:+.3f}  (p = {spearman.pvalue:.3f})")

    set_style()

    fig, (ax_raw, ax_rank) = plt.subplots(
        1, 2, figsize=(FIG_WIDTH, 1.9),
        layout="constrained",
    )

    draw_panel(
        ax_raw, x, y,
        stat_name="Pearson r", stat_value=pearson.statistic,
        xlabel="LLM-rater", ylabel="Human-rater",
        subtitle="Raw scores",
        limits=RAW_LIMITS,
        ticks=list(np.arange(RAW_LIMITS[0], RAW_LIMITS[1] + 1e-9, RAW_TICK_STEP)),
    )

    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    draw_panel(
        ax_rank, rx, ry,
        stat_name="Spearman ρ", stat_value=spearman.statistic,
        xlabel="LLM-rater", ylabel="",
        subtitle="Ranks",
        limits=(0, n + 1),
        ticks=[1, n // 2, n],
    )

    fig.suptitle("LLM- vs. human-DELTA score")

    # CreationDate=None keeps the PDF byte-identical across runs.
    fig.savefig(OUT_PATH, bbox_inches="tight", pad_inches=0.02,
                metadata={"CreationDate": None})
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
