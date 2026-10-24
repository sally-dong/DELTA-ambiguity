"""Two-panel rank-rank scatterplot for the adversarial dataset:
    left  — LLM-DELTA score rank vs LLM-judge ambiguity-rating rank
    right — LLM-DELTA score rank vs LLM-judge disagreement-rating rank

Inputs:
  results/adversarial/scores/delta_scores.csv
                              LLM-DELTA score per policy (flat mean over
                              cases), joined to candidate ids through
                              data/adversarial/policies.json
  llm_ratings/<rubric>/       one directory per LLM-as-a-judge rubric, each
                              holding one subdirectory per judge model with
                              that model's ratings_raw.csv (1--7 scale)

The two rubrics are `ambiguity` (vagueness read off the policy text alone) and
`delta` (the prompt that mirrors the DELTA mechanism by asking the model to
reason about stakeholder disagreement). Neither is the LLM-DELTA score. A
rubric's rating is the mean across judge models.

Read ratings_raw.csv rather than results.csv: a model that fails to return a
parseable rating is absent from results.csv, and the repaired value is written
back only to ratings_raw.csv.

Writes joined.csv and adversarial_ratings_republic.pdf (the name the paper
includes) to the same folder.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import FIG_WIDTH, set_style

ADV_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADV_DIR.parents[1]
SCORES_PATH = REPO_ROOT / "results" / "adversarial" / "scores" / "delta_scores.csv"
POLICIES_PATH = REPO_ROOT / "data" / "adversarial" / "policies.json"
RUBRICS = ("ambiguity", "delta")
TITLE = "LLM-as-a-judge ratings vs LLM-DELTA scores"

# The rating runs covered several candidate domains; the paper reports this one.
# Their candidate_ids are topic-prefixed, e.g. "republic-consensus1".
TOPIC = "republic"


def load_rubric(rubric: str) -> pd.Series:
    """Mean rating across judge models, indexed by unprefixed candidate_id."""
    per_model = [
        pd.read_csv(p).set_index("candidate_id")["rating"]
        for p in sorted(ADV_DIR.glob(f"llm_ratings/{rubric}/*/ratings_raw.csv"))
    ]
    mean = pd.concat(per_model, axis=1).mean(axis=1)
    mean = mean[mean.index.str.startswith(f"{TOPIC}-")]
    mean.index = mean.index.str.removeprefix(f"{TOPIC}-")
    return mean.rename(f"rating_{rubric}")


def build_table() -> pd.DataFrame:
    # delta_scores.csv indexes policies by position in policies.json.
    policies = pd.DataFrame(json.loads(POLICIES_PATH.read_text(encoding="utf-8")))
    scores = pd.read_csv(SCORES_PATH).set_index("policy")["score"]
    df = policies[["candidate_id", "text"]].assign(delta_score=scores).set_index("candidate_id")
    for rubric in RUBRICS:
        df = df.join(load_rubric(rubric), how="outer")
    return df.sort_index().reset_index()


def draw_panel(ax, df, *, y_col: str, n: int, xlabel: str, ylabel: str, subtitle: str):
    sns.scatterplot(
        data=df, x="rank_delta_score", y=y_col,
        ax=ax, s=14, color="tab:blue", edgecolor="none", zorder=2,
    )
    ax.set_xlim(0, n + 1)
    ax.set_ylim(0, n + 1)
    ax.set_xticks([1, n // 2, n])
    ax.set_yticks([1, n // 2, n])
    ax.set_xlabel(xlabel, labelpad=3)
    ax.set_ylabel(ylabel, labelpad=3)
    ax.set_title(subtitle, pad=2, fontsize=plt.rcParams["axes.labelsize"])
    ax.tick_params(axis="both", length=3, pad=2)
    sns.despine(ax=ax)
    ax.spines["bottom"].set_linewidth(0.8)


def main():
    set_style()

    df = build_table()
    out_csv = ADV_DIR / "joined.csv"
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}  ({len(df)} rows, {len(df.columns)} cols)")

    df["rank_delta_score"] = df["delta_score"].rank(method="average")
    df["rank_rating_ambiguity"] = df["rating_ambiguity"].rank(method="average")
    df["rank_rating_delta"] = df["rating_delta"].rank(method="average")
    n = len(df)

    fig, (ax_amb, ax_del) = plt.subplots(
        1, 2, figsize=(FIG_WIDTH, 1.9), layout="constrained"
    )
    draw_panel(
        ax_amb, df, y_col="rank_rating_ambiguity", n=n,
        xlabel="LLM-DELTA score rank", ylabel="LLM-as-a-judge rank",
        subtitle="Ambiguity rating",
    )
    draw_panel(
        ax_del, df, y_col="rank_rating_delta", n=n,
        xlabel="LLM-DELTA score rank", ylabel="",
        subtitle="Disagreement rating",
    )
    fig.suptitle(TITLE)

    out_pdf = ADV_DIR / "adversarial_ratings_republic.pdf"
    # CreationDate=None keeps the PDF byte-identical across runs.
    fig.savefig(out_pdf, pad_inches=0.02, bbox_inches="tight",
                metadata={"CreationDate": None})
    plt.close(fig)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
