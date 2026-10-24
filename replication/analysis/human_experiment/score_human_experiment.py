"""Score the university-speech policies under DELTA, for both rater populations.

Aggregation is a flat mean over cases, not the library's case-range weighting:
each human rated only two policies, so a case's disagreement range across
policies is estimated from a different subset of raters for every case and is
not comparable between them.

Both populations judge the same 20 policies against the same 20 cases:
  probs.csv        simulated (LLM) raters, in results/human_experiment/
  human_probs.csv  human raters, alongside this script

Writes all_scores.csv: per-policy DELTA for both populations, alongside the
baselines the paper compares against — the dictionary concreteness measure and
the two LLM-as-a-judge rubrics — and prints the pairwise correlation matrices.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from delta import load_probs
from delta.scoring import disagreement, revealed_probability

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SIM_DIR = REPO_ROOT / "results" / "human_experiment"


def delta_scores(input_dir: Path, input_file: str, col: str) -> pd.DataFrame:
    """Per-policy DELTA score, aggregating over cases with a flat mean."""
    df = load_probs(str(input_dir), input_file)
    comply = df[df["prompt"] % 2 == 0].copy()
    comply["prompt"] = comply["prompt"] // 2
    violate = df[df["prompt"] % 2 == 1].copy()
    violate["prompt"] = violate["prompt"] // 2
    merged = comply.merge(
        violate, on=["policy", "case", "rater", "prompt"],
        suffixes=("_comply", "_violate"),
    )
    merged["prob_yes"] = revealed_probability(
        merged["prob_yes_comply"], merged["prob_yes_violate"]
    )
    df = merged.groupby(["policy", "case", "rater"])["prob_yes"].mean().reset_index()

    def case_score(probs):
        return disagreement(float(np.mean(probs)))

    per_cell = (
        df.groupby(["policy", "case"])["prob_yes"]
        .apply(case_score)
        .reset_index(name="score")
    )
    return per_cell.groupby("policy")["score"].mean().reset_index(name=col)


def main():
    # Dictionary concreteness, from score_policy_concreteness.py (policy p{i} -> i-1).
    conc = pd.read_csv(HERE / "policy_concreteness.csv")
    conc["policy"] = conc["candidate_id"].str.lstrip("p").astype(int) - 1
    conc["concreteness_mean"] = conc["score_sum"] / conc["matched"]

    df = (
        delta_scores(SIM_DIR, "probs.csv", "sim_delta")
        .merge(delta_scores(HERE, "human_probs.csv", "hum_delta"), on="policy")
        .merge(conc[["policy", "concreteness_mean"]], on="policy")
    )

    # LLM-as-a-judge rubrics. `delta` is the rubric prompted to reason about
    # stakeholder disagreement; it is a judge rating, not a DELTA score.
    for rubric, col in (("ambiguity", "judge_ambiguity"), ("delta", "judge_disagreement")):
        rated = pd.read_csv(HERE / "llm_ratings" / rubric / "ratings_merged.csv")
        rated = rated.rename(columns={rubric: col})
        rated["policy"] = rated["candidate_id"].str[1:].astype(int) - 1
        df = df.merge(rated[["policy", col]], on="policy", how="left")

    score_cols = [
        "sim_delta", "hum_delta",
        "concreteness_mean", "judge_ambiguity", "judge_disagreement",
    ]
    master_path = HERE / "all_scores.csv"
    df[["policy"] + score_cols].to_csv(master_path, index=False)
    print(f"Wrote {master_path}")

    def fmt(m, title):
        print(f"\n=== {title} ===")
        abbr = {c: c.replace("concreteness_mean", "concrete")[:14] for c in score_cols}
        with pd.option_context("display.float_format", "{:+.2f}".format,
                               "display.width", 200, "display.max_columns", 20):
            print(m.rename(index=abbr, columns=abbr).round(2).to_string())

    fmt(df[score_cols].corr(method="pearson"), "Pearson r")
    fmt(df[score_cols].corr(method="spearman"), "Spearman rho")


if __name__ == "__main__":
    main()
