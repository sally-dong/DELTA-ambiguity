"""Merge the three judge models' ratings_raw.csv into one ratings_merged.csv.

A statement's baseline score is the plain arithmetic mean of the three judges'
parsed integer ratings (no z-score standardization). The merged column is
named after the rubric folder (`ambiguity` or `delta`) unless --col overrides
it; `score_human_experiment.py` reads it under that name.

We deliberately use the parsed integer rating rather than a logprob-based
expected rating for gpt-5.2: on the delta rubric it prepends a
"**Consequential..." preamble instead of emitting a bare digit, so the
first-token logprob distribution has no mass on 1-7 and a logprob-weighted
rating collapses to a uniform 4.0.

Usage (from replication/):
    python analysis/llm_judge/merge_ratings.py \
        --input_folder analysis/human_experiment/llm_ratings/ambiguity
"""

import argparse
import os

import pandas as pd

JUDGES = {
    "gpt-5.2": "rating_gpt52",
    "gpt-5": "rating_gpt5",
    "gemini-3-pro-preview": "rating_gemini",
}

KEYS = ["candidate_id", "iteration", "round_id"]


def merge(input_folder: str, col: str | None = None) -> pd.DataFrame:
    """Inner-join the per-model ratings on statement identity and average them."""
    df = None
    for model, rating_col in JUDGES.items():
        part = pd.read_csv(f"{input_folder}/{model}/ratings_raw.csv")
        part = part[KEYS + ["rating"]].rename(columns={"rating": rating_col})
        df = part if df is None else df.merge(part, on=KEYS, how="inner")

    df["overall_rating"] = df[list(JUDGES.values())].mean(axis=1)

    out_col = col or os.path.basename(os.path.normpath(input_folder))
    out = df[["candidate_id", "overall_rating"]].rename(columns={"overall_rating": out_col})
    out_path = f"{input_folder}/ratings_merged.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}  ({len(out)} rows)")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Rubric folder containing the per-model subfolders")
    parser.add_argument("--col", type=str, default=None,
                        help="Name for the merged rating column "
                             "(default: the rubric folder's basename)")
    args = parser.parse_args()
    merge(args.input_folder, args.col)
