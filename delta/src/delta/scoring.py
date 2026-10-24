"""Policy-level DELTA scores from per-(rater, policy, case) judgments.

A policy's DELTA score is the average disagreement its raters show about how
it applies to a set of cases. The pipeline is:

    probs.csv                       one row per rater x policy x case x prompt
        |   load_probs
        v
    per-(rater, policy, case) revealed probability
        |   revealed_probability     pairs the comply and violate prompts
        v
    per-(policy, case) disagreement
        |   disagreement             4 * mean * (1 - mean) over raters
        v
    per-policy DELTA score
        |   aggregation over cases
        v
    scores/delta_scores.csv
"""

import os

import numpy as np
import pandas as pd

__all__ = [
    "load_probs",
    "revealed_probability",
    "disagreement",
    "compute_delta_scores",
]


def load_probs(results_dir, input_file="probs.csv"):
    """Load a probs file into columns: rater, policy, case, prompt, prob_yes.

    Accepts three on-disk shapes:

    - ``custom_id, prob_yes`` -- the canonical pipeline output. The custom_id
      is parsed into its component fields, whose names are read off the first
      row (e.g. ``rater-0_policy-3_case-12_prompt-1``).
    - ``custom_id, answer, prob`` -- one row per answer token; the ``answer==1``
      rows are kept and ``prob`` renamed to ``prob_yes``.
    - Already-parsed columns, which pass through untouched.

    Read-only; writes nothing.
    """
    df = pd.read_csv(os.path.join(results_dir, input_file))

    if "answer" in df.columns:
        df = df[df["answer"] == 1].copy()
        if "prob" in df.columns:
            df.rename(columns={"prob": "prob_yes"}, inplace=True)
        df.drop(columns=["answer"], inplace=True)

    if "custom_id" in df.columns:
        first_id = df["custom_id"].iloc[0]
        column_names = [part.split("-")[0] for part in first_id.split("_")]
        pattern = "_".join([f"{col}-(\\d+)" for col in column_names])
        df[column_names] = df["custom_id"].str.extract(pattern).astype(int)
        df.drop(columns=["custom_id"], inplace=True)

    return df


def revealed_probability(p_comply, p_violate):
    """Collapse a rater's comply/violate answers into one probability.

    Each rater answers two framings of the same question: does the policy
    permit this case, and does it prohibit it. In principle the second is the
    complement of the first, but a policy that is merely silent on a case
    draws a "no" to both. Averaging the positive framing with the complement
    of the negative one keeps that asymmetry visible: a case the policy does
    not address lands near 0.5 rather than at either extreme.

    Elementwise; accepts scalars or same-shaped arrays/Series.
    """
    return (p_comply + (1 - p_violate)) / 2


def disagreement(mean):
    """Per-(policy, case) disagreement: ``4 * mean * (1 - mean)``.

    ``mean`` is taken over raters' revealed probabilities for one policy-case
    pair. Reading each rater's revealed probability as their probability of
    issuing the verdict "permitted", ``2 * mean * (1 - mean)`` is the
    probability that two independently sampled verdicts conflict (Gini
    impurity); the score is that probability normalized by its maximum of
    one half, so it is 0 under total consensus and 1 when the pooled verdict
    is a coin flip.

    By the law of total variance it splits exactly into a within-rater and a
    between-rater component::

        4*mean*(1 - mean) = 4*avg(x*(1 - x)) + 4*var(x)

    where ``x`` are the per-rater revealed probabilities and ``var`` is the
    population variance. The first term is self-disagreement (a rater whose
    verdict is undetermined conflicts with a redraw of themselves), the
    second is the extra disagreement attributable to raters differing.
    :func:`compute_delta_scores` reports both alongside the score.

    Elementwise; accepts scalars or same-shaped arrays/Series.
    """
    return 4 * mean * (1 - mean)


def _aggregate_over_cases(per_cell, weighting, value_cols=("score",)):
    """Average per-(policy, case) disagreement into one score per policy.

    Cells with NaN disagreement are dropped as missing data. Every column in
    ``value_cols`` is aggregated with the same weights, which are always
    derived from ``score`` — so component columns still sum to the score.
    """
    valid = per_cell.dropna(subset=["score"]).copy()
    value_cols = list(value_cols)

    if weighting == "mean":
        return valid.groupby("policy")[value_cols].mean().reset_index()

    if weighting == "case_range":
        case_weights = (
            valid.groupby("case")["score"].apply(np.ptp).reset_index(name="weight")
        )
        merged = valid.merge(case_weights, on="case")
        return (
            merged.groupby("policy")
            .apply(
                lambda g: pd.Series(
                    {c: np.average(g[c], weights=g["weight"] + 1e-5) for c in value_cols}
                ),
                include_groups=False,
            )
            .reset_index()
        )

    raise ValueError(
        f"Unknown weighting: {weighting!r}. Expected 'mean' or 'case_range'."
    )


def compute_delta_scores(
    results_dir,
    has_negative_questions=True,
    weighting="case_range",
    output_name="delta_scores.csv",
    input_file="probs.csv",
):
    """Compute a DELTA score per policy and write them to disk.

    Args:
        results_dir: directory holding ``input_file``. Scores are written to
            ``{results_dir}/scores/{output_name}``.
        has_negative_questions: True when the run used paired comply/violate
            prompts, so even prompt indices are the comply framing and odd
            ones the violate framing. Set False for single-prompt runs, where
            ``prob_yes`` is used directly.
        weighting: how to average case disagreement into a policy score.

            ``'case_range'`` (default) weights each case by the range of its
            disagreement across policies, so cases that discriminate between
            policies count for more than cases every policy handles alike.
            Because those weights depend on which policies are in the run, a
            case-range score is comparable only within a single run: scoring a
            subset of policies gives different weights and so different scores.

            ``'mean'`` weights every case equally. Use it when case-range
            weights are not meaningful -- for instance when raters each see
            only a couple of policies, or when cases are subsampled.

        output_name: filename written under ``{results_dir}/scores/``.
        input_file: filename read from ``results_dir``.

    Returns:
        DataFrame with columns ``policy``, ``score``, ``within`` and
        ``between``, where a score of 0 means total consensus and higher
        values mean more disagreement. ``within`` and ``between`` are the
        exact decomposition of ``score`` described in :func:`disagreement`;
        they sum to it.
    """
    df = load_probs(results_dir, input_file)

    if has_negative_questions:
        comply = df[df["prompt"] % 2 == 0].copy()
        comply["prompt"] = comply["prompt"] // 2
        violate = df[df["prompt"] % 2 == 1].copy()
        violate["prompt"] = violate["prompt"] // 2
        merged = comply.merge(
            violate,
            on=["policy", "case", "rater", "prompt"],
            suffixes=("_comply", "_violate"),
        )
        merged["prob_yes"] = revealed_probability(
            merged["prob_yes_comply"], merged["prob_yes_violate"]
        )
        df = merged[["policy", "case", "rater", "prompt", "prob_yes"]]

    if "prompt" in df.columns:
        df = df.groupby(["policy", "case", "rater"])["prob_yes"].mean().reset_index()

    def case_score(probs):
        x = probs.to_numpy(dtype=float)
        within = float(4 * np.mean(x * (1 - x)))
        between = float(4 * np.var(x))
        return pd.Series(
            {"score": within + between, "within": within, "between": between}
        )

    per_cell = (
        df.groupby(["policy", "case"])["prob_yes"]
        .apply(case_score)
        .unstack()
        .reset_index()
    )
    policy_scores = _aggregate_over_cases(
        per_cell, weighting, ("score", "within", "between")
    )

    scores_dir = os.path.join(results_dir, "scores")
    os.makedirs(scores_dir, exist_ok=True)
    policy_scores.to_csv(os.path.join(scores_dir, output_name), index=False)
    return policy_scores
