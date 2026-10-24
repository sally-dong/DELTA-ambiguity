"""Tests for the DELTA scoring formula.

These exercise pure functions over a small probs table, so they need no API
access and no model.
"""

import numpy as np
import pandas as pd
import pytest

from delta import compute_delta_scores, disagreement, load_probs, revealed_probability


# --- revealed_probability -------------------------------------------------


def test_revealed_probability_consistent_answers():
    # "yes it complies" + "no it does not violate" -> unambiguously permitted
    assert revealed_probability(1.0, 0.0) == 1.0
    # the mirror image -> unambiguously prohibited
    assert revealed_probability(0.0, 1.0) == 0.0


def test_revealed_probability_policy_silent_on_case():
    # "no it does not comply" AND "no it does not violate": the policy simply
    # does not speak to this case, which should land mid-scale rather than at
    # either extreme.
    assert revealed_probability(0.0, 0.0) == 0.5
    # the contradictory rater lands in the same place
    assert revealed_probability(1.0, 1.0) == 0.5


def test_revealed_probability_is_elementwise():
    out = revealed_probability(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert np.allclose(out, [1.0, 0.0])


# --- disagreement ---------------------------------------------------------


def test_disagreement_is_zero_under_consensus():
    # everyone issues the same verdict: two sampled verdicts never conflict
    assert disagreement(1.0) == 0.0
    assert disagreement(0.0) == 0.0


def test_disagreement_is_maximal_at_a_half():
    # a coin-flip pooled verdict: two sampled verdicts conflict half the
    # time, the maximum, so the normalized score is 1
    assert disagreement(0.5) == 1.0


def test_disagreement_credits_shared_uncertainty():
    # every rater answers 0.5, so they agree with each other but none can tell
    # whether the policy applies. That is still ambiguity: each rater's own
    # verdict is a coin flip, so sampled verdicts still conflict.
    assert disagreement(0.5) == 1.0


def test_disagreement_is_symmetric_about_a_half():
    assert disagreement(0.3) == pytest.approx(disagreement(0.7))


# --- end to end -----------------------------------------------------------


def _write_probs(path, rows):
    pd.DataFrame(rows, columns=["rater", "policy", "case", "prompt", "prob_yes"]).to_csv(
        path / "probs.csv", index=False
    )


def test_unambiguous_policy_scores_below_contested_one(tmp_path):
    """Policy 0 draws unanimous judgments; policy 1 splits its raters."""
    rows = []
    for case in range(2):
        for rater in range(4):
            # policy 0: every rater says "permitted" (comply yes, violate no)
            rows += [[rater, 0, case, 0, 1.0], [rater, 0, case, 1, 0.0]]
            # policy 1: raters split down the middle
            verdict = float(rater % 2)
            rows += [[rater, 1, case, 0, verdict], [rater, 1, case, 1, 1.0 - verdict]]
    _write_probs(tmp_path, rows)

    scores = compute_delta_scores(str(tmp_path), weighting="mean")
    by_policy = dict(zip(scores["policy"], scores["score"]))

    assert by_policy[0] == pytest.approx(0.0)
    assert by_policy[1] == pytest.approx(1.0)


def test_score_decomposes_into_within_and_between(tmp_path):
    """Polarized and undetermined cells score alike, but the within/between
    columns tell them apart and sum exactly to the score."""
    rows = []
    for rater in range(4):
        # policy 0: every rater's revealed judgment is 0.5 (policy silent)
        rows += [[rater, 0, 0, 0, 0.0], [rater, 0, 0, 1, 0.0]]
        # policy 1: raters split down the middle between 0 and 1
        verdict = float(rater % 2)
        rows += [[rater, 1, 0, 0, verdict], [rater, 1, 0, 1, 1.0 - verdict]]
    _write_probs(tmp_path, rows)

    scores = compute_delta_scores(str(tmp_path), weighting="mean")
    by_policy = scores.set_index("policy")

    # both cells have mean 0.5, so both score 4 * 0.5 * 0.5 = 1 ...
    assert by_policy.loc[0, "score"] == pytest.approx(1.0)
    assert by_policy.loc[1, "score"] == pytest.approx(1.0)
    # ... but the decomposition tells them apart
    assert by_policy.loc[0, "within"] == pytest.approx(1.0)
    assert by_policy.loc[0, "between"] == pytest.approx(0.0)
    assert by_policy.loc[1, "within"] == pytest.approx(0.0)
    assert by_policy.loc[1, "between"] == pytest.approx(1.0)
    # identity: score = within + between
    assert np.allclose(
        by_policy["score"], by_policy["within"] + by_policy["between"]
    )


def test_scores_are_written_to_disk(tmp_path):
    rows = [[r, 0, 0, p, 1.0 if p == 0 else 0.0] for r in range(3) for p in (0, 1)]
    _write_probs(tmp_path, rows)

    returned = compute_delta_scores(str(tmp_path))
    written = pd.read_csv(tmp_path / "scores" / "delta_scores.csv")

    pd.testing.assert_frame_equal(returned.reset_index(drop=True), written)


def test_case_range_downweights_undiscriminating_cases(tmp_path):
    """A case every policy handles identically should not move the ranking."""
    rows = []
    for rater in range(4):
        # case 0 separates the policies
        rows += [[rater, 0, 0, 0, 1.0], [rater, 0, 0, 1, 0.0]]
        split = float(rater % 2)
        rows += [[rater, 1, 0, 0, split], [rater, 1, 0, 1, 1.0 - split]]
        # case 1 is contested identically under both policies, so it carries
        # no information about which policy is more ambiguous
        for policy in (0, 1):
            same = float(rater % 2)
            rows += [[rater, policy, 1, 0, same], [rater, policy, 1, 1, 1.0 - same]]
    _write_probs(tmp_path, rows)

    flat = compute_delta_scores(str(tmp_path), weighting="mean", output_name="a.csv")
    ranged = compute_delta_scores(
        str(tmp_path), weighting="case_range", output_name="b.csv"
    )

    gap_flat = flat.set_index("policy")["score"].diff().iloc[-1]
    gap_ranged = ranged.set_index("policy")["score"].diff().iloc[-1]
    # both orderings agree, but case_range separates the policies more sharply
    assert gap_flat > 0 and gap_ranged > gap_flat


def test_rejects_unknown_weighting(tmp_path):
    _write_probs(tmp_path, [[0, 0, 0, 0, 1.0], [0, 0, 0, 1, 0.0]])
    with pytest.raises(ValueError, match="Unknown weighting"):
        compute_delta_scores(str(tmp_path), weighting="nonsense")


# --- load_probs -----------------------------------------------------------


def test_load_probs_parses_custom_id(tmp_path):
    pd.DataFrame(
        {"custom_id": ["rater-0_policy-1_case-2_prompt-1"], "prob_yes": [0.25]}
    ).to_csv(tmp_path / "probs.csv", index=False)

    df = load_probs(str(tmp_path))

    assert set(df.columns) == {"rater", "policy", "case", "prompt", "prob_yes"}
    assert df.iloc[0][["rater", "policy", "case", "prompt"]].tolist() == [0, 1, 2, 1]


def test_load_probs_keeps_yes_rows_of_long_format(tmp_path):
    pd.DataFrame(
        {
            "custom_id": ["rater-0_policy-0_case-0_prompt-0"] * 2,
            "answer": [0, 1],
            "prob": [0.3, 0.7],
        }
    ).to_csv(tmp_path / "probs.csv", index=False)

    df = load_probs(str(tmp_path))

    assert len(df) == 1
    assert df.iloc[0]["prob_yes"] == 0.7
