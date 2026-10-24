"""DELTA: measuring ambiguity in prescriptive text as interpretive disagreement.

A policy is ambiguous to the extent that a diverse set of raters disagree
about what it implies for concrete cases. Pair a policy with cases, ask
raters whether each case is permitted, and measure how far apart they land.

    from delta import compute_delta_scores

    scores = compute_delta_scores("results/my_experiment")

See the README for generating cases and rater personas, and for running the
judgments through a language model.
"""

from .scoring import (
    compute_delta_scores,
    disagreement,
    load_probs,
    revealed_probability,
)

__all__ = [
    "compute_delta_scores",
    "disagreement",
    "load_probs",
    "revealed_probability",
]

__version__ = "0.3.0"
