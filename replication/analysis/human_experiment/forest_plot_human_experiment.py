"""Forest plot: Spearman correlation of hum_delta against four comparators,
with bootstrap 95% CIs.

Bootstrap resamples human raters (stratified by
policy × party), recomputing hum_delta each iteration, hold the comparators
fixed. Treats the 20 policies as fixed stimuli; the CI captures
"if we'd seen different humans rating the same policies, how stable is this
correlation?"
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from delta.scoring import disagreement, revealed_probability

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import FIG_WIDTH, set_style

HERE = Path(__file__).resolve().parent
SCORES_PATH = HERE / "all_scores.csv"
PARTY_PATH = HERE / "rater_party.csv"
PROBS_PATH = HERE / "human_probs.csv"
OUT_PATH = HERE / "forest_plot_human_experiment.pdf"

TARGET = "hum_delta"
# (column, label, sign). sign=-1 flips the comparator so all axes point the
# same way (higher = more ambiguous). Concreteness measures the opposite of
# ambiguity, so we negate it before correlating; the plotted ρ has flipped
# sign accordingly. CIs come out clean (vs. taking |ρ|, which folds the
# bootstrap distribution and biases it upward for true ρ near zero).
COMPARATORS = [
    ("sim_delta",          "LLM\nDELTA score",       +1),
    ("concreteness_mean", "dictionary\nmeasure",    -1),
    ("judge_ambiguity",    "LLM judge:\nambiguity",  +1),
    ("judge_disagreement", "LLM judge:\ndisagreement", +1),
]

N_BOOT = 2000
BOOT_SEED = 0
CI_LO, CI_HI = 2.5, 97.5


def spearman(a, b):
    """Spearman ρ as Pearson on ranks (avoids the scipy dependency)."""
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return np.corrcoef(ra, rb)[0, 1]


def build_rater_party_map():
    """Party per rater, indexed by the integer rater id used in
    human_probs.csv. rater_party.csv is written by convert_human_ratings.py
    alongside human_probs.csv."""
    party = pd.read_csv(PARTY_PATH).sort_values("rater")
    assert (party["rater"].to_numpy() == np.arange(len(party))).all(), \
        "rater ids must be contiguous from 0"
    return party["party"].to_numpy()  # indexed by integer rater id


def build_revealed_array():
    """Dense (n_policies, n_cases, n_raters) array of per-rater revealed-prob.

    revealed_prob_ipc = ( permit_ipc + (1 − prohibit_ipc) ) / 2,
    with NaN where the rater didn't see (policy, case).
    """
    df = pd.read_csv(PROBS_PATH)
    comply = df[df["prompt"] == 0][["rater", "policy", "case", "prob_yes"]].rename(
        columns={"prob_yes": "permit"}
    )
    violate = df[df["prompt"] == 1][["rater", "policy", "case", "prob_yes"]].rename(
        columns={"prob_yes": "prohibit"}
    )
    rp = comply.merge(violate, on=["rater", "policy", "case"], how="inner")
    rp["revealed"] = revealed_probability(rp["permit"], rp["prohibit"])

    policies = sorted(rp["policy"].unique())
    cases = sorted(rp["case"].unique())
    raters = sorted(rp["rater"].unique())
    p_idx = {p: i for i, p in enumerate(policies)}
    c_idx = {c: i for i, c in enumerate(cases)}
    r_idx = {r: i for i, r in enumerate(raters)}

    arr = np.full((len(policies), len(cases), len(raters)), np.nan)
    arr[
        rp["policy"].map(p_idx).to_numpy(),
        rp["case"].map(c_idx).to_numpy(),
        rp["rater"].map(r_idx).to_numpy(),
    ] = rp["revealed"].to_numpy()
    return arr, policies, raters


def policy_delta_scores(arr):
    """Per-policy hum_delta on the unweighted-mean aggregation:
        per-cell  = 4 * mean_r rp * (1 − mean_r rp)
        per-policy = mean over cases.
    """
    with np.errstate(invalid="ignore"):
        mean_r = np.nanmean(arr, axis=2)
    per_cell = disagreement(mean_r)
    return np.nanmean(per_cell, axis=1)


def main():
    scores = pd.read_csv(SCORES_PATH).sort_values("policy").reset_index(drop=True)
    arr, policy_ids, rater_ids = build_revealed_array()
    rater_to_party = build_rater_party_map()
    # rater_to_party is indexed by 0..N-1 in tidy order — same id space as
    # the `rater` integers in human_probs.csv (set up by convert_human_ratings.py).
    assert len(rater_to_party) >= max(rater_ids) + 1, "rater id space mismatch"
    party_by_rater = rater_to_party[np.array(rater_ids)]
    unique_parties = sorted(set(party_by_rater))

    # Sanity-check: scores file row order must match arr's policy axis.
    assert list(scores["policy"]) == policy_ids, "policy ordering mismatch"

    # Pre-organize per-policy (party → array of rater rows). Each rater either
    # rated all of a policy's cases or none, so coverage is detected on the
    # first case. (A small minority of raters cover only a subset of cases
    # for a policy; nanmean/nanstd handle that.)
    policy_data = []
    for p_i in range(arr.shape[0]):
        # A rater is "in" policy p_i if they rated any case of it.
        valid = np.where(np.any(~np.isnan(arr[p_i]), axis=0))[0]
        parties_for_valid = party_by_rater[valid]
        by_party = []
        for party in unique_parties:
            mask = parties_for_valid == party
            if mask.any():
                # arr[p_i] is (n_cases, n_raters); pick the valid raters for this party.
                by_party.append(arr[p_i][:, valid[mask]])
        policy_data.append(by_party)

    print("Stratified-bootstrap cell sizes (raters per policy × party):")
    print("  policy  " + "  ".join(f"{p:>11}" for p in unique_parties) + "   total")
    for p_i in range(arr.shape[0]):
        sizes = {party: 0 for party in unique_parties}
        for party, party_arr in zip(unique_parties, [None] * len(unique_parties)):
            pass  # placeholder
        # Re-derive sizes for the printout (cheap):
        valid = np.where(np.any(~np.isnan(arr[p_i]), axis=0))[0]
        parties_for_valid = party_by_rater[valid]
        for party in unique_parties:
            sizes[party] = int((parties_for_valid == party).sum())
        cells_row = "  ".join(f"{sizes[party]:>11}" for party in unique_parties)
        print(f"  {p_i:>6}  {cells_row}   {sum(sizes.values()):>5}")

    def bootstrap_delta(rng):
        out = np.empty(arr.shape[0])
        for p_i, by_party in enumerate(policy_data):
            resamples = []
            for party_arr in by_party:
                k = party_arr.shape[1]
                ix = rng.integers(0, k, size=k)
                resamples.append(party_arr[:, ix])
            stacked = np.concatenate(resamples, axis=1)  # (n_cases, n_total)
            with np.errstate(invalid="ignore"):
                mean_r = np.nanmean(stacked, axis=1)
            per_cell = disagreement(mean_r)
            out[p_i] = np.nanmean(per_cell)
        return out

    # Point estimate from the unresampled array, to confirm we reproduce the
    # value in all_scores.csv.
    point_delta = policy_delta_scores(arr)
    target_csv = scores[TARGET].to_numpy()
    print(f"\nReproduction check: max |bootstrap_delta − csv| = "
          f"{np.max(np.abs(point_delta - target_csv)):.2e}")

    rng = np.random.default_rng(BOOT_SEED)
    rho_boot = {col: np.empty(N_BOOT) for col, _, _ in COMPARATORS}
    for b in range(N_BOOT):
        leg = bootstrap_delta(rng)
        for col, _, sign in COMPARATORS:
            rho_boot[col][b] = spearman(leg, sign * scores[col].to_numpy())

    rows = []
    for col, label, sign in COMPARATORS:
        rho = spearman(point_delta, sign * scores[col].to_numpy())
        lo, hi = np.nanpercentile(rho_boot[col], [CI_LO, CI_HI])
        rows.append({"col": col, "label": label, "rho": rho, "lo": lo, "hi": hi})
    res = pd.DataFrame(rows)
    print()
    print(res.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    set_style()
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 1.8))

    bar_color = "tab:blue"
    y_pos = np.arange(len(res))[::-1]
    for yp, (_, r) in zip(y_pos, res.iterrows()):
        ax.errorbar(
            r["rho"], yp,
            xerr=[[r["rho"] - r["lo"]], [r["hi"] - r["rho"]]],
            fmt="o", color=bar_color, ecolor=bar_color,
            markersize=3.5, capsize=2, elinewidth=0.9, capthick=0.9,
        )

    ax.axvline(0, color="0.6", lw=0.7, ls=(0, (4, 3)), zorder=0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(res["label"])
    ax.set_ylim(-0.6, len(res) - 0.4)
    ax.set_xlim(-0.4, 0.6)
    ax.set_xticks(np.arange(-0.4, 0.61, 0.2))
    ax.tick_params(axis="y", length=3, pad=4)
    ax.tick_params(axis="x", length=3, pad=2)

    ax.set_xlabel("Spearman ρ with human-rated DELTA scores", labelpad=6)
    ax.set_title("Predictors of human-DELTA scores", pad=8)

    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_linewidth(0.8)

    # CreationDate=None keeps the PDF byte-identical across runs.
    fig.savefig(OUT_PATH, bbox_inches="tight", pad_inches=0.02,
                metadata={"CreationDate": None})
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
