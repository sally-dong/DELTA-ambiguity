"""Convert human_ratings_tidy.csv to the probs.csv format expected by
scoring.delta.

Tidy format (one row per rater × policy × case, both prompts side by side):
    participant_id (P001..P309), party, policy (p1..p20), case (c1..c28),
    permit (0/1), prohibit (0/1)

probs.csv format (one row per rater × policy × case × prompt):
    rater, policy, case, prompt, prob_yes

Mapping:
    rater    = unique integer per participant_id
    policy   = (int after stripping 'p') - 1, aligning p1..p20 to 0..19 to
               match probs.csv
    case     = int after stripping 'c' (kept raw — case sets differ between
               probs.csv and tidy)
    prompt=0 = permit answer  (comply prompt)
    prompt=1 = prohibit answer (violate prompt)
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
SRC = DATA_DIR / "human_ratings_tidy.csv"
DST = DATA_DIR / "human_probs.csv"
PARTY_DST = DATA_DIR / "rater_party.csv"


def main():
    tidy = pd.read_csv(SRC)

    # c28 is an attention-check case: raters were instructed to mark BOTH
    # "yes permitted" and "yes prohibited". Drop any participant who failed
    # on any appearance of c28 (strict criterion, matching human-validation.py),
    # then drop the c28 rows themselves so the case isn't scored as data.
    ac = tidy[tidy["case"] == "c28"]
    ac_pass = ac.assign(passed=(ac["permit"] == 1) & (ac["prohibit"] == 1))
    attentive = (
        ac_pass.groupby("participant_id")["passed"].all()
        .loc[lambda s: s].index
    )
    n_before = tidy["participant_id"].nunique()
    tidy = tidy[tidy["participant_id"].isin(attentive)].reset_index(drop=True)
    tidy = tidy[tidy["case"] != "c28"].reset_index(drop=True)
    print(f"Attention filter: kept {len(attentive)} of {n_before} participants")

    rater_ids = {pid: i for i, pid in enumerate(sorted(tidy["participant_id"].unique()))}
    tidy["rater"] = tidy["participant_id"].map(rater_ids)

    # Rater → party map, in the same integer id space as human_probs.csv;
    # the forest plot's party-stratified bootstrap reads it.
    party = (
        tidy.drop_duplicates("participant_id")
        .set_index("rater")["party"]
        .sort_index()
        .rename_axis("rater")
        .reset_index()
    )
    party.to_csv(PARTY_DST, index=False)
    print(f"Wrote {PARTY_DST} — {len(party)} raters, "
          f"parties: {party['party'].value_counts().to_dict()}")
    tidy["policy"] = tidy["policy"].str.lstrip("p").astype(int) - 1
    tidy["case"] = tidy["case"].str.lstrip("c").astype(int)

    comply = tidy[["rater", "policy", "case", "permit"]].rename(columns={"permit": "prob_yes"})
    comply["prompt"] = 0
    violate = tidy[["rater", "policy", "case", "prohibit"]].rename(columns={"prohibit": "prob_yes"})
    violate["prompt"] = 1

    out = pd.concat([comply, violate], ignore_index=True)
    out = out[["rater", "policy", "case", "prompt", "prob_yes"]]
    out = out.sort_values(["rater", "policy", "case", "prompt"]).reset_index(drop=True)
    out.to_csv(DST, index=False)
    print(f"Wrote {DST} — {len(out)} rows, {out['rater'].nunique()} raters, "
          f"{out['policy'].nunique()} policies, {out['case'].nunique()} cases")


if __name__ == "__main__":
    main()
