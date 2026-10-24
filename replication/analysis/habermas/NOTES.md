# Habermas assembly data

Participant-level ratings from the Virtual Citizens' Assembly in the
Habermas Machine study (Tessler et al. 2024), and the agreement-distribution
analysis built on them.

`habermas_agreement_distribution.py` writes
`agreement_distribution_lowess.pdf`: how participant agreement with a
statement varies with that statement's DELTA score, LOWESS-smoothed, with a
cluster bootstrap over deliberation rounds.

```bash
python analysis/habermas/habermas_agreement_distribution.py
```

`--out` redirects the figure. The output is byte-reproducible.

## Source

Download from
[google-deepmind/habermas_machine](https://github.com/google-deepmind/habermas_machine)
into this directory:

- `hm_all_candidate_comparisons.parquet` (~400 MB)
- `hm_all_position_statement_ratings.parquet`

Both are gitignored. The two CSVs below are derived from them and are
tracked, so `habermas_agreement_distribution.py` runs without the download —
only `extract.py` needs the parquets.

## Derived files

`python analysis/habermas/extract.py` writes:

| File | Rows | Contents |
|---|---|---|
| `participant_agreements.csv` | 3,060 | One row per participant per winning statement: `agreement` and `quality` on the 1–7 scale, and that participant's `rank` of it |
| `participant_positions.csv` | 1,545 | One row per participant per round: agreement with the question's affirming statement, before and after deliberation |

Both cover the eight assembly questions, restricted to the 309 rounds that
completed both iterations. `agreement` is blank for the `MOCK` control
condition (710 rows) and `rank` is blank where a participant submitted no
ranking; the release encodes the latter as `-1`, which is not carried over.

## Reading the raw release

Judgments are stored as parallel arrays: `rankings.candidate_ids` indexes
positionally into `rankings.numerical_ranks`, `ratings.agreement` and
`ratings.quality` on the same row. They must be unnested together.

## The ninth question

The release covers nine questions; the paper uses eight. "Has the Internet
been a net positive for humanity?" is dropped — it asks for a judgment of
fact rather than a prescription, so its statements have no policy content
whose interpretation could diverge.

## Statement DELTA scores

Not stored here. They are read from `results/q[0-9]/scores/delta_scores.csv`
and joined back to statement IDs through `data/q*/policies.json`, which is
what `load_delta_scores()` in `habermas_agreement_distribution.py` does.
The rater judgments behind those scores (`results/q*/probs.csv`, 166 MB) are
not in the repository; they are available on request from the corresponding
author (sally@sallydong.me).
