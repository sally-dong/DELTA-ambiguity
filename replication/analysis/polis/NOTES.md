# Pol.is bridging analysis

Relates each statement's DELTA score to how well it bridges the two opinion
groups Pol.is identifies, across all four Pol.is datasets — `vtaiwan`,
`produktion`, `wohnen`, `energie`.

Bridging is `min(p_A, p_B)`: the lower of the two groups' agreement rates, so
a statement scores highly only if both groups endorse it. Each rate is
Laplace-smoothed, `(agrees + 1) / (agrees + disagrees + 2)`, which keeps
lightly-voted statements from reaching 0 or 1. Passes are not counted as
votes. Group vote counts come from each dataset's `comments-grouped.csv`;
DELTA scores come from `results/<dataset>/scores/delta_scores.csv`, joined on
the policy index.

## Files

| File | Contents |
|---|---|
| `correlate_bridging_vs_delta.py` | the analysis; everything below is its output |
| `bridging_vs_delta_<dataset>.csv` | per-statement join of bridging metrics and DELTA score, one per dataset |
| `bridging_min_scatter_panel.pdf` | four-panel scatter of bridging against DELTA, raw scores |
| `bridging_min_scatter_panel_ranks.pdf` | the same panel on ranks |

## Running it

```bash
python analysis/polis/correlate_bridging_vs_delta.py --dir all      # the four CSVs
python analysis/polis/correlate_bridging_vs_delta.py --panels --panels-metric bridging_min
```

`--dir <dataset>` restricts to one. `--metric` selects a different group-vote
measure (`bridging`, `divergence`, `in_group_support`, and others); `--sweep`
plots correlation against a fractional-reach threshold rather than a fixed
panel.

Statements need enough votes in both groups to give a meaningful rate. The
filter is on *fractional* reach, not raw counts: each group's vote count on a
statement is divided by that group's estimated size (the most votes it cast on
any single statement), and statements below `min_frac` in either group are
dropped. Fractions rather than counts, because the two clusters differ in size
by orders of magnitude across the four datasets.

The `--dir` CSVs are byte-reproducible. The panel PDFs are not: their
`savefig` does not pin `CreationDate`, so the bytes change on every run even
when the figure does not.
