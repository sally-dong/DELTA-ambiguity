# Convergence diagnostics

How many cases and raters a DELTA run actually needs. For each
`(n_cases, n_raters)` cell, the script draws `N_REPS = 100` fresh subsamples
of that size, scores each one, and reports the mean Spearman correlation
between the subsample's policy ranking and the full-data ranking.

Correlation is low in the bottom-left corner, where few cases and few raters
leave the ranking dominated by sampling noise, and saturates toward 1.0 in the
top-right. Where it flattens is the point past which more inference buys
little.

Two datasets are covered, chosen because they are the two with dense
rater × case coverage: `human_experiment` (the campus-speech set, read from
`results/human_experiment/`) and `vtaiwan` (read from `results/vtaiwan/`).

## Files

| File | Contents |
|---|---|
| `case_rater_heatmap.py` | the analysis; everything below is its output |
| `case_rater_heatmap_<dataset>.csv` | mean rho per (n_cases, n_raters) cell |
| `case_rater_heatmap_<dataset>.pdf` | that grid as a heatmap |

## Running it

```bash
python analysis/convergence/case_rater_heatmap.py
```

Scoring is inlined rather than taken from the library, and aggregates over
cases with a flat mean: the library's case-range weights would be recomputed
inside every subsample, adding variance that has nothing to do with what is
being measured. Cell score is `4 * mean_p * (1 - mean_p)` over raters'
revealed probabilities (`delta.scoring.disagreement`), and a policy's score is
the mean of its cell scores — the same formula `delta.scoring` applies,
without the weighting.

Outputs are byte-reproducible.
