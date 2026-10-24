# DELTA — paper analysis

Analysis scripts, datasets, and figure generation for *A Contextual Measure of
Ambiguity in Prescriptive Texts* (Dong, Thorburn, Konya, Bakker, 2026).

## Layout

| Path | Contents |
|---|---|
| `data/` | Per-dataset experiment inputs: `policies.json`, `cases.json`, rater personas, `scenario_params.json`, plus dataset-specific preprocessing ([notes](data/NOTES.md)) |
| `results/` | Model outputs, one directory per dataset, named to match `data/` |
| `analysis/` | Everything downstream of a run: per-dataset score tables, figures, and the scripts that build them |
| `analysis/habermas/` | Participant agreement and position ratings from the Habermas assembly, and the agreement-distribution figure ([notes](analysis/habermas/NOTES.md)) |
| `analysis/human_experiment/` | De-identified human ratings and study materials, human-validation score tables, LLM-as-a-judge ratings, the Brysbaert et al. (2014) concreteness norms and their scorer, and the forest/scatter figures |
| `analysis/adversarial/` | Adversarial-set LLM-as-a-judge ratings, and the rank-rank figure against the DELTA scores in `results/adversarial/` |
| `analysis/llm_judge/` | The judge scripts that produced both datasets' LLM-as-a-judge ratings ([notes](analysis/llm_judge/NOTES.md)) |
| `analysis/polis/` | Bridging-vs-DELTA analysis across the four Pol.is datasets ([notes](analysis/polis/NOTES.md)) |
| `analysis/convergence/` | How many cases and raters a run needs ([notes](analysis/convergence/NOTES.md)) |

The four datasets in the paper: `human_experiment` (campus speech, with the
human-DELTA ground truth), `adversarial` (the republic domain, 20 policies), the Pol.is
set (`vtaiwan`, `produktion`, `wohnen`, `energie`), and the Habermas `q1`–`q9`
topic series (eight questions; there is no `q8`).

Each `results/<name>` holds exactly what the `delta` package writes — the two
run manifests, the raw inference dumps, `probs.csv`, and one
`scores/delta_scores.csv` from `compute_delta_scores`. The Pol.is and
Habermas scores use its default case-range weighting; `human_experiment` and
`adversarial` use `weighting="mean"`, a flat mean over cases, which is also
what their analysis scripts compute. Analysis outputs are written under
`analysis/`, not back into `results/`.

Every directory in `data/` and `results/` belongs to a reported result.
Re-runs that varied the case draw, the rater draw or the judge model are not
kept here: they score the same statements a second time, so a glob like
`results/q*` would silently double-count them.

## Setup

```bash
uv sync
```

Installs `delta-ambiguity` from `../delta` as an editable dependency, so edits
to the scoring code are picked up immediately.

## Score formula

The per-case disagreement is `4 * mean * (1 - mean)` over the raters'
revealed probabilities — the probability that two independently sampled rater
verdicts conflict, normalized so the score runs
from 0 to 1. Each `delta_scores.csv` reports it per policy as `score`,
alongside its exact decomposition into `within` (rater indeterminacy,
`4 * avg(x * (1 - x))`) and `between` (between-rater disagreement,
`4 * var(x)`), which sum to `score`.

## What is and is not tracked

`results/` holds roughly 1.2 GB of raw inference output. Tracked from it: the
run manifests, the computed `scores/`, and `probs.csv` for the runs the
analysis actually reads — the four Pol.is runs, `human_experiment` and
`adversarial` (~37 MB) — enough to reproduce every reported number from a clean
checkout.

Not tracked: `logprobs_raw.csv` and `results_raw.csv` (~1.1 GB), `probs.csv`
for the q1–q9 Habermas series (166 MB, no current script reads it).

The Habermas judgments are too large to include, but are available on request
from the corresponding author (sally@sallydong.me). Without them, the committed
`results/q*/scores/delta_scores.csv` still reproduce every Habermas figure, but
the scores themselves cannot be recomputed.

`analysis/habermas/*.parquet` is untracked too, but it is a
public download rather than something we paid to generate; see
[analysis/habermas/NOTES.md](analysis/habermas/NOTES.md). The small CSVs
derived from it are tracked, so the figure reproduces without it.

`analysis/human_experiment/human_ratings_tidy.csv` is the de-identified human
ratings file and is tracked: participants are keyed by opaque ids
(`P001`–`P309`) assigned by a seeded shuffle, and no Prolific or Qualtrics id
is retained. The identified export it was derived from,
`human_ratings_identified.csv`, is gitignored and must never be committed.

## Licensing and attribution

The code and the datasets we authored are released under the [MIT
License](LICENSE); see [CITATION.cff](CITATION.cff) for how to cite. Data
derived from third-party sources keeps its upstream terms:

- **Pol.is exports** (`data/{vtaiwan,produktion,wohnen,energie}`): CC-BY 4.0,
  attribution to The Computational Democracy Project.
- **Habermas assembly data** (`data/q*`, `analysis/habermas/`): derived from
  the [Habermas Machine release](https://github.com/google-deepmind/habermas_machine)
  (Tessler et al. 2024), whose data files are CC-BY 4.0.
- **Concreteness norms**
  (`analysis/human_experiment/concreteness_norms_dictionary.csv`): from the
  supplementary materials of Brysbaert, Warriner & Kuperman (2014), *Behavior
  Research Methods*; redistributed here for research use with attribution.
