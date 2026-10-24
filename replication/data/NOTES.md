# Dataset notes

## Standard layout

Every DELTA experiment directory holds the same seven files, which is what
`delta-generate` writes and `delta-run` reads:

```
policies.json  scenario_params.json  cases.json  sampled_cases.json
base_raters.json  expanded_raters.json  longest_raters.json
```

All twelve generated datasets — `vtaiwan`, `produktion`, `wohnen`, `energie`
and `q1`–`q9` — carry the full set, so `delta-run --dir data/<name>` works
uniformly. Each `data/<name>` pairs with `results/<name>`, which holds that
run's outputs under the same name.

The two validation datasets, `human_experiment` and `adversarial`, were
authored by hand rather than generated, so each departs from the layout where
its construction did. Both add a `custom_prompt_fn.py`: their
`scenario_params.json` names no `prompt_template`, so `delta-run` loads the
prompt from that file. See [Validation datasets](#validation-datasets).

Anything else in a directory is dataset-specific preprocessing. The four
Pol.is sets add their source exports and a `preprocess.py`; each `q*` adds a
`q<n>.csv` of the assembly's per-statement participant ratings. The `q*`
statement sets — every `q<n>.csv` and `policies.json` — are built from the
Habermas Machine release by [`extract_assembly_statements.py`](extract_assembly_statements.py)
with the manual truncations in `cleaned_statements.csv`; the committed files
reproduce byte-for-byte.

Everything downstream of a run — score tables, figures, and the scripts that
make them — lives under [`../analysis/`](../analysis), not here.

| Dataset | Policies | Sampled cases |
|---|---|---|
| `vtaiwan` | 72 | 60 |
| `produktion` | 275 | 40 |
| `wohnen` | 314 | 40 |
| `energie` | 515 | 40 |
| `q1`–`q9` | 72–82 | 60 |
| `human_experiment` | 20 | 20 (hand-written) |
| `adversarial` | 20 | 32 (hand-selected from 60) |

## Pol.is preprocessing

All four Pol.is datasets run the same three-phase `preprocess.py`, which turns
a raw comment export into `policies.json`. Each keeps its source files under
the same names, so nothing needs a per-dataset lookup:

| File | Contents |
|---|---|
| `comments.csv` | raw comment export, 8 columns |
| `comments-grouped.csv` | per-opinion-group vote counts, 14 columns; the bridging input |
| `policies_full.json` | every surviving comment with its English translation, written by phase 1 |

**Phase 0 — filter.** Keep `moderated == '1'`, which is the platform's own
moderation flag and removes roughly half of every export. `energie` adds a
second filter, `MIN_GROUP_VOTES = 10`: a comment must have at least ten votes
in *both* Pol.is opinion groups, computed from `comments-grouped.csv`. On this
export that filter removes nothing beyond moderation, but it runs before
translation so the setting is worth knowing about.

**Phase 1 — translate.** None of the four exports is in English. Each comment
is translated by `gpt-5` (thinking off, `temperature=1`, `max_output_tokens=300`)
under a per-dataset system prompt that preserves hedging, sentiment, and
policy terminology, and keeps institutional names (Klimarat, EU,
Bundesregierung) as they are. Output is cached to `policies_full.json` and the
phase is resumable, so a re-run translates only what is missing. **Every
policy DELTA scores is a machine translation, not the participant's original
wording.**

**Phase 2 — curate.** Drop the ids listed in that dataset's `DROP_IDS`, then
sort what remains by English text length, longest first. The sort is what
`n_top_policies_for_cases: 30` selects from, so case generation is seeded by
the thirty longest statements.

### Counts at each stage

| Dataset | Source language | Raw | `moderated == '1'` | Dropped in curation | `policies.json` |
|---|---|---|---|---|---|
| `vtaiwan` | Traditional Chinese | 199 | 99 | 27 | **72** |
| `produktion` | Austrian German | 522 | 277 | 2 | **275** |
| `wohnen` | Austrian German | 613 | 315 | 1 | **314** |
| `energie` | Austrian German | 1041 | 515 | 0 | **515** |

For the three Klimarat sets, curation only removes link spam and comments
about the consultation process itself — `produktion` drops a Zoom-link-only
post and "The Klimarat is publicly known"; `wohnen` drops one comment that is
half slogan and half URL; `energie` drops nothing, and its `DROP_IDS` is an
empty set. Every id is listed with its text in that dataset's
`preprocess.py`.

### vTaiwan: which comments become policies

`vtaiwan` is the one dataset whose curation is substantive rather than
janitorial, and it matters for how its numbers should be read. Two sets take
99 moderated comments down to 72:

| Step | Kept |
|---|---|
| `moderated == '1'` in `comments.csv` | 99 |
| minus `DROP_IDS` — factual self-disclosures | 94 |
| minus `DESCRIPTIVE_IDS` — descriptive comparisons | **72** |

`DROP_IDS` removes statements that report a fact about the speaker ("I have a
driver's license", "I have used Uber outside Taiwan").

`DESCRIPTIVE_IDS` removes statements that compare Uber and taxis **as they
already operate** rather than proposing what should be done — "UberX is
usually cheaper than taxis", "Uber drivers are less likely to weave through
traffic", "taxi drivers earn more than Uber drivers". DELTA scores a
statement by whether raters agree on which concrete actions it permits and
rules out, so a claim about how things already are gives a rater nothing to
apply to a case. What survives is prescriptive: "Uber should pay taxes where
it operates", "all UberX drivers should first obtain a professional driver's
license".

Both sets are listed with the text of each dropped comment in
[`vtaiwan/preprocess.py`](vtaiwan/preprocess.py).

### What `preprocess.py` reproduces

`policies.json` is the authoritative input to every run; `preprocess.py`
regenerates it exactly for one dataset and approximately for the other three.

| Dataset | Re-running `preprocess.py` |
|---|---|
| `vtaiwan` | reproduces `policies.json` exactly, ids, text and order |
| `produktion` | same ids and order; restores a `Klimarat:` prefix on 17 texts |
| `wohnen` | same ids and order; restores a `Klimarat:` prefix on 8 texts |
| `energie` | same ids; different order |

Two differences to know about, both in the committed files rather than in the
script:

**The `Klimarat:` prefix.** Klimarat-authored propositions carry a `Klimarat:`
prefix in `comments.csv`, and the translation prompt preserves it, so it
reaches `policies_full.json` in all three Klimarat datasets. The three then
diverge: `produktion` and `wohnen` have it stripped from `policies.json` (17
and 8 policies), while `energie` keeps it on all 17 of its own. The stripping
happens after the length sort, which is why those two files are ordered by a
length that includes a prefix their text no longer carries.

**`energie` ordering.** Its `policies.json` is a 200-policy block followed by
315 policies sorted longest-first, rather than a single sorted run. Since
`n_top_policies_for_cases: 30` seeds case generation from the head of the
file, the thirty seed policies for `energie` are the first thirty of that
200-block, not its thirty longest statements.

Neither affects a reported score: `results/<name>/input_construction.json`
records the exact policy and case text each run consumed, and the committed
`policies.json` is what produced it.

## Case sampling

Across every dataset, `sampled_cases.json` holds exactly the cases the
reported run used — it matches the `cases` list in that run's
`results/*/input_construction.json`, and `n_sampled_cases` matches its
length. So `delta-run` reproduces the reported run with no extra flags.

One caveat for the three Klimarat datasets. Their 40 cases are the **first
40 of a 60-case sample** at `sampling_seed: 42`, not a 40-case sample:

```
sample_cases(n=60, seed=42)[:40]   !=   sample_cases(n=40, seed=42)
```

Re-running `delta-generate --action sample_cases` on these datasets will
therefore produce a *different* 40 cases, which is fine for a fresh
experiment but will not reproduce the reported one. The committed
`sampled_cases.json` is the authoritative record.

## Validation datasets

### `human_experiment` — campus speech (Dataset 1)

| File | Contents |
|---|---|
| `policies.json` | the 20 university speech policies, `p1`–`p20` |
| `cases.json` | the 20 campus-speech incidents, paraphrased by hand from FIRE databases |
| `base_raters.json`, `expanded_raters.json`, `longest_raters.json` | rater personas, generated as for every other dataset; `longest_raters.json` holds the 30 used |
| `scenario_params.json` | run configuration, including the permitted/prohibited questions |
| `custom_prompt_fn.py` | the judgment prompt |
| `case-selection/` | provenance of the cases: the FIRE source URLs and export dates, and the script used to shortlist incidents ([README](human_experiment/case-selection/README.md)) |

There is no `sampled_cases.json`: the cases were written, not sampled, so
`delta-run` falls back to `cases.json`. The LLM run is in
`results/human_experiment/`. The human ratings and study materials are in
`analysis/human_experiment/`.

### `adversarial` — Australian republic (Dataset 2)

| File | Contents |
|---|---|
| `policies.json` | 20 policies: `consensus1`–`consensus10`, `conflict1`–`conflict10` |
| `cases.json` | all 60 generated implementations |
| `sampled_cases.json` | the 32 implementations the reported run used |
| `longest_raters.json` | 30 personas; indices 0–14 are monarchist, 15–29 republican |
| `scenario_params.json` | run configuration, including the consistent/inconsistent questions |
| `custom_prompt_fn.py` | the judgment prompt |
| `README.md` | ground truth: the two groups, the consensus and conflict axes, and the expected judgment for each base case |
| `construction/` | the generation procedure (paper appendix) and the minimal 2-policy, 4-case, 2-rater template it starts from |

The 32 cases were filtered from the 60 by hand, not drawn with
`sample_cases`, so `delta-generate --action sample_cases` will not reproduce
them; the committed `sampled_cases.json` is the authoritative record. The
personas come from the adversarial procedure, which scales each of the two
groups separately, not from `delta-generate`. So there are no
`base_raters.json` or `expanded_raters.json`, and `scenario_params.json`
carries only the keys `delta-run` needs.

