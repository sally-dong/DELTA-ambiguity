# LLM-as-a-judge baselines

Scripts that produced the ratings committed under
`analysis/{adversarial,human_experiment}/llm_ratings/`. Three judges rate
every statement under two rubrics (`ambiguity` and `delta`, see
`rate_statements.py`); a statement's baseline score is the mean of the three
judges' integer ratings.

Ported from the archived Habermas repository's `rate_ambiguity.py`, with
inference re-pointed at `delta.inference` (same machinery, packaged with the
public library) and the unused rubrics dropped. Judge runs read
`OPENAI_API_KEY` and `GEMINI_API_KEY` from the environment and cost real
money; the committed outputs are the paper's.

## Reproducing a full rubric run

From `replication/`, for each dataset input:

```bash
# human_experiment input is generated (adversarial's is committed)
python analysis/llm_judge/build_human_experiment_input.py

INPUT=analysis/llm_judge/human_experiment_input.csv       # or
INPUT=analysis/adversarial/adversarial_policies.csv

for RUBRIC in ambiguity delta; do
    OUT=analysis/<dataset>/llm_ratings/$RUBRIC
    python analysis/llm_judge/rate_statements.py --property $RUBRIC --input $INPUT \
        --out-dir $OUT --model gpt-5.2
    python analysis/llm_judge/rate_statements.py --property $RUBRIC --input $INPUT \
        --out-dir $OUT --model gpt-5 --thinking
    python analysis/llm_judge/rate_statements.py --property $RUBRIC --input $INPUT \
        --out-dir $OUT --model gemini-3-pro-preview --thinking
    python analysis/llm_judge/merge_ratings.py --input_folder $OUT
done
```

`--max 2` on `rate_statements.py` smoke-tests without paying for a full run.
The merge step is read by `score_human_experiment.py`;
`plot_adversarial_ratings.py` instead averages the per-model
`ratings_raw.csv` itself, so the adversarial directories carry no
`ratings_merged.csv`.

## Provenance quirks in the committed outputs

Verified: `save_rubric_to_dir` and `parse_ratings` reproduce the committed
`prompt.md` and `ratings_raw.csv` files byte-for-byte, with two explainable
exceptions.

- **`human_experiment/llm_ratings/delta/prompt.md`** predates the final
  sentence of the delta rubric ("Respond with ONLY a single digit (1-7). Do
  not include any other text."), which was added before the adversarial run.
  The committed file records what that run actually saw; `rate_statements.py`
  carries the final rubric, which is the one in the paper appendix.
- **`adversarial/llm_ratings/delta/gpt-5/ratings_raw.csv`** has one row
  (`republic-conflict9`) absent from its `results_raw.csv`: the model failed
  to return a parseable rating, the query was retried, and the repaired value
  was written back only to `ratings_raw.csv` (see the note in
  `plot_adversarial_ratings.py`). Re-parsing raw dumps therefore yields 59 of
  its 60 rows.
- The committed `adversarial/llm_ratings/` outputs hold 60 statements from
  the original three-domain run (republic, housing, speech), but
  `adversarial_policies.csv` carries only the 20 republic policies the paper
  reports, so a re-run rates those 20; `plot_adversarial_ratings.py` joins on
  `delta_scores.csv` and uses the republic rows either way.
- The human_experiment `ratings_raw.csv` files were regenerated post-run: the
  archived pipeline's `parse_custom_ids` split custom ids on `_`, which
  mangled `round_id="human_experiment"`. `parse_ratings` here uses an
  anchored regex instead, so the bug cannot recur.
