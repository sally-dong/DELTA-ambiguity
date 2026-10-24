# DELTA-ambiguity

Code and data for *A Contextual Measure of Ambiguity in Prescriptive Text*
(Dong, Thorburn, Konya, Bakker; EMNLP 2026).

DELTA measures the ambiguity of a policy as disagreement about its
consequences. It pairs the policy with concrete cases, asks a diverse pool of
raters whether each case is permitted, and scores how far their verdicts
diverge. LLM-DELTA uses language models to generate the cases, construct the
rater personas, and simulate their judgments.

| Directory | Contents |
|---|---|
| [`delta/`](delta) | The `delta-ambiguity` Python package: case and persona generation, judgment collection, and scoring |
| [`replication/`](replication) | Datasets, model outputs, analysis scripts, and figures for every result in the paper |

## Install the package

```bash
pip install "git+https://github.com/sally-dong/DELTA-ambiguity#subdirectory=delta"
```

See [`delta/README.md`](delta/README.md) for the full pipeline and every
setting.

## Reproduce the paper

```bash
cd replication
uv sync    # installs ../delta as an editable dependency
```

[`replication/README.md`](replication/README.md) lists each dataset, the
script behind each figure, and which files are not included. The rater
judgments for the Habermas datasets are too large to include and are
available on request.

## Citation

```bibtex
@inproceedings{dong2026contextual,
  title     = {A Contextual Measure of Ambiguity in Prescriptive Text},
  author    = {Dong, Sally and Thorburn, Luke and Konya, Andrew and Bakker, Michiel},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```

## License

Code and the datasets we authored are released under the [MIT License](LICENSE).
Data derived from third-party sources (Pol.is, the Habermas Machine release,
and the Brysbaert et al. concreteness norms) keeps its upstream terms; see
[`replication/README.md`](replication/README.md#licensing-and-attribution).
