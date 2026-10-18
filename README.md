# tomography

Representation Engineering (RepE) reimplemented on small open language
models, with every result checked against controls.

RepE ([Zou et al., arXiv:2310.01405](https://arxiv.org/abs/2310.01405);
[reference code](https://github.com/andyzoujm/representation-engineering))
treats a concept such as truthfulness, morality or an emotion as a direction
in a model's residual stream. You read the concept by projecting activations
onto that direction, and you steer the model by adding or removing it while it
generates. This repository runs that recipe on ten checkpoints: GPT-2 small,
medium, large and XL, and Qwen3 0.6B, 1.7B and 4B (base and post-trained).
It is independent research code, not the authors' code.

Results: [project site](https://javierdejesusda.github.io/tomography/) and
[`docs/findings.md`](docs/findings.md).

## Why it matters

RepE claims are easy to overstate. A direction can score well because the
text gives the answer away, or because a random direction scores well too.
Here every reading number sits next to a random direction, a shuffled-label
control and, where text can leak the label, a text-only baseline. Every
checkpoint and data file is pinned, so a run can be repeated.

## Key results

Each result is one run per model and checkpoint. Intervals and caveats are in
[`docs/findings.md`](docs/findings.md).

- **Reading scales for some concepts only.** Morality (R3) accuracy rises from
  0.623 on GPT-2 small to 0.768 on GPT-2 XL and 0.940 on Qwen3 4B, against
  random directions at 0.52 to 0.63 on those three. Sentiment and honesty transfer do not
  order by size.
- **TruthfulQA does not beat text.** A reader scores 0.64 to 0.74 on MC1
  (R6), but a TF-IDF classifier on the answer texts alone scores 0.676 and
  0.671.
- **The paper's PCA finder does not read on GPT-2.** In the reference cell of
  each task `pca_diff` is not above a random direction on any GPT-2 size;
  supervised finders such as `logistic` are (P4).
- **Readings depend on wording.** Rewording the prompt moves a fixed
  sentiment reader by 0.09 (GPT-2) to 0.20 (Qwen3 0.6B) in accuracy (P1).
- **"Known versus unknown" is a direction, but not a better error
  predictor.** It separates questions better than subject popularity, yet it
  predicts errors no better than the model's own answer log-probability
  (P9).
- **Steering stays fluent only for small pushes.** At a relative coefficient
  of 0.1 the text stays fluent; at 0.3 the sentiment score saturates on
  repetitive text (C1).

## Quickstart

Install [uv](https://docs.astral.sh/uv/) 0.12.17 (the CI version). It installs
the pinned Python 3.12 when missing.

```bash
git clone https://github.com/javierdejesusda/tomography.git
cd tomography
uv sync                                        # CUDA 13.0 PyTorch build
uv sync --locked --no-group cu130 --group cpu  # or: CPU-only build
export UV_NO_SYNC=1                            # PowerShell: $env:UV_NO_SYNC = "1"
```

Run the sentiment reading experiment (R1) on GPT-2 small, then draw its
figures:

```bash
uv run tomography run r1
uv run tomography plot r1
```

The first run downloads the model and the data into a cache
(`$TOMOGRAPHY_CACHE`, default `~/.cache/tomography`). Results go to
`results/r1/gpt2/<config-hash>/`: `config.json`, `meta.json` (commit, device,
library versions), `summary.json` and the tables and curves. Figures go to the
run's `figures/` folder as PDF, PNG and SVG.

Options of `tomography run <id>`: `--model KEY` (a checkpoint key from
`MODELS` in [`model.py`](src/tomography/model.py), for example
`qwen3-1.7b-base`), `--seed N`, `--set KEY=VALUE` (any config field) and
`--out DIR`. An unknown id lists the available ones. A minimal end-to-end
example of an experiment is
[`experiments/r1_sentiment.py`](src/tomography/experiments/r1_sentiment.py).

## Experiments

Every id is the argument of `tomography run`. Coverage varies: the full list of
models per experiment, with run identities, is in
[`docs/findings.md`](docs/findings.md#run-identity).

| Id | Question | Models |
| --- | --- | --- |
| `r1` | Can a direction separate positive and negative SST-2 phrases? | all 10 |
| `r2` | Does one read pleasantness (ETHICS utility)? | all 10 |
| `r3` | Does one read morality (ETHICS commonsense)? | all 10 |
| `r4` | Does one read six emotions, one against the rest? | all 10 |
| `r5` | Does an honesty direction from personas transfer to plain statements? (exploratory) | all 10 |
| `r6` | Does a reader beat text-only baselines on TruthfulQA MC1? | 9 |
| `r7` | Does one read truth on facts and on Geometry of Truth sets? | all 10 |
| `p1` | How much does prompt wording move a sentiment reader? | GPT-2, Qwen3 0.6B and 1.7B |
| `p2` | How do token position and a misleading sentence matter? | 8 |
| `p3` | How does training-set size matter? | 8 |
| `p4` | Which direction finder, pairing and preprocessing work best? | 8 |
| `p5` | Do persona wording and instructions confound the reading? | 8 |
| `p6` | How does accuracy change per decade of parameters? | GPT-2 sizes |
| `p7` | Does the extraction template change steering? | 5 |
| `p8` | Base against post-trained checkpoints | 0.6B and 1.7B pairs |
| `c1` | Does steering move sentiment, and does the text stay fluent? | 8 |
| `c2` | Does steering move emotions? | 6 |
| `c4` | Does steering or a LoRRA adapter improve TruthfulQA? | 6 |
| `c5` | Can steering produce lies on TriviaQA? | 4 |
| `m1` | How do six finders compare on reading, steering and erasure? | 5 |
| `lr1` | Train one LoRRA adapter on Alpaca: loss curve and held-out loss | 8 |
| `a1` | How much do directions from different seeds agree? | all 10 |
| `a2` | What does the logit lens show for a direction? | all 10 |
| `a5` | Do directions align with Qwen-Scope SAE features? | code only |
| `p9a` to `p9d` | Label questions known, unknown or uncertain, read that direction, use it to predict errors, steer along it | 10, 10, 10, 9 |
| `p10a`, `p10b` | Do models answer approximate quantities with ranges, and can a reader tell? | all 10 |
| `p10c` | Does the range direction link to knowledge and hedging? | code only |

## Reproduce at scale

Experiments that need a GPU run through a chain spec (a TOML file of jobs):

```bash
uv run tomography chain run chain.toml --dry-run   # validate and print commands
uv run tomography chain run chain.toml
```

[`cloud.py`](src/tomography/cloud.py) rents a RunPod pod, runs a chain on it
and pulls the results back (`python -m tomography.cloud up|push|setup|start|
status|pull|down`; read `--help` of each). The API key comes from
`RUNPOD_API_KEY` only; always pass `--max-hours`.

To collect finished runs into a committable tree and redraw every figure
(figures are not committed; raw `readers*.npz` vectors are not either):

```bash
uv run tomography export --out results <folder-with-runs>
uv run tomography plot all --out results
```

Reproducibility rules:

- Every checkpoint pins a 40-character commit SHA; every data file pins a
  commit or a SHA-256 digest, checked on each read.
- Splits come from a hash of the example id, and seeds derive from a SHA-256,
  so they do not depend on row order or process.
- `meta.json` records the commit, whether tracked files were modified, the
  device and the library versions. The config hash covers the config only, so
  compare `meta.json` before pooling runs from different machines.
- Offline: every loader takes `offline=True`; for weights set
  `HF_HUB_OFFLINE=1`.

Deviations from the paper and the statistics behind the intervals are in
[`docs/methodology.md`](docs/methodology.md).

## Repository layout

```text
src/tomography/
  model.py        checkpoint registry (MODELS) and load_model
  hooks.py        read and edit the residual stream at chosen layers
  activations.py  collect_activations
  templates.py    chat formats and the paper's templates
  pairing.py      seeded strategies that turn stimuli into pairs
  reading/        direction finders, RepReader, metrics
  control/        steering operators, controllers, generation
  datasets/       pinned sources and one loader per dataset
  judges.py       pinned sentiment and emotion classifiers
  lorra.py        low-rank representation adaptation
  p9/, p10/       known-versus-unknown and approximate-answer helpers
  experiments/    one module per experiment id, plus shared sweeps
  viz/            figure style and plots
  chain.py        chain runner
  cloud.py        RunPod helper
  export.py       run export
  cli.py          the `tomography` command (also `python -m tomography`)
tests/            unit tests, roughly one file per module
docs/             findings, methodology, related work, references
results/          committed run outputs
```

## Data sources and licenses

All downloads are pinned in
[`datasets/sources.py`](src/tomography/datasets/sources.py) and
[`model.py`](src/tomography/model.py). Licenses below are those the sources
state; the repository's MIT license covers its code only, and none of these
files is redistributed here.

| Source | License as stated |
| --- | --- |
| GPT-2 checkpoints | MIT (the upstream repository says "Modified MIT") |
| Qwen3 checkpoints | Apache-2.0 |
| Judge models (sentiment, emotion) | Apache-2.0 or MIT; two state none |
| Qwen-Scope SAE decoders (`a5`) | Hub tag "other"; `LICENSE` file is Apache-2.0 |
| RepE statements (facts, emotions) | MIT |
| ETHICS | MIT |
| TruthfulQA | Apache-2.0 |
| KUQ | MIT |
| Wikidata snapshots | CC0 |
| ARC, NQ-Open, WikiText-103 | share-alike (CC-BY-SA-3.0, GFDL for WikiText) |
| Alpaca | CC BY-NC 4.0 (non-commercial) |
| SyntHal | research and review terms only |
| FalseQA, Geometry of Truth | none stated |
| PopQA, SST-2, TriviaQA | none on the Hub card; code repositories differ |

Check each source's terms before reuse. The `PinnedSource` rows and the
`MODELS` entries in the code carry the exact revisions.

## Development

```bash
uv sync --locked --no-group cu130 --group cpu
export UV_NO_SYNC=1
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run codespell src tests README.md pyproject.toml
uv run pytest -q
```

Tests that need a GPU are marked `slow` and tests that download data or
weights are marked `network`; both are deselected by default (`-m slow`,
`-m network`).

## Citation

Please cite the original work:

```bibtex
@misc{zou2025representationengineeringtopdownapproach,
      title={Representation Engineering: A Top-Down Approach to AI Transparency},
      author={Andy Zou and Long Phan and Sarah Chen and James Campbell and Phillip Guo and Richard Ren and Alexander Pan and Xuwang Yin and Mantas Mazeika and Ann-Kathrin Dombrowski and Shashwat Goel and Nathaniel Li and Michael J. Byun and Zifan Wang and Alex Mallen and Steven Basart and Sanmi Koyejo and Dawn Song and Matt Fredrikson and J. Zico Kolter and Dan Hendrycks},
      year={2025},
      eprint={2310.01405},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2310.01405},
}
```

## License

MIT, see [`LICENSE`](LICENSE).
