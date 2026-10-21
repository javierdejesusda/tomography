# tomography

Representation Engineering (RepE) reimplemented on small open language
models, with controls reported next to the results.

RepE ([Zou et al., arXiv:2310.01405](https://arxiv.org/abs/2310.01405);
[reference code](https://github.com/andyzoujm/representation-engineering))
treats a concept such as truthfulness, morality or an emotion as a direction
in a model's residual stream. You read the concept by projecting activations
onto that direction, and you steer the model by adding or removing it while it
generates. This repository runs that recipe on ten checkpoints: GPT-2 small,
medium, large and XL, and Qwen3 0.6B, 1.7B and 4B (base and post-trained).
It is independent research code, not the authors' code.

Results and full write-up: the
[project site](https://javierdejesusda.github.io/tomography/), rebuilt from the
committed results, and [`docs/findings.md`](docs/findings.md). The site's View
PDF button opens the whole page as a PDF, rebuilt with every deploy;
[`docs/tomography-results.pdf`](docs/tomography-results.pdf) is a snapshot copy
that can lag behind the site.

## Why it matters

RepE claims are easy to overstate. A direction can score well because the text
gives the answer away, or because a random direction scores well too. Here
every reading run stores a random-direction and a shuffled-label control, and
`docs/findings.md` reports them with the numbers. The readers of sentiment,
utility, morality, truth and honesty are compared with text-only baselines
(word and character n-grams, TB1), as are the TruthfulQA (R6) and
known-versus-unknown (P9) readers. Every checkpoint and data file is pinned, so
a run can be repeated.

## Key results

Each result is one run per checkpoint. Counts are per role (GPT-2, Qwen3 base,
Qwen3 post-trained) and never pooled; "above" means that the 95 percent
interval of the difference excludes zero. The [project
site](https://javierdejesusda.github.io/tomography/) has the full write-up and
[`docs/findings.md`](docs/findings.md) the tables.

- **The paper's PCA finder is weaker than logistic regression.** For sentiment
  `pca_diff` is above a random direction on 0 of 4 GPT-2, 1 of 3 Qwen3 base and
  0 of 3 post-trained checkpoints, but for utility on every Qwen3 checkpoint,
  where matching is confounded with the concept and the data. Logistic
  regression is above a random direction for sentiment on 4 of 4, 3 of 3 and 3
  of 3 (P4).
- **Utility and morality are read everywhere, truth only on Qwen3.** A logistic
  direction is above its random direction for utility and morality on 4 of 4
  GPT-2, 3 of 3 base and 3 of 3 post-trained checkpoints, and for truth on 0 of
  4, 3 of 3 and 3 of 3. Morality accuracy rises across GPT-2 from 0.623 (small,
  random 0.516) to 0.768 (XL, random 0.517); Qwen3-4B reads 0.940 against a
  random 0.626 (post-trained) and 0.917 against 0.598 (base). Words alone (word
  and character n-grams) score 0.520 and 0.525 on utility and 0.517 and 0.527
  on morality. Changing the template wording moves a sentiment reader by more
  than test-pair noise on 2 of 4, 3 of 3 and 3 of 3 (R1 to R7, P1).
- **"Known versus unknown" is a direction, not a better error predictor.** It
  ranks PopQA questions above popularity and the text classifiers on 0 of 3
  GPT-2, 3 of 3 base and 2 of 3 post-trained checkpoints, but adds to the
  model's own confidence on 0 of 3, 0 of 3 and 0 of 3 (P9b, P9c). GPT-2 small
  has no such direction: its labels hold no known versus unknown contrast
  (P9a). A range direction is above its random control on 0 of 4, 2 of 3 and
  2 of 3 (P10b).
- **A truth reader does not beat the words alone.** On TruthfulQA MC1 (173 test
  questions) the reader is not above the better text classifier, 0.676 (word)
  and 0.671 (character) with chance at 0.228, on any checkpoint: 0 of 4 GPT-2,
  0 of 3 base and 0 of 3 post-trained intervals exclude zero (R6).
- **Steering moves sentiment at small coefficients and breaks fluency at larger
  ones.** At relative coefficient 0.1, steering moves judged sentiment on 4 of
  4, 3 of 3 and 2 of 3, and keeps the fluency rule (neutral-text perplexity
  ratio at most 1.5 and distinct-1 at least 0.15) on 3 of 4, 3 of 3 and 3 of 3
  (C1). Utility steering at 0.3 beats every random direction on 1 of 4, 2 of 3
  and 2 of 3, but its text keeps the fluency rule on 0 of 3 GPT-2, 1 of 3 base
  and 0 of 3 post-trained checkpoints (GPT-2 small has no fluency measure; M1).
- **Post-training lowers held-out sentiment reading** on 2 of 3 Qwen3 sizes,
  with an interval below zero (P3 held-out split), and 3 of 8 other seed-paired
  differences exclude zero (P8).

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

Every id is the argument of `tomography run`. Coverage is read from the committed
`results/` tree; the models per experiment, with run identities, are in
[`docs/findings.md`](docs/findings.md#run-identity).

| Id | Question | Models |
| --- | --- | --- |
| `r1` | Can a direction separate positive and negative SST-2 phrases? | all 10 |
| `r2` | Does one read pleasantness (ETHICS utility)? | all 10 |
| `r3` | Does one read morality (ETHICS commonsense)? | all 10 |
| `r4` | Does one read six emotions, one against the rest? | all 10 |
| `r5` | Does an honesty direction from personas transfer to plain statements? (exploratory) | all 10 |
| `r6` | Does a reader beat text-only baselines on TruthfulQA MC1? | all 10 |
| `r7` | Does one read truth on facts and on Geometry of Truth sets? | all 10 |
| `tb1` | How much of a reader's accuracy do word and character n-gram classifiers reach with no model? | none (one run) |
| `p1` | How much does prompt wording move a sentiment reader? | all 10 |
| `p2` | How do token position and a misleading sentence matter? | all 10 |
| `p3` | How does training-set size matter? | all 10 |
| `p4` | Which direction finder, pairing and preprocessing work best? | all 10 |
| `p5` | Do persona wording and instructions confound the reading? | all 10 |
| `p6` | How does accuracy change per decade of parameters? | the 4 GPT-2 sizes (one run) |
| `p7` | Does the extraction template change steering? | all 10 |
| `p8` | Base against post-trained checkpoints | the 0.6B and 1.7B pairs (one run) |
| `c1` | Does steering move sentiment, and does the text pass the fluency rule? | all 10 |
| `c2` | Does steering move emotions? | all 10 |
| `c4` | Does steering or a LoRRA adapter improve TruthfulQA? | all 10 |
| `c5` | Can steering produce lies on TriviaQA? | all 10 |
| `m1` | How do six finders compare on reading, steering and erasure? | all 10 |
| `lr1` | Train one LoRRA adapter on Alpaca: loss curve and held-out loss | all 10 |
| `a1` | How much do directions from different seeds agree? | all 10 |
| `a2` | What does the logit lens show for a direction? | all 10 |
| `a5` | Do directions align with Qwen-Scope SAE features? | code only |
| `p9a` to `p9d` | Label questions known, unknown or uncertain, read that direction, use it to predict errors, steer along it | `p9a` to `p9c` all 10 (GPT-2 small has no direction), `p9d` 9 |
| `p10a`, `p10b`, `p10c` | Do models answer approximate quantities with ranges, can a reader tell, and does the range direction link to knowledge and hedging? | all 10 |

## Reproduce at scale

Experiments that need a GPU run through a chain spec (a TOML file of jobs):

```bash
uv run tomography chain run chain.toml --dry-run   # validate and print commands
uv run tomography chain run chain.toml
```

[`cloud.py`](src/tomography/cloud.py) rents a RunPod pod, runs a chain on it
and pulls the results back (`python -m tomography.cloud up|push|setup|start|
status|pull|down|ssh|resume`; read `--help` of each). The API key comes from
`RUNPOD_API_KEY` only; `start` and `resume` require `--max-hours`.

To collect finished runs into a committable tree and redraw every figure
(figures are not committed; readers, sweeps, per-item files and signals are not exported):

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
- The reference code is read at commit `5455d8a`.

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
  reading/        direction finders, reader.py (RepReader), metrics
  control/        steering operators, controllers, generation
  datasets/       pinned sources and one loader per dataset
  judges.py       pinned sentiment, emotion and entailment classifiers
  knowledge.py    known, unknown and uncertain question labels
  analysis.py     layer-wise PCA and t-SNE maps
  sae.py          pinned Qwen-Scope SAE decoders
  lorra.py        low-rank representation adaptation
  p9/, p10/       known-versus-unknown and approximate-answer helpers
  experiments/    one module per experiment id (including tb1, the text
                  baselines), plus shared sweeps
  site/           results site builder (`python -m tomography.site`)
  viz/            figure style, reading curves and geometry plots
  chain.py        chain runner
  cloud.py        RunPod helper
  export.py       run export
  cli.py          the `tomography` command (also `python -m tomography`)
tests/            unit tests, roughly one file per module
docs/             findings, methodology, related work, references
results/          committed run outputs
```

## Data sources and licenses

All downloads are pinned in the
[`datasets/`](src/tomography/datasets) package (see
[`sources.py`](src/tomography/datasets/sources.py)),
[`model.py`](src/tomography/model.py), `judges.py` and `sae.py`. Licenses below are those the sources
state; the repository's MIT license covers its code only, and none of these
files is redistributed here.

| Source | License as stated |
| --- | --- |
| GPT-2 checkpoints | Modified MIT (the Hub card tag is MIT) |
| Qwen3 checkpoints | Apache-2.0 |
| Judge models (sentiment, emotion) | Apache-2.0 or MIT; two state none |
| Entailment model (DeBERTa-large MNLI) | MIT |
| Qwen-Scope SAE decoders (`a5`) | Hub tag "other"; `LICENSE` file is Apache-2.0 |
| RepE statements (facts, emotions) | MIT |
| ETHICS | MIT |
| TruthfulQA | Apache-2.0 |
| KUQ | MIT |
| Wikidata snapshots | CC0 |
| ARC | CC-BY-SA-4.0 |
| NQ-Open | CC-BY-SA-3.0 |
| WikiText-103 | CC-BY-SA-3.0 and GFDL |
| Alpaca | CC BY-NC 4.0 plus OpenAI terms (non-commercial) |
| SyntHal | research and review terms only |
| FalseQA, Geometry of Truth | none stated |
| PopQA | none on the dataset card (code repository: MIT) |
| TriviaQA | none on the dataset card (code repository: Apache-2.0) |
| SST-2 | unknown (cite Socher et al., 2013) |

The Qwen-Scope authors ask for a citation of arXiv:2605.11887 and ask that
interpretability tools are not used to interfere with model capabilities
or to produce harmful content (`a5` steers nothing). Check each source's
terms before reuse. The `PinnedSource` rows and the
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
