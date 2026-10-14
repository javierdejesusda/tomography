# tomography

A reimplementation of Representation Engineering (Zou et al., 2023, arXiv:2310.01405) on GPT-2 and Qwen3:
representation reading with linear artificial tomography (LAT), representation control, and new studies of how
prompt and stimulus design shape what is read and how steering works.

Status: research code under active development. The code is MIT licensed. Datasets and models are downloaded
at runtime from pinned revisions and keep their own licenses; none are redistributed here.

## What is here

The package lives in `src/tomography/`:

- `model.py`: the checkpoint registry (`MODELS`), the architecture specs, `load_model`, and
  `LoadedModel.encode`, the single tokenizer, BOS, padding and character-offset policy shared by activation
  collection and generation.
- `hooks.py`: `capture_residuals` and `edit_residuals`, context managers that read or replace the residual
  stream at chosen layers.
- `activations.py`: `collect_activations`, residual-stream activations at chosen token positions as numpy
  arrays.
- `templates.py`: chat formats and stimulus templates with exact character spans, the paper's templates
  (`PAPER_TEMPLATES`), and `ablate` for prompt-ablation variants.
- `pairing.py`: seeded strategies that turn stimuli into reading pairs.
- `reading/`: direction finders (`directions.py`), the per-layer `RepReader` (`reader.py`), and accuracy,
  AUROC and interval metrics (`metrics.py`).
- `control/`: edit operators (`operators.py`), the reading-vector controller (`controllers.py`), and batched
  generation with residual edits (`generate.py`).
- `datasets/`: pinned data sources and one loader per dataset, example records, and stimulus-set builders.
- `judges.py`: pinned sentiment and emotion classifiers, and language-model scoring helpers (perplexity,
  sequence log-probability, multiple-choice accuracy, a verbalized-score baseline).
- `knowledge.py`: behavioral known, unknown and uncertain labels for closed-book questions.
- `experiments/`: the experiment framework (`base.py`): configs, run directories, seeding and results I/O.
- `viz/`: figure style and plots (layer curves, token scans, matrices and embeddings).
- `cli.py`, `__main__.py`: the `tomography` command line.

## Supported models

`load_model(key)` loads one of the checkpoints below from the Hugging Face Hub. Every entry pins a full
40-character commit SHA, and `load_model` always passes it as the revision, so a download never follows a
moving branch. It uses CUDA when available and the CPU otherwise.

| Key               | Family | Post-trained | Chat format | Default dtype |
| ----------------- | ------ | ------------ | ----------- | ------------- |
| `gpt2`            | GPT-2  | no           | plain       | float32       |
| `gpt2-medium`     | GPT-2  | no           | plain       | float32       |
| `gpt2-large`      | GPT-2  | no           | plain       | float32       |
| `gpt2-xl`         | GPT-2  | no           | plain       | float16       |
| `qwen3-0.6b-base` | Qwen3  | no           | plain       | bfloat16      |
| `qwen3-0.6b`      | Qwen3  | yes          | template    | bfloat16      |
| `qwen3-1.7b-base` | Qwen3  | no           | plain       | bfloat16      |
| `qwen3-1.7b`      | Qwen3  | yes          | template    | bfloat16      |

The GPT-2 checkpoints record the "Modified MIT" license and the Qwen3 checkpoints Apache-2.0. Two
architectures are registered, `gpt2` and `qwen3`; any other raises `UnsupportedArchitectureError`.

## Representation reading

Layers are numbered `0..L`: layer 0 is the embedding output, layer `l >= 1` is the output of block `l`, and
layer `L` is the last block output before the final norm.

`collect_activations` reads the residual stream at the last token (`last`), at every token of a marked span
(`response`), at the last token of a concept span (`concept`), or as the mean over a span (`mean`).

`fit_direction` offers six direction finders. The method is an explicit argument of `fit_direction` and
`RepReader.fit`; neither has an implicit default.

| Method      | Direction                                                | Input of `fit_direction` |
| ----------- | -------------------------------------------------------- | ------------------------ |
| `pca_diff`  | first principal component of the within-pair differences | `pairs`                  |
| `pca_raw`   | first principal component of the rows themselves         | rows only                |
| `mean_diff` | difference of the two class means                        | `labels`                 |
| `logistic`  | coefficients of an L2-regularized logistic regression    | `labels`                 |
| `kmeans`    | centroid difference of a two-cluster K-means             | rows only                |
| `random`    | Gaussian draw from a seed (a baseline)                   | rows only                |

`pca_diff` is the paper's PCA on pair differences. The `preprocess` argument is `center` (the default, as in
the reference implementation), `unit` (scale each vector to unit length first, as the paper describes) or
`none`.

`RepReader.fit` fits one unit direction per layer with any of these methods and always takes `labels`, which
set the sign of each layer so that a higher score means more of the positive pole. A reader holds one
direction, one mean and one sign per layer, and can be saved to and loaded from an `.npz` file.
`reading/metrics.py` provides `group_accuracy`, `auroc`, `bootstrap_ci`, `wilson_ci`, `best_layer` and
`cosine_matrix`.

```python
import numpy as np

from tomography.reading.metrics import auroc
from tomography.reading.reader import RepReader

rng = np.random.default_rng(0)
labels = np.array([0, 1] * 20)
h = rng.normal(size=(2, 40, 16))  # (layers, rows, d_model)
h[:, labels == 1, 0] += 3.0

reader = RepReader.fit("logistic", h, [3, 7], labels=labels)
scores = reader.score(h)  # shape (2, 40)
print(auroc(scores, labels))  # one AUROC per layer
```

## Representation control

Control adds or removes a reading vector in the residual stream.

- `apply_operator` (in `control/operators.py`) edits residual vectors `h` along a direction `v` with
  coefficient `c`. The operators are `linear` (`h + c v`), `piecewise` (`h + c s v`, with `s` the sign of
  `h.v`) and `projection` (`h - c (h.v / |v|^2) v`). It also takes an optional token mask and
  `preserve_norm`, which rescales each edited vector back to its prior norm.
- `ReadingVectorController` (in `control/controllers.py`) turns the oriented directions of a fitted reader
  into one residual edit per layer, through `ReadingVectorController.from_reader`. The coefficient is
  `absolute`, or `relative` to the mean residual norm of each layer (not available for `projection`), and
  the edit applies to `all` tokens or only to the `response` tokens.
- `generate` (in `control/generate.py`) runs batched, seeded generation, greedy or sampled, and takes the
  controller as `edits=` to steer the model while it generates.

## Experiments

An experiment is a module `src/tomography/experiments/<id>_<slug>.py` that defines `Config` (a subclass of
`ExperimentConfig`), `run(config, run_dir)` and `plot(run_dir, out_dir)`. The id starts with a lowercase
letter, continues with lowercase letters and digits, and names the experiment on the command line.

`experiments/base.py` provides the shared pieces:

- `ExperimentConfig`: a frozen dataclass with `experiment`, `model` (a checkpoint key) and `seed`, plus the
  fields each experiment adds. Equal configs hash to the same run directory.
- `RunDir`: the folder `<out>/<experiment>/<model>/<config hash>/`, holding `config.json`, `meta.json` (git
  state, device, library versions, start and finish times), result files (`write_json`, `write_arrays`) and
  `figures/`.
- `derive_seed` and `seed_everything` for reproducible seeding.

`tomography run` lists the ids it finds when it is given an unknown one.

## Running

The `tomography` command (from `[project.scripts]`; `python -m tomography` is equivalent) has two subcommands:

```bash
uv run tomography --help
uv run tomography run <id> [--model KEY] [--seed N] [--set KEY=VALUE] [--out DIR]
uv run tomography plot <id|all> [--run DIR] [--out DIR]
```

- `run` runs one experiment and prints the run directory. `--model` takes a checkpoint key from the table
  above, `--seed` sets the base seed, and `--set` sets any config field of the experiment (repeatable; the
  value is JSON or plain text). `--out` is the results root and defaults to `results`. A finished run
  directory is never overwritten: delete it to rerun.
- `plot` regenerates the figures of saved runs into the `figures/` folder of each run. `all` plots every
  finished run under `--out`, and `--run` plots one run directory, finished or not.

## Datasets

The dataset loaders fetch their data at runtime and cache it outside the repository, in `$TOMOGRAPHY_CACHE`
when set and `~/.cache/tomography` otherwise. A source is either a Hub dataset repository pinned to a commit
SHA, or plain files at an https URL, each with a required SHA-256 digest. Digests are checked on every read,
cache hits included. `fetch(..., offline=True)` never touches the network. Each source records its own license
in its `PinnedSource` row; some are non-commercial or state none, so check them before reuse. Nothing is
redistributed here.

The loaders:

- Labeled statements, scenarios and phrases: `load_facts` (true and false statements),
  `load_geometry_of_truth`, `load_emotions` and `load_emotion_replies`, `load_ethics_utility` and
  `load_ethics_morality` (ETHICS), `load_sst2`.
- Questions and prompts: `load_truthfulqa` (MC1), `load_arc` (ARC-Challenge), `load_triviaqa`, `load_nq_open`,
  `load_popqa`, `load_falseqa`, `load_kuq`, `load_synthal`.
- Other corpora: `load_alpaca` (instructions without an input) and `load_wikitext` (a neutral corpus for
  perplexity).
- Wikidata: `load_post_cutoff` and `load_unknowable_detail` run SPARQL queries against the public Wikidata
  Query Service. Each response is cached as a snapshot with the query text, the retrieval date and its
  SHA-256, so a later run reuses the exact data.
- `generate_fictional` invents names that match real names on token count; it downloads nothing.

Loaders return `Example`, `ChoiceExample` or `QAExample` records, each in a `train`, `val` or `test` split.
ARC, SST-2 and WikiText keep their source splits; the other loaders use `assign_splits`, which hashes the
example id.
`build_stimulus_set` renders examples through a template and a chat format and pairs them (`random`,
`label_contrastive`, `persona`, `one_vs_rest` or `by_group`) into a `StimulusSet`.

## Development

Install [uv](https://docs.astral.sh/uv/) 0.12.17, the version CI runs. uv installs the pinned Python 3.12 when
it is missing.

A plain `uv sync` installs the CUDA 13.0 build of PyTorch (NVIDIA GPU, Linux or Windows), and `uv run` keeps it.
CPU-only setups (CI, macOS, machines without an NVIDIA GPU) install the CPU build instead and then set
`UV_NO_SYNC=1` (CI does), so that a plain `uv run` keeps it instead of switching back to CUDA:

```bash
uv sync --locked --no-group cu130 --group cpu
export UV_NO_SYNC=1   # PowerShell: $env:UV_NO_SYNC = "1"
uv run pytest -q
```

CI runs these checks, and they should pass before every push:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run codespell src tests README.md pyproject.toml
uv run pytest -q
```

Tests that need a local GPU are marked `slow` and tests that download datasets or model weights are marked
`network`. Both are deselected by default, even when a test id is named on the command line, so pass `-m slow`
or `-m network`, for example `uv run pytest -m slow`.
