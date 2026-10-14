# tomography

A reimplementation of Representation Engineering (Zou et al., 2023, arXiv:2310.01405) on GPT-2 and Qwen3:
representation reading with linear artificial tomography (LAT), representation control, and new studies of how
prompt and stimulus design shape what is read and how steering works.

Status: under construction. Code is MIT licensed. Datasets and models are downloaded at runtime from pinned
revisions and keep their own licenses; none are redistributed here.

## Development

Requires [uv](https://docs.astral.sh/uv/) 0.12 or later. uv installs the pinned Python 3.12 when it is missing.

Pick the PyTorch build that matches the machine. The two extras conflict, so install only one:

```bash
uv sync --extra cu130   # NVIDIA GPU, CUDA 13.0 wheels (Linux or Windows)
uv sync --extra cpu     # CPU only
```

CI runs these checks, and they should pass before every push:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run codespell src tests README.md pyproject.toml
uv run pytest -q
```

Tests that load real model weights are marked `slow` and tests that download data are marked `network`. Both are
skipped by default; run them with `uv run pytest -m slow` or `uv run pytest -m network`.
