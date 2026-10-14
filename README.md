# tomography

A reimplementation of Representation Engineering (Zou et al., 2023, arXiv:2310.01405) on GPT-2 and Qwen3:
representation reading with linear artificial tomography (LAT), representation control, and new studies of how
prompt and stimulus design shape what is read and how steering works.

Status: under construction. Code is MIT licensed. Datasets and models are downloaded at runtime from pinned
revisions and keep their own licenses; none are redistributed here.

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
