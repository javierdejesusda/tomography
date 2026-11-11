"""Shared fixtures: tiny random models with an offline tokenizer.

Every model has 3 layers and d_model 32 with random weights from a fixed
seed, and every tokenizer is a byte-level BPE trained on an inline corpus, so
the default suite never touches the network. Other test modules reuse these
fixtures, so their names and shapes are stable. `fixed_machine` names the user
and the host that an export must not leak. `--shard` runs one of several
disjoint parts of the suite, so CI can run the parts as parallel jobs.
"""

import argparse
from collections.abc import Callable
from collections.abc import Iterator
import functools
import getpass
import importlib.metadata
import os
from pathlib import Path
import re
import socket
import zlib

import pytest
import threadpoolctl
from tokenizers import Tokenizer
from tokenizers import decoders
from tokenizers import models
from tokenizers import pre_tokenizers
from tokenizers import processors
from tokenizers import trainers
import torch
from transformers import GPT2Config
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast
from transformers import Qwen3Config
from transformers import Qwen3ForCausalLM

from tomography.experiments import base as experiments_base
from tomography.model import ARCHS
from tomography.model import LoadedModel
from tomography.model import wrap_model

pytest_plugins = ("pytester",)

END_OF_TEXT = "<|endoftext|>"
QWEN_SPECIAL_TOKENS = (
    END_OF_TEXT,
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
)
QWEN_CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content']"
    " + '<|im_end|>' + '\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}"
    "{{- '<think>\\n\\n</think>\\n\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
)
GIT_VARIABLES = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")
VOCAB_SIZE = 400
SEED = 0
CORPUS = (
    "The quick brown fox jumps over the lazy dog.",
    "USER: Is the movie good? ASSISTANT: Yes, the movie is very good.",
    "The model reads the stream of the residual after every block.",
    "A short text, a longer text, and another text with more words in it.",
    "Honest answers are true, and dishonest answers are false.",
    "She felt happy, then sad, then angry, afraid, disgusted and surprised.",
    "Pick a number between 0 and 9: 0123456789.",
) * 8


@functools.cache
def _trained_bpe(special_tokens: tuple[str, ...]) -> str:
    """Trains a byte-level BPE once and returns it serialized."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=list(special_tokens),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(CORPUS, trainer)
    return tokenizer.to_str()


def _tiny_tokenizer(
    special_tokens: tuple[str, ...], **kwargs: object
) -> PreTrainedTokenizerFast:
    """Builds a fresh fast tokenizer so no state leaks between tests."""
    backend = Tokenizer.from_str(_trained_bpe(special_tokens))
    return PreTrainedTokenizerFast(tokenizer_object=backend, **kwargs)


def _build_tiny_gpt2() -> LoadedModel:
    tokenizer = _tiny_tokenizer(
        (END_OF_TEXT,),
        bos_token=END_OF_TEXT,
        eos_token=END_OF_TEXT,
        unk_token=END_OF_TEXT,
    )
    end_id = tokenizer.eos_token_id
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_layer=3,
        n_embd=32,
        n_head=4,
        n_positions=128,
        initializer_range=0.5,
        bos_token_id=end_id,
        eos_token_id=end_id,
    )
    torch.manual_seed(SEED)
    return wrap_model(GPT2LMHeadModel(config), tokenizer)


def _build_tiny_qwen3() -> LoadedModel:
    """Mimics Qwen3: no BOS token, pad and eos are the document separator."""
    tokenizer = _tiny_tokenizer(
        QWEN_SPECIAL_TOKENS,
        eos_token=END_OF_TEXT,
        pad_token=END_OF_TEXT,
        chat_template=QWEN_CHAT_TEMPLATE,
    )
    end_id = tokenizer.eos_token_id
    config = Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        initializer_range=0.5,
        tie_word_embeddings=True,
        bos_token_id=end_id,
        eos_token_id=end_id,
    )
    torch.manual_seed(SEED)
    return wrap_model(Qwen3ForCausalLM(config), tokenizer)


TINY_BUILDERS: dict[str, Callable[[], LoadedModel]] = {
    "gpt2": _build_tiny_gpt2,
    "qwen3": _build_tiny_qwen3,
}


@pytest.fixture
def tiny_gpt2() -> LoadedModel:
    return _build_tiny_gpt2()


@pytest.fixture
def tiny_qwen3() -> LoadedModel:
    return _build_tiny_qwen3()


@pytest.fixture(params=sorted(ARCHS), ids=sorted(ARCHS))
def tiny_arch(request: pytest.FixtureRequest) -> LoadedModel:
    return TINY_BUILDERS[request.param]()


@pytest.fixture
def fixed_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Names the user and the host that the export checks must refuse."""
    monkeypatch.setattr(getpass, "getuser", lambda: "tester-user")
    monkeypatch.setattr(socket, "gethostname", lambda: "testhost-box.lan")


@pytest.fixture(autouse=True, scope="session")
def _one_blas_thread() -> Iterator[None]:
    """Runs the BLAS libraries of numpy and scipy on one thread.

    The suite's matrices are small, so a pool of BLAS threads costs more to
    wake and join than the products it splits. Torch keeps its own threads:
    the forward passes of the tiny models still gain from them.
    """
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        yield


@pytest.fixture(autouse=True, scope="session")
def _run_metadata_once() -> Iterator[None]:
    """Reads the repository's git state and the library versions once.

    Every `RunDir.create` runs two git commands and looks up ten library
    versions, which costs tens of milliseconds and adds up over the hundreds
    of runs the suite creates. The suite edits no tracked file of the
    repository and installs no package, so both answers hold for the whole
    session. A test that points `_SOURCE_DIR` at a repository of its own,
    sets a git variable or replaces `importlib.metadata.version` still gets
    a fresh answer.
    """
    git_state = experiments_base._git_state
    library_versions = experiments_base._library_versions
    real_version = importlib.metadata.version
    source = experiments_base._SOURCE_DIR
    states: dict[tuple[str | None, ...], tuple[str | None, bool | None]] = {}
    versions: list[dict[str, str]] = []

    def cached_git_state(cwd: Path) -> tuple[str | None, bool | None]:
        if cwd != source:
            return git_state(cwd)
        key = tuple(os.environ.get(name) for name in GIT_VARIABLES)
        if key not in states:
            states[key] = git_state(cwd)
        return states[key]

    def cached_library_versions() -> dict[str, str]:
        if importlib.metadata.version is not real_version:
            return library_versions()
        if not versions:
            versions.append(library_versions())
        return dict(versions[0])

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(experiments_base, "_git_state", cached_git_state)
        patch.setattr(
            experiments_base, "_library_versions", cached_library_versions
        )
        yield


def _parse_shard(text: str) -> tuple[int, int]:
    """Reads a shard as INDEX/COUNT, with INDEX counted from 0.

    Raises:
        argparse.ArgumentTypeError: The text is not INDEX/COUNT with
            0 <= INDEX < COUNT.
    """
    match = re.fullmatch(r"([0-9]+)/([0-9]+)", text)
    if match is None or int(match[1]) >= int(match[2]):
        raise argparse.ArgumentTypeError(
            f"expected INDEX/COUNT with 0 <= INDEX < COUNT, got {text!r}"
        )
    return int(match[1]), int(match[2])


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--shard",
        type=_parse_shard,
        metavar="INDEX/COUNT",
        help=(
            "run only shard INDEX of COUNT disjoint shards that together "
            "hold every selected test"
        ),
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Deselects the tests outside the shard that `--shard` names.

    A shard holds whole test files, picked by the CRC32 of their path, so
    what the tests of a file share (a module-scoped fixture or a run cached
    in a module global) is built in one shard only.
    """
    shard: tuple[int, int] | None = config.getoption("shard")
    if shard is None:
        return
    index, count = shard
    kept: list[pytest.Item] = []
    dropped: list[pytest.Item] = []
    for item in items:
        path = item.nodeid.split("::", 1)[0]
        bucket = zlib.crc32(path.encode()) % count
        (kept if bucket == index else dropped).append(item)
    config.hook.pytest_deselected(items=dropped)
    items[:] = kept
