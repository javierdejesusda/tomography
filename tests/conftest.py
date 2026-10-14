"""Shared fixtures: tiny random models with an offline tokenizer.

Every model has 3 layers and d_model 32 with random weights from a fixed
seed, and every tokenizer is a byte-level BPE trained on an inline corpus, so
the default suite never touches the network. Other test modules reuse these
fixtures, so their names and shapes are stable.
"""

from collections.abc import Callable
import functools

import pytest
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

from tomography.model import ARCHS
from tomography.model import LoadedModel
from tomography.model import wrap_model

END_OF_TEXT = "<|endoftext|>"
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
def _tiny_tokenizer(
    special_tokens: tuple[str, ...],
) -> PreTrainedTokenizerFast:
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
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=END_OF_TEXT,
        eos_token=END_OF_TEXT,
        unk_token=END_OF_TEXT,
    )


def _build_tiny_gpt2() -> LoadedModel:
    tokenizer = _tiny_tokenizer((END_OF_TEXT,))
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


TINY_BUILDERS: dict[str, Callable[[], LoadedModel]] = {
    "gpt2": _build_tiny_gpt2,
}


@pytest.fixture
def tiny_gpt2() -> LoadedModel:
    return _build_tiny_gpt2()


@pytest.fixture(params=sorted(ARCHS), ids=sorted(ARCHS))
def tiny_arch(request: pytest.FixtureRequest) -> LoadedModel:
    return TINY_BUILDERS[request.param]()
