from collections.abc import Callable
from typing import Any

import huggingface_hub
import numpy as np
import pytest
import torch
from transformers import AutoTokenizer
from transformers import GPT2Config
from transformers import GPT2LMHeadModel

from tomography.activations import Activations
from tomography.activations import _shared_prefix_length
from tomography.activations import collect_activations
from tomography.hooks import capture_residuals
from tomography.knowledge import DEFAULT_SHOTS
from tomography.knowledge import few_shot_prompt
from tomography.model import MODELS
from tomography.model import BosMode
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.model import wrap_model

TEXT = "USER: Is the movie good? ASSISTANT: Yes, the movie is very good."
RESPONSE = (TEXT.index("Yes"), len(TEXT))
CONCEPT = (TEXT.index("the movie"), TEXT.index("the movie") + len("the movie"))
TEXTS = [
    TEXT,
    "The quick brown fox jumps over the lazy dog.",
    "USER: A short one? ASSISTANT: Yes.",
    "She felt happy, then sad.",
]
SPANS = [
    RESPONSE,
    (TEXTS[1].index("jumps"), len(TEXTS[1])),
    (TEXTS[2].index("Yes"), len(TEXTS[2])),
    (TEXTS[3].index("happy"), len(TEXTS[3])),
]
TOLERANCE = {"rtol": 1e-4, "atol": 1e-3}


def overlapping(lm: LoadedModel, text: str, span: tuple[int, int]) -> list[int]:
    """Token indices of `text` (no BOS) overlapping `span`, by offsets."""
    offsets = lm.tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True
    )["offset_mapping"]
    start, end = span
    return [
        t for t, (a, b) in enumerate(offsets) if a < end and b > start and b > a
    ]


def bos_shift(lm: LoadedModel) -> int:
    return 1 if lm.default_bos == "prepend" else 0


def stream(lm: LoadedModel, text: str) -> np.ndarray:
    """Residual stream of one text at layers 0..L, shape (L + 1, T, d)."""
    encoded = lm.encode([text])
    layers = list(range(lm.n_layers + 1))
    with capture_residuals(lm, layers) as captured, torch.no_grad():
        lm.model(
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask,
            position_ids=encoded.position_ids,
        )
    return np.stack([captured[layer][0].float().numpy() for layer in layers])


def hook_count(lm: LoadedModel) -> int:
    modules = [lm.decoder, *lm.blocks]
    return sum(
        len(m._forward_hooks) + len(m._forward_pre_hooks) for m in modules
    )


def test_last_position_reads_the_last_real_token_at_every_layer(
    tiny_arch: LoadedModel,
) -> None:
    lm = tiny_arch

    act = collect_activations(lm, TEXTS, position="last", batch_size=3)

    assert act.data.dtype == np.float32
    assert act.data.shape == (lm.n_layers + 1, len(TEXTS), lm.d_model)
    assert act.layers == tuple(range(lm.n_layers + 1))
    assert act.position == "last"
    assert act.n_stimuli == len(TEXTS)
    assert act.row_stimulus.tolist() == [0, 1, 2, 3]
    assert act.row_rank.tolist() == [0, 0, 0, 0]
    for i, text in enumerate(TEXTS):
        last = len(overlapping(lm, text, (0, len(text)))) - 1
        token = last + bos_shift(lm)
        assert act.row_token[i] == token
        np.testing.assert_allclose(
            act.data[:, i], stream(lm, text)[:, token], **TOLERANCE
        )
        ids = lm.encode([text]).input_ids[0]
        assert act.row_token_id[i] == ids[token].item()


def test_layers_subset_keeps_the_requested_order(
    tiny_gpt2: LoadedModel,
) -> None:
    full = collect_activations(tiny_gpt2, TEXTS)

    subset = collect_activations(tiny_gpt2, TEXTS, layers=[3, 0])

    assert subset.layers == (3, 0)
    np.testing.assert_array_equal(subset.data[0], full.data[3])
    np.testing.assert_array_equal(subset.data[1], full.data[0])


@pytest.mark.parametrize("bad", [[-1], [4], []], ids=["neg", "top+1", "empty"])
def test_layers_outside_zero_to_l_or_empty_raise(
    tiny_gpt2: LoadedModel, bad: list[int]
) -> None:
    with pytest.raises(ValueError, match="layer"):
        collect_activations(tiny_gpt2, TEXTS, layers=bad)


@pytest.mark.parametrize("position", ["last", "response", "concept", "mean"])
def test_batched_collection_matches_per_example_collection(
    tiny_arch: LoadedModel, position: str
) -> None:
    lm = tiny_arch
    kwargs = {
        "position": position,
        "response_spans": SPANS,
        "concept_spans": SPANS,
    }

    batched = collect_activations(lm, TEXTS, batch_size=4, **kwargs)  # type: ignore[arg-type]
    single = collect_activations(lm, TEXTS, batch_size=1, **kwargs)  # type: ignore[arg-type]

    np.testing.assert_allclose(batched.data, single.data, **TOLERANCE)
    np.testing.assert_array_equal(batched.row_stimulus, single.row_stimulus)
    np.testing.assert_array_equal(batched.row_token, single.row_token)


def test_response_position_has_one_row_per_overlapping_token(
    tiny_arch: LoadedModel,
) -> None:
    lm = tiny_arch
    shift = bos_shift(lm)

    act = collect_activations(
        lm, TEXTS, position="response", response_spans=SPANS, batch_size=2
    )

    expected_tokens = [
        [t + shift for t in overlapping(lm, text, span)]
        for text, span in zip(TEXTS, SPANS, strict=True)
    ]
    assert act.row_token.tolist() == [
        t for tokens in expected_tokens for t in tokens
    ]
    assert act.row_stimulus.tolist() == [
        i for i, tokens in enumerate(expected_tokens) for _ in tokens
    ]
    assert act.row_rank.tolist() == [
        r for tokens in expected_tokens for r in range(len(tokens))
    ]
    assert act.row_token_id.tolist() == [
        lm.encode([text]).input_ids[0, t].item()
        for text, tokens in zip(TEXTS, expected_tokens, strict=True)
        for t in tokens
    ]
    assert act.data.shape[1] == sum(len(t) for t in expected_tokens)
    row = 0
    for text, tokens in zip(TEXTS, expected_tokens, strict=True):
        reference = stream(lm, text)
        for token in tokens:
            np.testing.assert_allclose(
                act.data[:, row], reference[:, token], **TOLERANCE
            )
            row += 1


def test_response_span_keeps_the_token_carrying_the_leading_space(
    tiny_gpt2: LoadedModel,
) -> None:
    lm = tiny_gpt2
    offsets = lm.encode([TEXT]).offsets[0]

    act = collect_activations(
        lm, [TEXT], position="response", response_spans=[RESPONSE]
    )

    assert offsets[act.row_token[0]][0] == RESPONSE[0] - 1
    assert offsets[act.row_token[-1]][1] == len(TEXT)


def test_bos_is_never_part_of_a_span(tiny_gpt2: LoadedModel) -> None:
    lm = tiny_gpt2

    act = collect_activations(
        lm,
        [TEXT],
        position="response",
        response_spans=[(0, len(TEXT))],
        bos="prepend",
    )

    assert act.row_token.min() == 1
    assert lm.encode([TEXT]).input_ids[0, 0].item() == lm.bos_id
    assert act.row_token_id[0] != lm.bos_id


def test_concept_position_is_one_row_at_the_last_concept_token(
    tiny_arch: LoadedModel,
) -> None:
    lm = tiny_arch
    tokens = overlapping(lm, TEXT, CONCEPT)

    act = collect_activations(
        lm, [TEXT, TEXT], position="concept", concept_spans=[CONCEPT, CONCEPT]
    )

    last = tokens[-1] + bos_shift(lm)
    assert len(tokens) >= 2
    assert act.row_stimulus.tolist() == [0, 1]
    assert act.row_rank.tolist() == [0, 0]
    assert act.row_token.tolist() == [last] * 2
    assert (
        act.row_token_id.tolist()
        == [lm.encode([TEXT]).input_ids[0][last].item()] * 2
    )
    np.testing.assert_allclose(
        act.data[:, 0],
        stream(lm, TEXT)[:, last],
        **TOLERANCE,
    )


def test_mean_position_averages_the_response_tokens(
    tiny_arch: LoadedModel,
) -> None:
    lm = tiny_arch
    rows = collect_activations(
        lm, TEXTS, position="response", response_spans=SPANS
    )

    mean = collect_activations(lm, TEXTS, position="mean", response_spans=SPANS)

    assert mean.data.shape == (lm.n_layers + 1, len(TEXTS), lm.d_model)
    assert mean.row_stimulus.tolist() == [0, 1, 2, 3]
    assert mean.row_rank.tolist() == [0, 0, 0, 0]
    assert mean.row_token.tolist() == [-1] * 4
    assert mean.row_token_id.tolist() == [-1] * 4
    for i in range(len(TEXTS)):
        selected = rows.row_stimulus == i
        np.testing.assert_allclose(
            mean.data[:, i],
            rows.data[:, selected].mean(axis=1),
            rtol=1e-5,
            atol=1e-6,
        )


def test_bos_none_shifts_every_token_index_by_one(
    tiny_gpt2: LoadedModel,
) -> None:
    with_bos = collect_activations(tiny_gpt2, TEXTS, bos="prepend")
    without = collect_activations(tiny_gpt2, TEXTS, bos="none")

    np.testing.assert_array_equal(with_bos.row_token, without.row_token + 1)


@pytest.mark.parametrize("position", ["last", "response", "mean"])
def test_a_bfloat16_model_still_yields_float32_rows(
    tiny_gpt2: LoadedModel, position: str
) -> None:
    half = wrap_model(tiny_gpt2.model.to(torch.bfloat16), tiny_gpt2.tokenizer)

    act = collect_activations(
        half,
        TEXTS,
        position=position,  # type: ignore[arg-type]
        response_spans=SPANS,
    )

    assert half.dtype == torch.bfloat16
    assert act.data.dtype == np.float32
    assert np.isfinite(act.data).all()


@pytest.mark.parametrize(
    ("position", "message"),
    [
        ("response", "response_spans"),
        ("mean", "response_spans"),
        ("concept", "concept_spans"),
    ],
)
def test_a_position_that_needs_spans_rejects_a_missing_argument(
    tiny_gpt2: LoadedModel, position: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        collect_activations(tiny_gpt2, TEXTS, position=position)  # type: ignore[arg-type]


def test_a_missing_span_names_the_stimulus(tiny_gpt2: LoadedModel) -> None:
    spans = [RESPONSE, None, *SPANS[2:]]

    with pytest.raises(ValueError, match="stimulus 1"):
        collect_activations(
            tiny_gpt2, TEXTS, position="response", response_spans=spans
        )


@pytest.mark.parametrize("span", [(0, 0), (500, 600)], ids=["empty", "outside"])
def test_a_span_without_tokens_names_the_stimulus(
    tiny_gpt2: LoadedModel, span: tuple[int, int]
) -> None:
    spans = [*SPANS[:2], span, SPANS[3]]

    with pytest.raises(ValueError, match="stimulus 2"):
        collect_activations(
            tiny_gpt2, TEXTS, position="concept", concept_spans=spans
        )


def test_invalid_arguments_raise(tiny_gpt2: LoadedModel) -> None:
    with pytest.raises(ValueError, match="empty"):
        collect_activations(tiny_gpt2, [])
    with pytest.raises(ValueError, match="batch_size"):
        collect_activations(tiny_gpt2, TEXTS, batch_size=0)
    with pytest.raises(ValueError, match="position"):
        collect_activations(tiny_gpt2, TEXTS, position="first")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="response_spans"):
        collect_activations(
            tiny_gpt2, TEXTS, position="response", response_spans=SPANS[:2]
        )


def test_collection_skips_the_lm_head_and_leaves_no_hooks(
    tiny_arch: LoadedModel,
) -> None:
    lm = tiny_arch

    def refuse(*args: object) -> None:
        raise AssertionError("the LM head must not run")

    handle = lm.model.get_output_embeddings().register_forward_hook(refuse)
    try:
        collect_activations(lm, TEXTS)
    finally:
        handle.remove()

    assert hook_count(lm) == 0


SHOTS = DEFAULT_SHOTS[:2]
QUESTIONS = [
    "Is the dog lazy?",
    "She felt happy, then sad.",
    "Pick a number between 0 and 9.",
    "The model reads the stream",
    "A short text?",
]
PROMPTS = [few_shot_prompt(question, SHOTS) for question in QUESTIONS]
ANSWER_LINE = 3


def common_prefix(
    lm: LoadedModel, texts: list[str], bos: BosMode | None
) -> int:
    """Longest shared token prefix of `texts`, one suffix token kept each."""
    rows = [lm.encode([text], bos=bos).input_ids[0].tolist() for text in texts]
    shared = 0
    while shared < min(map(len, rows)) - 1 and (
        len({row[shared] for row in rows}) == 1
    ):
        shared += 1
    return shared


def decoder_calls(
    lm: LoadedModel, run: Callable[[], object]
) -> list[tuple[int, ...]]:
    """Shapes of the `input_ids` of every decoder call made by `run`."""
    calls: list[tuple[int, ...]] = []

    def record(
        module: torch.nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, Any],
    ) -> None:
        calls.append(tuple(kwargs["input_ids"].shape))

    handle = lm.decoder.register_forward_pre_hook(record, with_kwargs=True)
    try:
        run()
    finally:
        handle.remove()
    return calls


def assert_same_rows(got: Activations, expected: Activations) -> None:
    assert got.layers == expected.layers
    assert got.position == expected.position
    assert got.n_stimuli == expected.n_stimuli
    for name in ("row_stimulus", "row_rank", "row_token", "row_token_id"):
        np.testing.assert_array_equal(
            getattr(got, name), getattr(expected, name)
        )


@pytest.mark.parametrize("bos", ["prepend", "none"])
@pytest.mark.parametrize("batch_size", [1, 2, 8], ids=["b1", "b2", "b8"])
def test_shared_prefix_collection_equals_the_uncached_path(
    tiny_arch: LoadedModel, bos: BosMode, batch_size: int
) -> None:
    lm = tiny_arch

    plain = collect_activations(lm, PROMPTS, bos=bos, batch_size=batch_size)
    cached = collect_activations(
        lm, PROMPTS, bos=bos, batch_size=batch_size, share_prefix=True
    )

    np.testing.assert_allclose(cached.data, plain.data, **TOLERANCE)
    assert_same_rows(cached, plain)


@pytest.mark.parametrize("position", ["response", "concept", "mean"])
def test_shared_prefix_reads_span_positions_in_full_prompt_coordinates(
    tiny_arch: LoadedModel, position: str
) -> None:
    lm = tiny_arch
    spans = [(len(prompt) - ANSWER_LINE, len(prompt)) for prompt in PROMPTS]
    kwargs: dict[str, Any] = {
        "position": position,
        "response_spans": spans,
        "concept_spans": spans,
        "batch_size": 2,
    }

    plain = collect_activations(lm, PROMPTS, **kwargs)
    cached = collect_activations(lm, PROMPTS, share_prefix=True, **kwargs)

    np.testing.assert_allclose(cached.data, plain.data, **TOLERANCE)
    assert_same_rows(cached, plain)


def test_shared_prefix_runs_the_prefix_once_and_then_only_suffix_tokens(
    tiny_arch: LoadedModel,
) -> None:
    lm = tiny_arch
    lengths = lm.encode(PROMPTS).lengths
    shared = common_prefix(lm, PROMPTS, None)

    calls = decoder_calls(
        lm,
        lambda: collect_activations(
            lm, PROMPTS, batch_size=2, share_prefix=True
        ),
    )

    assert shared > 0
    assert calls == [
        (1, shared),
        (2, max(lengths[0:2]) - shared),
        (2, max(lengths[2:4]) - shared),
        (1, lengths[4] - shared),
    ]


@pytest.mark.parametrize("batch_size", [1, 2, 3, 8], ids=str)
def test_shared_prefix_length_is_the_token_level_prefix_for_any_batch_size(
    tiny_arch: LoadedModel, batch_size: int
) -> None:
    lm = tiny_arch

    for bos in ("prepend", "none"):
        assert _shared_prefix_length(
            lm, PROMPTS, bos=bos, batch_size=batch_size
        ) == common_prefix(lm, PROMPTS, bos)


def test_a_prompt_that_differs_early_shortens_the_shared_prefix(
    tiny_arch: LoadedModel,
) -> None:
    lm = tiny_arch
    early = PROMPTS[1].replace("seven", "eight", 1)
    texts = [*PROMPTS, early]

    wide = _shared_prefix_length(lm, PROMPTS, bos=None, batch_size=2)
    narrow = _shared_prefix_length(lm, texts, bos=None, batch_size=2)
    plain = collect_activations(lm, texts, batch_size=2)
    cached = collect_activations(lm, texts, batch_size=2, share_prefix=True)

    assert 0 < narrow < wide
    assert narrow == common_prefix(lm, texts, None)
    np.testing.assert_allclose(cached.data, plain.data, **TOLERANCE)


def test_prompts_that_differ_only_in_the_last_token_share_all_the_rest(
    tiny_arch: LoadedModel,
) -> None:
    lm = tiny_arch
    texts = [PROMPTS[0] + "x", PROMPTS[0] + "y"]
    length = lm.encode(texts[:1]).lengths[0]

    plain = collect_activations(lm, texts)
    cached = collect_activations(lm, texts, share_prefix=True)

    assert (
        _shared_prefix_length(lm, texts, bos=None, batch_size=8) == length - 1
    )
    np.testing.assert_allclose(cached.data, plain.data, **TOLERANCE)
    assert not np.allclose(cached.data[:, 0], cached.data[:, 1])


@pytest.mark.parametrize(
    "texts",
    [
        [PROMPTS[0], PROMPTS[0][:-ANSWER_LINE], PROMPTS[0] + " Yes"],
        [PROMPTS[0]] * 3,
        [PROMPTS[0]],
    ],
    ids=["prefix-of-the-others", "identical", "single"],
)
def test_a_prompt_inside_the_common_prefix_keeps_one_suffix_token(
    tiny_arch: LoadedModel, texts: list[str]
) -> None:
    lm = tiny_arch
    shortest = min(lm.encode([text]).lengths[0] for text in texts)

    plain = collect_activations(lm, texts)
    cached = collect_activations(lm, texts, share_prefix=True)

    assert _shared_prefix_length(lm, texts, bos=None, batch_size=8) == (
        shortest - 1
    )
    np.testing.assert_allclose(cached.data, plain.data, **TOLERANCE)
    assert_same_rows(cached, plain)


def test_without_a_shared_prefix_the_uncached_path_runs(
    tiny_arch: LoadedModel,
) -> None:
    lm = tiny_arch
    texts = [QUESTIONS[0], QUESTIONS[1], QUESTIONS[2]]

    plain = collect_activations(lm, texts, bos="none")
    calls = decoder_calls(
        lm,
        lambda: collect_activations(lm, texts, bos="none", share_prefix=True),
    )
    cached = collect_activations(lm, texts, bos="none", share_prefix=True)

    assert _shared_prefix_length(lm, texts, bos="none", batch_size=8) == 0
    assert calls == [(3, max(lm.encode(texts, bos="none").lengths))]
    np.testing.assert_array_equal(cached.data, plain.data)


def test_a_position_inside_the_shared_prefix_is_refused(
    tiny_arch: LoadedModel,
) -> None:
    spans: list[tuple[int, int] | None] = [(0, 5)] * len(PROMPTS)

    with pytest.raises(ValueError, match=r"stimulus 0.*shared prefix"):
        collect_activations(
            tiny_arch,
            PROMPTS,
            position="response",
            response_spans=spans,
            share_prefix=True,
        )


def test_shared_prefix_collection_skips_the_lm_head_and_leaves_no_hooks(
    tiny_arch: LoadedModel,
) -> None:
    lm = tiny_arch

    def refuse(*args: object) -> None:
        raise AssertionError("the LM head must not run")

    handle = lm.model.get_output_embeddings().register_forward_hook(refuse)
    try:
        collect_activations(lm, PROMPTS, share_prefix=True)
    finally:
        handle.remove()

    assert hook_count(lm) == 0


def make_activations(counts: list[int]) -> Activations:
    row_stimulus = np.repeat(np.arange(len(counts)), counts).astype(np.int64)
    row_rank = np.concatenate([np.arange(c) for c in counts]).astype(np.int64)
    n_rows = int(sum(counts))
    return Activations(
        data=np.zeros((1, n_rows, 2), dtype=np.float32),
        layers=(1,),
        position="response",
        row_stimulus=row_stimulus,
        row_rank=row_rank,
        row_token=np.arange(n_rows, dtype=np.int64),
        row_token_id=np.arange(n_rows, dtype=np.int64),
        n_stimuli=len(counts),
    )


def test_row_values_repeats_each_stimulus_value_for_its_rows() -> None:
    act = make_activations([2, 1, 3])
    labels = np.array([1, 0, 1])

    assert act.row_values(labels).tolist() == [1, 1, 0, 1, 1, 1]
    matrix = np.arange(6).reshape(3, 2)
    assert act.row_values(matrix).shape == (6, 2)


def test_row_values_rejects_a_wrong_length() -> None:
    with pytest.raises(ValueError, match="n_stimuli"):
        make_activations([1, 1]).row_values(np.array([1, 0, 1]))


def test_row_groups_is_the_identity_for_one_row_positions() -> None:
    act = make_activations([1, 1, 1, 1, 1])
    groups = np.array([[0, 1, 2], [3, 4, -1]], dtype=np.int64)

    assert act.row_groups(groups).tolist() == groups.tolist()


def test_row_groups_aligns_members_by_rank_and_keeps_padding() -> None:
    act = make_activations([2, 2, 2, 2])
    pairs = np.array([[0, 1], [2, 3]], dtype=np.int64)
    padded = np.array([[0, 1, -1]], dtype=np.int64)

    assert act.row_groups(pairs).tolist() == [[0, 2], [1, 3], [4, 6], [5, 7]]
    assert act.row_groups(padded).tolist() == [[0, 2, -1], [1, 3, -1]]


def test_row_groups_rejects_or_truncates_unequal_row_counts() -> None:
    act = make_activations([3, 2, 1])
    groups = np.array([[0, 1]], dtype=np.int64)

    with pytest.raises(ValueError, match="group 0"):
        act.row_groups(groups)
    assert act.row_groups(groups, ragged="truncate").tolist() == [
        [0, 3],
        [1, 4],
    ]
    assert act.row_groups(
        np.array([[0, 2]], dtype=np.int64), ragged="truncate"
    ).tolist() == [[0, 5]]


def test_row_groups_rejects_malformed_groups() -> None:
    act = make_activations([1, 1])

    with pytest.raises(ValueError, match="2D"):
        act.row_groups(np.array([0, 1], dtype=np.int64))
    with pytest.raises(ValueError, match="no members"):
        act.row_groups(np.array([[-1, -1]], dtype=np.int64))
    with pytest.raises(ValueError, match="n_stimuli"):
        act.row_groups(np.array([[0, 2]], dtype=np.int64))
    with pytest.raises(ValueError, match="ragged"):
        act.row_groups(np.array([[0, 1]], dtype=np.int64), ragged="pad")  # type: ignore[arg-type]


@pytest.mark.network
def test_real_gpt2_tokenizer_response_span_keeps_the_leading_space() -> None:
    checkpoint = MODELS["gpt2"]
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint.repo_id, revision=checkpoint.revision
    )
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_layer=3,
        n_embd=32,
        n_head=4,
        n_positions=128,
        bos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    lm = wrap_model(GPT2LMHeadModel(config), tokenizer)

    act = collect_activations(
        lm, [TEXT], position="response", response_spans=[RESPONSE]
    )

    offsets = lm.encode([TEXT]).offsets[0]
    assert offsets[act.row_token[0]][0] == RESPONSE[0] - 1
    assert offsets[act.row_token[-1]][1] == len(TEXT)
    assert lm.encode([TEXT]).input_ids[0, 0].item() == 50256
    assert act.row_token.min() >= 1


REAL_QUESTIONS = [
    "Who wrote the play Hamlet?",
    "What is the capital city of Peru?",
    "Which metal is liquid at room temperature?",
]


@pytest.fixture(scope="module", params=["gpt2", "qwen3-0.6b-base"])
def real_cached(request: pytest.FixtureRequest) -> LoadedModel:
    checkpoint = MODELS[request.param]
    cached = huggingface_hub.try_to_load_from_cache(
        checkpoint.repo_id, "config.json", revision=checkpoint.revision
    )
    if not isinstance(cached, str):
        pytest.skip(f"{request.param} weights are not in the local cache")
    return load_model(request.param, device="cpu", dtype=torch.float32)


@pytest.mark.slow
def test_real_few_shot_prompts_share_the_demonstrations_by_token(
    real_cached: LoadedModel,
) -> None:
    lm = real_cached
    prompts = [few_shot_prompt(question) for question in REAL_QUESTIONS]
    demos = prompts[0][: prompts[0].index(REAL_QUESTIONS[0]) - len("Q: ")]
    demo_ids = lm.tokenizer(demos, add_special_tokens=False)["input_ids"]

    shared = _shared_prefix_length(lm, prompts, bos=None, batch_size=2)
    plain = collect_activations(lm, prompts, batch_size=2)
    cached = collect_activations(lm, prompts, batch_size=2, share_prefix=True)

    assert shared == common_prefix(lm, prompts, None)
    assert shared >= len(demo_ids)
    if lm.spec.model_type == "gpt2":
        full = lm.encode(prompts[:1], bos="none").input_ids[0].tolist()
        assert full[: len(demo_ids)] != demo_ids
    np.testing.assert_allclose(cached.data, plain.data, **TOLERANCE)
    assert_same_rows(cached, plain)
