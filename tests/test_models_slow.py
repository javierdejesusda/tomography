"""Real-checkpoint integration tests and the Qwen3 BOS probe.

Tests marked slow load real weights and want a GPU; the lead runs them with
`pytest -m slow`. The unmarked tests check the probe helpers on tiny random
models, so the probe logic is exercised by the default suite.
"""

import dataclasses
import statistics

import numpy as np
import pytest
import torch

from tomography.activations import collect_activations
from tomography.hooks import ForwardInfo
from tomography.hooks import capture_residuals
from tomography.hooks import edit_residuals
from tomography.model import ARCHS
from tomography.model import BosMode
from tomography.model import LoadedModel
from tomography.model import load_model

TEXTS = [
    "USER: Is the movie good? ASSISTANT: Yes, it is very good.",
    "The quick brown fox jumps over the lazy dog.",
    "Honest.",
]
PROBE_TEXTS = [
    "The quick brown fox jumps over the lazy dog near the river.",
    "She opened the old letter and read it slowly, twice.",
    "Three plus four equals seven, and seven is a prime number.",
    "The committee will meet on Tuesday to discuss the new budget.",
    "He walked along the shore while the tide was coming in.",
    "A small bakery on the corner sells fresh bread every morning.",
]
SINK_RATIO = 5.0


@dataclasses.dataclass(frozen=True)
class SinkRatios:
    """Median residual-norm ratios that locate the attention sink.

    Each ratio divides the norm at one position by the median norm of the
    positions after it, so an ordinary token sits near 1 and a sink far
    above.

    Attributes:
        first_without_bos: Position 0 (the first content token) in a run
            without BOS.
        separator_with_bos: Position 0 (the separator) in a run with BOS
            prepended.
        content_with_bos: Position 1 (the first content token) in the run
            with BOS prepended.
    """

    first_without_bos: float
    separator_with_bos: float
    content_with_bos: float


def position_ratio(norms: torch.Tensor, position: int) -> float:
    return (norms[position] / norms[position + 1 :].median()).item()


def run_norms(
    lm: LoadedModel, text: str, bos: BosMode, layers: list[int]
) -> dict[int, torch.Tensor]:
    """Returns the per-token residual norms of one text at each layer."""
    encoded = lm.encode([text], bos=bos)
    with capture_residuals(lm, layers) as captured, torch.inference_mode():
        lm.decoder(
            input_ids=encoded.input_ids.to(lm.device),
            attention_mask=encoded.attention_mask.to(lm.device),
            position_ids=encoded.position_ids.to(lm.device),
            use_cache=False,
        )
    return {
        layer: captured[layer][0].float().norm(dim=-1).cpu() for layer in layers
    }


def measure_sink(
    lm: LoadedModel, texts: list[str], layers: list[int]
) -> SinkRatios:
    """Measures where the attention sink sits with and without BOS.

    Runs under any hooks the caller has active.

    Args:
        lm: The wrapped model.
        texts: Texts of at least four tokens.
        layers: Layers whose residual norms are measured.

    Returns:
        The medians over texts and layers.
    """
    first, separator, content = [], [], []
    for text in texts:
        for norms in run_norms(lm, text, "none", layers).values():
            first.append(position_ratio(norms, 0))
        for norms in run_norms(lm, text, "prepend", layers).values():
            separator.append(position_ratio(norms, 0))
            content.append(position_ratio(norms, 1))
    return SinkRatios(
        first_without_bos=statistics.median(first),
        separator_with_bos=statistics.median(separator),
        content_with_bos=statistics.median(content),
    )


def decide_bos(ratios: SinkRatios) -> BosMode:
    """Picks the BOS policy for plain-format runs of a base model.

    Prepend only when the first content token is a sink without BOS and the
    separator takes over the sink role when prepended, leaving the first
    content token ordinary. Anything else keeps the tokenizer default.
    """
    first_is_sink = ratios.first_without_bos > SINK_RATIO
    separator_is_sink = ratios.separator_with_bos > SINK_RATIO
    content_is_ordinary = ratios.content_with_bos < SINK_RATIO
    if first_is_sink and separator_is_sink and content_is_ordinary:
        return "prepend"
    return "none"


@pytest.mark.parametrize(
    ("first", "separator", "content", "expected"),
    [
        (50.0, 60.0, 1.2, "prepend"),
        (1.1, 1.3, 1.2, "none"),
        (50.0, 1.5, 1.2, "none"),
        (50.0, 60.0, 40.0, "none"),
    ],
    ids=["separator-takes-over", "no-sink", "separator-not-a-sink", "both"],
)
def test_decide_bos_prepends_only_when_the_separator_takes_the_sink(
    first: float, separator: float, content: float, expected: str
) -> None:
    ratios = SinkRatios(first, separator, content)

    assert decide_bos(ratios) == expected


def test_probe_sees_no_sink_in_a_random_model(tiny_arch: LoadedModel) -> None:
    ratios = measure_sink(tiny_arch, PROBE_TEXTS, layers=[1, 2])

    assert ratios.first_without_bos < SINK_RATIO
    assert ratios.separator_with_bos < SINK_RATIO
    assert decide_bos(ratios) == "none"


def test_probe_finds_a_planted_sink_on_whatever_sits_at_position_zero(
    tiny_arch: LoadedModel,
) -> None:
    def plant(h: torch.Tensor, info: ForwardInfo) -> torch.Tensor:
        planted = h.clone()
        planted[:, 0] *= 100
        return planted

    with edit_residuals(tiny_arch, {1: plant}):
        ratios = measure_sink(tiny_arch, PROBE_TEXTS, layers=[1, 2])

    assert ratios.first_without_bos > SINK_RATIO
    assert ratios.separator_with_bos > SINK_RATIO
    assert ratios.content_with_bos < SINK_RATIO
    assert decide_bos(ratios) == "prepend"


@pytest.fixture(scope="module", params=["gpt2", "qwen3-0.6b-base"])
def real(request: pytest.FixtureRequest) -> LoadedModel:
    return load_model(request.param, dtype=torch.float32)


def forward_with_capture(
    lm: LoadedModel, texts: list[str], padding_side: str
) -> tuple[dict[int, torch.Tensor], tuple[torch.Tensor, ...]]:
    encoded = lm.encode(texts, padding_side=padding_side)  # type: ignore[arg-type]
    layers = list(range(lm.n_layers + 1))
    with capture_residuals(lm, layers) as captured, torch.inference_mode():
        output = lm.model(
            input_ids=encoded.input_ids.to(lm.device),
            attention_mask=encoded.attention_mask.to(lm.device),
            position_ids=encoded.position_ids.to(lm.device),
            output_hidden_states=True,
        )
    return dict(captured), tuple(output.hidden_states)


@pytest.mark.slow
def test_real_capture_equals_hf_hidden_states_below_the_top(
    real: LoadedModel,
) -> None:
    captured, hidden_states = forward_with_capture(real, TEXTS, "right")

    for layer in range(real.n_layers):
        assert torch.equal(captured[layer], hidden_states[layer])


@pytest.mark.slow
def test_real_final_norm_of_the_top_capture_is_the_last_hidden_state(
    real: LoadedModel,
) -> None:
    captured, hidden_states = forward_with_capture(real, TEXTS, "right")

    with torch.inference_mode():
        normed = real.final_norm(captured[real.n_layers])

    assert torch.equal(normed, hidden_states[real.n_layers])


@pytest.mark.slow
def test_real_left_padding_with_explicit_positions_matches_unpadded(
    real: LoadedModel,
) -> None:
    padded, _ = forward_with_capture(real, TEXTS, "left")
    width = max(real.encode(TEXTS).lengths)

    for row, text in enumerate(TEXTS):
        alone, _ = forward_with_capture(real, [text], "right")
        length = real.encode([text]).lengths[0]
        pad = width - length
        for layer in (1, real.n_layers // 2, real.n_layers):
            assert torch.allclose(
                padded[layer][row, pad:],
                alone[layer][0, :length],
                rtol=1e-4,
                atol=1e-3,
            )


@pytest.mark.slow
def test_real_batched_collection_matches_per_example_collection(
    real: LoadedModel,
) -> None:
    spans = [(len(text) // 2, len(text)) for text in TEXTS]
    kwargs = {"position": "response", "response_spans": spans}

    batched = collect_activations(real, TEXTS, batch_size=3, **kwargs)  # type: ignore[arg-type]
    single = collect_activations(real, TEXTS, batch_size=1, **kwargs)  # type: ignore[arg-type]

    np.testing.assert_allclose(batched.data, single.data, rtol=1e-4, atol=1e-3)
    assert batched.row_token.tolist() == single.row_token.tolist()


@pytest.mark.slow
def test_qwen3_bf16_capture_equals_hf_hidden_states() -> None:
    lm = load_model("qwen3-0.6b-base")

    captured, hidden_states = forward_with_capture(lm, TEXTS, "right")

    assert lm.dtype == torch.bfloat16
    for layer in range(lm.n_layers):
        assert torch.equal(captured[layer], hidden_states[layer])


@pytest.mark.slow
@pytest.mark.parametrize("key", ["qwen3-0.6b-base", "qwen3-1.7b-base"])
def test_qwen3_bos_probe_agrees_with_the_registered_default(key: str) -> None:
    lm = load_model(key)
    layers = [lm.n_layers // 2, (3 * lm.n_layers) // 4]

    ratios = measure_sink(lm, PROBE_TEXTS, layers)

    print(f"{key} layers {layers}: {ratios}")
    assert ARCHS["qwen3"].default_bos == decide_bos(ratios), (
        f"The probe measured {ratios}; set ArchSpec.default_bos of qwen3 to "
        f"{decide_bos(ratios)!r}"
    )
