"""Controllers that steer a model with reading vectors.

A controller is a factory of residual edits: it turns a reading vector per
layer into the `ResidualEdit` mapping that `edit_residuals` expects.
"""

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Literal
from typing import Protocol
from typing import Self

import numpy as np
from numpy.typing import NDArray
import torch

from tomography.control.operators import Operator
from tomography.control.operators import apply_operator
from tomography.hooks import ForwardInfo
from tomography.hooks import ResidualEdit
from tomography.hooks import capture_residuals
from tomography.model import BosMode
from tomography.model import EncodedBatch
from tomography.model import LoadedModel

CoeffMode = Literal["absolute", "relative"]
ResponseMask = Literal["all", "response"]


class VectorSource(Protocol):
    """What a controller needs from a fitted reader."""

    @property
    def layers(self) -> tuple[int, ...]:
        """Layers with a direction, in the order of `directions`."""
        ...

    @property
    def directions(self) -> NDArray[np.float32]:
        """Unit-norm directions with shape `(K, d_model)`."""
        ...

    @property
    def signs(self) -> NDArray[np.float32]:
        """Orientation of each direction, `(K,)` with values in {-1, +1}."""
        ...


def mean_residual_norms(
    loaded: LoadedModel,
    texts: Sequence[str],
    layers: Sequence[int],
    *,
    bos: BosMode | None = None,
    batch_size: int = 8,
    skip_first: bool = True,
) -> dict[int, float]:
    """Measures the mean norm of the residual stream per layer.

    The mean is pooled over all real tokens of all texts, not a mean of
    per-text means, and the norms are accumulated in float32 even for a
    bfloat16 model. Batches are right padded, so the result does not depend
    on `batch_size` beyond float rounding. Pass held-out text.

    Args:
        loaded: The wrapped model.
        texts: The texts to run.
        layers: Layer indices in 0..L; duplicates collapse.
        bos: BOS policy of the encoding; None uses the model default.
        batch_size: Texts per forward.
        skip_first: Whether to drop token 0 of every row. GPT-2's first
            token is an attention sink with a much larger norm that would
            dominate the mean.

    Returns:
        Maps each layer to its mean Euclidean norm per real token.

    Raises:
        ValueError: If `texts` or `layers` is empty, a layer is outside
            0..L, `batch_size` is below 1, or no token is left to average.
    """
    if not texts:
        raise ValueError("texts is empty")
    if not layers:
        raise ValueError("layers is empty")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    wanted = list(dict.fromkeys(layers))
    totals = {layer: torch.zeros((), device=loaded.device) for layer in wanted}
    count = 0
    for start in range(0, len(texts), batch_size):
        encoded = loaded.encode(texts[start : start + batch_size], bos=bos)
        real = (encoded.attention_mask > 0).to(loaded.device)
        if skip_first:
            real[:, 0] = False
        with (
            torch.inference_mode(),
            capture_residuals(loaded, wanted) as seen,
        ):
            loaded.decoder(
                input_ids=encoded.input_ids.to(loaded.device),
                attention_mask=encoded.attention_mask.to(loaded.device),
                position_ids=encoded.position_ids.to(loaded.device),
            )
        for layer in wanted:
            norms = seen[layer].float().norm(dim=-1)
            totals[layer] += norms[real].sum()
        count += int(real.sum())
    if count == 0:
        raise ValueError("no token left to average (skip_first drops token 0)")
    return {layer: float(total) / count for layer, total in totals.items()}


def _token_index(
    info: ForwardInfo, batch: int, seq: int, device: torch.device
) -> torch.Tensor:
    """Returns the `(batch, seq)` index of each column within its row.

    Taken from `position_ids` when present, else from the attention mask,
    else the column number. A column rule alone fails with left padding and
    with decode steps, which hold one token with a growing mask.
    """
    if info.position_ids is not None:
        return info.position_ids[:, -seq:]
    if info.attention_mask is not None:
        counted = info.attention_mask.cumsum(dim=-1) - 1
        return counted[:, -seq:].clamp(min=0)
    return torch.arange(seq, device=device).expand(batch, seq)


def _build_edit_mask(
    info: ForwardInfo,
    batch: int,
    seq: int,
    device: torch.device,
    prompt_lengths: int | tuple[int, ...] | None,
) -> torch.Tensor | None:
    """Returns the `(batch, seq)` bool mask of the tokens to edit, or None.

    Real tokens only; with `prompt_lengths`, also only the tokens from the
    last prompt token on, whose output predicts the first generated token.

    Raises:
        ValueError: If `prompt_lengths` has a different size than the batch.
    """
    mask: torch.Tensor | None = None
    if info.attention_mask is not None:
        mask = info.attention_mask[:, -seq:] > 0
    if prompt_lengths is None:
        return mask
    if isinstance(prompt_lengths, int):
        first = torch.tensor([[prompt_lengths - 1]], device=device)
    else:
        if len(prompt_lengths) != batch:
            raise ValueError(
                f"got {len(prompt_lengths)} prompt lengths for {batch} rows"
            )
        first = torch.tensor(prompt_lengths, device=device).unsqueeze(-1) - 1
    response = _token_index(info, batch, seq, device) >= first
    return response if mask is None else mask & response


@dataclass(frozen=True, eq=False)
class ReadingVectorController:
    """Adds a scaled reading vector to the residual stream at some layers.

    The controller is an `EditFactory`: `controller(batch)` returns the edits
    for that batch, so it can be passed to `generate` as `edits=controller`.

    The edit runs its math in float32 on purpose and casts back to the dtype
    of the stream. Never run a forward under `torch.autocast`: under CPU
    bfloat16 autocast the dot product `h.v` is computed in bfloat16, and a
    projection at coefficient 1 leaves a residual of about 3.9e-2.

    Attributes:
        vectors: Maps a layer to its oriented `(d_model,)` float32 vector on
            the model device; the controller adds `coeff * vector`.
        coeff: Finite coefficient. Negative values suppress the concept.
        op: The edit operator.
        mode: `absolute` uses `coeff`; `relative` uses `coeff * norms[l]`.
        norms: Mean residual norm per layer, required for `relative`; layers
            without a vector are ignored.
        mask: `all` edits every real token; `response` edits only the tokens
            from the last prompt token on.
        preserve_norm: Whether to rescale edited vectors to their prior norm.
    """

    vectors: Mapping[int, torch.Tensor]
    coeff: float
    op: Operator = "linear"
    mode: CoeffMode = "absolute"
    norms: Mapping[int, float] | None = None
    mask: ResponseMask = "all"
    preserve_norm: bool = False

    def __post_init__(self) -> None:
        """Checks every vector once, so the edits can skip value checks.

        `apply_operator` treats a zero direction as a no-op without raising,
        so a zero reading vector would silently switch the steering off.

        Raises:
            ValueError: If `vectors` is empty, a vector is not 1-D, not
                finite, zero, or of a different length than the others,
                `coeff` is not finite, `mode` is `relative` with a missing
                or non-positive norm or with `op="projection"`.
        """
        if not self.vectors:
            raise ValueError("vectors is empty")
        if not math.isfinite(self.coeff):
            raise ValueError(f"coefficient must be finite, got {self.coeff}")
        sizes = set()
        for layer, vector in self.vectors.items():
            if vector.ndim != 1:
                raise ValueError(
                    f"vector at layer {layer} must be 1-D, "
                    f"got shape {tuple(vector.shape)}"
                )
            if not bool(torch.isfinite(vector).all()):
                raise ValueError(f"vector at layer {layer} is not finite")
            if not bool((vector != 0).any()):
                raise ValueError(f"vector at layer {layer} is zero")
            sizes.add(vector.shape[0])
        if len(sizes) > 1:
            raise ValueError(f"vectors have different lengths: {sorted(sizes)}")
        if self.mode == "relative":
            self._check_relative()

    def _check_relative(self) -> None:
        if self.op == "projection":
            raise ValueError(
                "relative coefficients are not supported for the projection "
                "operator: its coefficient is dimensionless"
            )
        norms = self.norms or {}
        for layer in self.vectors:
            norm = norms.get(layer)
            if norm is None or not math.isfinite(norm) or norm <= 0:
                raise ValueError(
                    f"relative mode needs a finite positive norm for layer "
                    f"{layer}, got {norm}"
                )

    @classmethod
    def from_reader(
        cls,
        reader: VectorSource,
        layers: Sequence[int] | None = None,
        *,
        coeff: float,
        op: Operator = "linear",
        mode: CoeffMode = "absolute",
        norms: Mapping[int, float] | None = None,
        mask: ResponseMask = "all",
        preserve_norm: bool = False,
        device: torch.device | str = "cpu",
    ) -> Self:
        """Builds a controller from the oriented directions of a reader.

        Args:
            reader: Source of layers, unit directions and signs.
            layers: Layers to steer; None takes every reader layer.
            coeff: Finite coefficient.
            op: The edit operator.
            mode: `absolute` or `relative`.
            norms: Mean residual norm per layer, for `relative`.
            mask: `all` or `response`.
            preserve_norm: Whether to rescale edited vectors to their prior
                norm.
            device: Device of the vectors; pass the model device.

        Returns:
            The controller, with float32 vectors `signs[k] * directions[k]`.

        Raises:
            ValueError: If `layers` is empty, the reader lacks a wanted
                layer, or the arguments fail the checks of `__post_init__`.
        """
        wanted = list(reader.layers if layers is None else layers)
        if not wanted:
            raise ValueError("layers is empty")
        vectors: dict[int, torch.Tensor] = {}
        for layer in dict.fromkeys(wanted):
            if layer not in reader.layers:
                raise ValueError(
                    f"layer {layer} is not in the reader layers "
                    f"{tuple(reader.layers)}"
                )
            k = reader.layers.index(layer)
            oriented = reader.signs[k] * reader.directions[k]
            vectors[layer] = torch.as_tensor(
                oriented, dtype=torch.float32, device=device
            )
        return cls(
            vectors,
            coeff,
            op=op,
            mode=mode,
            norms=norms,
            mask=mask,
            preserve_norm=preserve_norm,
        )

    def _effective_coeff(self, layer: int) -> float:
        if self.mode == "relative" and self.norms is not None:
            return self.coeff * self.norms[layer]
        return self.coeff

    def _normalize_lengths(
        self, prompt_lengths: int | Sequence[int] | None
    ) -> int | tuple[int, ...] | None:
        if self.mask == "all":
            return None
        if prompt_lengths is None:
            raise ValueError("mask='response' needs prompt_lengths")
        values = (
            [prompt_lengths]
            if isinstance(prompt_lengths, int)
            else list(prompt_lengths)
        )
        if not values or min(values) < 1:
            raise ValueError(
                f"prompt lengths must be at least 1, got {prompt_lengths}"
            )
        return (
            prompt_lengths if isinstance(prompt_lengths, int) else tuple(values)
        )

    def edits(
        self, prompt_lengths: int | Sequence[int] | None = None
    ) -> dict[int, ResidualEdit]:
        """Builds one residual edit per vector layer.

        Args:
            prompt_lengths: Prompt token counts, BOS included, as an int for
                all rows or one per row.

        Returns:
            Maps each vector layer to its edit.

        Raises:
            ValueError: If `mask` is `response` and `prompt_lengths` is
                missing or holds a length below 1.
        """
        lengths = self._normalize_lengths(prompt_lengths)

        cache: list[tuple[ForwardInfo, torch.Tensor | None]] = []

        def edit_mask(
            info: ForwardInfo, h: torch.Tensor
        ) -> torch.Tensor | None:
            if not cache or cache[0][0] is not info:
                built = _build_edit_mask(
                    info, h.shape[0], h.shape[1], h.device, lengths
                )
                cache[:] = [(info, built)]
            return cache[0][1]

        def make(layer: int) -> ResidualEdit:
            vector = self.vectors[layer]
            coeff = self._effective_coeff(layer)

            def edit(h: torch.Tensor, info: ForwardInfo) -> torch.Tensor:
                return apply_operator(
                    h,
                    vector,
                    coeff,
                    self.op,
                    mask=edit_mask(info, h),
                    preserve_norm=self.preserve_norm,
                )

            return edit

        return {layer: make(layer) for layer in self.vectors}

    def __call__(self, batch: EncodedBatch) -> dict[int, ResidualEdit]:
        """Returns the edits for a batch, using its prompt lengths."""
        return self.edits(batch.lengths)
