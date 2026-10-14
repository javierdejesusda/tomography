"""Experiment configuration, run directories, seeding and results I/O."""

from collections.abc import Callable
from collections.abc import Mapping
import dataclasses
import hashlib
import json
import types
import typing
from typing import Any
from typing import Literal
from typing import Self


class _TypeMismatchError(Exception):
    """Raised when a value does not fit a field annotation."""


_SCALAR_CHECKS: Mapping[Any, Callable[[Any], bool]] = {
    bool: lambda value: isinstance(value, bool),
    int: lambda value: isinstance(value, int) and not isinstance(value, bool),
    float: lambda value: (
        isinstance(value, int | float) and not isinstance(value, bool)
    ),
    str: lambda value: isinstance(value, str),
}


def _coerce(annotation: Any, value: Any) -> Any:
    """Checks a JSON value against an annotation, turning lists into tuples.

    Annotations other than scalars, None, Literal, unions and tuples are not
    checked.

    Args:
        annotation: A resolved type annotation.
        value: A value decoded from JSON.

    Returns:
        The value, with lists converted where the annotation asks for a tuple.

    Raises:
        _TypeMismatchError: If the value does not fit the annotation.
    """
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if annotation is None or annotation is type(None):
        if value is not None:
            raise _TypeMismatchError
        return None
    if origin is Literal:
        if value not in args:
            raise _TypeMismatchError
        return value
    if origin in (typing.Union, types.UnionType):
        for member in args:
            try:
                return _coerce(member, value)
            except _TypeMismatchError:
                continue
        raise _TypeMismatchError
    if origin is tuple:
        if not isinstance(value, list | tuple):
            raise _TypeMismatchError
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], item) for item in value)
        if len(args) != len(value):
            raise _TypeMismatchError
        return tuple(
            _coerce(member, item)
            for member, item in zip(args, value, strict=True)
        )
    check = _SCALAR_CHECKS.get(annotation)
    if check is not None and not check(value):
        raise _TypeMismatchError
    return value


@dataclasses.dataclass(frozen=True, kw_only=True)
class ExperimentConfig:
    """Base class of every experiment configuration.

    Subclasses are frozen, keyword-only dataclasses that add JSON-serializable
    fields. Two configs with equal fields share one run directory.

    Attributes:
        experiment: Lowercase experiment id used in paths and on the command
            line, such as "r1" or "p3".
        model: Key of a registered checkpoint, kept as a plain string.
        seed: Base seed of the run.
    """

    experiment: str
    model: str
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Returns the fields as JSON values, with tuples turned into lists."""
        data: dict[str, Any] = json.loads(json.dumps(dataclasses.asdict(self)))
        return data

    def config_hash(self) -> str:
        """Returns a stable 12-character hex digest of the fields.

        Returns:
            The first 12 hex digits of the SHA-256 of the canonical JSON
            (sorted keys, no whitespace).

        Raises:
            ValueError: If a field holds a non-finite float.
        """
        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Builds a config from decoded JSON, checking names and types.

        Args:
            data: Field values as produced by `to_dict`. Missing fields take
                their defaults; lists become tuples where a field is a tuple.

        Returns:
            The config.

        Raises:
            ValueError: If a key is not a field, a required field is missing
                or a value does not fit its field's annotation.
        """
        fields = {field.name: field for field in dataclasses.fields(cls)}
        unknown = sorted(set(data) - set(fields))
        if unknown:
            raise ValueError(f"unknown fields for {cls.__name__}: {unknown}")
        missing = sorted(
            name
            for name, field in fields.items()
            if name not in data
            and field.default is dataclasses.MISSING
            and field.default_factory is dataclasses.MISSING
        )
        if missing:
            raise ValueError(
                f"missing required fields for {cls.__name__}: {missing}"
            )
        hints = typing.get_type_hints(cls)
        values: dict[str, Any] = {}
        for name, value in data.items():
            try:
                values[name] = _coerce(hints[name], value)
            except _TypeMismatchError:
                raise ValueError(
                    f"{cls.__name__}.{name}: expected {hints[name]}, "
                    f"got {value!r}"
                ) from None
        return cls(**values)
