"""Name-to-primitive registry."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import TypeVar

from tusk.exceptions import PrimitiveError
from tusk.primitives.base import Primitive

_REGISTRY: dict[str, type[Primitive]] = {}
_P = TypeVar("_P", bound=Primitive)


def register(cls: type[_P]) -> type[_P]:
    """Register a primitive class under its ``name``.

    Args:
        cls: The primitive class to register.

    Returns:
        The class unchanged, so this works as a decorator.

    Raises:
        PrimitiveError: If the name is already registered to a different
            class. Re-registering the same class under its own name is a
            no-op.
    """
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        raise PrimitiveError(f"primitive name {cls.name!r} is already registered")
    _REGISTRY[cls.name] = cls
    return cls


def resolve(spec: str | Primitive) -> Primitive:
    """Turn a name or instance into a primitive instance.

    Args:
        spec: A registered primitive name, or an already-built instance.

    Returns:
        A primitive instance.

    Raises:
        PrimitiveError: If the name is not registered, or the primitive is not
            a frozen dataclass with equality enabled.
    """
    if isinstance(spec, Primitive):
        primitive = spec
    else:
        try:
            primitive = _REGISTRY[spec]()
        except KeyError:
            known = ", ".join(sorted(_REGISTRY))
            msg = f"unknown primitive {spec!r}; available: {known}"
            raise PrimitiveError(msg) from None
    _require_frozen_dataclass(primitive)
    return primitive


def _require_frozen_dataclass(primitive: Primitive) -> None:
    """Check that a primitive is a frozen dataclass with equality enabled.

    Args:
        primitive: The resolved primitive to check.

    Raises:
        PrimitiveError: If it is not.
    """
    cls = type(primitive)
    params = getattr(cls, "__dataclass_params__", None)
    is_value_semantic = (
        dataclasses.is_dataclass(cls)
        and params is not None
        and params.frozen
        and params.eq
    )
    if is_value_semantic:
        return
    raise PrimitiveError(
        f"primitive {cls.__name__!r} needs @dataclass(frozen=True): "
        "features deduplicate by value.",
    )


def resolve_all(specs: Iterable[str | Primitive]) -> tuple[Primitive, ...]:
    """Resolve a collection of names or instances.

    Args:
        specs: Names or instances.

    Returns:
        Primitive instances in the given order.
    """
    return tuple(resolve(spec) for spec in specs)
