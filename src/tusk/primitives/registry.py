"""Name-to-primitive registry."""

from __future__ import annotations

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
        PrimitiveError: If the name is not registered.
    """
    if isinstance(spec, Primitive):
        return spec
    try:
        return _REGISTRY[spec]()
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        msg = f"unknown primitive {spec!r}; available: {known}"
        raise PrimitiveError(msg) from None


def resolve_all(specs: Iterable[str | Primitive]) -> tuple[Primitive, ...]:
    """Resolve a collection of names or instances.

    Args:
        specs: Names or instances.

    Returns:
        Primitive instances in the given order.
    """
    return tuple(resolve(spec) for spec in specs)
