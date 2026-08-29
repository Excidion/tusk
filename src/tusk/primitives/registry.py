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
    _require_value_semantics(primitive)
    return primitive


def _require_value_semantics(primitive: Primitive) -> None:
    """Reject a primitive that compares by identity rather than by value.

    Features are frozen dataclasses holding a primitive, so a feature's
    equality and hash are only as good as its primitive's. Deduplication
    during synthesis and matching a saved feature set against a fresh one both
    depend on that.

    Args:
        primitive: The resolved primitive to check.

    Raises:
        PrimitiveError: If it is not a frozen dataclass with equality enabled.
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
        f"primitive {cls.__name__!r} must be a frozen dataclass with equality "
        "enabled (eq=True, the default). tusk compares primitives by value to "
        "deduplicate features and to match a saved feature set against a "
        "fresh one; a primitive that compares by identity silently produces "
        "duplicate features. Decorate it with @dataclass(frozen=True) -- see "
        "docs/guide/custom-primitives.md.",
    )


def resolve_all(specs: Iterable[str | Primitive]) -> tuple[Primitive, ...]:
    """Resolve a collection of names or instances.

    Args:
        specs: Names or instances.

    Returns:
        Primitive instances in the given order.
    """
    return tuple(resolve(spec) for spec in specs)
