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
        PrimitiveError: If the name is not registered, names a primitive that
            needs arguments, or the primitive is not a frozen dataclass with
            equality enabled.
    """
    if isinstance(spec, Primitive):
        primitive = spec
    else:
        try:
            cls = _REGISTRY[spec]
        except KeyError:
            known = ", ".join(sorted(_REGISTRY))
            msg = f"unknown primitive {spec!r}; available: {known}"
            raise PrimitiveError(msg) from None
        _require_no_arguments(spec, cls)
        primitive = cls()
    _require_frozen_dataclass(primitive)
    return primitive


def _require_no_arguments(name: str, cls: type[Primitive]) -> None:
    """Check that a primitive class can be built without arguments.

    Args:
        name: The name the class is registered under.
        cls: The primitive class.

    Raises:
        PrimitiveError: If the class has a field without a default.
    """
    required = _required_fields(cls)
    if not required:
        return
    arguments = ", ".join(f"{field}=..." for field in required)
    raise PrimitiveError(
        f"primitive {name!r} needs arguments; "
        f"pass {cls.__name__}({arguments}) instead of its name",
    )


def _required_fields(cls: type[Primitive]) -> list[str]:
    """List the dataclass fields a primitive class cannot be built without.

    Args:
        cls: A primitive class.

    Returns:
        The names of the fields without a default, none for a class that is
        not a dataclass.
    """
    if not dataclasses.is_dataclass(cls):
        return []
    return [
        field.name
        for field in dataclasses.fields(cls)
        if field.init
        and field.default is dataclasses.MISSING
        and field.default_factory is dataclasses.MISSING
    ]


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
