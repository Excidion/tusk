"""Name-to-primitive registry and the decorator sugar for simple primitives."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, TypeVar

from tusk.dtypes import DtypeFamily
from tusk.exceptions import PrimitiveError
from tusk.primitives.base import Primitive, TransformPrimitive

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


def transform_primitive(
    *,
    name: str,
    input_dtypes: tuple[DtypeFamily, ...],
    output_dtype: Any = None,
    order_dependent: bool = False,
) -> Callable[[Callable[..., Any]], type[Primitive]]:
    """Build and register a transform primitive from a plain function.

    Sugar for the zero-parameter case only; parameterized primitives should
    subclass :class:`~tusk.primitives.base.TransformPrimitive` directly so
    the parameters can be dataclass fields.

    Args:
        name: Registry key.
        input_dtypes: One dtype family per input.
        output_dtype: Fixed output dtype, or None to preserve the first
            input's.
        order_dependent: Whether the expression needs an explicit ordering.

    Returns:
        A decorator producing a registered primitive class.
    """

    def decorate(fn: Callable[..., Any]) -> type[Primitive]:
        def build(self: Primitive, *inputs: Any) -> Any:
            """Delegate to the decorated function.

            Args:
                self: The primitive instance; unused, since the decorated
                    function is stateless.
                *inputs: One expression per declared input.

            Returns:
                The decorated function's expression.
            """
            return fn(*inputs)

        fn_name = getattr(fn, "__name__", name)
        cls = type(
            fn_name,
            (TransformPrimitive,),
            {
                "__doc__": fn.__doc__ or f"The {name} transform primitive.",
                "name": name,
                "input_dtypes": input_dtypes,
                "output_dtype": output_dtype,
                "order_dependent": order_dependent,
                "build": build,
            },
        )
        return register(dataclass(frozen=True)(cls))

    return decorate
