"""Primitive protocol.

A primitive builds narwhals expressions and never sees a value. That is what
lets feature computation push down into the backend engine. Featuretools'
model -- a callable over a materialized pandas Series -- is deliberately not
used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

import narwhals as nw

from tusk.dtypes import DtypeFamily


class Primitive(ABC):
    """Base class for every primitive.

    Attributes:
        name: Registry key, also the upper-cased stem of generated names.
        input_dtypes: One dtype family per input. Empty means the primitive
            takes no column input, e.g. ``count``.
        output_dtype: Fixed output dtype, or None to preserve the first
            input's.
        commutative: Whether argument order is irrelevant, so that only one
            of ``f(a, b)`` and ``f(b, a)`` is generated.
        stack_on_self: Whether this primitive may be applied to its own
            output.
        default_value: Value substituted for empty groups after a left join.
    """

    name: ClassVar[str]
    input_dtypes: ClassVar[tuple[DtypeFamily, ...]] = ()
    output_dtype: ClassVar[Any] = None
    commutative: ClassVar[bool] = False
    stack_on_self: ClassVar[bool] = True
    default_value: ClassVar[Any] = None

    @property
    def number_of_outputs(self) -> int:
        """How many columns this primitive produces."""
        return 1

    def return_dtype(self, input_dtypes: tuple[Any, ...]) -> Any:
        """Compute the output dtype without touching data.

        Args:
            input_dtypes: Dtypes of the input features, in order.

        Returns:
            The dtype of this primitive's output.
        """
        if self.output_dtype is not None:
            return self.output_dtype
        return input_dtypes[0]

    def generate_name(self, arg_names: Sequence[str]) -> str:
        """Build the display name for an application of this primitive.

        Args:
            arg_names: Names of the inputs. For a zero-input aggregation
                this is the child table's name, giving e.g.
                ``COUNT(transactions)``.

        Returns:
            The feature name.
        """
        return f"{self.name.upper()}({', '.join(arg_names)})"

    def output_names(self, base_name: str) -> tuple[str, ...]:
        """Expand a feature name into one name per output column.

        Args:
            base_name: The name from :meth:`generate_name`.

        Returns:
            One name per output column; indexed when there is more than one.
        """
        if self.number_of_outputs == 1:
            return (base_name,)
        return tuple(f"{base_name}[{i}]" for i in range(self.number_of_outputs))

    def outputs(self, *inputs: nw.Expr) -> tuple[nw.Expr, ...]:
        """Normalize :meth:`build` to a tuple of expressions.

        Args:
            *inputs: One expression per declared input.

        Returns:
            One expression per output column.
        """
        built = self.build(*inputs)
        if isinstance(built, Sequence):
            return tuple(built)
        return (built,)

    @abstractmethod
    def build(self, *inputs: nw.Expr) -> nw.Expr | Sequence[nw.Expr]:
        """Build this primitive's narwhals expression.

        Args:
            *inputs: One expression per declared input.

        Returns:
            A single expression, or a sequence for multi-output primitives.
        """


class AggregationPrimitive(Primitive):
    """A primitive applied to a child table's rows, grouped by foreign key."""


class TransformPrimitive(Primitive):
    """A primitive applied row-wise within a single table.

    Attributes:
        order_dependent: Whether the expression needs an explicit ordering.
            Narwhals requires ``.over(order_by=...)`` for these on lazy
            backends, so tusk requires a ``row_creation_time`` on the table.
    """

    order_dependent: ClassVar[bool] = False
