"""Primitive protocol.

A primitive builds narwhals expressions and never sees a value. That is what
lets feature computation push down into the backend engine. Featuretools'
model -- a callable over a materialized pandas Series -- is deliberately not
used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from typing import Any, ClassVar, cast

import narwhals as nw

from tusk.dtypes import DtypeFamily
from tusk.exceptions import PrimitiveError


class Primitive(ABC):
    """Base class for every primitive.

    Attributes:
        name: Registry key, and the default :attr:`stem` of generated names.
        input_dtypes: One dtype family per input, or several such tuples if
            the primitive accepts alternative input shapes. Empty means the
            primitive takes no column input, e.g. ``count``. Read it through
            :attr:`signatures`.
        output_dtype: Fixed output dtype that every output column is cast
            to, or None to preserve the first input's.
        commutative: Whether argument order is irrelevant, so that only one
            of ``f(a, b)`` and ``f(b, a)`` is generated.
        stack_on_self: Whether this primitive may be applied to its own
            output.
        default_value: Value substituted for empty groups after a left join.
    """

    name: ClassVar[str]
    input_dtypes: ClassVar[
        tuple[DtypeFamily, ...] | tuple[tuple[DtypeFamily, ...], ...]
    ] = ()
    output_dtype: ClassVar[Any] = None
    commutative: ClassVar[bool] = False
    stack_on_self: ClassVar[bool] = True
    default_value: ClassVar[Any] = None

    @property
    def stem(self) -> str:
        """The upper-cased start of every name this primitive generates."""
        return self.name.upper()

    @property
    def number_of_outputs(self) -> int:
        """How many columns this primitive produces."""
        return 1

    @property
    def signatures(self) -> tuple[tuple[DtypeFamily, ...], ...]:
        """Every input shape this primitive accepts.

        Returns:
            One tuple of dtype families per accepted shape. A primitive that
            takes no column input has none.

        Raises:
            PrimitiveError: If input_dtypes mixes a flat shape with nested
                alternatives, declares an alternative that is not a
                non-empty tuple of dtype families, or the shapes do not all
                take the same number of inputs.
        """
        declared = self.input_dtypes
        if not declared:
            return ()
        if all(isinstance(family, DtypeFamily) for family in declared):
            # A type checker cannot narrow the declared union from this
            # element-wise check, so the flat shape is asserted explicitly.
            return (cast("tuple[DtypeFamily, ...]", declared),)
        signatures = _validated_alternatives(self.name, declared)
        if len({len(signature) for signature in signatures}) > 1:
            raise PrimitiveError(
                f"primitive {self.name!r} declares input shapes that do not "
                f"all take the same number of inputs; build() has one "
                f"parameter list, so every shape must match it.",
            )
        return signatures

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
        """Build the column name for an application of this primitive.

        Every part is joined with ``__`` so the result is a plain SQL
        identifier. Parentheses and commas would be parsed as a function call
        by any backend that generates SQL; see
        :meth:`generate_display_name` for the readable form.

        Args:
            arg_names: Names of the inputs. For a zero-input aggregation
                this is the child table's name, giving e.g.
                ``COUNT__transactions``.

        Returns:
            The feature name.
        """
        return "__".join([self.stem, *arg_names])

    def generate_display_name(self, arg_names: Sequence[str]) -> str:
        """Build the readable name for an application of this primitive.

        Args:
            arg_names: Display names of the inputs.

        Returns:
            The conventional parenthesised form, e.g. ``MEAN(amount)``.
        """
        return f"{self.stem}({', '.join(arg_names)})"

    def output_names(self, base_name: str) -> tuple[str, ...]:
        """Expand a feature name into one name per output column.

        Args:
            base_name: The name from :meth:`generate_name`.

        Returns:
            One name per output column; indexed when there is more than one.
        """
        if self.number_of_outputs == 1:
            return (base_name,)
        return tuple(f"{base_name}__{i}" for i in range(self.number_of_outputs))

    def display_output_names(self, base_name: str) -> tuple[str, ...]:
        """Expand a display name into one readable name per output column.

        Args:
            base_name: The name from :meth:`generate_display_name`.

        Returns:
            One name per output column; indexed when there is more than one.
        """
        if self.number_of_outputs == 1:
            return (base_name,)
        return tuple(f"{base_name}[{i}]" for i in range(self.number_of_outputs))

    def outputs(self, *inputs: nw.Expr) -> tuple[nw.Expr, ...]:
        """Normalize :meth:`build` to a tuple of expressions of the output dtype.

        Args:
            *inputs: One expression per declared input.

        Returns:
            One expression per output column, cast to :attr:`output_dtype`.
        """
        return _cast_to_output_dtype(self.output_dtype, self.build(*inputs))

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


class OrderedAggregationPrimitive(AggregationPrimitive):
    """An aggregation that reads its group's rows in ``row_creation_time`` order.

    The compiler passes the ordering columns in at the moment the expression
    is built, so the child table needs a ``row_creation_time``.
    """

    def outputs(
        self,
        *inputs: nw.Expr,
        order_by: Sequence[str],
    ) -> tuple[nw.Expr, ...]:
        """Normalize :meth:`build` to a tuple of expressions of the output dtype.

        Args:
            *inputs: One expression per declared input.
            order_by: The columns that order the group's rows.

        Returns:
            One expression per output column, cast to :attr:`output_dtype`.
        """
        return _cast_to_output_dtype(
            self.output_dtype,
            self.build(*inputs, order_by=order_by),
        )

    @abstractmethod
    def build(
        self,
        *inputs: nw.Expr,
        order_by: Sequence[str],
    ) -> nw.Expr | Sequence[nw.Expr]:
        """Build this primitive's narwhals expression.

        Args:
            *inputs: One expression per declared input.
            order_by: The columns that order the group's rows.

        Returns:
            A single expression, or a sequence for multi-output primitives.
        """


class GroupRelativeAggregationPrimitive(AggregationPrimitive):
    """An aggregation that measures each row against its group before reducing.

    SQL backends reject an aggregate nested in an aggregate, such as
    ``SUM(x > AVG(x))``. The compiler therefore adds :meth:`build_per_row`
    to the child as a column computed within each foreign-key group, and
    :meth:`build` and :meth:`outputs` take that one column as their input.
    Its per-row expression must read an aggregate of the group, such as
    ``expr.mean()``.
    """

    @abstractmethod
    def build_per_row(self, *inputs: nw.Expr) -> nw.Expr:
        """Build the per-row expression, which may read its group's aggregates.

        Args:
            *inputs: One expression per declared input.

        Returns:
            A narwhals expression with one value per row.
        """

    @abstractmethod
    def build(self, per_row: nw.Expr) -> nw.Expr:
        """Build the expression reducing the per-row column.

        Args:
            per_row: The column :meth:`build_per_row` produced.

        Returns:
            A narwhals expression.
        """


class ValueCountAggregationPrimitive(GroupRelativeAggregationPrimitive):
    """A group-relative aggregation reading each row's value and its count.

    It takes one input column. :meth:`build_per_row` receives that column and,
    per row, how often the row's value occurs in its group.
    """

    @abstractmethod
    def build_per_row(self, values: nw.Expr, counts: nw.Expr) -> nw.Expr:
        """Build the per-row expression from each row's value and its count.

        Args:
            values: The input column.
            counts: How often each row's value occurs in its group; null
                where the value is null.

        Returns:
            A narwhals expression with one value per row.
        """


class TransformPrimitive(Primitive):
    """A primitive applied row-wise within a single table."""


class GroupTransformPrimitive(TransformPrimitive):
    """A transform that reads the other rows sharing its row's foreign key.

    Synthesis applies it within each foreign-key group: the compiler wraps its
    expression in ``.over(foreign_key)``.
    """


class OrderedTransformPrimitive(GroupTransformPrimitive):
    """A group transform that reads its group's rows in ``row_creation_time`` order.

    The compiler wraps its expression in
    ``.over(foreign_key, order_by=(row_creation_time, primary_key))``, so the
    table needs a ``row_creation_time``.
    """


class NeedsCutoffTime(Primitive):
    """A primitive that measures against the cutoff time.

    The cutoff time describes the question being asked rather than the
    feature, so it is never stored on a primitive: the compiler passes it in
    at the moment the expression is built. That is what lets one
    :class:`~tusk.FeatureList` be applied at several cutoff times.
    """

    def outputs(self, *inputs: nw.Expr, cutoff_time: datetime) -> tuple[nw.Expr, ...]:
        """Normalize :meth:`build` to a tuple of expressions of the output dtype.

        Args:
            *inputs: One expression per declared input.
            cutoff_time: The moment the values are measured against.

        Returns:
            One expression per output column, cast to :attr:`output_dtype`.
        """
        return _cast_to_output_dtype(
            self.output_dtype,
            self.build(*inputs, cutoff_time=cutoff_time),
        )

    @abstractmethod
    def build(
        self,
        *inputs: nw.Expr,
        cutoff_time: datetime,
    ) -> nw.Expr | Sequence[nw.Expr]:
        """Build this primitive's narwhals expression.

        Args:
            *inputs: One expression per declared input.
            cutoff_time: The moment the values are measured against.

        Returns:
            A single expression, or a sequence for multi-output primitives.
        """


def _validated_alternatives(
    name: str,
    declared: tuple[Any, ...],
) -> tuple[tuple[DtypeFamily, ...], ...]:
    """Validate a declared input_dtypes known not to be a single flat shape.

    Args:
        name: The declaring primitive's name, for the error message.
        declared: The primitive's raw ``input_dtypes``.

    Returns:
        The declared alternatives, unchanged.

    Raises:
        PrimitiveError: If any alternative is not a non-empty tuple of
            dtype families.
    """
    for signature in declared:
        if (
            not isinstance(signature, tuple)
            or not signature
            or not all(isinstance(family, DtypeFamily) for family in signature)
        ):
            raise PrimitiveError(
                f"primitive {name!r} declares input_dtypes={declared!r}; each "
                f"alternative must be a non-empty tuple of DtypeFamily "
                f"members, e.g. (F.NUMERIC, F.NUMERIC).",
            )
    return cast("tuple[tuple[DtypeFamily, ...], ...]", declared)


def _cast_to_output_dtype(
    output_dtype: Any,
    built: nw.Expr | Sequence[nw.Expr],
) -> tuple[nw.Expr, ...]:
    """Cast what a primitive built to its declared output dtype.

    Args:
        output_dtype: The primitive's ``output_dtype``.
        built: What :meth:`Primitive.build` returned.

    Returns:
        One expression per output column; uncast when ``output_dtype`` is
        None or a parametric dtype class without its parameters.
    """
    expressions = _as_tuple(built)
    if output_dtype is None or _is_bare_parametric_dtype(output_dtype):
        return expressions
    return tuple(expression.cast(output_dtype) for expression in expressions)


def _is_bare_parametric_dtype(dtype: Any) -> bool:
    """Report whether a dtype is a parametric dtype class, not an instance.

    Args:
        dtype: A narwhals dtype class or instance.

    Returns:
        True for e.g. ``nw.Duration``, False for ``nw.Duration("ms")``.
    """
    # Casting to the bare class would impose narwhals' default parameters,
    # e.g. turn a backend's millisecond Duration into microseconds.
    parametric_dtypes = (
        nw.Array,
        nw.Datetime,
        nw.Decimal,
        nw.Duration,
        nw.Enum,
        nw.List,
        nw.Struct,
    )
    return isinstance(dtype, type) and issubclass(dtype, parametric_dtypes)


def _as_tuple(built: nw.Expr | Sequence[nw.Expr]) -> tuple[nw.Expr, ...]:
    """Normalize what a primitive built to a tuple of expressions.

    Args:
        built: What :meth:`Primitive.build` returned.

    Returns:
        One expression per output column.
    """
    if isinstance(built, Sequence):
        return tuple(built)
    return (built,)
