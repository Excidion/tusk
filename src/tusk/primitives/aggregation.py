"""Built-in aggregation primitives.

Every expression here is legal inside a lazy ``group_by().agg()``. Length-changing
expressions such as ``mode()`` are not -- narwhals rejects them on lazy frames --
which is why ``quantiles`` rather than ``n_most_common`` is the multi-output
primitive.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import narwhals as nw

from tusk.dtypes import DtypeFamily as F
from tusk.primitives.base import AggregationPrimitive
from tusk.primitives.registry import register

AGG_DEFAULTS: tuple[str, ...] = (
    "count",
    "sum",
    "mean",
    "min",
    "max",
    "std",
    "n_unique",
)


@register
@dataclass(frozen=True)
class Count(AggregationPrimitive):
    """Number of child rows in the group."""

    name = "count"
    input_dtypes = ()
    output_dtype = nw.Int64
    default_value = 0
    stack_on_self = False

    def build(self) -> nw.Expr:
        """Build the row-count expression.

        Returns:
            A narwhals expression counting rows.
        """
        return nw.len().cast(nw.Int64)


@register
@dataclass(frozen=True)
class Sum(AggregationPrimitive):
    """Sum of a numeric column."""

    name = "sum"
    input_dtypes = (F.NUMERIC,)
    # Sum over an empty group is 0, the additive identity -- not unknown. We
    # already report COUNT = 0 for the same rows, asserting we know there were
    # zero child rows; a null total would contradict that known-zero count.
    # Do not remove this to "fix" a perceived inconsistency with MEAN/MIN/MAX:
    # those are genuinely undefined over an empty set (0/0, and min/max of
    # nothing), which is why they stay null.
    default_value = 0

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the sum expression.

        Args:
            expr: The column to sum.

        Returns:
            A narwhals expression.
        """
        return expr.sum()


@register
@dataclass(frozen=True)
class Mean(AggregationPrimitive):
    """Arithmetic mean of a numeric column."""

    name = "mean"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the mean expression.

        Args:
            expr: The column to average.

        Returns:
            A narwhals expression.
        """
        return expr.mean()


@register
@dataclass(frozen=True)
class Min(AggregationPrimitive):
    """Smallest value of a numeric column."""

    name = "min"
    input_dtypes = (F.NUMERIC,)

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the minimum expression.

        Args:
            expr: The column to reduce.

        Returns:
            A narwhals expression.
        """
        return expr.min()


@register
@dataclass(frozen=True)
class Max(AggregationPrimitive):
    """Largest value of a numeric column."""

    name = "max"
    input_dtypes = (F.NUMERIC,)

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the maximum expression.

        Args:
            expr: The column to reduce.

        Returns:
            A narwhals expression.
        """
        return expr.max()


@register
@dataclass(frozen=True)
class Std(AggregationPrimitive):
    """Sample standard deviation of a numeric column."""

    name = "std"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the standard-deviation expression.

        Args:
            expr: The column to reduce.

        Returns:
            A narwhals expression.
        """
        return expr.std()


@register
@dataclass(frozen=True)
class Median(AggregationPrimitive):
    """Median of a numeric column."""

    name = "median"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the median expression.

        Args:
            expr: The column to reduce.

        Returns:
            A narwhals expression.
        """
        return expr.median()


@register
@dataclass(frozen=True)
class NUnique(AggregationPrimitive):
    """Number of distinct *known* values in a column; nulls are not a value."""

    name = "n_unique"
    input_dtypes = (F.ANY,)
    output_dtype = nw.Int64
    # No rows means 0 distinct values -- and, by the same logic, so does a
    # group whose only rows are null.
    default_value = 0
    stack_on_self = False

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the distinct-count expression, excluding null.

        Polars counts null as one more distinct value, which contradicts this
        primitive's own ``default_value``: a customer whose only session had no
        transactions would report 0 rows and 1 distinct value at once.
        featuretools' ``NUM_UNIQUE`` excludes nulls, so counting them would
        also diverge silently on any data containing them.

        Subtracting a null indicator is deliberate rather than
        ``expr.drop_nulls().n_unique()``: ``drop_nulls`` is a length-changing
        (filtration) expression, and narwhals rejects those inside a lazy
        ``group_by().agg()`` on backends that cannot express them. Both
        operands here are plain reductions.

        Args:
            expr: The column to count distinct values of.

        Returns:
            A narwhals expression.
        """
        return (expr.n_unique() - expr.is_null().any().cast(nw.Int64)).cast(nw.Int64)


@register
@dataclass(frozen=True)
class PercentTrue(AggregationPrimitive):
    """Fraction of rows where a boolean column is true; a null counts as false."""

    name = "percent_true"
    input_dtypes = (F.BOOLEAN,)
    output_dtype = nw.Float64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the true-fraction expression, treating null as false.

        Matches featuretools' ``PercentTrue``, which fills nulls with
        ``False`` before averaging, so a null lowers the fraction instead of
        being skipped. ``fill_null`` is elementwise rather than
        length-changing, so -- like ``cast`` -- it composes with the
        trailing ``mean()`` reduction the way a filtration such as
        ``drop_nulls`` could not.

        Args:
            expr: The boolean column.

        Returns:
            A narwhals expression.
        """
        return expr.fill_null(False).cast(nw.Int64).mean()


@register
@dataclass(frozen=True)
class Quantiles(AggregationPrimitive):
    """Several quantiles of a numeric column, one output column per quantile.

    Attributes:
        name: Registry key.
        input_dtypes: Tuple containing one dtype family (NUMERIC).
        output_dtype: The output dtype (Float64).
        qs: The quantiles to compute, each in [0, 1].
    """

    name = "quantiles"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64

    qs: tuple[float, ...] = field(default=(0.25, 0.5, 0.75))

    @property
    def number_of_outputs(self) -> int:
        """One output column per requested quantile."""
        return len(self.qs)

    def build(self, expr: nw.Expr) -> Sequence[nw.Expr]:
        """Build one expression per quantile.

        Args:
            expr: The column to reduce.

        Returns:
            One narwhals expression per quantile.
        """
        return [expr.quantile(q, interpolation="linear") for q in self.qs]
