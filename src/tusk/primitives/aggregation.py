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
    """Number of distinct values in a column."""

    name = "n_unique"
    input_dtypes = (F.ANY,)
    output_dtype = nw.Int64
    default_value = 0
    stack_on_self = False

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the distinct-count expression.

        Args:
            expr: The column to count distinct values of.

        Returns:
            A narwhals expression.
        """
        return expr.n_unique().cast(nw.Int64)


@register
@dataclass(frozen=True)
class PercentTrue(AggregationPrimitive):
    """Fraction of rows where a boolean column is true."""

    name = "percent_true"
    input_dtypes = (F.BOOLEAN,)
    output_dtype = nw.Float64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the true-fraction expression.

        Args:
            expr: The boolean column.

        Returns:
            A narwhals expression.
        """
        return expr.cast(nw.Int64).mean()


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
