"""Built-in aggregation primitives.

Every expression here is legal inside a lazy ``group_by().agg()``. Length-changing
expressions such as ``mode()`` are not -- narwhals rejects them on lazy frames --
which is why ``quantiles`` rather than ``n_most_common`` is the multi-output
primitive.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

import narwhals as nw

from tusk.dtypes import DtypeFamily as F
from tusk.primitives.base import AggregationPrimitive, NeedsCutoffTime
from tusk.primitives.registry import register

AGG_DEFAULTS: tuple[str, ...] = (
    "count",
    "sum",
    "mean",
    "min",
    "max",
    "std",
    "n_unique",
    "skew",
    "percent_true",
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
    """Number of distinct values in a column; a null counts as one value."""

    name = "n_unique"
    input_dtypes = (F.ANY,)
    output_dtype = nw.Int64
    # Zero rows means zero distinct values. A group whose only rows are null
    # is 1, not 0, since null now counts as a value.
    default_value = 0
    stack_on_self = False

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the distinct-count expression, counting null as a value.

        Args:
            expr: The column to count distinct values of.

        Returns:
            A narwhals expression.
        """
        # polars' n_unique returns UInt32, not the declared Int64; duckdb's
        # is already Int64.
        return expr.n_unique().cast(nw.Int64)


@register
@dataclass(frozen=True)
class PercentTrue(AggregationPrimitive):
    """Fraction of rows where a boolean column is true; a null counts as false."""

    name = "percent_true"
    input_dtypes = (F.BOOLEAN,)
    output_dtype = nw.Float64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the true-fraction expression, treating null as false.

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


@register
@dataclass(frozen=True)
class TimeSinceFirst(NeedsCutoffTime, AggregationPrimitive):
    """Time elapsed from the group's earliest datetime to the cutoff time."""

    name = "time_since_first"
    input_dtypes = (F.HAS_DATE,)
    output_dtype = nw.Duration

    def build(self, expr: nw.Expr, *, cutoff_time: datetime) -> nw.Expr:
        """Build the elapsed-time expression.

        Args:
            expr: The datetime column to reduce.
            cutoff_time: The moment the values are measured against.

        Returns:
            A narwhals expression.
        """
        return nw.lit(cutoff_time) - expr.min()


@register
@dataclass(frozen=True)
class TimeSinceLast(NeedsCutoffTime, AggregationPrimitive):
    """Time elapsed from the group's latest datetime to the cutoff time."""

    name = "time_since_last"
    input_dtypes = (F.HAS_DATE,)
    output_dtype = nw.Duration

    def build(self, expr: nw.Expr, *, cutoff_time: datetime) -> nw.Expr:
        """Build the elapsed-time expression.

        Args:
            expr: The datetime column to reduce.
            cutoff_time: The moment the values are measured against.

        Returns:
            A narwhals expression.
        """
        return nw.lit(cutoff_time) - expr.max()


@register
@dataclass(frozen=True)
class TimeSinceLastTrue(NeedsCutoffTime, AggregationPrimitive):
    """Time elapsed from the group's latest true row to the cutoff time."""

    name = "time_since_last_true"
    input_dtypes = (F.HAS_DATE, F.BOOLEAN)
    output_dtype = nw.Duration

    def build(
        self,
        timestamps: nw.Expr,
        flags: nw.Expr,
        *,
        cutoff_time: datetime,
    ) -> nw.Expr:
        """Build the elapsed-time expression.

        Args:
            timestamps: The datetime column the elapsed time is measured from.
            flags: The boolean column selecting which rows count.
            cutoff_time: The moment the values are measured against.

        Returns:
            A narwhals expression.
        """
        return _time_since_last_selected(timestamps, flags, cutoff_time)


@register
@dataclass(frozen=True)
class TimeSinceLastFalse(NeedsCutoffTime, AggregationPrimitive):
    """Time elapsed from the group's latest false row to the cutoff time."""

    name = "time_since_last_false"
    input_dtypes = (F.HAS_DATE, F.BOOLEAN)
    output_dtype = nw.Duration

    def build(
        self,
        timestamps: nw.Expr,
        flags: nw.Expr,
        *,
        cutoff_time: datetime,
    ) -> nw.Expr:
        """Build the elapsed-time expression.

        Args:
            timestamps: The datetime column the elapsed time is measured from.
            flags: The boolean column selecting which rows count.
            cutoff_time: The moment the values are measured against.

        Returns:
            A narwhals expression.
        """
        return _time_since_last_selected(timestamps, ~flags, cutoff_time)


@register
@dataclass(frozen=True)
class AllTrue(AggregationPrimitive):
    """Whether every known value of a boolean column is true."""

    name = "all_true"
    input_dtypes = (F.BOOLEAN,)
    output_dtype = nw.Boolean

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the every-value-true expression, ignoring nulls.

        Args:
            expr: The boolean column.

        Returns:
            A narwhals expression.
        """
        return expr.all()


@register
@dataclass(frozen=True)
class AnyTrue(AggregationPrimitive):
    """Whether any known value of a boolean column is true."""

    name = "any_true"
    input_dtypes = (F.BOOLEAN,)
    output_dtype = nw.Boolean
    default_value = False

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the some-value-true expression, ignoring nulls.

        Args:
            expr: The boolean column.

        Returns:
            A narwhals expression.
        """
        return expr.any()


@register
@dataclass(frozen=True)
class NTrue(AggregationPrimitive):
    """Number of rows where a boolean column is true."""

    name = "n_true"
    input_dtypes = (F.BOOLEAN,)
    output_dtype = nw.Int64
    default_value = 0

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the true-count expression; a null is not true.

        Args:
            expr: The boolean column.

        Returns:
            A narwhals expression.
        """
        # duckdb's SUM widens the integer type
        return expr.fill_null(False).cast(nw.Int64).sum().cast(nw.Int64)


@register
@dataclass(frozen=True)
class Skew(AggregationPrimitive):
    """Skewness of a numeric column, without bias correction."""

    name = "skew"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the skewness expression; a constant group is null.

        Args:
            expr: The column to reduce.

        Returns:
            A narwhals expression.
        """
        return _where_the_column_varies(expr, expr.skew())


@register
@dataclass(frozen=True)
class Kurtosis(AggregationPrimitive):
    """Excess kurtosis of a numeric column, without bias correction."""

    name = "kurtosis"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the excess-kurtosis expression; a constant group is null.

        Args:
            expr: The column to reduce.

        Returns:
            A narwhals expression.
        """
        return _where_the_column_varies(expr, expr.kurtosis())


@register
@dataclass(frozen=True)
class Variance(AggregationPrimitive):
    """Sample variance of a numeric column."""

    name = "variance"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the variance expression.

        Args:
            expr: The column to reduce.

        Returns:
            A narwhals expression.
        """
        return expr.var()


@register
@dataclass(frozen=True)
class MaxMinDelta(AggregationPrimitive):
    """Difference between the largest and smallest value of a numeric column."""

    name = "max_min_delta"
    input_dtypes = (F.NUMERIC,)

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the largest-minus-smallest expression.

        Args:
            expr: The column to reduce.

        Returns:
            A narwhals expression.
        """
        return expr.max() - expr.min()


@register
@dataclass(frozen=True)
class IsUnique(AggregationPrimitive):
    """Whether no value of a column repeats; a null counts as a value."""

    name = "is_unique"
    input_dtypes = (F.ANY,)
    output_dtype = nw.Boolean

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the no-value-repeats expression.

        Args:
            expr: The column to test.

        Returns:
            A narwhals expression.
        """
        return expr.n_unique() == nw.len()


@register
@dataclass(frozen=True)
class PercentUnique(AggregationPrimitive):
    """Distinct values of a column as a fraction of its rows."""

    name = "percent_unique"
    input_dtypes = (F.ANY,)
    output_dtype = nw.Float64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the distinct-fraction expression; a null counts as a value and a row.

        Args:
            expr: The column to count distinct values of.

        Returns:
            A narwhals expression.
        """
        return expr.n_unique() / nw.len()


@register
@dataclass(frozen=True)
class FirstLastTimeDelta(AggregationPrimitive):
    """Time between a group's earliest and latest datetime."""

    name = "first_last_time_delta"
    input_dtypes = (F.HAS_DATE,)
    output_dtype = nw.Duration

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the latest-minus-earliest expression.

        Args:
            expr: The datetime column to reduce.

        Returns:
            A narwhals expression.
        """
        # duckdb subtracts two Dates into a day count instead of an interval
        timestamps = expr.cast(nw.Datetime)
        return timestamps.max() - timestamps.min()


@register
@dataclass(frozen=True)
class NUniqueDays(AggregationPrimitive):
    """Number of distinct calendar dates in a datetime column."""

    name = "n_unique_days"
    input_dtypes = (F.HAS_DATE,)
    output_dtype = nw.Int64
    default_value = 0

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the distinct-date count; a null counts as one value.

        Args:
            expr: The datetime column to reduce.

        Returns:
            A narwhals expression.
        """
        # polars' n_unique returns UInt32, not the declared Int64.
        return expr.dt.date().n_unique().cast(nw.Int64)


@register
@dataclass(frozen=True)
class NUniqueDaysOfCalendarYear(AggregationPrimitive):
    """Number of distinct month-and-day pairs in a datetime column."""

    name = "n_unique_days_of_calendar_year"
    input_dtypes = (F.HAS_DATE,)
    output_dtype = nw.Int64
    default_value = 0

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the distinct month-and-day count; a null counts as one value.

        Args:
            expr: The datetime column to reduce.

        Returns:
            A narwhals expression.
        """
        # month and day come back as Int8, which month * 100 overflows
        month = expr.dt.month().cast(nw.Int32)
        day = expr.dt.day().cast(nw.Int32)
        # polars' n_unique returns UInt32, not the declared Int64.
        return (month * 100 + day).n_unique().cast(nw.Int64)


@register
@dataclass(frozen=True)
class NUniqueDaysOfMonth(AggregationPrimitive):
    """Number of distinct days of the month in a datetime column."""

    name = "n_unique_days_of_month"
    input_dtypes = (F.HAS_DATE,)
    output_dtype = nw.Int64
    default_value = 0

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the distinct day-of-month count; a null counts as one value.

        Args:
            expr: The datetime column to reduce.

        Returns:
            A narwhals expression.
        """
        # polars' n_unique returns UInt32, not the declared Int64.
        return expr.dt.day().n_unique().cast(nw.Int64)


@register
@dataclass(frozen=True)
class NUniqueMonths(AggregationPrimitive):
    """Number of distinct calendar months, year included, in a datetime column."""

    name = "n_unique_months"
    input_dtypes = (F.HAS_DATE,)
    output_dtype = nw.Int64
    default_value = 0

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the distinct year-and-month count; a null counts as one value.

        Args:
            expr: The datetime column to reduce.

        Returns:
            A narwhals expression.
        """
        # year * 12 + month must not overflow the narrow ints year()/month() return
        year = expr.dt.year().cast(nw.Int32)
        month = expr.dt.month().cast(nw.Int32)
        # polars' n_unique returns UInt32, not the declared Int64.
        return (year * 12 + month).n_unique().cast(nw.Int64)


def _time_since_last_selected(
    timestamps: nw.Expr,
    selected: nw.Expr,
    cutoff_time: datetime,
) -> nw.Expr:
    """Build the time from the latest selected row's datetime to the cutoff.

    Rows are blanked rather than filtered out because ``expr.filter()`` is a
    length-changing expression, which narwhals rejects inside a lazy
    ``group_by().agg()``. A null flag selects no row, so a group without a
    selected row reduces to null.

    Args:
        timestamps: The datetime column the elapsed time is measured from.
        selected: The boolean column marking which rows count.
        cutoff_time: The moment the values are measured against.

    Returns:
        A narwhals expression.
    """
    return nw.lit(cutoff_time) - nw.when(selected).then(timestamps).max()


def _where_the_column_varies(expr: nw.Expr, moment: nw.Expr) -> nw.Expr:
    """Keep a standardized moment only for a group whose values vary.

    Args:
        expr: The column the moment was computed from.
        moment: The moment's expression.

    Returns:
        The moment, or null where the column's standard deviation is zero or
        unknown.
    """
    # A constant group divides zero by zero, which polars answers with NaN
    # and duckdb with 0.0 or null.
    return nw.when(expr.std() > 0).then(moment)
