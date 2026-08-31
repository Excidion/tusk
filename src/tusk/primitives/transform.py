"""Built-in transform primitives.

Primitives with ``order_dependent = True`` must be wrapped by the compiler in
``.over(..., order_by=...)``; narwhals requires this on lazy backends and will
raise otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import narwhals as nw

from tusk.dtypes import DtypeFamily as F
from tusk.primitives.base import NeedsCutoffTime, TransformPrimitive
from tusk.primitives.registry import register

TRANS_DEFAULTS: tuple[str, ...] = ("year", "month", "weekday")


@register
@dataclass(frozen=True)
class Year(TransformPrimitive):
    """Calendar year."""

    name = "year"
    input_dtypes = (F.DATETIME,)
    output_dtype = nw.Int32

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the calendar-year expression.

        Args:
            expr: A temporal expression.

        Returns:
            A narwhals expression of the calendar year.
        """
        return expr.dt.year()


@register
@dataclass(frozen=True)
class Month(TransformPrimitive):
    """Calendar month, 1-12."""

    name = "month"
    input_dtypes = (F.DATETIME,)
    output_dtype = nw.Int8

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the calendar-month expression.

        Args:
            expr: A temporal expression.

        Returns:
            A narwhals expression of the calendar month.
        """
        return expr.dt.month()


@register
@dataclass(frozen=True)
class Day(TransformPrimitive):
    """Day of month, 1-31."""

    name = "day"
    input_dtypes = (F.DATETIME,)
    output_dtype = nw.Int8

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the day-of-month expression.

        Args:
            expr: A temporal expression.

        Returns:
            A narwhals expression of the day of month.
        """
        return expr.dt.day()


@register
@dataclass(frozen=True)
class Hour(TransformPrimitive):
    """Hour of day, 0-23."""

    name = "hour"
    input_dtypes = (F.TIMESTAMP,)
    output_dtype = nw.Int8

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the hour-of-day expression.

        Args:
            expr: A temporal expression.

        Returns:
            A narwhals expression of the hour of day.
        """
        return expr.dt.hour()


@register
@dataclass(frozen=True)
class Weekday(TransformPrimitive):
    """ISO weekday, 1 (Monday) to 7 (Sunday)."""

    name = "weekday"
    input_dtypes = (F.DATETIME,)
    output_dtype = nw.Int8

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the ISO-weekday expression.

        Args:
            expr: A temporal expression.

        Returns:
            A narwhals expression of the ISO weekday.
        """
        return expr.dt.weekday()


@register
@dataclass(frozen=True)
class IsWeekend(TransformPrimitive):
    """Whether the date falls on a Saturday or Sunday."""

    name = "is_weekend"
    input_dtypes = (F.DATETIME,)
    output_dtype = nw.Boolean

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the weekend-indicator expression.

        Args:
            expr: A temporal expression.

        Returns:
            A narwhals boolean expression.
        """
        return expr.dt.weekday() >= 6


@register
@dataclass(frozen=True)
class Absolute(TransformPrimitive):
    """Absolute value."""

    name = "absolute"
    input_dtypes = (F.NUMERIC,)

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the absolute-value expression.

        Args:
            expr: A numeric expression.

        Returns:
            A narwhals expression with absolute values.
        """
        return expr.abs()


@register
@dataclass(frozen=True)
class NaturalLog(TransformPrimitive):
    """Natural logarithm. Non-positive inputs yield null or negative infinity."""

    name = "natural_log"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the natural-logarithm expression.

        Args:
            expr: A numeric expression.

        Returns:
            A narwhals expression of natural logarithms.
        """
        return expr.log()


@register
@dataclass(frozen=True)
class SubtractNumeric(TransformPrimitive):
    """Difference of two numeric columns."""

    name = "subtract_numeric"
    input_dtypes = (F.NUMERIC, F.NUMERIC)

    def build(self, left: nw.Expr, right: nw.Expr) -> nw.Expr:
        """Build the subtraction expression.

        Args:
            left: First numeric expression.
            right: Second numeric expression.

        Returns:
            A narwhals expression of the difference.
        """
        return left - right


@register
@dataclass(frozen=True)
class DivideNumeric(TransformPrimitive):
    """Ratio of two numeric columns."""

    name = "divide_numeric"
    input_dtypes = (F.NUMERIC, F.NUMERIC)
    output_dtype = nw.Float64

    def build(self, left: nw.Expr, right: nw.Expr) -> nw.Expr:
        """Build the division expression.

        Args:
            left: First numeric expression (numerator).
            right: Second numeric expression (denominator).

        Returns:
            A narwhals expression of the ratio.
        """
        return left / right


@register
@dataclass(frozen=True)
class AddNumeric(TransformPrimitive):
    """Sum of two numeric columns."""

    name = "add_numeric"
    input_dtypes = (F.NUMERIC, F.NUMERIC)
    commutative = True

    def build(self, left: nw.Expr, right: nw.Expr) -> nw.Expr:
        """Build the addition expression.

        Args:
            left: First column.
            right: Second column.

        Returns:
            A narwhals expression.
        """
        return left + right


@register
@dataclass(frozen=True)
class MultiplyNumeric(TransformPrimitive):
    """Product of two numeric columns."""

    name = "multiply_numeric"
    input_dtypes = (F.NUMERIC, F.NUMERIC)
    commutative = True

    def build(self, left: nw.Expr, right: nw.Expr) -> nw.Expr:
        """Build the multiplication expression.

        Args:
            left: First column.
            right: Second column.

        Returns:
            A narwhals expression.
        """
        return left * right


@register
@dataclass(frozen=True)
class CumSum(TransformPrimitive):
    """Running total in row-creation order."""

    name = "cum_sum"
    input_dtypes = (F.NUMERIC,)
    order_dependent = True

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the cumulative-sum expression.

        Args:
            expr: A numeric expression.

        Returns:
            A narwhals expression of cumulative sum.
        """
        return expr.cum_sum()


@register
@dataclass(frozen=True)
class CumCount(TransformPrimitive):
    """Running count of non-null values in row-creation order."""

    name = "cum_count"
    input_dtypes = (F.ANY,)
    output_dtype = nw.Int64
    order_dependent = True

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the cumulative-count expression.

        Args:
            expr: An expression.

        Returns:
            A narwhals expression of cumulative count.
        """
        return expr.cum_count().cast(nw.Int64)


@register
@dataclass(frozen=True)
class CumMin(TransformPrimitive):
    """Running minimum in row-creation order."""

    name = "cum_min"
    input_dtypes = (F.NUMERIC,)
    order_dependent = True

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the cumulative-minimum expression.

        Args:
            expr: A numeric expression.

        Returns:
            A narwhals expression of cumulative minimum.
        """
        return expr.cum_min()


@register
@dataclass(frozen=True)
class CumMax(TransformPrimitive):
    """Running maximum in row-creation order."""

    name = "cum_max"
    input_dtypes = (F.NUMERIC,)
    order_dependent = True

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the cumulative-maximum expression.

        Args:
            expr: A numeric expression.

        Returns:
            A narwhals expression of cumulative maximum.
        """
        return expr.cum_max()


@register
@dataclass(frozen=True)
class Diff(TransformPrimitive):
    """Change from the previous row in row-creation order."""

    name = "diff"
    input_dtypes = (F.NUMERIC,)
    order_dependent = True

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the row-difference expression.

        Args:
            expr: A numeric expression.

        Returns:
            A narwhals expression of differences.
        """
        return expr.diff()


@register
@dataclass(frozen=True)
class TimeSincePrevious(TransformPrimitive):
    """Seconds elapsed since the previous row in row-creation order."""

    name = "time_since_previous"
    input_dtypes = (F.DATETIME,)
    output_dtype = nw.Float64
    order_dependent = True

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the elapsed-seconds expression.

        Args:
            expr: A temporal expression.

        Returns:
            A narwhals expression of time elapsed in seconds.
        """
        return expr.diff().dt.total_seconds().cast(nw.Float64)


@register
@dataclass(frozen=True)
class TimeSince(NeedsCutoffTime, TransformPrimitive):
    """Time elapsed from a datetime to the cutoff time."""

    name = "time_since"
    input_dtypes = (F.DATETIME,)
    output_dtype = nw.Duration

    def build(self, expr: nw.Expr, *, cutoff_time: datetime) -> nw.Expr:
        """Build the elapsed-time expression.

        Args:
            expr: A datetime expression.
            cutoff_time: The moment the values are measured against.

        Returns:
            A narwhals expression of the duration since each value.
        """
        return nw.lit(cutoff_time) - expr
