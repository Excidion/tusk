"""Built-in transform primitives.

Primitives with ``order_dependent = True`` must be wrapped by the compiler in
``.over(..., order_by=...)``; narwhals requires this on lazy backends and will
raise otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import narwhals as nw

from tusk.dtypes import DtypeFamily as F
from tusk.primitives.base import TransformPrimitive
from tusk.primitives.registry import register, transform_primitive

TRANS_DEFAULTS: tuple[str, ...] = ("year", "month", "weekday")


@transform_primitive(name="year", input_dtypes=(F.TEMPORAL,), output_dtype=nw.Int32)
def year(expr: nw.Expr) -> nw.Expr:
    """Calendar year.

    Args:
        expr: A temporal expression.

    Returns:
        A narwhals expression of the calendar year.
    """
    return expr.dt.year()


@transform_primitive(name="month", input_dtypes=(F.TEMPORAL,), output_dtype=nw.Int8)
def month(expr: nw.Expr) -> nw.Expr:
    """Calendar month, 1-12.

    Args:
        expr: A temporal expression.

    Returns:
        A narwhals expression of the calendar month.
    """
    return expr.dt.month()


@transform_primitive(name="day", input_dtypes=(F.TEMPORAL,), output_dtype=nw.Int8)
def day(expr: nw.Expr) -> nw.Expr:
    """Day of month, 1-31.

    Args:
        expr: A temporal expression.

    Returns:
        A narwhals expression of the day of month.
    """
    return expr.dt.day()


@transform_primitive(name="hour", input_dtypes=(F.TEMPORAL,), output_dtype=nw.Int8)
def hour(expr: nw.Expr) -> nw.Expr:
    """Hour of day, 0-23.

    Args:
        expr: A temporal expression.

    Returns:
        A narwhals expression of the hour of day.
    """
    return expr.dt.hour()


@transform_primitive(name="weekday", input_dtypes=(F.TEMPORAL,), output_dtype=nw.Int8)
def weekday(expr: nw.Expr) -> nw.Expr:
    """ISO weekday, 1 (Monday) to 7 (Sunday).

    Args:
        expr: A temporal expression.

    Returns:
        A narwhals expression of the ISO weekday.
    """
    return expr.dt.weekday()


@transform_primitive(
    name="is_weekend", input_dtypes=(F.TEMPORAL,), output_dtype=nw.Boolean
)
def is_weekend(expr: nw.Expr) -> nw.Expr:
    """Whether the date falls on a Saturday or Sunday.

    Args:
        expr: A temporal expression.

    Returns:
        A narwhals boolean expression.
    """
    return expr.dt.weekday() >= 6


@transform_primitive(name="absolute", input_dtypes=(F.NUMERIC,))
def absolute(expr: nw.Expr) -> nw.Expr:
    """Absolute value.

    Args:
        expr: A numeric expression.

    Returns:
        A narwhals expression with absolute values.
    """
    return expr.abs()


@transform_primitive(
    name="natural_log", input_dtypes=(F.NUMERIC,), output_dtype=nw.Float64
)
def natural_log(expr: nw.Expr) -> nw.Expr:
    """Natural logarithm. Non-positive inputs yield null or negative infinity.

    Args:
        expr: A numeric expression.

    Returns:
        A narwhals expression of natural logarithms.
    """
    return expr.log()


@transform_primitive(name="subtract_numeric", input_dtypes=(F.NUMERIC, F.NUMERIC))
def subtract_numeric(left: nw.Expr, right: nw.Expr) -> nw.Expr:
    """Difference of two numeric columns.

    Args:
        left: First numeric expression.
        right: Second numeric expression.

    Returns:
        A narwhals expression of the difference.
    """
    return left - right


@transform_primitive(
    name="divide_numeric", input_dtypes=(F.NUMERIC, F.NUMERIC), output_dtype=nw.Float64
)
def divide_numeric(left: nw.Expr, right: nw.Expr) -> nw.Expr:
    """Ratio of two numeric columns.

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


@transform_primitive(name="cum_sum", input_dtypes=(F.NUMERIC,), order_dependent=True)
def cum_sum(expr: nw.Expr) -> nw.Expr:
    """Running total in row-creation order.

    Args:
        expr: A numeric expression.

    Returns:
        A narwhals expression of cumulative sum.
    """
    return expr.cum_sum()


@transform_primitive(
    name="cum_count", input_dtypes=(F.ANY,), output_dtype=nw.Int64, order_dependent=True
)
def cum_count(expr: nw.Expr) -> nw.Expr:
    """Running count of non-null values in row-creation order.

    Args:
        expr: An expression.

    Returns:
        A narwhals expression of cumulative count.
    """
    return expr.cum_count().cast(nw.Int64)


@transform_primitive(name="cum_min", input_dtypes=(F.NUMERIC,), order_dependent=True)
def cum_min(expr: nw.Expr) -> nw.Expr:
    """Running minimum in row-creation order.

    Args:
        expr: A numeric expression.

    Returns:
        A narwhals expression of cumulative minimum.
    """
    return expr.cum_min()


@transform_primitive(name="cum_max", input_dtypes=(F.NUMERIC,), order_dependent=True)
def cum_max(expr: nw.Expr) -> nw.Expr:
    """Running maximum in row-creation order.

    Args:
        expr: A numeric expression.

    Returns:
        A narwhals expression of cumulative maximum.
    """
    return expr.cum_max()


@transform_primitive(name="diff", input_dtypes=(F.NUMERIC,), order_dependent=True)
def diff(expr: nw.Expr) -> nw.Expr:
    """Change from the previous row in row-creation order.

    Args:
        expr: A numeric expression.

    Returns:
        A narwhals expression of differences.
    """
    return expr.diff()


@transform_primitive(
    name="time_since_previous",
    input_dtypes=(F.TEMPORAL,),
    output_dtype=nw.Float64,
    order_dependent=True,
)
def time_since_previous(expr: nw.Expr) -> nw.Expr:
    """Seconds elapsed since the previous row in row-creation order.

    Args:
        expr: A temporal expression.

    Returns:
        A narwhals expression of time elapsed in seconds.
    """
    return expr.diff().dt.total_seconds().cast(nw.Float64)
