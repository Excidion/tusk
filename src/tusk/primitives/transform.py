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

_COMPARABLE_PAIRS: tuple[tuple[F, ...], ...] = (
    (F.NUMERIC, F.NUMERIC),
    (F.HAS_DATE, F.HAS_DATE),
)
_EQUATABLE_PAIRS: tuple[tuple[F, ...], ...] = (
    *_COMPARABLE_PAIRS,
    (F.BOOLEAN, F.BOOLEAN),
    (F.STRING, F.STRING),
)


@register
@dataclass(frozen=True)
class Year(TransformPrimitive):
    """Calendar year."""

    name = "year"
    input_dtypes = (F.HAS_DATE,)
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
    input_dtypes = (F.HAS_DATE,)
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
    input_dtypes = (F.HAS_DATE,)
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
    input_dtypes = (F.HAS_TIME,)
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
    input_dtypes = (F.HAS_DATE,)
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
    input_dtypes = (F.HAS_DATE,)
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
class ModuloNumeric(TransformPrimitive):
    """Remainder after division, taking the sign of the divisor.

    A zero divisor is backend-defined: duckdb nulls it for every numeric
    dtype, polars nulls it for integers but produces NaN for floats.
    """

    name = "modulo_numeric"
    input_dtypes = (F.NUMERIC, F.NUMERIC)

    def build(self, left: nw.Expr, right: nw.Expr) -> nw.Expr:
        """Build the floored-modulo expression.

        Args:
            left: The dividend.
            right: The divisor.

        Returns:
            A narwhals expression of the remainder.
        """
        # `%` alone truncates on duckdb but floors on polars: -7 % 2 is -1 on
        # one and 1 on the other. Adding the divisor back turns a truncated
        # remainder into a floored one, but only when the signs disagree —
        # that's the one case where |remainder| < |right| already guarantees
        # the sum can't overflow the input's dtype. Doing it unconditionally
        # (as an earlier version did) overflows narrow integer types even
        # though the true remainder fits, e.g. Int16 20000 % 30000.
        remainder = left % right
        return (
            nw.when((remainder != 0) & ((remainder < 0) != (right < 0)))
            .then(
                remainder + right,
            )
            .otherwise(remainder)
        )


@register
@dataclass(frozen=True)
class MultiplyNumericBoolean(TransformPrimitive):
    """A number where the flag is true, zero where it is false."""

    name = "multiply_numeric_boolean"
    input_dtypes = (F.NUMERIC, F.BOOLEAN)

    def build(self, left: nw.Expr, right: nw.Expr) -> nw.Expr:
        """Build the masking expression.

        Args:
            left: A numeric expression.
            right: A boolean expression.

        Returns:
            A narwhals expression of the number masked by the flag.
        """
        return left * right.cast(nw.Int8)


@register
@dataclass(frozen=True)
class GreaterThan(TransformPrimitive):
    """Whether the first value exceeds the second. A null gives a null."""

    name = "greater_than"
    input_dtypes = _COMPARABLE_PAIRS
    output_dtype = nw.Boolean

    def build(self, left: nw.Expr, right: nw.Expr) -> nw.Expr:
        """Build the greater-than expression.

        Args:
            left: First comparable expression.
            right: Second comparable expression.

        Returns:
            A narwhals expression that is true where left exceeds right.
        """
        return left > right


@register
@dataclass(frozen=True)
class GreaterThanEqualTo(TransformPrimitive):
    """Whether the first value is at least the second. A null gives a null."""

    name = "greater_than_equal_to"
    input_dtypes = _COMPARABLE_PAIRS
    output_dtype = nw.Boolean

    def build(self, left: nw.Expr, right: nw.Expr) -> nw.Expr:
        """Build the greater-than-or-equal expression.

        Args:
            left: First comparable expression.
            right: Second comparable expression.

        Returns:
            A narwhals expression that is true where left is at least right.
        """
        return left >= right


@register
@dataclass(frozen=True)
class LessThan(TransformPrimitive):
    """Whether the first value is less than the second. A null gives a null."""

    name = "less_than"
    input_dtypes = _COMPARABLE_PAIRS
    output_dtype = nw.Boolean

    def build(self, left: nw.Expr, right: nw.Expr) -> nw.Expr:
        """Build the less-than expression.

        Args:
            left: First comparable expression.
            right: Second comparable expression.

        Returns:
            A narwhals expression that is true where left is less than right.
        """
        return left < right


@register
@dataclass(frozen=True)
class LessThanEqualTo(TransformPrimitive):
    """Whether the first value is at most the second. A null gives a null."""

    name = "less_than_equal_to"
    input_dtypes = _COMPARABLE_PAIRS
    output_dtype = nw.Boolean

    def build(self, left: nw.Expr, right: nw.Expr) -> nw.Expr:
        """Build the less-than-or-equal expression.

        Args:
            left: First comparable expression.
            right: Second comparable expression.

        Returns:
            A narwhals expression that is true where left is at most right.
        """
        return left <= right


@register
@dataclass(frozen=True)
class Equal(TransformPrimitive):
    """Whether two values are equal. A null gives a null."""

    name = "equal"
    input_dtypes = _EQUATABLE_PAIRS
    output_dtype = nw.Boolean
    commutative = True

    def build(self, left: nw.Expr, right: nw.Expr) -> nw.Expr:
        """Build the equality expression.

        Args:
            left: First comparable expression.
            right: Second comparable expression.

        Returns:
            A narwhals expression that is true where left equals right.
        """
        return left == right


@register
@dataclass(frozen=True)
class NotEqual(TransformPrimitive):
    """Whether two values differ. A null gives a null."""

    name = "not_equal"
    input_dtypes = _EQUATABLE_PAIRS
    output_dtype = nw.Boolean
    commutative = True

    def build(self, left: nw.Expr, right: nw.Expr) -> nw.Expr:
        """Build the inequality expression.

        Args:
            left: First comparable expression.
            right: Second comparable expression.

        Returns:
            A narwhals expression that is true where left differs from right.
        """
        return left != right


@register
@dataclass(frozen=True)
class EqualCategorical(TransformPrimitive):
    """Whether two label columns carry the same label."""

    name = "equal_categorical"
    input_dtypes = (F.CATEGORICAL, F.CATEGORICAL)
    output_dtype = nw.Boolean
    commutative = True

    def build(self, left: nw.Expr, right: nw.Expr) -> nw.Expr:
        """Build the label-equality expression.

        Args:
            left: First categorical expression.
            right: Second categorical expression.

        Returns:
            A narwhals expression that is true where the labels match.
        """
        # Compared as text because two Enum columns with different member
        # lists cannot be compared directly on polars, though duckdb allows
        # it. The label is the value; its encoding is not.
        return left.cast(nw.String) == right.cast(nw.String)


@register
@dataclass(frozen=True)
class NotEqualCategorical(TransformPrimitive):
    """Whether two label columns carry different labels."""

    name = "not_equal_categorical"
    input_dtypes = (F.CATEGORICAL, F.CATEGORICAL)
    output_dtype = nw.Boolean
    commutative = True

    def build(self, left: nw.Expr, right: nw.Expr) -> nw.Expr:
        """Build the label-inequality expression.

        Args:
            left: First categorical expression.
            right: Second categorical expression.

        Returns:
            A narwhals expression that is true where the labels differ.
        """
        # Compared as text because two Enum columns with different member
        # lists cannot be compared directly on polars, though duckdb allows
        # it. The label is the value; its encoding is not.
        return left.cast(nw.String) != right.cast(nw.String)


@register
@dataclass(frozen=True)
class Not(TransformPrimitive):
    """Logical negation. A null stays null."""

    name = "not"
    input_dtypes = (F.BOOLEAN,)
    output_dtype = nw.Boolean
    stack_on_self = False

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the negation expression.

        Args:
            expr: A boolean expression.

        Returns:
            A narwhals expression of the negated values.
        """
        return ~expr


@register
@dataclass(frozen=True)
class And(TransformPrimitive):
    """Logical conjunction of two boolean columns."""

    name = "and"
    input_dtypes = (F.BOOLEAN, F.BOOLEAN)
    output_dtype = nw.Boolean
    commutative = True

    def build(self, left: nw.Expr, right: nw.Expr) -> nw.Expr:
        """Build the conjunction expression.

        Args:
            left: First boolean expression.
            right: Second boolean expression.

        Returns:
            A narwhals expression that is true where both are true.
        """
        return left & right


@register
@dataclass(frozen=True)
class Or(TransformPrimitive):
    """Logical disjunction of two boolean columns."""

    name = "or"
    input_dtypes = (F.BOOLEAN, F.BOOLEAN)
    output_dtype = nw.Boolean
    commutative = True

    def build(self, left: nw.Expr, right: nw.Expr) -> nw.Expr:
        """Build the disjunction expression.

        Args:
            left: First boolean expression.
            right: Second boolean expression.

        Returns:
            A narwhals expression that is true where either is true.
        """
        return left | right


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
    """Time elapsed since the previous row in row-creation order."""

    name = "time_since_previous"
    input_dtypes = (F.HAS_DATE,)
    output_dtype = nw.Duration
    order_dependent = True

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the elapsed-time expression.

        Args:
            expr: A datetime expression.

        Returns:
            A narwhals expression of the duration since the previous row.
        """
        return expr.diff()


@register
@dataclass(frozen=True)
class TimeSince(NeedsCutoffTime, TransformPrimitive):
    """Time elapsed from a datetime to the cutoff time."""

    name = "time_since"
    input_dtypes = (F.HAS_DATE,)
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
