"""Built-in transform primitives.

Group transforms subclass ``GroupTransformPrimitive`` and are wrapped by the
compiler in ``.over(foreign_key)``. Ordered transforms subclass
``OrderedTransformPrimitive`` and are wrapped in
``.over(foreign_key, order_by=...)``; narwhals requires the ordering on lazy
backends and raises otherwise.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import cast

import narwhals as nw

from tusk.dtypes import DtypeFamily as F
from tusk.exceptions import PrimitiveError
from tusk.primitives.base import (
    GroupTransformPrimitive,
    NeedsCutoffTime,
    OrderedTransformPrimitive,
    TransformPrimitive,
)
from tusk.primitives.registry import register

TRANS_DEFAULTS: tuple[str, ...] = ("year", "month", "day", "weekday")

_COMPARABLE_PAIRS = (
    (F.NUMERIC, F.NUMERIC),
    (F.HAS_DATE, F.HAS_DATE),
)
_EQUATABLE_PAIRS = (
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
class Minute(TransformPrimitive):
    """Minute of the hour, 0-59."""

    name = "minute"
    input_dtypes = (F.HAS_TIME,)
    output_dtype = nw.Int8

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the minute-of-hour expression.

        Args:
            expr: A temporal expression.

        Returns:
            A narwhals expression of the minute of the hour.
        """
        return expr.dt.minute()


@register
@dataclass(frozen=True)
class Second(TransformPrimitive):
    """Second of the minute, 0-59."""

    name = "second"
    input_dtypes = (F.HAS_TIME,)
    output_dtype = nw.Int8

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the second-of-minute expression.

        Args:
            expr: A temporal expression.

        Returns:
            A narwhals expression of the second of the minute.
        """
        return expr.dt.second()


@register
@dataclass(frozen=True)
class DayOfYear(TransformPrimitive):
    """Day of the year, 1-366."""

    name = "day_of_year"
    input_dtypes = (F.HAS_DATE,)
    output_dtype = nw.Int16

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the day-of-year expression.

        Args:
            expr: A temporal expression.

        Returns:
            A narwhals expression of the day of the year.
        """
        return expr.dt.ordinal_day()


@register
@dataclass(frozen=True)
class IsLeapYear(TransformPrimitive):
    """Whether the date falls in a leap year. A null date gives null."""

    name = "is_leap_year"
    input_dtypes = (F.HAS_DATE,)
    output_dtype = nw.Boolean

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the leap-year expression.

        Args:
            expr: A temporal expression.

        Returns:
            A narwhals boolean expression.
        """
        year = expr.dt.year().cast(nw.Int32)
        return ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)


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
    """Natural logarithm.

    A negative input gives null, where the unguarded answer varies by
    backend; zero gives negative infinity.
    """

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
        return nw.when(expr >= 0).then(expr.log())


@register
@dataclass(frozen=True)
class IsNull(TransformPrimitive):
    """Whether the value is null."""

    name = "is_null"
    input_dtypes = (F.ANY,)
    output_dtype = nw.Boolean

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the null-indicator expression.

        Args:
            expr: An expression of any dtype.

        Returns:
            A narwhals boolean expression.
        """
        return expr.is_null()


@register
@dataclass(frozen=True)
class Negate(TransformPrimitive):
    """The value with its sign flipped, as a float.

    An integer above 2**53 in magnitude, or a high-precision decimal, loses
    precision in the round trip through ``Float64``.
    """

    name = "negate"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the negation expression.

        Args:
            expr: A numeric expression.

        Returns:
            A narwhals expression of the negated values.
        """
        # an integer dtype's minimum has no negation inside the same dtype
        return expr.cast(nw.Float64) * -1


@register
@dataclass(frozen=True)
class SquareRoot(TransformPrimitive):
    """Square root.

    A negative input gives null, where the unguarded answer varies by
    backend; zero gives ``0.0``.
    """

    name = "square_root"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the square-root expression.

        Args:
            expr: A numeric expression.

        Returns:
            A narwhals expression of square roots.
        """
        return nw.when(expr >= 0).then(expr.sqrt())


@register
@dataclass(frozen=True)
class Sine(TransformPrimitive):
    """Sine of a value in radians."""

    name = "sine"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the sine expression.

        Args:
            expr: A numeric expression in radians.

        Returns:
            A narwhals expression of sines.
        """
        return expr.sin()


@register
@dataclass(frozen=True)
class Cosine(TransformPrimitive):
    """Cosine of a value in radians."""

    name = "cosine"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the cosine expression.

        Args:
            expr: A numeric expression in radians.

        Returns:
            A narwhals expression of cosines.
        """
        return expr.cos()


@register
@dataclass(frozen=True)
class NCharacters(TransformPrimitive):
    """Number of characters in a text string."""

    name = "n_characters"
    input_dtypes = (F.STRING,)
    output_dtype = nw.Int64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the character-count expression.

        Args:
            expr: A string expression.

        Returns:
            A narwhals expression of how many characters each value holds.
        """
        return expr.str.len_chars()


@register
@dataclass(frozen=True)
class NWords(TransformPrimitive):
    """Number of whitespace-separated words in a string."""

    name = "n_words"
    input_dtypes = (F.STRING,)
    output_dtype = nw.Int64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the word-count expression.

        Args:
            expr: A string expression.

        Returns:
            A narwhals expression of how many words each value holds.
        """
        words = _split_into_words(expr)
        return _count_unless_empty(words, words.str.split(" ").list.len())


@register
@dataclass(frozen=True)
class NUniqueWords(TransformPrimitive):
    """Number of distinct words in a string, ignoring case.

    Words are separated by whitespace and keep the punctuation inside them, so
    `a-b` is one word; punctuation around a word is not part of it, and a run of
    punctuation alone is not a word. A string holding no word at all is `0`.
    """

    name = "n_unique_words"
    input_dtypes = (F.STRING,)
    output_dtype = nw.Int64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the distinct-word-count expression.

        Args:
            expr: A string expression.

        Returns:
            A narwhals expression of how many distinct words each value holds.
        """
        words = _split_into_words(expr.str.to_lowercase())
        return _count_unless_empty(
            words,
            words.str.split(" ").list.unique().list.len(),
        )


@register
@dataclass(frozen=True)
class Domain(TransformPrimitive):
    """Domain of a URL or an email address.

    Whitespace around the value is ignored. The domain follows any scheme such
    as `https://`, anything up to the last `@` before the first `/` or `?`, and
    a leading `www.`, and runs up to the first `:`, `/` or `?`. A value that
    leaves no domain is null.
    """

    name = "domain"
    input_dtypes = (F.STRING,)
    output_dtype = nw.String

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the domain expression.

        Args:
            expr: A string expression of URLs or email addresses.

        Returns:
            A narwhals expression of each value's domain.
        """
        return _extract_domain(expr)


@register
@dataclass(frozen=True)
class TopLevelDomain(TransformPrimitive):
    """Top-level domain of a URL or an email address, after its domain's last dot.

    The domain is the one `domain` gives. A domain without a dot, or ending in
    one, is null.
    """

    name = "top_level_domain"
    input_dtypes = (F.STRING,)
    output_dtype = nw.String

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the top-level-domain expression.

        Args:
            expr: A string expression of URLs or email addresses.

        Returns:
            A narwhals expression of each value's top-level domain.
        """
        domain = _extract_domain(expr)
        return nw.when(domain.str.contains(r"\.[^.]+$")).then(
            domain.str.replace_all(r"^.*\.", "")
        )


@register
@dataclass(frozen=True)
class URLToProtocol(TransformPrimitive):
    """Protocol of a URL, `http` or `https`; null for any other or none."""

    name = "url_to_protocol"
    input_dtypes = (F.STRING,)
    output_dtype = nw.String

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the URL-protocol expression.

        Args:
            expr: A string expression of URLs.

        Returns:
            A narwhals expression of each URL's protocol.
        """
        return (
            nw.when(expr.str.starts_with("https:"))
            .then(nw.lit("https"))
            .otherwise(nw.when(expr.str.starts_with("http:")).then(nw.lit("http")))
        )


@register
@dataclass(frozen=True)
class Percentile(GroupTransformPrimitive):
    """Rank of the value among the known values of its group, from above 0 to 1.

    Tied values share their average rank. A null stays null and is not
    counted.
    """

    name = "percentile"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the percentile-rank expression.

        Args:
            expr: A numeric expression.

        Returns:
            A narwhals expression of percentile ranks.
        """
        return expr.rank("average") / expr.count()


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

    A null dividend or divisor gives a null. A zero divisor is
    backend-defined: duckdb nulls it for every numeric dtype, polars nulls it
    for integers but produces NaN for floats.
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
        # Backends disagree on %'s sign, so the divisor floors the remainder only when
        # the signs disagree, keeping the sum inside the input's dtype.
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
    """A number where the flag is true, zero where it is false. A null gives a null."""

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
    """Whether two label columns carry the same label. A null gives a null."""

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
        # compare the label, not its encoding
        return left.cast(nw.String) == right.cast(nw.String)


@register
@dataclass(frozen=True)
class NotEqualCategorical(TransformPrimitive):
    """Whether two label columns carry different labels. A null gives a null."""

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
        # compare the label, not its encoding
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
class CumSum(OrderedTransformPrimitive):
    """Running total in row-creation order."""

    name = "cum_sum"
    input_dtypes = (F.NUMERIC,)

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
class CumCount(OrderedTransformPrimitive):
    """Running count of non-null values in row-creation order."""

    name = "cum_count"
    input_dtypes = (F.ANY,)
    output_dtype = nw.Int64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the cumulative-count expression.

        Args:
            expr: An expression.

        Returns:
            A narwhals expression of cumulative count.
        """
        return expr.cum_count()


@register
@dataclass(frozen=True)
class CumMin(OrderedTransformPrimitive):
    """Running minimum in row-creation order."""

    name = "cum_min"
    input_dtypes = (F.NUMERIC,)

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
class CumMax(OrderedTransformPrimitive):
    """Running maximum in row-creation order."""

    name = "cum_max"
    input_dtypes = (F.NUMERIC,)

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
class Diff(OrderedTransformPrimitive):
    """Change from the previous row in row-creation order."""

    name = "diff"
    input_dtypes = (F.NUMERIC,)

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
class TimeSincePrevious(OrderedTransformPrimitive):
    """Time elapsed since the previous row in row-creation order."""

    name = "time_since_previous"
    input_dtypes = (F.HAS_DATE,)
    output_dtype = nw.Duration

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
class CumMean(OrderedTransformPrimitive):
    """Running mean of the known values in row-creation order. A null row is null."""

    name = "cum_mean"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the cumulative-mean expression.

        Args:
            expr: A numeric expression.

        Returns:
            A narwhals expression of the running mean.
        """
        return expr.cum_sum() / expr.cum_count()


@register
@dataclass(frozen=True)
class SameAsPrevious(OrderedTransformPrimitive):
    """Whether the value equals the previous row's in row-creation order.

    The first row, and a row whose own or previous value is null, is null.
    """

    name = "same_as_previous"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Boolean

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the equal-to-previous expression.

        Args:
            expr: A numeric expression.

        Returns:
            A narwhals boolean expression.
        """
        return expr == expr.shift(1)


@register
@dataclass(frozen=True)
class AbsoluteDiff(OrderedTransformPrimitive):
    """Size of the change from the previous row in row-creation order.

    The first row, and a row whose own or previous value is null, is null.
    An integer above 2**53 in magnitude, or a high-precision decimal, loses
    precision in the round trip through ``Float64``.
    """

    name = "absolute_diff"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the absolute-difference expression.

        Args:
            expr: A numeric expression.

        Returns:
            A narwhals expression of absolute differences.
        """
        # an integer dtype's difference can overflow or wrap around it
        return expr.cast(nw.Float64).diff().abs()


@register
@dataclass(frozen=True)
class PercentChange(OrderedTransformPrimitive):
    """Relative change from the previous row in row-creation order.

    The first row, and a row whose own or previous value is null, is null. A
    zero previous value gives positive or negative infinity, or NaN when the
    value is zero too.
    """

    name = "percent_change"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the relative-change expression.

        Args:
            expr: A numeric expression.

        Returns:
            A narwhals expression of relative changes, where 0.5 is a 50% rise.
        """
        return expr / expr.shift(1) - 1


@register
@dataclass(frozen=True)
class CumulativeTimeSinceLastTrue(OrderedTransformPrimitive):
    """Time elapsed since the latest row whose flag is true, in row-creation order.

    Null until the first true flag, and on a row whose datetime is null. A
    null flag is not true. A matching row whose datetime is null is skipped,
    so later rows measure from the match before it. For a time-zone-aware
    column, a gap across a daylight-saving change can vary by backend.
    """

    name = "cumulative_time_since_last_true"
    input_dtypes = (F.HAS_DATE, F.BOOLEAN)
    output_dtype = nw.Duration

    def build(self, moment: nw.Expr, flag: nw.Expr) -> nw.Expr:
        """Build the time-since-last-true expression.

        Args:
            moment: The datetime of each row.
            flag: The boolean flag of each row.

        Returns:
            A narwhals expression of the elapsed time.
        """
        return _time_since_last_match(moment, flag)


@register
@dataclass(frozen=True)
class CumulativeTimeSinceLastFalse(OrderedTransformPrimitive):
    """Time elapsed since the latest row whose flag is false, in row-creation order.

    Null until the first false flag, and on a row whose datetime is null. A
    null flag is not false. A matching row whose datetime is null is
    skipped, so later rows measure from the match before it. For a
    time-zone-aware column, a gap across a daylight-saving change can vary by
    backend.
    """

    name = "cumulative_time_since_last_false"
    input_dtypes = (F.HAS_DATE, F.BOOLEAN)
    output_dtype = nw.Duration

    def build(self, moment: nw.Expr, flag: nw.Expr) -> nw.Expr:
        """Build the time-since-last-false expression.

        Args:
            moment: The datetime of each row.
            flag: The boolean flag of each row.

        Returns:
            A narwhals expression of the elapsed time.
        """
        return _time_since_last_match(moment, ~flag)


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


@dataclass(frozen=True)
class _HolidayCalendarPrimitive(TransformPrimitive):
    """A transform that reads a date against a calendar of holidays.

    A datetime is read by its calendar date, and a null date gives null. The
    expression grows with each holiday, so a calendar of a few hundred dates
    is the intended scale.

    Attributes:
        input_dtypes: Tuple containing one dtype family (HAS_DATE).
        holidays: Each holiday's date mapped to its name. Stored as
            ``(date, name)`` pairs sorted by date, so equal calendars give
            equal primitives.

    Raises:
        PrimitiveError: If ``holidays`` is empty, or a key is a datetime or
            not a date at all.
    """

    input_dtypes = (F.HAS_DATE,)

    holidays: Mapping[date, str]

    def __post_init__(self) -> None:
        """Store the holidays as sorted ``(date, name)`` pairs."""
        object.__setattr__(self, "holidays", _sorted_holidays(self.holidays))

    @property
    def _holiday_pairs(self) -> tuple[tuple[date, str], ...]:
        """The holidays as ``(date, name)`` pairs sorted by date."""
        # The field is declared as the mapping callers pass, but holds the
        # pairs __post_init__ stored, which keep the primitive hashable.
        return cast("tuple[tuple[date, str], ...]", self.holidays)

    @property
    def _holiday_day_numbers(self) -> list[int]:
        """The holidays' day numbers, as ``date.toordinal`` counts them, in order."""
        return [holiday.toordinal() for holiday, _ in self._holiday_pairs]


@register
@dataclass(frozen=True)
class IsHoliday(_HolidayCalendarPrimitive):
    """Whether the date is one of the given holidays.

    ``holidays`` maps each holiday's date to its name, as in
    ``{date(2024, 12, 25): "Christmas"}``.
    """

    name = "is_holiday"
    output_dtype = nw.Boolean

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the is-a-holiday expression.

        Args:
            expr: A date or datetime expression.

        Returns:
            A narwhals expression of whether each date is a holiday.
        """
        return expr.dt.date().is_in([holiday for holiday, _ in self._holiday_pairs])


@register
@dataclass(frozen=True)
class HolidayName(_HolidayCalendarPrimitive):
    """Name of the holiday on the date; null on any other day.

    ``holidays`` maps each holiday's date to its name, as in
    ``{date(2024, 12, 25): "Christmas"}``.
    """

    name = "holiday_name"
    output_dtype = nw.String

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the holiday-name expression.

        Args:
            expr: A date or datetime expression.

        Returns:
            A narwhals expression of each date's holiday name.
        """
        return expr.dt.date().replace_strict(
            dict(self._holiday_pairs),
            default=None,
            return_dtype=nw.String,
        )


@register
@dataclass(frozen=True)
class DaysToHoliday(_HolidayCalendarPrimitive):
    """Signed days from the date to the nearest holiday; 0 on a holiday.

    Positive when the nearest holiday is ahead, negative when it is behind.
    Between two equally near holidays, the one behind counts.

    ``holidays`` maps each holiday's date to its name, as in
    ``{date(2024, 12, 25): "Christmas"}``.
    """

    name = "days_to_holiday"
    output_dtype = nw.Int64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the days-to-nearest-holiday expression.

        Args:
            expr: A date or datetime expression.

        Returns:
            A narwhals expression of the signed days to each nearest holiday.
        """
        return _days_to_nearest(_day_number(expr), self._holiday_day_numbers)


@register
@dataclass(frozen=True)
class DaysUntilHoliday(_HolidayCalendarPrimitive):
    """Days from the date to the next holiday; 0 on a holiday.

    Null after the last holiday.

    ``holidays`` maps each holiday's date to its name, as in
    ``{date(2024, 12, 25): "Christmas"}``.
    """

    name = "days_until_holiday"
    output_dtype = nw.Int64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the days-until-next-holiday expression.

        Args:
            expr: A date or datetime expression.

        Returns:
            A narwhals expression of the days until each next holiday.
        """
        return _days_until_next(_day_number(expr), self._holiday_day_numbers)


@register
@dataclass(frozen=True)
class DaysSinceHoliday(_HolidayCalendarPrimitive):
    """Days from the last holiday to the date; 0 on a holiday.

    Null before the first holiday.

    ``holidays`` maps each holiday's date to its name, as in
    ``{date(2024, 12, 25): "Christmas"}``.
    """

    name = "days_since_holiday"
    output_dtype = nw.Int64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the days-since-last-holiday expression.

        Args:
            expr: A date or datetime expression.

        Returns:
            A narwhals expression of the days since each last holiday.
        """
        return _days_since_last(_day_number(expr), self._holiday_day_numbers)


def _time_since_last_match(moment: nw.Expr, is_match: nw.Expr) -> nw.Expr:
    """Build the time elapsed since the latest row that matches.

    Args:
        moment: The datetime of each row.
        is_match: Where the row matches; a null does not match.

    Returns:
        A narwhals expression of the elapsed time, to be ordered by the caller.
    """
    # duckdb subtracts two Dates into a day count instead of an interval
    timestamps = moment.cast(nw.Datetime)
    latest_match = nw.when(is_match).then(timestamps).fill_null(strategy="forward")
    return timestamps - latest_match


def _split_into_words(expr: nw.Expr) -> nw.Expr:
    """Reduce a string to its words, separated by one space each.

    Words are whitespace-separated and keep the punctuation inside them, so
    ``"a-b"`` is one word; punctuation around a word is not part of it, and a
    run of punctuation alone is not a word.

    Args:
        expr: A string expression.

    Returns:
        A narwhals expression of the words, joined by single spaces, empty
        where the value holds no word at all.
    """
    # Codepoint ranges rather than escaped literals: polars' regex engine
    # rejects a redundant escape such as `\!` inside a character class.
    punctuation = r"!-/:-@\x5B-\x60{-~"
    collapsed = expr.str.replace_all(r"\s+", " ")
    return (
        collapsed.str.replace_all(f"(^|[ ])[{punctuation}]+", " ")
        .str.replace_all(f"[{punctuation}]+([ ]|$)", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def _count_unless_empty(words: nw.Expr, count: nw.Expr) -> nw.Expr:
    """Apply a counting expression, answering zero where there is no word.

    Args:
        words: A string expression of words joined by single spaces.
        count: A counting expression over ``words``.

    Returns:
        A narwhals expression of the count, zero where ``words`` is empty,
        which splits into one empty word rather than into no word.
    """
    return nw.when(words == "").then(nw.lit(0)).otherwise(count)


def _extract_domain(address: nw.Expr) -> nw.Expr:
    """Extract the domain of a URL or an email address.

    Args:
        address: A string expression of URLs or email addresses.

    Returns:
        A narwhals expression of each value's domain, null where there is none.
    """
    domain = (
        address.str.strip_chars()
        .str.replace_all(r"^[A-Za-z][A-Za-z0-9+.-]*://", "")
        .str.replace_all(r"^[^/?]*@", "")
        .str.replace_all(r"^www\.", "")
        .str.replace_all(r"[:/?][\s\S]*$", "")
    )
    return nw.when(domain != "").then(domain)


def _sorted_holidays(
    holidays: Mapping[date, str] | Iterable[tuple[date, str]],
) -> tuple[tuple[date, str], ...]:
    """Validate a holiday calendar and sort it by date.

    Args:
        holidays: Each holiday's date mapped to its name, or the
            ``(date, name)`` pairs.

    Returns:
        The ``(date, name)`` pairs, sorted by date.

    Raises:
        PrimitiveError: If ``holidays`` is empty, or a key is a datetime or
            not a date at all.
    """
    names_by_date = dict(holidays)
    if not names_by_date:
        raise PrimitiveError("holidays is empty; pass at least one date")
    for holiday in names_by_date:
        _require_date(holiday)
    return tuple(sorted(names_by_date.items()))


def _require_date(holiday: object) -> None:
    """Check that a holiday is a plain date.

    Args:
        holiday: One key of a holiday calendar.

    Raises:
        PrimitiveError: If it is a datetime, or not a date at all.
    """
    # datetime subclasses date; accepting one would silently drop its time.
    if isinstance(holiday, datetime):
        raise PrimitiveError(
            f"holiday {holiday!r} is a datetime; pass its date, {holiday.date()!r}",
        )
    if not isinstance(holiday, date):
        raise PrimitiveError(f"holiday {holiday!r} is not a date; pass a date")


def _day_number(expr: nw.Expr) -> nw.Expr:
    """Number each date's day as ``date.toordinal`` does, 0001-01-01 being day 1.

    Args:
        expr: A date or datetime expression.

    Returns:
        A narwhals expression of each value's day number.
    """
    # Built from year and day of year because no backend-neutral route turns
    # a duration into days: duckdb subtracts two Dates into an integer, and
    # its interval keeps whole days apart from the seconds.
    years_before = expr.dt.year() - 1
    leap_days_before = years_before // 4 - years_before // 100 + years_before // 400
    return years_before * 365 + leap_days_before + expr.dt.ordinal_day()


def _days_to_nearest(day_number: nw.Expr, holiday_day_numbers: list[int]) -> nw.Expr:
    """Count the signed days to the nearest holiday; the earlier one on a tie.

    Args:
        day_number: Each value's day number.
        holiday_day_numbers: The holidays' day numbers, in ascending order.

    Returns:
        A narwhals expression of the days, negative for a holiday before.
    """
    nearest_holiday = _pick_holiday(
        holiday_day_numbers,
        lambda earlier, later: day_number * 2 > earlier + later,
    )
    return nearest_holiday - day_number


def _days_until_next(day_number: nw.Expr, holiday_day_numbers: list[int]) -> nw.Expr:
    """Count the days until the next holiday on or after each day.

    Args:
        day_number: Each value's day number.
        holiday_day_numbers: The holidays' day numbers, in ascending order.

    Returns:
        A narwhals expression of the days, null after the last holiday.
    """
    next_holiday = _pick_holiday(
        holiday_day_numbers,
        lambda earlier, later: day_number > earlier,
    )
    is_before_last = day_number <= holiday_day_numbers[-1]
    return nw.when(is_before_last).then(next_holiday - day_number)


def _days_since_last(day_number: nw.Expr, holiday_day_numbers: list[int]) -> nw.Expr:
    """Count the days since the last holiday on or before each day.

    Args:
        day_number: Each value's day number.
        holiday_day_numbers: The holidays' day numbers, in ascending order.

    Returns:
        A narwhals expression of the days, null before the first holiday.
    """
    last_holiday = _pick_holiday(
        holiday_day_numbers,
        lambda earlier, later: day_number >= later,
    )
    is_after_first = day_number >= holiday_day_numbers[0]
    return nw.when(is_after_first).then(day_number - last_holiday)


def _pick_holiday(
    holiday_day_numbers: list[int],
    lies_in_later_half: Callable[[int, int], nw.Expr],
) -> nw.Expr:
    """Pick one holiday per row by halving the holidays until one is left.

    Args:
        holiday_day_numbers: The holidays' day numbers, in ascending order.
        lies_in_later_half: Given the last day number of an earlier half and
            the first of the later half, builds whether a row's holiday lies in
            the later half.

    Returns:
        A narwhals expression of the picked holiday's day number.
    """
    # Halving rather than one condition per holiday keeps the nesting at a
    # logarithmic depth; narwhals recurses once per nested condition.
    if len(holiday_day_numbers) == 1:
        return nw.lit(holiday_day_numbers[0])
    middle = len(holiday_day_numbers) // 2
    earlier, later = holiday_day_numbers[:middle], holiday_day_numbers[middle:]
    return (
        nw.when(lies_in_later_half(earlier[-1], later[0]))
        .then(_pick_holiday(later, lies_in_later_half))
        .otherwise(_pick_holiday(earlier, lies_in_later_half))
    )
