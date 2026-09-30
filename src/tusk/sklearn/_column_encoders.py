"""Encoders for the columns of one narwhals dtype.

:class:`EnumEncoder` encodes ``Enum`` columns as codes. :class:`DateEncoder`,
:class:`TimeEncoder`, :class:`DatetimeEncoder` and :class:`DurationEncoder`
encode temporal columns as numeric components. :class:`StringEncoder`
encodes ``String`` columns as coordinates of their character n-grams.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import narwhals as nw
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from tusk.exceptions import EncoderError
from tusk.sklearn._narwhals import NarwhalsEncoder, collect

DATE_COMPONENTS = ("year", "month", "day", "weekday", "ordinal_day", "timestamp")
TIME_COMPONENTS = (
    "hour",
    "minute",
    "second",
    "millisecond",
    "microsecond",
    "nanosecond",
)
DURATION_COMPONENTS = (
    "total_minutes",
    "total_seconds",
    "total_milliseconds",
    "total_microseconds",
    "total_nanoseconds",
)


class EnumEncoder(NarwhalsEncoder):
    """An encoder of ``Enum`` columns as their position in the category order.

    The categories come from the dtype, so fitting reads no rows. A null
    becomes NaN. Output name: ``{column}_code``.
    """

    def _fit(self, table: nw.DataFrame | nw.LazyFrame, y: Any) -> None:
        """Check that every column is an ``Enum``.

        Args:
            table: The input table.
            y: Ignored.
        """
        reject_other_dtypes(self.schema_in_, nw.Enum, type(self).__name__)

    def _expressions(self) -> list[nw.Expr]:
        """Return one code expression per column.

        Returns:
            The expressions.
        """
        return [
            code_expression(name, list(dtype.categories))
            for name, dtype in self.schema_in_.items()
        ]

    def _output_names(self) -> list[str]:
        """Return ``{column}_code`` per column.

        Returns:
            The names.
        """
        return [f"{name}_code" for name in self.schema_in_]


class TemporalEncoder(NarwhalsEncoder):
    """The base of the encoders of temporal columns as numeric components.

    A subclass sets ``accepted_dtype`` and ``allowed_components``, and takes
    ``components`` in its ``__init__``. Each component is a method of
    narwhals' ``dt`` namespace. Output name: ``{column}_{component}``.

    Attributes:
        accepted_dtype: The dtype class the encoder encodes.
        allowed_components: The components ``components`` may name.
        components: The components to compute, in output order.
    """

    accepted_dtype: ClassVar[type[nw.dtypes.DType]]
    allowed_components: ClassVar[tuple[str, ...]]
    components: Sequence[str]

    def _fit(self, table: nw.DataFrame | nw.LazyFrame, y: Any) -> None:
        """Check the components and the column dtypes.

        Args:
            table: The input table.
            y: Ignored.

        Raises:
            TypeError: If ``components`` is a string.
            ValueError: If a component is not in ``allowed_components``.
        """
        if isinstance(self.components, str):
            raise TypeError(
                f"{type(self).__name__} takes components as a list, such as "
                f"[{self.allowed_components[0]!r}], not the string "
                f"{self.components!r}",
            )
        unrecognized = [c for c in self.components if c not in self.allowed_components]
        if unrecognized:
            raise ValueError(
                f"{type(self).__name__} has no component {unrecognized[0]!r}; "
                f"choose from {list(self.allowed_components)}",
            )
        reject_other_dtypes(self.schema_in_, self.accepted_dtype, type(self).__name__)

    def _expressions(self) -> list[nw.Expr]:
        """Return one expression per column and component.

        Returns:
            The expressions, column by column.
        """
        return [
            getattr(nw.col(name).dt, component)().alias(f"{name}_{component}")
            for name in self.schema_in_
            for component in self.components
        ]

    def _output_names(self) -> list[str]:
        """Return ``{column}_{component}``, column by column.

        Returns:
            The names.
        """
        return [
            f"{name}_{component}"
            for name in self.schema_in_
            for component in self.components
        ]


class DateEncoder(TemporalEncoder):
    """An encoder of ``Date`` columns as calendar components.

    Attributes:
        accepted_dtype: ``Date``.
        allowed_components: ``year``, ``month``, ``day``, ``weekday``,
            ``ordinal_day`` and ``timestamp``.
    """

    accepted_dtype = nw.Date
    allowed_components = DATE_COMPONENTS

    def __init__(self, components: Sequence[str] = ("month", "day")) -> None:
        """Configure the components.

        Args:
            components: Any of ``year``, ``month``, ``day``, ``weekday``,
                ``ordinal_day`` and ``timestamp``.
        """
        self.components = components


class TimeEncoder(TemporalEncoder):
    """An encoder of ``Time`` columns as clock components.

    Attributes:
        accepted_dtype: ``Time``.
        allowed_components: ``hour``, ``minute``, ``second``,
            ``millisecond``, ``microsecond`` and ``nanosecond``.
    """

    accepted_dtype = nw.Time
    allowed_components = TIME_COMPONENTS

    def __init__(self, components: Sequence[str] = ("hour", "minute")) -> None:
        """Configure the components.

        Args:
            components: Any of ``hour``, ``minute``, ``second``,
                ``millisecond``, ``microsecond`` and ``nanosecond``.
        """
        self.components = components


class DatetimeEncoder(TemporalEncoder):
    """An encoder of ``Datetime`` columns as calendar and clock components.

    Attributes:
        accepted_dtype: ``Datetime``.
        allowed_components: The components of :class:`DateEncoder` and
            :class:`TimeEncoder`.
    """

    accepted_dtype = nw.Datetime
    allowed_components = DATE_COMPONENTS + TIME_COMPONENTS

    def __init__(
        self, components: Sequence[str] = ("month", "day", "hour", "minute")
    ) -> None:
        """Configure the components.

        Args:
            components: Any component of :class:`DateEncoder` or
                :class:`TimeEncoder`.
        """
        self.components = components


class DurationEncoder(TemporalEncoder):
    """An encoder of ``Duration`` columns as their total length.

    Attributes:
        accepted_dtype: ``Duration``.
        allowed_components: ``total_minutes``, ``total_seconds``,
            ``total_milliseconds``, ``total_microseconds`` and
            ``total_nanoseconds``.
    """

    accepted_dtype = nw.Duration
    allowed_components = DURATION_COMPONENTS

    def __init__(self, components: Sequence[str] = ("total_seconds",)) -> None:
        """Configure the components.

        Args:
            components: Any of ``total_minutes``, ``total_seconds``,
                ``total_milliseconds``, ``total_microseconds`` and
                ``total_nanoseconds``.
        """
        self.components = components


class StringEncoder(NarwhalsEncoder):
    """An encoder of ``String`` columns as coordinates of their character n-grams.

    Each column is encoded separately. A value becomes a TF-IDF vector over
    the character 3- and 4-grams seen at fit. ``TruncatedSVD`` reduces that
    vector to ``n_components`` coordinates. A null is encoded as ``""``. Zero
    columns fill the output when the column has too few n-grams or rows for
    ``n_components``. Output name: ``{column}_svd_{i}``.

    Fitting sets ``vectorizers_``, column name to its fitted TF-IDF and SVD
    steps.
    """

    _fits_on_schema = False

    def __init__(self, n_components: int = 30) -> None:
        """Configure the output width.

        Args:
            n_components: The number of output columns per input column.
        """
        self.n_components = n_components

    def _fit(self, table: nw.DataFrame | nw.LazyFrame, y: Any) -> None:
        """Fit TF-IDF and SVD per column.

        Args:
            table: The input table.
            y: Ignored.
        """
        reject_other_dtypes(self.schema_in_, nw.String, type(self).__name__)
        eager = collect(table)
        self.vectorizers_ = {
            name: fit_string_column(read_strings(eager, name), self.n_components)
            for name in self.schema_in_
        }

    def _output_names(self) -> list[str]:
        """Return ``{column}_svd_{i}``, column by column.

        Returns:
            The names.
        """
        return [
            f"{name}_svd_{i}"
            for name in self.schema_in_
            for i in range(self.n_components)
        ]

    def _transform(self, table: nw.DataFrame | nw.LazyFrame) -> nw.DataFrame:
        """Encode each column.

        Args:
            table: The input table.

        Returns:
            The coordinates, column by column.
        """
        eager = collect(table)
        blocks = [
            encode_string_column(
                self.vectorizers_[name], read_strings(eager, name), self.n_components
            )
            for name in self.schema_in_
        ]
        coordinates = np.hstack([np.zeros((len(eager), 0)), *blocks])
        return nw.from_numpy(
            coordinates.reshape(len(eager), -1),
            schema=self._output_names(),
            backend=eager.implementation,
        )


def reject_other_dtypes(
    schema: dict[str, nw.dtypes.DType],
    accepted: type[nw.dtypes.DType],
    owner: str,
) -> None:
    """Raise if a column is not of the ``accepted`` dtype.

    Args:
        schema: Column name to dtype.
        accepted: The dtype class the encoder encodes.
        owner: The encoder's class name, for the error message.

    Raises:
        EncoderError: If a column has another dtype.
    """
    for name, dtype in schema.items():
        if dtype != accepted:
            raise EncoderError(
                f"{owner} encodes {accepted.__name__} columns; column {name!r} "
                f"is {dtype}",
            )


def code_expression(name: str, categories: list[str]) -> nw.Expr:
    """Return the position of each value of ``name`` in ``categories``.

    Args:
        name: The column name.
        categories: The categories, in order.

    Returns:
        A ``Float64`` expression, null where the value is null.
    """
    # default=None: pandas casts a null category to the string "nan", which
    # replace_strict would otherwise reject as a value with no code.
    return (
        nw.col(name)
        .cast(nw.String)
        .replace_strict(
            categories,
            [float(position) for position in range(len(categories))],
            default=None,
            return_dtype=nw.Float64,
        )
        .alias(f"{name}_code")
    )


def read_strings(table: nw.DataFrame, name: str) -> list[str]:
    """Return the values of column ``name``, with null as ``""``.

    Args:
        table: The table.
        name: The column name.

    Returns:
        The values.
    """
    return table[name].fill_null("").to_list()


def fit_string_column(
    values: list[str], n_components: int
) -> tuple[TfidfVectorizer, TruncatedSVD | None] | None:
    """Fit TF-IDF and SVD on one column's values.

    Args:
        values: The column's values.
        n_components: The number of coordinates wanted.

    Returns:
        The fitted TF-IDF and SVD steps. The SVD is None when there are fewer
        than two n-grams. None when there are no n-grams at all.
    """
    if not any(value.strip() for value in values):
        return None
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4))
    tfidf = vectorizer.fit_transform(values)
    # TruncatedSVD needs at least two features.
    if tfidf.shape[1] < 2:
        return vectorizer, None
    svd = TruncatedSVD(n_components=min(n_components, tfidf.shape[1]))
    svd.fit(tfidf)
    return vectorizer, svd


def encode_string_column(
    fitted: tuple[TfidfVectorizer, TruncatedSVD | None] | None,
    values: list[str],
    n_components: int,
) -> np.ndarray:
    """Encode one column's values, padded to ``n_components`` columns.

    Args:
        fitted: The steps from :func:`fit_string_column`.
        values: The column's values.
        n_components: The output width.

    Returns:
        An array of shape ``(len(values), n_components)``.
    """
    if fitted is None:
        return np.zeros((len(values), n_components))
    vectorizer, svd = fitted
    tfidf = vectorizer.transform(values)
    coordinates = tfidf.toarray() if svd is None else svd.transform(tfidf)
    # The SVD yields fewer components than asked when the column has fewer
    # rows or n-grams; zero columns keep the output width fixed.
    return np.pad(coordinates, ((0, 0), (0, n_components - coordinates.shape[1])))
