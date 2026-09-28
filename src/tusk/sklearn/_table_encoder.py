"""Encoding of every column of a table by its narwhals dtype.

:class:`TableEncoder` splits the columns into disjoint dtype groups and
encodes each group with its own estimator.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder

from tusk.exceptions import EncoderError
from tusk.sklearn._column_encoders import (
    DateEncoder,
    DatetimeEncoder,
    DurationEncoder,
    EnumEncoder,
    StringEncoder,
    TimeEncoder,
)
from tusk.sklearn._narwhals import NarwhalsEncoder, NarwhalsMixin, collect

GROUPS = (
    "numeric",
    "boolean",
    "string",
    "categorical",
    "enum",
    "date",
    "time",
    "datetime",
    "duration",
    "other",
)
GROUP_DTYPES = {
    "boolean": nw.Boolean,
    "string": nw.String,
    "categorical": nw.Categorical,
    "enum": nw.Enum,
    "date": nw.Date,
    "time": nw.Time,
    "datetime": nw.Datetime,
    "duration": nw.Duration,
}
KEYWORDS = ("passthrough", "drop")


class TableEncoder(NarwhalsEncoder):
    """An encoder of every column of a table by its narwhals dtype.

    Each parameter names a dtype group and takes an estimator,
    ``"passthrough"`` or ``"drop"``. Each column belongs to exactly one
    group. An estimator with :class:`NarwhalsMixin` receives a narwhals table
    and converts it itself. Any other estimator receives numpy.

    Output names are ``{group}__{name}``. Groups appear in parameter order.

    Fitting sets ``groups_``, group name to its fitted estimator or
    ``"passthrough"`` and its input columns. Empty and dropped groups are
    absent. It also sets ``output_names_``, the output column names.
    """

    _fits_on_schema = False

    def __init__(
        self,
        numeric: Any = "passthrough",
        boolean: Any = "passthrough",
        string: Any = StringEncoder(),  # noqa: B008
        categorical: Any = OneHotEncoder(handle_unknown="ignore", sparse_output=False),  # noqa: B008
        enum: Any = EnumEncoder(),  # noqa: B008
        date: Any = DateEncoder(),  # noqa: B008
        time: Any = TimeEncoder(),  # noqa: B008
        datetime: Any = DatetimeEncoder(),  # noqa: B008
        duration: Any = DurationEncoder(),  # noqa: B008
        other: Any = "drop",
    ) -> None:
        """Configure the estimator of each dtype group.

        Args:
            numeric: For ``Int*``, ``UInt*``, ``Float*`` and ``Decimal``.
                ``Decimal`` is cast to ``Float64`` first.
            boolean: For ``Boolean``.
            string: For ``String``.
            categorical: For ``Categorical``.
            enum: For ``Enum``.
            date: For ``Date``.
            time: For ``Time``.
            datetime: For ``Datetime``.
            duration: For ``Duration``.
            other: For every other dtype, such as ``List`` or ``Struct``.
        """
        self.numeric = numeric
        self.boolean = boolean
        self.string = string
        self.categorical = categorical
        self.enum = enum
        self.date = date
        self.time = time
        self.datetime = datetime
        self.duration = duration
        self.other = other

    def set_params(self, **params: Any) -> TableEncoder:
        """Set parameters, including nested ones such as ``date__components``.

        Args:
            **params: Parameter names and values.

        Returns:
            This encoder.
        """
        # The default estimators are shared by every TableEncoder. A nested
        # parameter would change the shared instance, so it is cloned first.
        defaults = type(self)().get_params(deep=False)
        for group in {key.split("__")[0] for key in params if "__" in key}:
            if getattr(self, group) is defaults[group]:
                setattr(self, group, clone(defaults[group]))
        return super().set_params(**params)

    def _fit(self, table: nw.DataFrame | nw.LazyFrame, y: Any) -> None:
        """Fit each group's estimator on the group's columns.

        Args:
            table: The input table, eager or lazy.
            y: Targets, passed to each estimator.
        """
        groups = split_into_groups(self.schema_in_)
        estimators = {group: self._get_estimator(group) for group in groups}
        kept = {g: c for g, c in groups.items() if estimators[g] != "drop"}
        cast = table.select(read_expressions(self.schema_in_))
        by_values = [g for g in kept if not fits_on_schema(estimators[g])]
        values = collect_columns(cast, [c for g in by_values for c in kept[g]])
        self.groups_ = {}
        for group, columns in kept.items():
            source = values if group in by_values else cast
            self.groups_[group] = (
                fit_group(estimators[group], source.select(columns), y),
                columns,
            )
        self.output_names_ = [
            f"{group}__{name}"
            for group, (estimator, columns) in self.groups_.items()
            for name in group_output_names(estimator, columns)
        ]

    def _output_names(self) -> list[str]:
        """Return the output names found at fit.

        Returns:
            The names.
        """
        return self.output_names_

    def _transform(self, table: nw.DataFrame | nw.LazyFrame) -> nw.DataFrame:
        """Encode each group and join the results by position.

        Args:
            table: The input table, eager or lazy.

        Returns:
            The encoded table.

        Raises:
            EncoderError: If a group's output has another row count than the
                input.
        """
        selected = nw.maybe_reset_index(collect(table.select(self._selection())))
        blocks = [self._encode_group(group, selected) for group in self.groups_]
        for group, block in zip(self.groups_, blocks, strict=True):
            if len(block) != len(selected):
                raise EncoderError(
                    f"the {group!r} group returned {len(block)} rows for "
                    f"{len(selected)} input rows",
                )
        if not blocks:
            return selected.select([])
        return nw.concat(blocks, how="horizontal")

    def _get_estimator(self, group: str) -> Any:
        """Return a clone of the group's estimator, or its keyword.

        Args:
            group: The group name.

        Returns:
            The estimator clone, ``"passthrough"`` or ``"drop"``.

        Raises:
            ValueError: If the parameter is neither an estimator nor a
                keyword.
        """
        value = getattr(self, group)
        if isinstance(value, str) and value in KEYWORDS:
            return value
        if not (hasattr(value, "fit") and hasattr(value, "transform")):
            raise ValueError(
                f"{group}={value!r} is not an estimator, 'passthrough' or 'drop'",
            )
        return clone(value)

    def _selection(self) -> list[nw.Expr]:
        """Return the one ``select`` that feeds every group.

        Schema-only estimators contribute their expressions. Every other group
        contributes its columns. Every name is prefixed with its group.

        Returns:
            The expressions.
        """
        read = dict(
            zip(self.schema_in_, read_expressions(self.schema_in_), strict=True)
        )
        expressions = []
        for group, (estimator, columns) in self.groups_.items():
            if estimator != "passthrough" and fits_on_schema(estimator):
                names = estimator.get_feature_names_out()
                sources = estimator._expressions()
                expressions += [
                    expression.alias(f"{group}__{name}")
                    for expression, name in zip(sources, names, strict=True)
                ]
                continue
            expressions += [read[c].alias(f"{group}__{c}") for c in columns]
        return expressions

    def _encode_group(self, group: str, selected: nw.DataFrame) -> nw.DataFrame:
        """Return the encoded columns of one group.

        Args:
            group: The group name.
            selected: The collected result of :meth:`_selection`.

        Returns:
            The group's output, with ``{group}__`` names.
        """
        estimator, columns = self.groups_[group]
        if estimator == "passthrough" or fits_on_schema(estimator):
            names = group_output_names(estimator, columns)
            return selected.select([f"{group}__{name}" for name in names])
        values = selected.select([f"{group}__{c}" for c in columns]).rename(
            {f"{group}__{c}": c for c in columns}
        )
        output = estimator.transform(to_estimator_input(estimator, values))
        return nw.from_numpy(
            to_dense(output),
            schema=[f"{group}__{n}" for n in group_output_names(estimator, columns)],
            backend=selected.implementation,
        )


def split_into_groups(schema: dict[str, nw.dtypes.DType]) -> dict[str, list[str]]:
    """Return group name to its columns, in group order, without empty groups.

    Args:
        schema: Column name to dtype.

    Returns:
        The groups and their columns, in table order.
    """
    columns: dict[str, list[str]] = {group: [] for group in GROUPS}
    for name, dtype in schema.items():
        columns[find_group(dtype)].append(name)
    return {group: names for group, names in columns.items() if names}


def find_group(dtype: nw.dtypes.DType) -> str:
    """Return the group a dtype belongs to.

    Args:
        dtype: A narwhals dtype.

    Returns:
        The group name.
    """
    if dtype.is_numeric():
        return "numeric"
    for group, group_dtype in GROUP_DTYPES.items():
        if dtype == group_dtype:
            return group
    return "other"


def read_expressions(schema: dict[str, nw.dtypes.DType]) -> list[nw.Expr]:
    """Return one expression per column, with ``Decimal`` cast to ``Float64``.

    Args:
        schema: Column name to dtype.

    Returns:
        The expressions, in column order.
    """
    return [
        nw.col(name).cast(nw.Float64) if dtype == nw.Decimal else nw.col(name)
        for name, dtype in schema.items()
    ]


def collect_columns(
    table: nw.DataFrame | nw.LazyFrame, columns: list[str]
) -> nw.DataFrame | nw.LazyFrame:
    """Return ``columns`` of ``table``, collected unless there are none.

    Args:
        table: The table, eager or lazy.
        columns: The columns to collect.

    Returns:
        The collected columns, or ``table`` itself when ``columns`` is empty.
    """
    if not columns:
        return table
    return collect(table.select(columns))


def fits_on_schema(estimator: Any) -> bool:
    """Report whether an estimator fits on the schema alone.

    Args:
        estimator: An estimator or ``"passthrough"``.

    Returns:
        True for ``"passthrough"`` and for tusk's schema-only encoders.
    """
    return estimator == "passthrough" or (
        isinstance(estimator, NarwhalsEncoder) and estimator._fits_on_schema
    )


def fit_group(estimator: Any, table: nw.DataFrame | nw.LazyFrame, y: Any) -> Any:
    """Fit a group's estimator on the group's columns.

    Args:
        estimator: The cloned estimator or ``"passthrough"``.
        table: The group's columns.
        y: Targets.

    Returns:
        The fitted estimator, or ``"passthrough"``.
    """
    if estimator == "passthrough":
        return estimator
    return estimator.fit(to_estimator_input(estimator, table), y)


def to_estimator_input(estimator: Any, table: nw.DataFrame | nw.LazyFrame) -> Any:
    """Return what the estimator receives: the table, or numpy.

    Args:
        estimator: The estimator.
        table: The group's columns.

    Returns:
        The narwhals table for an estimator with :class:`NarwhalsMixin`, else
        a numpy array.
    """
    if isinstance(estimator, NarwhalsMixin):
        return table
    return collect(table).to_numpy()


def group_output_names(estimator: Any, columns: list[str]) -> list[str]:
    """Return a group's output names, without the group prefix.

    Args:
        estimator: The fitted estimator or ``"passthrough"``.
        columns: The group's input columns.

    Returns:
        The names.
    """
    if estimator == "passthrough":
        return columns
    return list(estimator.get_feature_names_out(columns))


def to_dense(output: Any) -> np.ndarray:
    """Return an estimator's output as a dense 2-D array.

    Args:
        output: A numpy array, a sparse matrix or a table.

    Returns:
        The dense array.
    """
    if hasattr(output, "toarray"):
        return output.toarray()
    table = nw.from_native(output, pass_through=True)
    if isinstance(table, nw.DataFrame):
        return table.to_numpy()
    return np.asarray(output)
