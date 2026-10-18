"""Casts a table's columns and names their `sdm` semantic types."""

from __future__ import annotations

import warnings
from collections.abc import Iterable

import narwhals as nw

from tusk._codes import code_expression
from tusk.exceptions import UnsupportedDtypeWarning

_MICROSECONDS_PER_SECOND = 1_000_000


def semantic_type(dtype: nw.dtypes.DType) -> str | None:
    """Return the `sdm` semantic type of a dtype.

    Args:
        dtype: The narwhals dtype.

    Returns:
        `"numerical"`, `"categorical"` or `"datetime"`, or None if no semantic
        type fits the dtype.
    """
    if dtype == nw.Boolean or dtype == nw.Enum or dtype == nw.Duration:
        return "numerical"
    if dtype.is_numeric():
        return "numerical"
    if dtype == nw.String or dtype == nw.Categorical:
        return "categorical"
    if dtype == nw.Date or dtype == nw.Datetime:
        return "datetime"
    return None


def _cast_expression(name: str, dtype: nw.dtypes.DType) -> nw.Expr:
    """Return the expression that gives a column the numbers its dtype means.

    Args:
        name: The column name.
        dtype: The column's narwhals dtype.

    Returns:
        A `Float64` expression for a `Boolean`, `Enum`, `Duration` or `Decimal`
        column, and the column itself for any other dtype.
    """
    if dtype == nw.Boolean or dtype == nw.Decimal:
        return nw.col(name).cast(nw.Float64)
    if isinstance(dtype, nw.Enum):
        return code_expression(name, list(dtype.categories))
    if dtype == nw.Duration:
        microseconds = nw.col(name).dt.total_microseconds()
        return microseconds.cast(nw.Float64) / _MICROSECONDS_PER_SECOND
    return nw.col(name)


def _warn_about_left_out(table_name: str, left_out: Iterable[str]) -> None:
    """Issue one warning that names the columns left out of a table.

    Args:
        table_name: The table's name.
        left_out: The names of the columns left out.
    """
    names = ", ".join(f"{name!r}" for name in left_out)
    warnings.warn(
        f"table {table_name!r}: no sdm semantic type fits the dtype of "
        f"column(s) {names}, so the export leaves them out",
        UnsupportedDtypeWarning,
        stacklevel=2,
    )


def prepare_table(
    table_name: str,
    table: nw.DataFrame,
    key_columns: frozenset[str],
) -> tuple[nw.DataFrame, dict[str, str]]:
    """Cast a table's columns and name their `sdm` semantic types.

    A key column keeps its values and gets the semantic type `"id"`. A
    `Boolean`, `Enum`, `Duration` or `Decimal` column becomes `Float64`. A column whose
    dtype has no semantic type is left out, and one warning names these
    columns.

    Args:
        table_name: The table's name, for the warning.
        table: The collected table.
        key_columns: The names of the primary key and foreign key columns.

    Returns:
        The table with the cast columns, and a mapping of each of its columns
        to its semantic type.

    Warns:
        UnsupportedDtypeWarning: If a column that is not a key column has a
            dtype with no semantic type.
    """
    expressions: list[nw.Expr] = []
    types: dict[str, str] = {}
    left_out: list[str] = []
    for name, dtype in table.schema.items():
        if name in key_columns:
            expressions.append(nw.col(name))
            types[name] = "id"
        elif (kind := semantic_type(dtype)) is None:
            left_out.append(name)
        else:
            expressions.append(_cast_expression(name, dtype).alias(name))
            types[name] = kind
    if left_out:
        _warn_about_left_out(table_name, left_out)
    return table.select(expressions), types
