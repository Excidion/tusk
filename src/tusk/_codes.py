"""Numeric codes for the categories of an `Enum` column."""

from __future__ import annotations

import narwhals as nw


def code_expression(name: str, categories: list[str]) -> nw.Expr:
    """Return the position of each value of `name` in `categories`.

    Args:
        name: The column name.
        categories: The categories, in order.

    Returns:
        A `Float64` expression, null where the value is null.
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
    )
