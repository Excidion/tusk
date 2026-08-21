"""Checks that confirm a table's declarations hold against its data.

Nothing here runs unless the caller asks, through
:meth:`tusk.Database.validate` or ``add_table(validate=…)``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import narwhals as nw

from tusk.exceptions import ValidationError

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from tusk.database import TableSchema


def check_unique_primary_key(frame: nw.LazyFrame, schema: TableSchema) -> None:
    """Confirm the declared primary key holds no repeated value.

    A table with no ``primary_key`` is skipped. Nulls count as one distinct
    value, so repeated nulls fail and a single null passes.

    Args:
        frame: The table's lazy frame.
        schema: The table's schema, naming the column to check.

    Raises:
        ValidationError: If the primary key column holds a repeated value.
    """
    key = schema.primary_key
    if key is None:
        return

    counts = frame.select(total=nw.len(), distinct=nw.col(key).n_unique()).collect()
    total, distinct = counts["total"].item(), counts["distinct"].item()
    if total <= 1:
        # narwhals lowers n_unique on SQL backends to
        # count_distinct(x) + max(x IS NULL); MAX() over zero rows is SQL
        # NULL, so an empty relation makes `distinct` come back as None and
        # `total == distinct` would be `0 == None` -> a false positive. Zero
        # or one row cannot hold a duplicate regardless, so return early and
        # never reach that comparison.
        return
    if total == distinct:
        return

    raise ValidationError(
        f"primary_key {key!r} of {schema.name!r} is not unique: "
        f"{total} rows, {distinct} distinct values"
    )


CHECKS = {
    "unique_primary_key": check_unique_primary_key,
}


def validate_table(
    frame: nw.LazyFrame,
    schema: TableSchema,
    checks: bool | str | Iterable[str] = True,
) -> None:
    """Run the selected checks against one table.

    Checks run in the order given; the first failure raises
    :class:`~tusk.exceptions.ValidationError` and stops the run.

    Args:
        frame: The table's lazy frame.
        schema: The table's schema.
        checks: ``True`` for every check, ``False`` for none, a check name, or
            an iterable of check names.

    Raises:
        ValueError: If ``checks`` is not one of those forms, or names a check
            that is not in :data:`CHECKS`.
    """
    if checks is True:
        names = list(CHECKS)
    elif checks is False:
        names = []
    elif isinstance(checks, str):
        names = [checks]
    else:
        try:
            names = list(checks)
        except TypeError:
            raise ValueError(
                f"invalid checks selector {checks!r}; expected True, False, a "
                f"check name, or an iterable of check names"
            ) from None

    for name in names:
        try:
            check = CHECKS[name]
        except KeyError:
            available = ", ".join(repr(known) for known in CHECKS)
            raise ValueError(
                f"unknown check {name!r}; available checks: {available}"
            ) from None
        check(frame, schema)
