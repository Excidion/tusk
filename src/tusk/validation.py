"""Checks that confirm a table's declarations hold against its data.

The rest of the schema layer takes declarations on trust: naming a column as
``primary_key`` asserts that it identifies a row, and nothing confirms it.
These checks spend real queries to find out. Nothing here runs unless the user
asks, through :meth:`tusk.Database.validate` or ``add_table(validate=…)``.

Adding a check is a new function plus a new entry in :data:`CHECKS`; no call
signature changes and both entry points pick it up.
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

    One aggregate: the row count against the distinct-value count. The error
    reports both counts and stops there rather than querying again for example
    duplicates -- naming them would cost a second full scan of the table, which
    on a remote backend is a real bill for a nicety the caller did not ask for.

    A table with no ``primary_key`` is skipped rather than failed --
    ``add_table`` already warned about it, and failing here would make
    :meth:`tusk.Database.validate` unusable on any database that deliberately
    holds a keyless table.

    Nulls count as one distinct value, so repeated nulls fail this check while
    a single null passes it. A lone null key is a nullability defect, not a
    uniqueness one.

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

    Checks run in the order given and the first failure stops the run, so
    later checks do not run. A failing check raises
    :class:`~tusk.exceptions.ValidationError`.

    Args:
        frame: The table's lazy frame.
        schema: The table's schema.
        checks: ``True`` for every check, ``False`` for none, a check name, or
            any iterable of check names.

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
