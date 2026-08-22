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
    from tusk.database import Database, TableSchema


def check_non_null_primary_key(frame: nw.LazyFrame, schema: TableSchema) -> None:
    """Confirm the declared primary key holds no null.

    A table with no ``primary_key`` is skipped.

    Args:
        frame: The table's lazy frame.
        schema: The table's schema, naming the column to check.

    Raises:
        ValidationError: If the primary key column holds a null.
    """
    key = schema.primary_key
    if key is None:
        return

    nulls = frame.select(nulls=nw.col(key).is_null().sum()).collect()["nulls"].item()
    if not nulls:
        return

    raise ValidationError(
        f"primary_key {key!r} of {schema.name!r} has {nulls} null values"
    )


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


def check_datetime_row_creation_time(frame: nw.LazyFrame, schema: TableSchema) -> None:
    """Confirm the declared row creation time is a Datetime, not a Date.

    A table with no ``row_creation_time`` is skipped. Reads the schema only.

    Args:
        frame: The table's lazy frame. Unused.
        schema: The table's schema, naming the column to check.

    Raises:
        ValidationError: If the column is not a Datetime.
    """
    column = schema.row_creation_time
    if column is None:
        return

    dtype = schema.dtypes[column]
    if dtype == nw.Datetime:
        return

    raise ValidationError(
        f"row_creation_time {column!r} of {schema.name!r} is {dtype}, expected Datetime"
    )


def check_consistent_time_zones(database: Database) -> None:
    """Confirm every Datetime column in the database agrees on time zone awareness.

    Reads the schemas only. Time zone *values* may differ; only mixing aware
    with naive fails.

    Args:
        database: The database to check.

    Raises:
        ValidationError: If some Datetime columns are tz-aware and others naive.
    """
    aware, naive = [], []
    for table in database.table_names:
        for column, dtype in database.schema(table).dtypes.items():
            if dtype == nw.Datetime:
                name = f"{table}.{column}"
                if dtype.time_zone:
                    aware.append(name)
                else:
                    naive.append(name)

    if not (aware and naive):
        return

    raise ValidationError(
        f"database mixes tz-aware and tz-naive datetimes: "
        f"{len(aware)} are tz-aware, {len(naive)} are tz-naive."
        "Aware: " + " ,".join(aware),
        "Naive: " + ", ".join(naive),
    )


CHECKS = {
    "non_null_primary_key": check_non_null_primary_key,
    "unique_primary_key": check_unique_primary_key,
    "datetime_row_creation_time": check_datetime_row_creation_time,
}

DATABASE_CHECKS = {
    "consistent_time_zones": check_consistent_time_zones,
}


def _select_checks(
    checks: bool | str | Iterable[str],
    registry: dict,
    vocabulary: dict | None = None,
) -> list[str]:
    """Return the names in ``registry`` that ``checks`` selects.

    Args:
        checks: ``True`` for every name in ``registry``, ``False`` for none, a
            check name, or an iterable of check names.
        registry: The registry to select from.
        vocabulary: Every name a caller may legally give, defaulting to
            ``registry``. Names outside ``registry`` but inside ``vocabulary``
            are accepted and skipped, so one selector can span both registries.

    Returns:
        Selected names, in registry order for ``True`` and in the given order
        otherwise.

    Raises:
        ValueError: If ``checks`` is not one of those forms, or names a check
            outside ``vocabulary``.
    """
    known = registry if vocabulary is None else vocabulary
    if checks is True:
        return list(registry)
    if checks is False:
        return []
    if isinstance(checks, str):
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
        if name not in known:
            available = ", ".join(repr(other) for other in known)
            raise ValueError(f"unknown check {name!r}; available checks: {available}")
    return [name for name in names if name in registry]


def validate_table(
    frame: nw.LazyFrame,
    schema: TableSchema,
    checks: bool | str | Iterable[str] = True,
) -> None:
    """Run the selected table checks against one table.

    Checks run in the order given; the first failure raises
    :class:`~tusk.exceptions.ValidationError` and stops the run. A name that
    is not in :data:`CHECKS` raises :class:`ValueError`.

    Args:
        frame: The table's lazy frame.
        schema: The table's schema.
        checks: ``True`` for every table check, ``False`` for none, a check
            name, or an iterable of check names.
    """
    for name in _select_checks(checks, CHECKS):
        CHECKS[name](frame, schema)


def validate_database(
    database: Database, checks: bool | str | Iterable[str] = True
) -> None:
    """Run the selected checks against a database.

    Table checks run against every table in insertion order, then
    database-wide checks run once. The first failure raises
    :class:`~tusk.exceptions.ValidationError` and stops the run. A name in
    neither :data:`CHECKS` nor :data:`DATABASE_CHECKS` raises
    :class:`ValueError`.

    Args:
        database: The database to check.
        checks: ``True`` for every check, ``False`` for none, a check name, or
            an iterable of check names.
    """
    vocabulary = {**CHECKS, **DATABASE_CHECKS}
    table_checks = _select_checks(checks, CHECKS, vocabulary)
    database_checks = _select_checks(checks, DATABASE_CHECKS, vocabulary)

    for name in database.table_names:
        validate_table(database.frame(name), database.schema(name), table_checks)
    for name in database_checks:
        DATABASE_CHECKS[name](database)
