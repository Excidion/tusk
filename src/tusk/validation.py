"""Checks that confirm a table's declarations hold against its data.

Nothing here runs unless the caller asks, through
:meth:`tusk.Database.validate` or ``add_table(validate=…)``.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

import narwhals as nw

from tusk.exceptions import ValidationError

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from tusk.database import Database, Relationship, TableSchema


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
        f"primary_key {key!r} of {schema.name!r} has {nulls} null values",
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
        f"{total} rows, {distinct} distinct values",
    )


def check_dtype_row_creation_time(frame: nw.LazyFrame, schema: TableSchema) -> None:
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


def check_dtype_row_update_times(frame: nw.LazyFrame, schema: TableSchema) -> None:
    """Confirm every declared row update time is a Datetime, not a Date.

    A table with no ``row_update_times`` is skipped. Reads the schema only.

    Args:
        frame: The table's lazy frame. Unused.
        schema: The table's schema, naming the columns to check.

    Raises:
        ValidationError: If a row update time column is not a Datetime.
    """
    for column in schema.row_update_times:
        dtype = schema.dtypes[column]
        if dtype == nw.Datetime:
            continue
        raise ValidationError(
            f"row_update_time {column!r} of {schema.name!r} is {dtype}, "
            f"expected Datetime",
        )


def check_never_updated_primary_key(frame: nw.LazyFrame, schema: TableSchema) -> None:
    """Confirm no row update time rewrites the primary key.

    A table with no ``primary_key`` is skipped. Reads the schema only.

    Args:
        frame: The table's lazy frame. Unused.
        schema: The table's schema, naming the column to check.

    Raises:
        ValidationError: If a row update time lists the primary key.
    """
    update_time = _updating_row_update_time(schema, schema.primary_key)
    if update_time is None:
        return

    raise ValidationError(
        f"primary_key {schema.primary_key!r} of {schema.name!r} is listed "
        f"under row_update_time {update_time!r}; the primary key names the "
        f"feature matrix's rows, so it cannot be served an earlier value",
    )


def check_never_updated_row_creation_time(
    frame: nw.LazyFrame, schema: TableSchema
) -> None:
    """Confirm no row update time rewrites the row creation time.

    A table with no ``row_creation_time`` is skipped. Reads the schema only.

    Args:
        frame: The table's lazy frame. Unused.
        schema: The table's schema, naming the column to check.

    Raises:
        ValidationError: If a row update time lists the row creation time.
    """
    update_time = _updating_row_update_time(schema, schema.row_creation_time)
    if update_time is None:
        return

    raise ValidationError(
        f"row_creation_time {schema.row_creation_time!r} of {schema.name!r} "
        f"is listed under row_update_time {update_time!r}; every visible row "
        f"was created at or before the cutoff already",
    )


def check_unchained_row_update_times(frame: nw.LazyFrame, schema: TableSchema) -> None:
    """Confirm no row update time is rewritten by another row update time.

    The entry ``add_table`` adds for an update time that says nothing about
    itself is not a chain. Reads the schema only.

    Args:
        frame: The table's lazy frame. Unused.
        schema: The table's schema, naming the updates.

    Raises:
        ValidationError: If a row update time is listed under a different row
            update time.
    """
    for update_time, column, _ in schema.column_updates:
        if column == update_time or column not in schema.row_update_times:
            continue
        raise ValidationError(
            f"row_update_time {column!r} of {schema.name!r} is itself listed "
            f"under row_update_time {update_time!r}; an update time cannot say "
            f"what other columns held before if it is unknown itself",
        )


def check_singly_updated_columns(frame: nw.LazyFrame, schema: TableSchema) -> None:
    """Confirm no column is rewritten by two row update times.

    The entry ``add_table`` adds for an update time that says nothing about
    itself counts like any other. Reads the schema only.

    Args:
        frame: The table's lazy frame. Unused.
        schema: The table's schema, naming the updates.

    Raises:
        ValidationError: If one column appears under two row update times.
    """
    seen: dict[str, str] = {}
    for update_time, column, _ in schema.column_updates:
        if column in seen:
            raise ValidationError(
                f"{column!r} of {schema.name!r} is listed under both "
                f"row_update_time {seen[column]!r} and row_update_time "
                f"{update_time!r}; give it one row_update_time",
            )
        seen[column] = update_time


def check_matching_earlier_value_dtypes(
    frame: nw.LazyFrame, schema: TableSchema
) -> None:
    """Confirm every declared pre-update value fits the column it replaces.

    Reads the schema only. A null fits every column.

    Args:
        frame: The table's lazy frame. Unused.
        schema: The table's schema, naming the updates.

    Raises:
        ValidationError: If a pre-update value does not fit its column's dtype.
    """
    for update_time, column, value in schema.column_updates:
        dtype = schema.dtypes[column]
        if _fits_dtype(value, dtype):
            continue
        raise ValidationError(
            f"{value!r}, listed for {column!r} of {schema.name!r} under "
            f"row_update_time {update_time!r}, is not a {dtype} value",
        )


def check_ordered_row_times(frame: nw.LazyFrame, schema: TableSchema) -> None:
    """Confirm no row was updated before it was created.

    Scans the table. A table with no ``row_creation_time`` or no
    ``row_update_times`` is skipped, as is a row whose update time is null:
    a row that was never updated has no order to check.

    Args:
        frame: The table's lazy frame.
        schema: The table's schema, naming the columns to compare.

    Raises:
        ValidationError: If an update time is before the row creation time.
    """
    created = schema.row_creation_time
    if created is None or not schema.row_update_times:
        return

    early = frame.select(
        (nw.col(update_time) < nw.col(created)).sum().alias(update_time)
        for update_time in schema.row_update_times
    ).collect()

    for update_time in schema.row_update_times:
        rows = early[update_time].item()
        if not rows:
            continue
        raise ValidationError(
            f"row_update_time {update_time!r} of {schema.name!r} is before "
            f"row_creation_time {created!r} in {rows} rows",
        )


def check_cutoff_time_zone(database: Database, cutoff_time: datetime) -> None:
    """Confirm a cutoff matches the tz awareness of the database's Datetime columns.

    Reads the schemas only. A database with no Datetime column accepts any
    cutoff.

    Args:
        database: The database the cutoff will measure against.
        cutoff_time: The cutoff.

    Raises:
        ValidationError: If the cutoff's time zone awareness differs from the
            columns', or if those columns disagree among themselves.
    """
    database_aware = is_time_zone_aware(database)
    if database_aware is None:
        return

    cutoff_aware = cutoff_time.utcoffset() is not None
    if cutoff_aware == database_aware:
        return

    raise ValidationError(
        f"cutoff_time {cutoff_time!r} is "
        f"tz-{'aware' if cutoff_aware else 'naive'}, but the database's "
        f"datetimes are tz-{'aware' if database_aware else 'naive'}",
    )


def is_time_zone_aware(database: Database) -> bool | None:
    """Reduce every Datetime column in the database to one time zone awareness.

    Reads the schemas only. Time zone *values* may differ; only mixing aware
    with naive fails.

    Args:
        database: The database to inspect.

    Returns:
        awareness: ``True`` if every Datetime column is tz-aware, ``False`` if
            every one is naive, ``None`` if the database has no Datetime
            column.

    Raises:
        ValidationError: If some Datetime columns are tz-aware and others
            naive.
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

    if aware and naive:
        raise ValidationError(
            f"database mixes tz-aware and tz-naive datetimes: "
            f"{len(aware)} tz-aware ({', '.join(aware)}), "
            f"{len(naive)} tz-naive ({', '.join(naive)})",
        )
    if aware:
        return True
    if naive:
        return False
    return None


def check_matching_key_dtypes(database: Database, relationship: Relationship) -> None:
    """Confirm a foreign key has the same dtype as the primary key it points at.

    Reads the schemas only. The dtypes must match exactly: pyarrow refuses to
    join Int64 to Int32, polars refuses Int64 to Float64 and refuses every
    String/Categorical/Enum crossing, and polars also refuses two Enums whose
    categories differ. Anything looser passes validation here and then fails
    the join.

    Args:
        database: The database holding both tables.
        relationship: The link to check.

    Raises:
        ValidationError: If the two dtypes differ.
    """
    parent = database.schema(relationship.parent)
    child = database.schema(relationship.child)
    if parent.primary_key is None:
        return

    parent_dtype = parent.dtypes[parent.primary_key]
    child_dtype = child.dtypes[relationship.foreign_key]
    if parent_dtype == child_dtype:
        return

    raise ValidationError(
        f"foreign_key {relationship.foreign_key!r} of {child.name!r} is "
        f"{child_dtype}, but primary_key {parent.primary_key!r} of "
        f"{parent.name!r} is {parent_dtype}",
    )


def check_overlapping_keys(database: Database, relationship: Relationship) -> None:
    """Confirm the foreign key matches at least one of the parent's primary keys.

    Stops at the first match rather than proving every foreign key resolves.
    Orphan rows are ordinary in real data; no overlap at all means the link
    itself is wrong -- the wrong column, or two id spaces that never met.

    Null foreign keys are ignored, and a child holding no non-null foreign key
    is skipped: it has nothing to match.

    Args:
        database: The database holding both tables.
        relationship: The link to check.

    Raises:
        ValidationError: If no foreign key value appears in the parent.
    """
    foreign_key = relationship.foreign_key
    primary_key = database.schema(relationship.parent).primary_key
    if primary_key is None:
        return

    children = (
        database.frame(relationship.child)
        .select(nw.col(foreign_key))
        .filter(~nw.col(foreign_key).is_null())
    )
    parents = database.frame(relationship.parent).select(nw.col(primary_key))
    matched = children.join(
        parents,
        left_on=foreign_key,
        right_on=primary_key,
        how="semi",
    )
    if len(matched.head(1).collect()):
        return
    if not len(children.head(1).collect()):
        return

    raise ValidationError(
        f"no {foreign_key!r} in {relationship.child!r} matches any "
        f"{primary_key!r} in {relationship.parent!r}",
    )


def _updating_row_update_time(schema: TableSchema, column: str | None) -> str | None:
    """Return the row update time that rewrites a column.

    Args:
        schema: The table's schema.
        column: The column to look for, or None.

    Returns:
        The first row update time listing it, or None when nothing does.
    """
    if column is None:
        return None
    for update_time, updated, _ in schema.column_updates:
        if updated == column:
            return update_time
    return None


def _fits_dtype(value: Any, dtype: Any) -> bool:
    """Decide whether a Python value can stand in for a narwhals dtype.

    A null fits every column. A value of a Python type this function does not
    check for fits every dtype.

    Args:
        value: The declared pre-update value.
        dtype: The narwhals dtype of the column it replaces.

    Returns:
        True if the value fits the dtype.
    """
    # bool is a subclass of int and datetime is a subclass of date, so both
    # narrow types have to be tested before the wide ones.
    if value is None:
        return True
    if isinstance(value, bool):
        return dtype == nw.Boolean
    if isinstance(value, int):
        return bool(dtype.is_numeric())
    if isinstance(value, float):
        return bool(dtype.is_float()) or dtype == nw.Decimal
    if isinstance(value, str):
        return dtype in (nw.String, nw.Categorical, nw.Enum)
    if isinstance(value, datetime):
        return dtype == nw.Datetime
    if isinstance(value, date):
        return dtype == nw.Date
    # Refusing a value in a dtype family tusk cannot judge is worse than
    # letting the backend judge it at collect time.
    return True


TABLE_CHECKS = {
    "non_null_primary_key": check_non_null_primary_key,
    "unique_primary_key": check_unique_primary_key,
    "datetime_row_creation_time": check_dtype_row_creation_time,
    "datetime_row_update_times": check_dtype_row_update_times,
    "never_updated_primary_key": check_never_updated_primary_key,
    "never_updated_row_creation_time": check_never_updated_row_creation_time,
    "unchained_row_update_times": check_unchained_row_update_times,
    "singly_updated_columns": check_singly_updated_columns,
    "matching_earlier_value_dtypes": check_matching_earlier_value_dtypes,
    "ordered_row_times": check_ordered_row_times,
}

# Every check here answers from the declared schema, so add_table can run all
# of them without reading a row. A check that scans belongs in TABLE_CHECKS
# and not here.
DEFAULT_TABLE_CHECKS = (
    "datetime_row_creation_time",
    "datetime_row_update_times",
    "unchained_row_update_times",
    "singly_updated_columns",
    "never_updated_primary_key",
    "never_updated_row_creation_time",
    "matching_earlier_value_dtypes",
)

RELATIONSHIP_CHECKS = {
    "matching_key_dtypes": check_matching_key_dtypes,
    "overlapping_keys": check_overlapping_keys,
}

DATABASE_CHECKS = {
    "consistent_time_zones": is_time_zone_aware,
}


def _select_checks(checks: bool | str | Iterable[str], registry: dict) -> list[str]:
    """Return the names in ``registry`` that ``checks`` selects.

    Args:
        checks: ``True`` for every name in ``registry``, ``False`` for none, a
            check name, or an iterable of check names.
        registry: The registry to select from.

    Returns:
        names: Selected names, in registry order for ``True`` and in the given
            order otherwise.

    Raises:
        ValueError: If ``checks`` is not one of those forms, or names a check
            outside ``registry``.
    """
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
                f"check name, or an iterable of check names",
            ) from None

    for name in names:
        if name not in registry:
            available = ", ".join(repr(other) for other in registry)
            raise ValueError(f"unknown check {name!r}; available checks: {available}")
    return names


def validate_table(
    frame: nw.LazyFrame,
    schema: TableSchema,
    checks: bool | str | Iterable[str] = True,
) -> None:
    """Run the selected table checks against one table.

    Checks run in the order given; the first failure raises
    :class:`~tusk.exceptions.ValidationError` and stops the run. A name that
    is not in :data:`TABLE_CHECKS` raises :class:`ValueError`.

    Args:
        frame: The table's lazy frame.
        schema: The table's schema.
        checks: ``True`` for every table check, ``False`` for none, a check
            name, or an iterable of check names.
    """
    for name in _select_checks(checks, TABLE_CHECKS):
        TABLE_CHECKS[name](frame, schema)


def validate_relationship(
    database: Database,
    relationship: Relationship,
    checks: bool | str | Iterable[str] = True,
) -> None:
    """Run the selected relationship checks against one relationship.

    Checks run in the order given; the first failure raises
    :class:`~tusk.exceptions.ValidationError` and stops the run. A name that
    is not in :data:`RELATIONSHIP_CHECKS` raises :class:`ValueError`.

    Args:
        database: The database holding both tables.
        relationship: The link to check.
        checks: ``True`` for every relationship check, ``False`` for none, a
            check name, or an iterable of check names.
    """
    for name in _select_checks(checks, RELATIONSHIP_CHECKS):
        RELATIONSHIP_CHECKS[name](database, relationship)


def validate_database(
    database: Database,
    *,
    database_checks: bool | str | Iterable[str] = True,
    table_checks: bool | str | Iterable[str] = True,
    relationship_checks: bool | str | Iterable[str] = True,
) -> None:
    """Run the selected checks against a database.

    Table checks run against every table in insertion order, then relationship
    checks against every relationship, then database-wide checks once. The
    first failure raises :class:`~tusk.exceptions.ValidationError` and stops
    the run. A name outside the registry its selector draws from raises
    :class:`ValueError`.

    Args:
        database: The database to check.
        database_checks: Selects from :data:`DATABASE_CHECKS`.
        table_checks: Selects from :data:`TABLE_CHECKS`.
        relationship_checks: Selects from :data:`RELATIONSHIP_CHECKS`.
    """
    tables = _select_checks(table_checks, TABLE_CHECKS)
    relationships = _select_checks(relationship_checks, RELATIONSHIP_CHECKS)
    wide = _select_checks(database_checks, DATABASE_CHECKS)

    for name in database.table_names:
        validate_table(database.frame(name), database.schema(name), tables)
    for relationship in database.relationships:
        validate_relationship(database, relationship, relationships)
    for name in wide:
        DATABASE_CHECKS[name](database)
