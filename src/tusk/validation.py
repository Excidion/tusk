"""Checks that confirm a table's declarations hold against its data.

The rest of the schema layer takes declarations on trust: naming a column as
``primary_key`` asserts that it identifies a row, and nothing confirms it.
These checks spend real queries to find out. Nothing here runs unless the user
asks, through :meth:`tusk.Database.validate` or ``add_table(validate=…)``.

Adding a check is a new function plus a new entry in :data:`CHECKS`; no call
signature changes and both entry points pick it up.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING

import narwhals as nw

from tusk.exceptions import ValidationError

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from tusk.database import TableSchema

Check = Callable[["nw.LazyFrame", "TableSchema"], None]
"""A check: given a table's frame and schema, raise or return."""

Checks = bool | str | Iterable[str]
"""Selects checks: every check, none, one by name, or several by name."""

_SAMPLE_SIZE = 5
_COUNT = "__tusk_count"


def check_unique_primary_key(frame: nw.LazyFrame, schema: TableSchema) -> None:
    """Confirm the declared primary key holds no repeated value.

    Runs in two phases so the passing case stays cheap: one aggregate compares
    the row count to the distinct-value count, and only a disagreement pays for
    a second query naming offending values.

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

    duplicates = (
        frame.group_by(key)
        .agg(nw.len().alias(_COUNT))
        .filter(nw.col(_COUNT) > 1)
        .sort(key)
        .head(_SAMPLE_SIZE)
        .collect()
    )
    raise ValidationError(
        f"primary_key {key!r} of {schema.name!r} is not unique: "
        f"{total} rows, {distinct} distinct values, "
        f"e.g. {duplicates[key].to_list()}"
    )


CHECKS: Mapping[str, Check] = {
    "unique_primary_key": check_unique_primary_key,
}
"""Every available check, by name. Iteration order is the order checks run."""


def resolve_checks(selector: Checks) -> tuple[Check, ...]:
    """Turn a selector into the checks it names.

    Selected checks are deduplicated and returned in :data:`CHECKS` order, not
    argument order, so which failure a caller sees never depends on how they
    happened to type the list.

    Args:
        selector: ``True`` for every check, ``False`` (or an empty iterable)
            for none, a string for one by name, or an iterable of names.

    Returns:
        The selected checks, in registry order.

    Raises:
        ValueError: If ``selector`` is not a recognized selector form, or
            names a check that is not in :data:`CHECKS`.
    """
    if isinstance(selector, bool):
        return tuple(CHECKS.values()) if selector else ()

    if isinstance(selector, str):
        names = {selector}
    elif isinstance(selector, Iterable):
        names = set(selector)
    else:
        available = ", ".join(repr(name) for name in CHECKS)
        raise ValueError(
            f"invalid checks selector {selector!r}; expected bool, str, or an "
            f"iterable of check names; available checks: {available}"
        )

    unknown = sorted(names - set(CHECKS))
    if unknown:
        listed = ", ".join(repr(name) for name in unknown)
        available = ", ".join(repr(name) for name in CHECKS)
        raise ValueError(f"unknown check {listed}; available checks: {available}")
    return tuple(check for name, check in CHECKS.items() if name in names)


def validate_table(
    frame: nw.LazyFrame, schema: TableSchema, checks: Checks = True
) -> None:
    """Run the selected checks against one table.

    Running the checks calls :func:`resolve_checks`, which raises
    :class:`ValueError` if ``checks`` names a check that does not exist, and
    then runs each selected check, which raises
    :class:`~tusk.exceptions.ValidationError` on a data defect. The first
    failure stops the run; later checks do not run.

    Args:
        frame: The table's lazy frame.
        schema: The table's schema.
        checks: Which checks to run. See :func:`resolve_checks`.
    """
    for check in resolve_checks(checks):
        check(frame, schema)
