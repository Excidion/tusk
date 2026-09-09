"""The schema model: tables, relationships, and the database that holds them."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import narwhals as nw

from tusk.exceptions import (
    ImplicitRowUpdateTimeMaskWarning,
    MissingPrimaryKeyWarning,
    SchemaError,
)
from tusk.plotting import SchemaDiagram
from tusk.validation import (
    DEFAULT_TABLE_CHECKS,
    validate_database,
    validate_relationship,
    validate_table,
)


@dataclass(frozen=True)
class TableSchema:
    """Everything phase 1 knows about a table.

    Attributes:
        name: Table name within the database.
        primary_key: Column uniquely identifying a row, if declared.
        row_creation_time: Column recording when a row became knowable.
        dtypes: Mapping of column name to narwhals dtype.
        row_update_times: Mapping of each column recording an update time to
            the columns that update rewrote, each mapped to the value it held
            before the update.
    """

    name: str
    primary_key: str | None
    row_creation_time: str | None
    dtypes: Mapping[str, Any]
    row_update_times: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    @property
    def column_updates(self) -> tuple[tuple[str, str, Any], ...]:
        """Every update as a (row update time, column, pre-update value) triple."""
        return tuple(
            (update_time, column, value)
            for update_time, updated in self.row_update_times.items()
            for column, value in updated.items()
        )


@dataclass(frozen=True)
class Relationship:
    """A one-to-many link from a parent table to a child table.

    The parent side is always the parent's ``primary_key``; ``foreign_key``
    names the child's column.

    Attributes:
        parent: Name of the parent table.
        child: Name of the child table.
        foreign_key: Column on the child pointing at the parent's primary key.
    """

    parent: str
    child: str
    foreign_key: str


class Database:
    """A collection of related tables that DFS can synthesize features over."""

    def __init__(self, name: str) -> None:
        """Create an empty database.

        Args:
            name: Human-readable identifier for this database.
        """
        self.name = name
        self._frames: dict[str, nw.LazyFrame] = {}
        self._schemas: dict[str, TableSchema] = {}
        self._relationships: list[Relationship] = []
        self._backend: Any = None

    @property
    def table_names(self) -> tuple[str, ...]:
        """Names of every table in the database."""
        return tuple(self._schemas)

    @property
    def relationships(self) -> tuple[Relationship, ...]:
        """Every relationship in the database, in insertion order."""
        return tuple(self._relationships)

    def add_table(
        self,
        name: str,
        table: Any,
        primary_key: str | None = None,
        row_creation_time: str | None = None,
        row_update_times: Mapping[str, Mapping[str, Any]] | None = None,
        *,
        validate: bool | str | Iterable[str] = DEFAULT_TABLE_CHECKS,
    ) -> Database:
        """Add a table to the database.

        A selected check that fails raises
        :class:`~tusk.exceptions.ValidationError`.

        Args:
            name: Name to register the table under.
            table: A native frame or a narwhals frame, eager or lazy.
                It is lazified on the way in, so the two forms are
                interchangeable and may be mixed within one database.
            primary_key: Column uniquely identifying a row. Required for a
                table used as a relationship parent or as the DFS target.
            row_creation_time: Column recording when a row became knowable.
                Required for order-dependent primitives on this table.
            row_update_times: Maps each column recording when a row was
                updated to the columns that update rewrote, each mapped to the
                value it held before. Under a ``cutoff_time``, a row updated
                after the cutoff serves those earlier values instead. An update
                time column that no update time lists is added to its own
                mapping with a null value.
            validate: Pick a string or list of strings from
                [here](validation/#tusk.validation.TABLE_CHECKS)
                to enable specific checks.
                `True` runs every check, `False` runs none. The default runs
                every check that reads no rows.

        Returns:
            This database, to allow chaining.

        Raises:
            SchemaError: If the name is taken, a declared column is missing,
                a key is composite, or the backend differs from earlier tables.

        Warns:
            MissingPrimaryKeyWarning: If ``primary_key`` is omitted.
            ImplicitRowUpdateTimeMaskWarning: If a ``row_update_times`` key is
                absent from every mapping, so tusk gave it a null value.
        """
        if name in self._schemas:
            raise SchemaError(f"table {name!r} is already in this database")
        _reject_composite(primary_key, "primary_key")
        _reject_composite(row_creation_time, "row_creation_time")

        # nw.from_native and .lazy() are both idempotent, so this one line
        # accepts a native or narwhals frame, eager or lazy, and normalizes
        # every one of them to the lazy frame the rest of tusk works on.
        lazy = nw.from_native(table).lazy()

        # The backend is only assigned once validation passes, so a caught
        # ValidationError leaves this database exactly as it was -- otherwise
        # a failed first add_table would silently rewrite an empty database's
        # backend, rejecting frames it should still accept.
        backend = self._backend
        if self._backend is None:
            backend = lazy.implementation
        elif lazy.implementation != self._backend:
            raise SchemaError(
                f"table {name!r} uses backend {lazy.implementation}, but this "
                f"database uses {self._backend}; narwhals cannot join across backends",
            )

        dtypes = dict(lazy.collect_schema())
        for column, label in (
            (primary_key, "primary_key"),
            (row_creation_time, "row_creation_time"),
        ):
            if column is not None and column not in dtypes:
                raise SchemaError(f"{label} {column!r} is not a column of {name!r}")

        row_update_times = _normalize_row_update_times(row_update_times, name)
        _reject_unknown_updated_columns(row_update_times, dtypes, name)
        incomplete = _row_update_times_without_a_value(row_update_times)
        _warn_about_incomplete_row_update_times(incomplete, name)
        row_update_times = _insert_own_values(row_update_times, incomplete)

        if primary_key is None:
            warnings.warn(
                f"{name!r} has no primary_key: it cannot be used as a "
                f"relationship parent or as a DFS target, and order-dependent "
                f"primitives on it will have non-deterministic tiebreaks",
                MissingPrimaryKeyWarning,
                stacklevel=2,
            )

        schema = TableSchema(
            name, primary_key, row_creation_time, dtypes, row_update_times
        )
        validate_table(lazy, schema, validate)

        self._frames[name] = lazy
        self._schemas[name] = schema
        self._backend = backend
        return self

    def add_relationship(
        self,
        parent: str,
        child: str,
        foreign_key: str,
        *,
        validate: bool | str | Iterable[str] = "matching_key_dtypes",
    ) -> Database:
        """Link a parent table to a child table.

        Args:
            parent: Name of the parent table. Must have a ``primary_key``.
            child: Name of the child table.
            foreign_key: The child's column pointing at the parent's primary key.
            validate: Pick a string or list of strings from
                [here](validation/#tusk.validation.RELATIONSHIP_CHECKS)
                to enable specific checks.
                `True` runs every check, `False` runs none.

        Returns:
            This database, to allow chaining.

        Raises:
            SchemaError: If a table is unknown, the parent has no primary key,
                the foreign key is composite, or the child lacks that column.
        """
        _reject_composite(foreign_key, "foreign_key")
        for table in (parent, child):
            if table not in self._schemas:
                raise SchemaError(f"unknown table {table!r}")
        if self._schemas[parent].primary_key is None:
            raise SchemaError(f"parent table {parent!r} needs a primary_key")
        if foreign_key not in self._schemas[child].dtypes:
            raise SchemaError(
                f"child table {child!r} is missing foreign_key column {foreign_key!r}",
            )

        relationship = Relationship(parent, child, foreign_key)
        validate_relationship(self, relationship, validate)

        self._relationships.append(relationship)
        return self

    def validate(
        self,
        *,
        database: bool | str | Iterable[str] = True,
        tables: bool | str | Iterable[str] = True,
        relationships: bool | str | Iterable[str] = True,
    ) -> Database:
        """Run validation checks against the database.

        Table checks run against every table in insertion order, then
        relationship checks against every relationship, then database-wide
        checks once. The first failure raises.

        Args:
            database: [Checks](validation/#tusk.validation.DATABASE_CHECKS)
                spanning the whole database.
            tables: [Checks](validation/#tusk.validation.TABLE_CHECKS)
                run against each table.
            relationships: [Checks](validation/#tusk.validation.RELATIONSHIP_CHECKS)
                run against each relationship.

        Returns:
            This database, to allow chaining.
        """
        validate_database(
            self,
            database_checks=database,
            table_checks=tables,
            relationship_checks=relationships,
        )
        return self

    def plot(self, columns: bool | str = True) -> SchemaDiagram:
        """Draw the database's schema as an entity-relationship diagram.

        Reads no rows: the diagram is built entirely from declared schema.

        A ``columns`` value other than True, False, or ``"structural"`` raises
        :class:`ValueError`.

        Args:
            columns: True lists every column, False lists none, and
                ``"structural"`` lists only the primary key, the foreign keys,
                the ``row_creation_time``, the ``row_update_times``, and the
                columns they update.

        Returns:
            The diagram, which renders itself in a notebook and writes itself
            to a file with :meth:`~tusk.SchemaDiagram.save`.
        """
        return SchemaDiagram.from_database(self, columns)

    def schema(self, name: str) -> TableSchema:
        """Return a table's schema.

        Args:
            name: Table name.

        Returns:
            The table's schema.

        Raises:
            SchemaError: If the table is unknown.
        """
        try:
            return self._schemas[name]
        except KeyError:
            raise SchemaError(f"unknown table {name!r}") from None

    def frame(self, name: str) -> nw.LazyFrame:
        """Return a table's lazy frame.

        Args:
            name: Table name.

        Returns:
            The table's narwhals LazyFrame.

        Raises:
            SchemaError: If the table is unknown.
        """
        try:
            return self._frames[name]
        except KeyError:
            raise SchemaError(f"unknown table {name!r}") from None

    def children_of(self, name: str) -> list[Relationship]:
        """Return relationships where this table is the parent.

        Args:
            name: Table name.

        Returns:
            Matching relationships, in insertion order.
        """
        return [r for r in self._relationships if r.parent == name]

    def parents_of(self, name: str) -> list[Relationship]:
        """Return relationships where this table is the child.

        Args:
            name: Table name.

        Returns:
            Matching relationships, in insertion order.
        """
        return [r for r in self._relationships if r.child == name]

    def input_excluded_columns(self, name: str) -> frozenset[str]:
        """Return columns that may not be fed to a primitive as an input.

        Join keys only: the primary key and every foreign key. They identify
        rows rather than measure anything, so ``MEAN(customer_id)`` is noise.
        Foreign keys remain usable as groupby keys.

        The ``row_creation_time`` is deliberately **not** here. It is a real
        measurement — ``MONTH(signed_up_at)``-style temporal transforms, and
        ``N_UNIQUE`` or ``CUM_COUNT`` over a temporal column, are exactly the
        features this split unblocks — and excluding it would leave
        a zero-configuration run with no transform features at all. Contrast
        :meth:`output_excluded_columns`, which does exclude it; conflating the
        two sets is a bug this split exists to prevent.

        Args:
            name: Table name.

        Returns:
            The table's join-key column names.
        """
        schema = self.schema(name)
        keys: set[str] = set()
        if schema.primary_key is not None:
            keys.add(schema.primary_key)
        keys.update(r.foreign_key for r in self.parents_of(name))
        return frozenset(keys)

    def output_excluded_columns(self, name: str) -> frozenset[str]:
        """Return raw columns that never appear in the feature matrix.

        Everything in :meth:`input_excluded_columns`, plus the
        ``row_creation_time``: passing the time index through as a feature
        invites target leakage, and featuretools drops it from the matrix for
        the same reason. Derived features *over* the row creation time, such as
        ``MONTH(signed_up_at)``, are unaffected — only the raw column is
        dropped.

        Args:
            name: Table name.

        Returns:
            Column names to omit from the feature matrix.
        """
        schema = self.schema(name)
        keys = set(self.input_excluded_columns(name))
        if schema.row_creation_time is not None:
            keys.add(schema.row_creation_time)
        return frozenset(keys)


def _reject_composite(value: Any, label: str) -> None:
    """Raise if a key was given as a sequence.

    Args:
        value: The declared key.
        label: Parameter name, used in the message.

    Raises:
        SchemaError: If the value is a list or tuple.
    """
    if isinstance(value, (list, tuple)):
        raise SchemaError(f"composite {label} is not supported; got {value!r}")


def _normalize_row_update_times(
    row_update_times: Mapping[str, Mapping[str, Any]] | None,
    table: str,
) -> dict[str, dict[str, Any]]:
    """Copy a row update time declaration into plain nested dicts.

    Args:
        row_update_times: The declaration, or None.
        table: Table name, used in the message.

    Returns:
        The declaration as nested dicts, empty when None was given.

    Raises:
        SchemaError: If it is not a mapping of column name to mapping.
    """
    if row_update_times is None:
        return {}
    if not isinstance(row_update_times, Mapping):
        raise SchemaError(
            f"row_update_times of {table!r} must map each update time column "
            f"to the columns it updates; got {row_update_times!r}",
        )
    normalized = {}
    for update_time, updated in row_update_times.items():
        if not isinstance(updated, Mapping):
            raise SchemaError(
                f"row_update_time {update_time!r} of {table!r} must map each "
                f"column it updates to that column's pre-update value; "
                f"got {updated!r}",
            )
        normalized[update_time] = dict(updated)
    return normalized


def _reject_unknown_updated_columns(
    row_update_times: Mapping[str, Mapping[str, Any]],
    dtypes: Mapping[str, Any],
    table: str,
) -> None:
    """Raise if a row update time declaration names a column the table lacks.

    Args:
        row_update_times: The normalized declaration.
        dtypes: The table's columns.
        table: Table name, used in the message.

    Raises:
        SchemaError: If an update time or an updated column is not a column of
            the table.
    """
    for update_time, updated in row_update_times.items():
        if update_time not in dtypes:
            raise SchemaError(
                f"row_update_time {update_time!r} is not a column of {table!r}",
            )
        unknown = [column for column in updated if column not in dtypes]
        if unknown:
            raise SchemaError(
                f"{unknown[0]!r}, listed under row_update_time "
                f"{update_time!r}, is not a column of {table!r}",
            )


def _row_update_times_without_a_value(
    row_update_times: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Return the update time columns that do not list themselves.

    Args:
        row_update_times: The normalized declaration.

    Returns:
        Their names, in declaration order.
    """
    return [name for name, updated in row_update_times.items() if name not in updated]


def _warn_about_incomplete_row_update_times(incomplete: list[str], table: str) -> None:
    """Warn that tusk will read each named column as null before its update.

    Args:
        incomplete: Update time columns nothing gives a value, as built by
            :func:`_row_update_times_without_a_value`.
        table: Table name, used in the message.
    """
    for update_time in incomplete:
        warnings.warn(
            f"row_update_time {update_time!r} of {table!r} does not say what "
            f"it held before the update, so tusk reads it as null; list it "
            f"under itself to choose a value",
            ImplicitRowUpdateTimeMaskWarning,
            stacklevel=3,
        )


def _insert_own_values(
    row_update_times: Mapping[str, Mapping[str, Any]],
    incomplete: list[str],
) -> dict[str, dict[str, Any]]:
    """Give each update time column named in ``incomplete`` a null value.

    Args:
        row_update_times: The normalized declaration.
        incomplete: Update time columns nothing gives a value.

    Returns:
        The declaration with a null entry added for each of them.
    """
    completed = {key: dict(values) for key, values in row_update_times.items()}
    for update_time in incomplete:
        completed[update_time][update_time] = None
    return completed
