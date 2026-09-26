"""The schema model: tables, relationships, and the database that holds them."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import narwhals as nw

from tusk.exceptions import (
    ImplicitEarlierValueWarning,
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
        name: The table's name within the database.
        primary_key: The column that identifies a row uniquely, if declared.
        row_creation_time: The column that records when a row became
            knowable.
        dtypes: A mapping of column name to narwhals dtype.
        row_update_times: A mapping of each column recording an update time
            to the columns that update rewrote, each mapped to the value it
            held before the update.
        where: Named row conditions as narwhals expressions.
        when: Named row conditions as callables that take the cutoff time
            and return a narwhals expression.
    """

    name: str
    primary_key: str | None
    row_creation_time: str | None
    dtypes: Mapping[str, Any]
    row_update_times: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    where: Mapping[str, Any] = field(default_factory=dict)
    when: Mapping[str, Any] = field(default_factory=dict)

    @property
    def column_updates(self) -> tuple[tuple[str, str, Any], ...]:
        """Every update as a (row update time, column, pre-update value) triple."""
        return tuple(
            (update_time, column, value)
            for update_time, updated in self.row_update_times.items()
            for column, value in updated.items()
        )

    @property
    def conditions(self) -> tuple[tuple[str, str], ...]:
        """Every declared condition as a (kind, key) pair.

        The where conditions come first.
        """
        return tuple(
            [("where", key) for key in self.where]
            + [("when", key) for key in self.when],
        )


@dataclass(frozen=True)
class Relationship:
    """A one-to-many link from a parent table to a child table.

    The parent side is always the parent's ``primary_key``. ``foreign_key``
    names the child's column.

    Attributes:
        parent: The name of the parent table.
        child: The name of the child table.
        foreign_key: The column on the child that points at the parent's
            primary key.
    """

    parent: str
    child: str
    foreign_key: str


class Database:
    """A collection of related tables that DFS can synthesize features over."""

    def __init__(self, name: str) -> None:
        """Create an empty database.

        Args:
            name: A name for this database that people can read.
        """
        self.name = name
        self._tables: dict[str, nw.LazyFrame] = {}
        self._schemas: dict[str, TableSchema] = {}
        self._relationships: list[Relationship] = []
        self._backend: Any = None

    @property
    def table_names(self) -> tuple[str, ...]:
        """The names of every table in the database."""
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
        where: Mapping[str, Any] | None = None,
        when: Mapping[str, Any] | None = None,
        *,
        validate: bool | str | Iterable[str] = DEFAULT_TABLE_CHECKS,
    ) -> Database:
        """Add a table to the database.

        A selected check that fails raises
        :class:`~tusk.exceptions.ValidationError`.

        Args:
            name: The name to register the table under.
            table: A native table object or a narwhals table object, eager
                or lazy. It becomes a lazy narwhals object on input, so
                eager and lazy tables are interchangeable, and one database
                may mix both.
            primary_key: The column that identifies a row uniquely.
                Required for a table used as a relationship parent or as
                the DFS target.
            row_creation_time: The column that records when a row became
                knowable. Required for order-dependent primitives on this
                table.
            row_update_times: Maps each column recording when a row was
                updated to the columns that update rewrote, each mapped to the
                value it held before. Computing a feature matrix under a
                ``cutoff_time`` gives those columns their earlier value on
                every row updated after the cutoff time. An update time
                column that does not list itself is added to its own mapping
                with a null value.
            where: Named row conditions as narwhals expressions, used to mask
                a child table's rows before an aggregation groups them. Each
                key becomes a feature name part.
            when: Named row conditions as callables that take the cutoff
                time and return a narwhals expression. Use this for a
                condition measured against the cutoff time, such as a
                validity interval.
            validate: Pick a string or list of strings from
                [here](validation/#tusk.validation.TABLE_CHECKS)
                to enable specific checks.
                `True` runs every check, `False` runs none. The default runs
                every check that needs no query.

        Returns:
            This database, to allow chaining.

        Raises:
            SchemaError: If the name is taken, a declared column does not
                exist, a key is composite, or the backend differs from
                earlier tables.

        Warns:
            MissingPrimaryKeyWarning: If ``primary_key`` is omitted.
            ImplicitEarlierValueWarning: If a ``row_update_times`` key
                does not list itself, so tusk gave it a null value.
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

        where = dict(where or {})
        when = dict(when or {})

        schema = TableSchema(
            name=name,
            primary_key=primary_key,
            row_creation_time=row_creation_time,
            dtypes=dtypes,
            row_update_times=row_update_times,
            where=where,
            when=when,
        )
        validate_table(lazy, schema, validate)

        self._tables[name] = lazy
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
            parent: The name of the parent table. Must have a ``primary_key``.
            child: The name of the child table.
            foreign_key: The child's column that points at the parent's
                primary key.
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

        Runs checks in this order:

        1. Table checks against every table, in insertion order.
        2. Relationship checks against every relationship.
        3. Database-wide checks, once.

        The first failure raises.

        Args:
            database: The [checks](validation/#tusk.validation.DATABASE_CHECKS)
                that span the whole database.
            tables: The [checks](validation/#tusk.validation.TABLE_CHECKS)
                that run against each table.
            relationships: The
                [checks](validation/#tusk.validation.RELATIONSHIP_CHECKS)
                that run against each relationship.

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

        Reads no rows. The diagram is built entirely from the declared
        schema.

        A ``columns`` value other than True, False, or ``"structural"`` raises
        :class:`ValueError`.

        Args:
            columns: True lists every column. False lists no columns.
                ``"structural"`` lists only the primary key, the foreign
                keys, the ``row_creation_time``, the ``row_update_times``,
                and the columns they update.

        Returns:
            The diagram. It renders itself in a notebook and writes itself
            to a file with :meth:`~tusk.SchemaDiagram.save`.
        """
        return SchemaDiagram.from_database(self, columns)

    def get_schema(self, name: str) -> TableSchema:
        """Return a table's schema.

        Args:
            name: The table's name.

        Returns:
            The table's schema.

        Raises:
            SchemaError: If the table is unknown.
        """
        try:
            return self._schemas[name]
        except KeyError:
            raise SchemaError(f"unknown table {name!r}") from None

    def get_table(self, name: str) -> nw.LazyFrame:
        """Return a table as a narwhals LazyFrame.

        Args:
            name: The table's name.

        Returns:
            The table as a narwhals LazyFrame.

        Raises:
            SchemaError: If the table is unknown.
        """
        try:
            return self._tables[name]
        except KeyError:
            raise SchemaError(f"unknown table {name!r}") from None

    def children_of(self, name: str) -> list[Relationship]:
        """Return relationships where this table is the parent.

        Args:
            name: The table's name.

        Returns:
            Matching relationships, in insertion order.
        """
        return [r for r in self._relationships if r.parent == name]

    def parents_of(self, name: str) -> list[Relationship]:
        """Return relationships where this table is the child.

        Args:
            name: The table's name.

        Returns:
            Matching relationships, in insertion order.
        """
        return [r for r in self._relationships if r.child == name]

    def input_excluded_columns(self, name: str) -> frozenset[str]:
        """Return columns that may not be fed to a primitive as an input.

        Only join keys: the primary key and every foreign key. They identify
        rows, not measurements: ``MEAN(customer_id)`` would be noise.
        Foreign keys remain usable as groupby keys.

        The ``row_creation_time`` is not included here. Primitives can
        still use it, for example for ``MONTH(signed_up_at)``-style
        temporal transforms, or for ``N_UNIQUE`` or ``CUM_COUNT`` over a
        temporal column. Contrast :meth:`output_excluded_columns`, which
        does exclude it.

        Args:
            name: The table's name.

        Returns:
            The table's join-key column names.
        """
        schema = self.get_schema(name)
        keys: set[str] = set()
        if schema.primary_key is not None:
            keys.add(schema.primary_key)
        keys.update(r.foreign_key for r in self.parents_of(name))
        return frozenset(keys)

    def output_excluded_columns(self, name: str) -> frozenset[str]:
        """Return raw columns that never appear in the feature matrix.

        Everything in :meth:`input_excluded_columns`, plus the
        ``row_creation_time``. This also matches what featuretools drops
        from the feature matrix, to avoid target leakage. Derived features
        *over* the row creation time, such as ``MONTH(signed_up_at)``, are
        not affected. Only the raw column is dropped.

        Args:
            name: The table's name.

        Returns:
            Column names to omit from the feature matrix.
        """
        schema = self.get_schema(name)
        keys = set(self.input_excluded_columns(name))
        if schema.row_creation_time is not None:
            keys.add(schema.row_creation_time)
        return frozenset(keys)


def _reject_composite(value: Any, label: str) -> None:
    """Raise if the key is a sequence.

    Args:
        value: The declared key.
        label: The parameter name, used in the message.

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
        table: The table's name, used in the message.

    Returns:
        The declaration as nested dicts, or an empty dict if None is given.

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
        table: The table's name, used in the message.

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
        incomplete: The update time columns that have no given value, as
            built by :func:`_row_update_times_without_a_value`.
        table: The table's name, used in the message.
    """
    for update_time in incomplete:
        warnings.warn(
            f"row_update_time {update_time!r} of {table!r} does not say what "
            f"it held before the update, so tusk reads it as null; list it "
            f"under itself to choose a value",
            ImplicitEarlierValueWarning,
            stacklevel=3,
        )


def _insert_own_values(
    row_update_times: Mapping[str, Mapping[str, Any]],
    incomplete: list[str],
) -> dict[str, dict[str, Any]]:
    """Give each update time column named in ``incomplete`` a null value.

    Args:
        row_update_times: The normalized declaration.
        incomplete: The update time columns that have no given value.

    Returns:
        The declaration with a null entry added for each of them.
    """
    completed = {key: dict(values) for key, values in row_update_times.items()}
    for update_time in incomplete:
        completed[update_time][update_time] = None
    return completed
