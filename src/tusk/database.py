"""The schema model: tables, relationships, and the database that holds them."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import narwhals as nw

from tusk.exceptions import MissingPrimaryKeyWarning, SchemaError
from tusk.validation import Checks, validate_table


@dataclass(frozen=True)
class TableSchema:
    """Everything phase 1 knows about a table.

    Attributes:
        name: Table name within the database.
        primary_key: Column uniquely identifying a row, if declared.
        row_creation_time: Column recording when a row became knowable.
        dtypes: Mapping of column name to narwhals dtype.
    """

    name: str
    primary_key: str | None
    row_creation_time: str | None
    dtypes: Mapping[str, Any]


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
        self._is_eager: bool | None = None

    @property
    def is_eager(self) -> bool:
        """Whether the caller supplied eager frames."""
        return bool(self._is_eager)

    @property
    def table_names(self) -> tuple[str, ...]:
        """Names of every table in the database."""
        return tuple(self._schemas)

    def add_table(
        self,
        name: str,
        table: Any,
        primary_key: str | None = None,
        row_creation_time: str | None = None,
        *,
        validate: Checks = False,
    ) -> Database:
        """Add a table to the database.

        Args:
            name: Name to register the table under.
            table: A native frame or a narwhals frame.
            primary_key: Column uniquely identifying a row. Required for a
                table used as a relationship parent or as the DFS target.
            row_creation_time: Column recording when a row became knowable.
                Required for order-dependent primitives on this table.
            validate: Which validation checks to run against the data before
                registering the table. ``False`` (the default) runs none and
                reads no rows; ``True`` runs every check; a name or an
                iterable of names runs those. See
                :func:`tusk.validation.validate_table`.
                A requested check raising :class:`~tusk.exceptions.ValidationError`
                aborts the call before the table is registered, and an unknown
                check name or an invalid selector raises :class:`ValueError`.

        Returns:
            This database, to allow chaining.

        Raises:
            SchemaError: If the name is taken, a declared column is missing,
                a key is composite, or the backend differs from earlier tables.

        Warns:
            MissingPrimaryKeyWarning: If ``primary_key`` is omitted.
        """
        if name in self._schemas:
            raise SchemaError(f"table {name!r} is already in this database")
        _reject_composite(primary_key, "primary_key")
        _reject_composite(row_creation_time, "row_creation_time")

        frame = (
            table
            if isinstance(table, (nw.DataFrame, nw.LazyFrame))
            else nw.from_native(table)
        )
        is_eager = isinstance(frame, nw.DataFrame)
        lazy = frame.lazy() if is_eager else frame

        # Backend and eagerness are only assigned once validation passes, so a
        # caught ValidationError leaves this database exactly as it was --
        # otherwise a failed first add_table would silently rewrite an empty
        # database's backend, rejecting frames it should still accept.
        backend, was_eager = self._backend, self._is_eager
        if self._backend is None:
            backend, was_eager = lazy.implementation, is_eager
        elif lazy.implementation != self._backend:
            raise SchemaError(
                f"table {name!r} uses backend {lazy.implementation}, but this "
                f"database uses {self._backend}; narwhals cannot join across backends"
            )

        dtypes = dict(lazy.collect_schema())
        for column, label in (
            (primary_key, "primary_key"),
            (row_creation_time, "row_creation_time"),
        ):
            if column is not None and column not in dtypes:
                raise SchemaError(f"{label} {column!r} is not a column of {name!r}")

        if primary_key is None:
            warnings.warn(
                f"{name!r} has no primary_key: it cannot be used as a "
                f"relationship parent or as a DFS target, and order-dependent "
                f"primitives on it will have non-deterministic tiebreaks",
                MissingPrimaryKeyWarning,
                stacklevel=2,
            )

        schema = TableSchema(name, primary_key, row_creation_time, dtypes)
        validate_table(lazy, schema, validate)

        self._frames[name] = lazy
        self._schemas[name] = schema
        self._backend = backend
        self._is_eager = was_eager
        return self

    def add_relationship(self, parent: str, child: str, foreign_key: str) -> Database:
        """Link a parent table to a child table.

        Args:
            parent: Name of the parent table. Must have a ``primary_key``.
            child: Name of the child table.
            foreign_key: The child's column pointing at the parent's primary key.

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
                f"child table {child!r} is missing foreign_key column {foreign_key!r}"
            )
        self._relationships.append(Relationship(parent, child, foreign_key))
        return self

    def validate(self, checks: Checks = True) -> Database:
        """Run validation checks against every table in the database.

        Tables are checked in insertion order and the first failure raises, so
        a database with several defects reports the earliest one.

        Unlike the rest of the schema layer, this reads rows: each check runs
        real queries against the data. Running the checks may raise
        :class:`~tusk.exceptions.ValidationError` if a check finds a defect
        in a table's data, or :class:`ValueError` if ``checks`` names a check
        that does not exist or is not a recognized selector form.

        Args:
            checks: Which checks to run. ``True`` (the default) runs every
                check; ``False`` runs none; a name or an iterable of names
                runs those. See :func:`tusk.validation.validate_table`.

        Returns:
            This database, to allow chaining.
        """
        # One-shot iterators are accepted, so materialize before the loop:
        # otherwise the first table consumes the selector and every table
        # after it is silently skipped.
        if not isinstance(checks, (bool, str)):
            checks = list(checks)
        for name, schema in self._schemas.items():
            validate_table(self._frames[name], schema, checks)
        return self

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
