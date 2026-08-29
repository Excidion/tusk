"""The collection type synthesis hands back.

A :class:`FeatureList` is what makes "a feature set" a thing you can hold
rather than a list you have to keep facts about: it knows the one table its
features belong to, it can name every column they materialize, and it can
compute itself against a database. Its invariants -- non-empty, one table --
are the compiler's preconditions, checked here so a mistake surfaces where it
was made rather than one phase later.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime
from typing import Any, overload

from tusk.compiler import compile_features
from tusk.database import Database
from tusk.exceptions import SchemaError
from tusk.features import Feature
from tusk.validation import check_cutoff_time_zone


class FeatureList(Sequence[Feature]):
    """An immutable, single-table collection of feature definitions.

    Behaves as a :class:`~collections.abc.Sequence` of :class:`Feature`, so
    indexing, iteration, ``len`` and comprehensions all work as they would on
    a list. Slicing returns another ``FeatureList``.
    """

    def __init__(self, features: Iterable[Feature]) -> None:
        """Validate and freeze a collection of features.

        Args:
            features: The definitions to hold. All must be on one table.

        Raises:
            SchemaError: If ``features`` is empty or spans more than one table.
        """
        self._features = tuple(features)
        if not self._features:
            raise SchemaError("a feature set cannot be empty")
        tables = {f.table for f in self._features}
        if len(tables) > 1:
            raise SchemaError(f"features span multiple tables: {sorted(tables)}")
        self._target_table = tables.pop()

    @property
    def target_table(self) -> str:
        """The table every feature here is a column of."""
        return self._target_table

    @property
    def max_depth(self) -> int:
        """The deepest feature's number of stacked primitive applications."""
        return max(f.depth for f in self._features)

    @property
    def output_names(self) -> tuple[str, ...]:
        """Every column name the feature matrix will carry, in column order.

        Wider than ``len(self)`` when any feature is multi-output.
        """
        return tuple(n for f in self._features for n in f.output_names)

    @property
    def display_output_names(self) -> tuple[str, ...]:
        """The readable names, parallel to :attr:`output_names`."""
        return tuple(n for f in self._features for n in f.display_output_names)

    def apply(
        self,
        database: Database,
        cutoff_time: datetime | None = None,
    ) -> Any:
        """Compute these features against a database.

        Equivalent to :func:`tusk.apply_features`, which is the same call
        spelled as a free function for callers holding a plain sequence.

        Compilation raises :class:`~tusk.exceptions.SchemaError` if the target
        table has no ``primary_key``, or if two features compile to the same
        column name, and :class:`~tusk.exceptions.PrimitiveError` if an
        order-dependent primitive lands on a table with no
        ``row_creation_time``. ``check_cutoff_time_zone`` raises
        :class:`~tusk.exceptions.ValidationError` if ``cutoff_time`` disagrees
        with the database's row creation times in tz awareness.

        Args:
            database: The database to compute over.
            cutoff_time: Only rows whose ``row_creation_time`` is at or before
                this value are visible, on the target table as well as its
                relatives, so the matrix may have fewer rows than the target.
                Its tz awareness must match the database's row creation times'.
                None disables filtering.

        Returns:
            feature_matrix: An uncomputed query plan in the caller's native
                frame type, with one row per visible target row.

        Raises:
            TypeError: If ``cutoff_time`` is not a ``datetime``.
        """
        if cutoff_time is not None:
            if not isinstance(cutoff_time, datetime):
                raise TypeError("'cutoff_time' must be a datetime.")
            check_cutoff_time_zone(database, cutoff_time)
        return compile_features(self, database, cutoff_time).to_native()

    @overload
    def __getitem__(self, index: int) -> Feature: ...

    @overload
    def __getitem__(self, index: slice) -> FeatureList: ...

    def __getitem__(self, index: int | slice) -> Feature | FeatureList:
        """Index to a feature; slice to a narrower ``FeatureList``.

        A slice that selects nothing hits the empty-set invariant and raises
        :class:`~tusk.exceptions.SchemaError`, since there is no such thing as
        an empty feature set to hand back.

        Args:
            index: Position or slice.

        Returns:
            The feature at ``index``, or the sliced list.
        """
        if isinstance(index, slice):
            return FeatureList(self._features[index])
        return self._features[index]

    def __len__(self) -> int:
        """Count the definitions, not the columns they materialize."""
        return len(self._features)

    def __iter__(self) -> Iterator[Feature]:
        """Iterate the definitions in column order."""
        return iter(self._features)

    def __eq__(self, other: object) -> bool:
        """Compare structurally against another ``FeatureList``.

        A plain list of the same features is deliberately unequal, the way a
        tuple is: the type carries the single-table guarantee.
        """
        if not isinstance(other, FeatureList):
            return NotImplemented
        return self._features == other._features

    def __hash__(self) -> int:
        """Hash the frozen features, so feature sets can key a cache."""
        return hash(self._features)

    def __repr__(self) -> str:
        """Summarize, since a run of eight hundred features has to print."""
        return (
            f"FeatureList({len(self._features)} features "
            f"on {self._target_table!r}, max_depth={self.max_depth})"
        )
