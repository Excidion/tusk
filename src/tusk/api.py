"""The public entry points.

:func:`deep_feature_synthesis` builds feature definitions and computes them;
:func:`apply_features` computes existing ones.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any

from tusk.database import Database
from tusk.feature_list import FeatureList
from tusk.features import Feature
from tusk.primitives.base import Primitive
from tusk.synthesis import synthesize


def deep_feature_synthesis(
    database: Database,
    target_table: str,
    agg_primitives: Iterable[str | Primitive] | None = None,
    trans_primitives: Iterable[str | Primitive] | None = None,
    groupby_trans_primitives: Iterable[str | Primitive] | None = None,
    max_depth: int = 2,
    cutoff_time: datetime | None = None,
    features_only: bool = False,
) -> Any:
    """Run deep feature synthesis over a database.

    Synthesis raises :class:`~tusk.exceptions.SchemaError` if the target table
    is unknown or the walk generates no features at all, and
    :class:`~tusk.exceptions.PrimitiveError` for an unknown primitive name or
    an order-dependent primitive on a table with no ``row_creation_time``.
    Compilation raises :class:`~tusk.exceptions.SchemaError` if the target
    table has no ``primary_key``, and whatever :func:`apply_features`
    documents for ``cutoff_time``.

    Args:
        database: The tables and relationships to synthesize over.
        target_table: Table to build features for. The result has one
            row per *visible* row of this table, keyed by its ``primary_key``.
            With no ``cutoff_time`` that is every row; with one, the target is
            filtered like any other table, so the matrix may have fewer rows.
        agg_primitives: Aggregation primitives, as names or instances. None
            selects the documented defaults.
        trans_primitives: Transform primitives. None selects the defaults.
        groupby_trans_primitives: Transforms applied within foreign-key groups.
            None means none.
        max_depth: Maximum number of stacked primitive applications.
        cutoff_time: Only rows whose ``row_creation_time`` is at or before this
            value are visible, on the target table as well as its relatives.
            Its tz awareness must match the database's row creation times'.
            A table with no ``row_creation_time`` is timeless and passes
            through unfiltered, so a cutoff on a database that declares none is
            silently a no-op. None disables filtering. Ignored entirely when
            ``features_only`` is true, since nothing is computed: the cutoff
            belongs to compilation, and feature definitions do not record it.
        features_only: Return the feature definitions without computing them.

    Returns:
        feature_matrix: An uncomputed query plan in the caller's native frame
            type. On a backend with a lazy frame type you get one back, so call
            ``.collect()`` to compute it. Not returned when ``features_only``
            is true.
        features (FeatureList): The feature definitions, reusable with
            :meth:`~tusk.FeatureList.apply` or :func:`apply_features`.

    Warns:
        CategoricalDtypeWarning: If a Categorical or Enum column is skipped
            because a requested primitive requires a string input.
        UnmatchedPrimitiveWarning: If a requested primitive matched no column
            of its input dtypes anywhere in the walk.
    """
    features = synthesize(
        database=database,
        target_table=target_table,
        agg_primitives=agg_primitives,
        trans_primitives=trans_primitives,
        groupby_trans_primitives=groupby_trans_primitives,
        max_depth=max_depth,
    )
    if features_only:
        return features
    return features.apply(database, cutoff_time), features


def apply_features(
    features: Sequence[Feature],
    database: Database,
    cutoff_time: datetime | None = None,
) -> Any:
    """Apply existing feature definitions to a database.

    Use this to apply a feature set fitted on training data to new data. It
    accepts any sequence of features; a :class:`~tusk.FeatureList` can compute
    itself with :meth:`~tusk.FeatureList.apply` instead.

    Raises :class:`~tusk.exceptions.SchemaError` if ``features`` is empty,
    spans more than one table, or targets a table with no ``primary_key``, and
    :class:`~tusk.exceptions.PrimitiveError` if an order-dependent primitive
    lands on a table with no ``row_creation_time``.
    :class:`~tusk.exceptions.ValidationError` is raised if ``cutoff_time``
    differs from the database's row creation times in tz awareness, or if
    those disagree among themselves; or if a feature's primitive measures
    against ``cutoff_time`` and none was given. ``TypeError`` is raised if
    ``cutoff_time`` is not a ``datetime``.

    Args:
        features: Feature definitions, all on the same target table.
        database: The database to compute over.
        cutoff_time: Only rows whose ``row_creation_time`` is at or before this
            value are visible, on the target table as well as its relatives, so
            the matrix may have fewer rows than the target. Its tz awareness
            must match the database's row creation times'. A table with no
            ``row_creation_time`` is timeless and passes through unfiltered,
            so a cutoff on a database that declares none is silently a no-op.
            None disables filtering.

    Returns:
        feature_matrix: An uncomputed query plan in the caller's native frame
            type, with one row per visible target row. tusk never collects, so
            on a backend with a lazy frame type you decide when to compute.
    """
    return FeatureList(features).apply(database, cutoff_time)
