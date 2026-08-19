"""The public entry points.

:func:`deep_feature_synthesis` and :func:`apply_features`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from tusk.compiler import compile_features
from tusk.database import Database
from tusk.features import Feature
from tusk.primitives.aggregation import AGG_DEFAULTS
from tusk.primitives.base import Primitive
from tusk.primitives.transform import TRANS_DEFAULTS
from tusk.synthesis import synthesize


def deep_feature_synthesis(
    database: Database,
    target_table: str,
    agg_primitives: Iterable[str | Primitive] | None = None,
    trans_primitives: Iterable[str | Primitive] | None = None,
    groupby_trans_primitives: Iterable[str | Primitive] | None = None,
    max_depth: int = 2,
    cutoff_time: Any = None,
    features_only: bool = False,
) -> Any:
    """Run deep feature synthesis over a database.

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
            A table with no ``row_creation_time`` is timeless and passes
            through unfiltered, so a cutoff on a database that declares none
            is silently a no-op. None disables filtering. Ignored entirely when
            ``features_only`` is true, since nothing is computed: the cutoff
            belongs to compilation, and feature definitions do not record it.
        features_only: Return the feature definitions without computing them.

    Returns:
        A ``(feature_matrix, features)`` tuple, where the matrix is in the
        caller's native frame type; or just the feature list when
        ``features_only`` is true. Synthesis raises
        :class:`~tusk.exceptions.SchemaError` if the target table is unknown
        and :class:`~tusk.exceptions.PrimitiveError` for an unknown primitive
        name or an order-dependent primitive on a table with no
        ``row_creation_time``; compilation raises
        :class:`~tusk.exceptions.SchemaError` if the target table has no
        ``primary_key``.

    Warns:
        CategoricalDtypeWarning: If a Categorical or Enum column is skipped
            because a requested primitive requires a string input.
        UnmatchedPrimitiveWarning: If a requested primitive matched no column
            of its input dtypes anywhere in the walk.
    """
    features = synthesize(
        entityset=database,
        target_dataframe_name=target_table,
        agg_primitives=AGG_DEFAULTS if agg_primitives is None else agg_primitives,
        trans_primitives=TRANS_DEFAULTS
        if trans_primitives is None
        else trans_primitives,
        groupby_trans_primitives=groupby_trans_primitives or (),
        max_depth=max_depth,
    )
    if features_only:
        return features
    return apply_features(features, database, cutoff_time), features


def apply_features(
    features: Sequence[Feature],
    database: Database,
    cutoff_time: Any = None,
) -> Any:
    """Apply existing feature definitions to a database.

    Use this to apply a feature set fitted on training data to new data.

    Args:
        features: Feature definitions, all on the same target table.
        database: The database to compute over.
        cutoff_time: Only rows whose ``row_creation_time`` is at or before this
            value are visible, on the target table as well as its relatives, so
            the matrix may have fewer rows than the target. A table with no
            ``row_creation_time`` is timeless and passes through unfiltered, so
            a cutoff on a database that declares none is silently a no-op.
            None disables filtering.

    Returns:
        The feature matrix in the caller's native frame type, with one row per
        visible target row. ``compile_features`` raises
        :class:`~tusk.exceptions.SchemaError` if ``features`` is empty, spans
        more than one table, or targets a table with no ``primary_key``, and
        :class:`~tusk.exceptions.PrimitiveError` if an order-dependent
        primitive lands on a table with no ``row_creation_time``.
    """
    frame = compile_features(features, database, cutoff_time)
    if database.is_eager:
        return frame.collect().to_native()
    return frame.to_native()
