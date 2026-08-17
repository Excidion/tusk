"""The public entry points: :func:`dfs` and :func:`calculate_feature_matrix`."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from tusk.compiler import compile_features
from tusk.entityset import EntitySet
from tusk.features import Feature
from tusk.primitives.aggregation import AGG_DEFAULTS
from tusk.primitives.base import Primitive
from tusk.primitives.transform import TRANS_DEFAULTS
from tusk.synthesis import synthesize


def dfs(
    entityset: EntitySet,
    target_dataframe_name: str,
    agg_primitives: Iterable[str | Primitive] | None = None,
    trans_primitives: Iterable[str | Primitive] | None = None,
    groupby_trans_primitives: Iterable[str | Primitive] | None = None,
    max_depth: int = 2,
    cutoff_time: Any = None,
    features_only: bool = False,
) -> Any:
    """Run deep feature synthesis over an entity set.

    Args:
        entityset: The tables and relationships to synthesize over.
        target_dataframe_name: Table to build features for. One row of the
            result corresponds to one row of this table.
        agg_primitives: Aggregation primitives, as names or instances. None
            selects the documented defaults.
        trans_primitives: Transform primitives. None selects the defaults.
        groupby_trans_primitives: Transforms applied within foreign-key groups.
            None means none.
        max_depth: Maximum number of stacked primitive applications.
        cutoff_time: Only rows whose ``row_creation_time`` is at or before this
            value are visible. None disables filtering.
        features_only: Return the feature definitions without computing them.

    Returns:
        A ``(feature_matrix, features)`` tuple, where the matrix is in the
        caller's native frame type; or just the feature list when
        ``features_only`` is true.
    """
    features = synthesize(
        entityset=entityset,
        target_dataframe_name=target_dataframe_name,
        agg_primitives=AGG_DEFAULTS if agg_primitives is None else agg_primitives,
        trans_primitives=TRANS_DEFAULTS
        if trans_primitives is None
        else trans_primitives,
        groupby_trans_primitives=groupby_trans_primitives or (),
        max_depth=max_depth,
    )
    if features_only:
        return features
    return calculate_feature_matrix(features, entityset, cutoff_time), features


def calculate_feature_matrix(
    features: Sequence[Feature],
    entityset: EntitySet,
    cutoff_time: Any = None,
) -> Any:
    """Compute a feature matrix from existing feature definitions.

    Use this to apply a feature set fitted on training data to new data.

    Args:
        features: Feature definitions, all on the same target table.
        entityset: The entity set to compute over.
        cutoff_time: Only rows whose ``row_creation_time`` is at or before this
            value are visible. None disables filtering.

    Returns:
        The feature matrix in the caller's native frame type.
    """
    frame = compile_features(features, entityset, cutoff_time)
    if entityset.is_eager:
        return frame.collect().to_native()
    return frame.to_native()
