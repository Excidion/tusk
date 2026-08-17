"""Phase 2: turn feature definitions into a single lazy query plan.

The only ``collect()`` in tusk is the documented eager round-trip in
:func:`compile_features`' caller; nothing here materializes a frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import narwhals as nw

from tusk.entityset import EntitySet
from tusk.exceptions import SchemaError
from tusk.features import Feature, IdentityFeature, TransformFeature


def compile_features(
    features: Sequence[Feature],
    entityset: EntitySet,
    cutoff_time: Any = None,
) -> nw.LazyFrame:
    """Compile feature definitions into a lazy feature matrix.

    Args:
        features: Features to compute. All must be on the same table.
        entityset: The entity set holding the frames.
        cutoff_time: Only rows whose ``row_creation_time`` is at or before this
            value are visible. None disables filtering.

    Returns:
        A lazy frame with the target's primary key plus one column per feature
        output.

    Raises:
        SchemaError: If the features span tables, the list is empty, or the
            target table has no primary key.
    """
    if not features:
        raise SchemaError("no features to compile")
    tables = {f.table for f in features}
    if len(tables) > 1:
        raise SchemaError(f"features span multiple tables: {sorted(tables)}")
    target = tables.pop()
    primary_key = entityset.schema(target).primary_key
    if primary_key is None:
        raise SchemaError(
            f"target table {target!r} needs a primary_key: the feature "
            "matrix is keyed by it"
        )

    frame = _table_frame(entityset, target, _closure(features), cutoff_time)
    columns = [primary_key]
    for feature in features:
        columns.extend(feature.output_names)
    return frame.select(*dict.fromkeys(columns))


def _closure(features: Sequence[Feature]) -> set[Feature]:
    """Expand features to include every feature they are computed from.

    A requested feature's inputs must exist as columns before it can be
    computed, so the compiler always works over the transitive closure rather
    than the caller's list.

    Args:
        features: Starting features.

    Returns:
        The features plus all of their transitive bases.
    """
    out: set[Feature] = set()
    stack = list(features)
    while stack:
        feature = stack.pop()
        if feature in out:
            continue
        out.add(feature)
        stack.extend(feature.base_features)
    return out


def _base_frame(entityset: EntitySet, table: str, cutoff_time: Any) -> nw.LazyFrame:
    """Return a table's frame with the cutoff filter applied.

    Tables without a ``row_creation_time`` are timeless and pass through.

    Args:
        entityset: The entity set holding the frames.
        table: Table name.
        cutoff_time: The cutoff, or None.

    Returns:
        The filtered lazy frame.
    """
    frame = entityset.frame(table)
    row_creation_time = entityset.schema(table).row_creation_time
    if cutoff_time is not None and row_creation_time is not None:
        frame = frame.filter(nw.col(row_creation_time) <= cutoff_time)
    return frame


def _table_frame(
    entityset: EntitySet,
    table: str,
    needed: set[Feature],
    cutoff_time: Any,
) -> nw.LazyFrame:
    """Build a frame for ``table`` carrying a column for every needed feature.

    Args:
        entityset: The entity set holding the frames.
        table: Table to build.
        needed: Features on this table that must appear as columns.
        cutoff_time: The cutoff, or None.

    Returns:
        A lazy frame with the table's own columns plus the needed features.
    """
    frame = _base_frame(entityset, table, cutoff_time)
    derived = [f for f in needed if not isinstance(f, IdentityFeature)]
    for feature in sorted(derived, key=lambda f: f.depth):
        frame = _apply(frame, feature)
    return frame


def _apply(frame: nw.LazyFrame, feature: Feature) -> nw.LazyFrame:
    """Add a derived feature's columns to a frame.

    Args:
        frame: The frame to extend.
        feature: The feature to compute.

    Returns:
        The extended frame.

    Raises:
        SchemaError: If the feature type is not handled here.
    """
    if isinstance(feature, TransformFeature):
        inputs = [nw.col(b.name) for b in feature.base_features]
        exprs = feature.primitive.outputs(*inputs)
        named = [e.alias(n) for e, n in zip(exprs, feature.output_names, strict=True)]
        return frame.with_columns(*named)
    raise SchemaError(f"cannot compile feature type {type(feature).__name__}")
