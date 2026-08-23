"""Phase 2: turn feature definitions into a single lazy query plan.

Nothing here materializes a frame, and neither does the caller: the only
``collect()`` calls in tusk are the ones :mod:`tusk.validation` makes when a
check is explicitly requested.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import narwhals as nw

from tusk.database import Database, Relationship
from tusk.exceptions import PrimitiveError, SchemaError
from tusk.features import (
    AggregationFeature,
    DirectFeature,
    Feature,
    GroupByTransformFeature,
    IdentityFeature,
    TransformFeature,
)


def compile_features(
    features: Sequence[Feature],
    database: Database,
    cutoff_time: Any = None,
) -> nw.LazyFrame:
    """Compile feature definitions into a lazy feature matrix.

    Args:
        features: Features to compute. All must be on the same table.
        database: The database holding the frames.
        cutoff_time: Only rows whose ``row_creation_time`` is at or before this
            value are visible. None disables filtering.

    Returns:
        A lazy frame with the target's primary key plus one column per feature
        output, and one row per target row visible at ``cutoff_time`` -- which
        may be fewer rows than the target table holds, since the target is
        filtered like any other table. Building it calls ``_apply``, whose
        ``_order_by`` helper raises
        :class:`~tusk.exceptions.PrimitiveError` if an order-dependent
        primitive among ``features`` is applied to a table with no
        ``row_creation_time``.

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
    primary_key = database.schema(target).primary_key
    if primary_key is None:
        raise SchemaError(
            f"target table {target!r} needs a primary_key: the feature "
            "matrix is keyed by it",
        )

    _reject_colliding_names(features)

    frame = _table_frame(database, target, _closure(features), cutoff_time)
    columns = [primary_key]
    for feature in features:
        columns.extend(feature.output_names)
    return frame.select(*dict.fromkeys(columns))


def _reject_colliding_names(features: Sequence[Feature]) -> None:
    """Fail if two distinct features want the same column.

    Column names join their parts with ``__``, so a source column already
    containing ``__`` can in principle collide with a generated name -- e.g.
    ``MEAN(a.b)`` and ``MEAN(a__b)`` both want ``MEAN__a__b``. Silently
    keeping one and dropping the other would put the wrong values under a
    plausible-looking name, so it is refused instead.

    Args:
        features: The features to compile.

    Raises:
        SchemaError: If two distinct features share an output name.
    """
    owner: dict[str, Feature] = {}
    for feature in features:
        for name in feature.output_names:
            other = owner.setdefault(name, feature)
            if other != feature:
                raise SchemaError(
                    f"features {other.display_name!r} and "
                    f"{feature.display_name!r} both compile to column "
                    f"{name!r}; rename a source column to break the tie",
                )


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


def _base_frame(database: Database, table: str, cutoff_time: Any) -> nw.LazyFrame:
    """Return a table's frame with the cutoff filter applied.

    Tables without a ``row_creation_time`` are timeless and pass through
    unfiltered -- documented rather than warned, per spec section 8. A
    database that declares no ``row_creation_time`` anywhere therefore
    treats a cutoff as a silent no-op.

    The target table is filtered like any other, so a cutoff can leave the
    feature matrix with fewer rows than the target table has. That matches
    featuretools: a row that did not exist yet at the cutoff has no features to
    compute.

    Args:
        database: The database holding the frames.
        table: Table name.
        cutoff_time: The cutoff, or None.

    Returns:
        The filtered lazy frame.
    """
    frame = database.frame(table)
    row_creation_time = database.schema(table).row_creation_time
    if cutoff_time is not None and row_creation_time is not None:
        frame = frame.filter(nw.col(row_creation_time) <= cutoff_time)
    return frame


def _table_frame(
    database: Database,
    table: str,
    needed: set[Feature],
    cutoff_time: Any,
) -> nw.LazyFrame:
    """Build a frame for ``table`` carrying a column for every needed feature.

    Aggregations are folded in first, batched by relationship so that feature
    count does not drive join count; row-wise features are then applied in
    depth order, so each one's inputs already exist as columns.

    Args:
        database: The database holding the frames.
        table: Table to build.
        needed: Features on this table that must appear as columns.
        cutoff_time: The cutoff, or None.

    Returns:
        A lazy frame with the table's own columns plus the needed features.
        ``_apply``'s ``_order_by`` helper raises
        :class:`~tusk.exceptions.PrimitiveError` if ``needed`` contains an
        order-dependent primitive applied to a table with no
        ``row_creation_time``.

    Raises:
        SchemaError: If ``needed`` contains a feature type this compiler does
            not know how to compute.
    """
    frame = _base_frame(database, table, cutoff_time)
    needed = {f for f in needed if f.table == table}

    aggregations = [f for f in needed if isinstance(f, AggregationFeature)]
    for relationship in dict.fromkeys(f.relationship for f in aggregations):
        batch = [f for f in aggregations if f.relationship == relationship]
        frame = _add_aggregations(
            frame,
            database,
            table,
            relationship,
            batch,
            cutoff_time,
        )

    directs = [f for f in needed if isinstance(f, DirectFeature)]
    for relationship in dict.fromkeys(f.relationship for f in directs):
        batch = [f for f in directs if f.relationship == relationship]
        frame = _add_directs(frame, database, relationship, batch, cutoff_time)

    row_wise = [
        f for f in needed if isinstance(f, (TransformFeature, GroupByTransformFeature))
    ]
    for feature in sorted(row_wise, key=lambda f: f.depth):
        frame = _apply(frame, feature, database)

    handled = (
        IdentityFeature,
        AggregationFeature,
        DirectFeature,
        TransformFeature,
        GroupByTransformFeature,
    )
    unhandled = [f for f in needed if not isinstance(f, handled)]
    if unhandled:
        raise SchemaError(f"cannot compile feature type {type(unhandled[0]).__name__}")
    return frame


def _add_aggregations(
    frame: nw.LazyFrame,
    database: Database,
    table: str,
    relationship: Relationship,
    batch: Sequence[AggregationFeature],
    cutoff_time: Any,
) -> nw.LazyFrame:
    """Fold one child table's aggregations into the parent with a single join.

    Args:
        frame: The parent frame being built.
        database: The database holding the frames.
        table: The parent table's name.
        relationship: The relationship being aggregated across.
        batch: Every aggregation feature using that relationship.
        cutoff_time: The cutoff, or None.

    Returns:
        The parent frame with the batch's columns joined on.
    """
    child_needed: set[Feature] = set()
    for feature in batch:
        child_needed.update(_closure(feature.base_features))
    child = _table_frame(database, relationship.child, child_needed, cutoff_time)

    exprs = []
    for feature in batch:
        inputs = [nw.col(b.name) for b in feature.base_features]
        built = feature.primitive.outputs(*inputs)
        exprs.extend(
            e.alias(n) for e, n in zip(built, feature.output_names, strict=True)
        )

    grouped = child.group_by(relationship.foreign_key).agg(*exprs)
    frame = frame.join(
        grouped,
        left_on=database.schema(table).primary_key,
        right_on=relationship.foreign_key,
        how="left",
    )

    defaults = [
        nw.col(name).fill_null(feature.primitive.default_value).alias(name)
        for feature in batch
        if feature.primitive.default_value is not None
        for name in feature.output_names
    ]
    return frame.with_columns(*defaults) if defaults else frame


def _add_directs(
    frame: nw.LazyFrame,
    database: Database,
    relationship: Relationship,
    batch: Sequence[DirectFeature],
    cutoff_time: Any,
) -> nw.LazyFrame:
    """Join one parent table's features down onto the child with a single join.

    Args:
        frame: The child frame being built.
        database: The database holding the frames.
        relationship: The relationship being traversed.
        batch: Every direct feature using that relationship.
        cutoff_time: The cutoff, or None.

    Returns:
        The child frame with the batch's columns joined on.

    Raises:
        SchemaError: If the parent table has no primary key.
    """
    parent_key = database.schema(relationship.parent).primary_key
    if parent_key is None:
        raise SchemaError(f"parent table {relationship.parent!r} needs a primary_key")
    parent_needed: set[Feature] = set()
    for feature in batch:
        parent_needed.update(_closure(feature.base_features))
    parent = _table_frame(database, relationship.parent, parent_needed, cutoff_time)

    selected = [nw.col(parent_key)]
    for feature in batch:
        selected.append(nw.col(feature.base_feature.name).alias(feature.name))

    return frame.join(
        parent.select(*selected),
        left_on=relationship.foreign_key,
        right_on=parent_key,
        how="left",
    )


def _apply(frame: nw.LazyFrame, feature: Feature, database: Database) -> nw.LazyFrame:
    """Add a row-wise feature's columns to a frame.

    Order-dependent primitives are wrapped in ``.over(..., order_by=...)``
    rather than relying on a frame-level sort: on lazy backends a sort is not
    guaranteed to survive later operations, and narwhals requires ``order_by``
    for these expressions in any case.

    Args:
        frame: The frame to extend.
        feature: The feature to compute.
        database: The database, used to find ordering columns.

    Returns:
        The extended frame.

    Raises:
        SchemaError: If the feature type is not handled here.
    """
    if not isinstance(feature, (TransformFeature, GroupByTransformFeature)):
        raise SchemaError(f"cannot compile feature type {type(feature).__name__}")

    inputs = [nw.col(b.name) for b in feature.base_features]
    exprs = list(feature.primitive.outputs(*inputs))

    partition = (
        [feature.relationship.foreign_key]
        if isinstance(feature, GroupByTransformFeature)
        else []
    )
    if getattr(feature.primitive, "order_dependent", False):
        order_by = _order_by(database, feature.table, feature.primitive.name)
        exprs = [e.over(*partition, order_by=order_by) for e in exprs]
    elif partition:
        exprs = [e.over(*partition) for e in exprs]

    named = [e.alias(n) for e, n in zip(exprs, feature.output_names, strict=True)]
    return frame.with_columns(*named)


def _order_by(database: Database, table: str, primitive_name: str) -> tuple[str, ...]:
    """Build the ordering key for an order-dependent expression.

    Args:
        database: The database holding the schemas.
        table: The table being ordered.
        primitive_name: Used in the error message.

    Returns:
        The row creation time, followed by the primary key when one exists.

    Raises:
        PrimitiveError: If the table has no ``row_creation_time``.
    """
    schema = database.schema(table)
    if schema.row_creation_time is None:
        raise PrimitiveError(
            f"primitive {primitive_name!r} is order-dependent, so table {table!r} "
            f"needs a row_creation_time",
        )
    if schema.primary_key is None:
        return (schema.row_creation_time,)
    return (schema.row_creation_time, schema.primary_key)
