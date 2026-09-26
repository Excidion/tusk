"""Phase 2: the single lazy query plan built from feature definitions.

Nothing here materializes a table, and neither does the caller. The only
``collect()`` calls in tusk are the ones :mod:`tusk.validation` makes when a
check is explicitly requested.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING

import narwhals as nw

from tusk.database import Database, Relationship, TableSchema
from tusk.exceptions import PrimitiveError, SchemaError, ValidationError
from tusk.features import (
    AggregationFeature,
    DirectFeature,
    Feature,
    GroupByTransformFeature,
    IdentityFeature,
    TransformFeature,
)
from tusk.primitives.base import (
    GroupRelativeAggregationPrimitive,
    NeedsCutoffTime,
    OrderedAggregationPrimitive,
    OrderedTransformPrimitive,
    Primitive,
    ValueCountAggregationPrimitive,
)

if TYPE_CHECKING:
    from tusk.feature_list import FeatureList


def compile_features(
    features: FeatureList,
    database: Database,
    cutoff_time: datetime | None = None,
) -> nw.LazyFrame:
    """Compute a lazy feature matrix from feature definitions.

    It also raises :class:`~tusk.exceptions.ValidationError`, from
    :func:`_require_cutoff_time`, if a primitive measures against
    ``cutoff_time`` and none was given.

    Args:
        features: Features to compute.
        database: The database holding the tables.
        cutoff_time: Only rows whose ``row_creation_time`` is at or before this
            value are visible. None disables filtering.

    Returns:
        feature_matrix: A narwhals LazyFrame with the target's primary key
            plus one column per feature output, and one row per target row
            visible at ``cutoff_time``. This may have fewer rows than the
            target table holds, because the target is filtered like any
            other table.

    Raises:
        SchemaError: If the target table has no primary key.
    """
    target = features.target_table
    primary_key = database.get_schema(target).primary_key
    if primary_key is None:
        raise SchemaError(
            f"target table {target!r} needs a primary_key: the feature "
            "matrix is keyed by it",
        )

    _reject_colliding_names(features)
    closure = _closure(features)
    _require_cutoff_time(closure, cutoff_time)

    table = _read_table(database, target, closure, cutoff_time)
    columns = [primary_key, *features.output_names]
    return table.select(*dict.fromkeys(columns))


def _reject_colliding_names(features: Sequence[Feature]) -> None:
    """Fail if two distinct features want the same column.

    Column names join their parts with ``__``. A source column already
    containing ``__`` can collide with a built name. For example,
    ``MEAN(a.b)`` and ``MEAN(a__b)`` both want ``MEAN__a__b``. Silently
    keeping one and dropping the other would put the wrong values under a
    plausible-looking name, so this function raises an error instead.

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
    computed. So the compiler always works over the transitive closure, not
    only the caller's list.

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


def _require_cutoff_time(features: set[Feature], cutoff_time: datetime | None) -> None:
    """Fail if a primitive or `when` condition needs a cutoff time that was not given.

    Args:
        features: The transitive closure of features to compile.
        cutoff_time: The cutoff time, or None.

    Raises:
        ValidationError: If a primitive or `when` condition measures against
            ``cutoff_time`` and none was given.
    """
    if cutoff_time is not None:
        return
    measuring: set[str] = set()
    for feature in features:
        measuring.update(_names_measuring_against_cutoff_time(feature))
    if measuring:
        raise ValidationError(
            f"{', '.join(sorted(measuring))} needs a cutoff_time; pass one "
            "when applying the features",
        )


def _names_measuring_against_cutoff_time(feature: Feature) -> tuple[str, ...]:
    """Name whatever in a feature is measured against the cutoff time.

    Args:
        feature: The feature to inspect.

    Returns:
        The primitive's name when it measures against the cutoff time, the
        condition's name when it is a ``when`` condition, or both.
    """
    names = []
    primitive = getattr(feature, "primitive", None)
    if isinstance(primitive, NeedsCutoffTime):
        names.append(primitive.name)
    condition = getattr(feature, "condition", None)
    if condition is not None and condition[0] == "when":
        names.append(f"when condition {condition[1]!r}")
    return tuple(names)


def base_table(
    database: Database,
    table_name: str,
    cutoff_time: datetime | None,
) -> nw.LazyFrame:
    """Return a table's narwhals LazyFrame as it stood at the cutoff time.

    Rows created after the cutoff time are dropped. Columns updated after
    the cutoff time hold the value they held before. A table with no
    ``row_creation_time`` keeps every row, and a table declaring no
    ``row_update_times`` keeps every value.

    This function filters the target table the same way as any other table.
    So a cutoff time can leave the feature matrix with fewer rows than the
    target table holds.

    Args:
        database: The database holding the tables.
        table_name: Table name.
        cutoff_time: The cutoff time, or None.

    Returns:
        The table's narwhals LazyFrame as it stood at the cutoff time.
    """
    table = database.get_table(table_name)
    if cutoff_time is None:
        return table

    schema = database.get_schema(table_name)
    if schema.row_creation_time is not None:
        table = table.filter(nw.col(schema.row_creation_time) <= cutoff_time)
    return _restore_updated_columns(table, schema, cutoff_time)


def _restore_updated_columns(
    table: nw.LazyFrame,
    schema: TableSchema,
    cutoff_time: datetime,
) -> nw.LazyFrame:
    """Give every column updated after the cutoff time the value it held before.

    Args:
        table: The table, already filtered to the cutoff time.
        schema: The table's schema, naming the updates.
        cutoff_time: The cutoff time.

    Returns:
        The table, with one replaced column per declared update.
    """
    if not schema.column_updates:
        return table

    return table.with_columns(
        nw.when(_was_updated_by(update_time, cutoff_time))
        .then(nw.col(column))
        .otherwise(nw.lit(value, dtype=schema.dtypes[column]))
        .alias(column)
        for update_time, column, value in schema.column_updates
    )


def _was_updated_by(update_time: str, cutoff_time: datetime) -> nw.Expr:
    """Build the test for a row's update having already happened.

    A null update time counts as never updated.

    Args:
        update_time: Column recording when the row was updated.
        cutoff_time: The cutoff time.

    Returns:
        A boolean expression, true where the update has already happened.
    """
    updated = nw.col(update_time)
    return updated.is_null() | (updated <= cutoff_time)


def _read_table(
    database: Database,
    table_name: str,
    needed: set[Feature],
    cutoff_time: datetime | None,
) -> nw.LazyFrame:
    """Build a table for ``table_name`` carrying a column for every needed feature.

    This function joins aggregations onto the table first, batched by
    relationship. This keeps join count independent of feature count. It
    then applies row-wise features in depth order, so each one's inputs
    already exist as columns.

    Args:
        database: The database holding the tables.
        table_name: Table to build.
        needed: Features on this table that must appear as columns.
        cutoff_time: The cutoff time, or None.

    Returns:
        table: A narwhals LazyFrame with the table's own columns plus the
            needed features.

    Raises:
        SchemaError: If ``needed`` contains a feature type this compiler does
            not know how to compute.
    """
    table = base_table(database, table_name, cutoff_time)
    needed = {f for f in needed if f.table == table_name}

    aggregations = [f for f in needed if isinstance(f, AggregationFeature)]
    for relationship in dict.fromkeys(f.relationship for f in aggregations):
        batch = [f for f in aggregations if f.relationship == relationship]
        table = _add_aggregations(
            table,
            database,
            table_name,
            relationship,
            batch,
            cutoff_time,
        )

    directs = [f for f in needed if isinstance(f, DirectFeature)]
    for relationship in dict.fromkeys(f.relationship for f in directs):
        batch = [f for f in directs if f.relationship == relationship]
        table = _add_directs(table, database, relationship, batch, cutoff_time)

    row_wise = [
        f for f in needed if isinstance(f, (TransformFeature, GroupByTransformFeature))
    ]
    for feature in sorted(row_wise, key=lambda f: f.depth):
        table = _apply(table, feature, database, cutoff_time)

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
    return table


def _add_aggregations(
    table: nw.LazyFrame,
    database: Database,
    table_name: str,
    relationship: Relationship,
    batch: Sequence[AggregationFeature],
    cutoff_time: datetime | None,
) -> nw.LazyFrame:
    """Fold one child table's aggregations into the parent, one join per condition.

    Unconditioned features share a single join over the unfiltered child.
    Each distinct condition adds one further join over the child filtered to
    that condition's mask.

    Args:
        table: The parent table being built.
        database: The database holding the tables.
        table_name: The parent table's name.
        relationship: The relationship being aggregated across.
        batch: Every aggregation feature using that relationship.
        cutoff_time: The cutoff time, or None.

    Returns:
        The parent table with the batch's columns joined on.
    """
    child_needed: set[Feature] = set()
    for feature in batch:
        child_needed.update(_closure(feature.base_features))
    child = _read_table(database, relationship.child, child_needed, cutoff_time)

    child_schema = database.get_schema(relationship.child)
    for condition, features in _group_by_condition(batch):
        mask = _build_condition_mask(child_schema, condition, cutoff_time)
        table = _join_condition_aggregations(
            table,
            child if mask is None else child.filter(mask),
            database,
            table_name,
            relationship,
            features,
            cutoff_time,
        )
    return table


def _build_condition_mask(
    schema: TableSchema,
    condition: tuple[str, str] | None,
    cutoff_time: datetime | None,
) -> nw.Expr | None:
    """Build the mask a condition selects on the child's rows.

    Args:
        schema: The child table's schema, holding the declared conditions.
        condition: The (kind, key) pair, or None for an unconditioned feature.
        cutoff_time: The cutoff time, passed to a ``when`` condition's callable.

    Returns:
        The mask expression, or None when the feature has no condition.

    Raises:
        SchemaError: If the key is not declared on the child table.
    """
    if condition is None:
        return None

    kind, key = condition
    declared = schema.where if kind == "where" else schema.when
    if key not in declared:
        raise SchemaError(
            f"{kind} condition {key!r} is not declared on table {schema.name!r}; "
            f"declare it in add_table({kind}=...) or drop the feature",
        )
    return declared[key] if kind == "where" else declared[key](cutoff_time)


def _group_by_condition(
    batch: Sequence[AggregationFeature],
) -> list[tuple[tuple[str, str] | None, list[AggregationFeature]]]:
    """Group a relationship's aggregations by the condition masking them.

    Args:
        batch: Every aggregation feature using one relationship.

    Returns:
        One (condition, features) pair per distinct condition, unconditioned
        first.
    """
    grouped: dict[tuple[str, str] | None, list[AggregationFeature]] = {}
    for feature in batch:
        grouped.setdefault(feature.condition, []).append(feature)
    return sorted(grouped.items(), key=lambda item: item[0] is not None)


def _join_condition_aggregations(
    table: nw.LazyFrame,
    child: nw.LazyFrame,
    database: Database,
    table_name: str,
    relationship: Relationship,
    batch: Sequence[AggregationFeature],
    cutoff_time: datetime | None,
) -> nw.LazyFrame:
    """Fold a condition's aggregations into the parent with a single join.

    Args:
        table: The parent table being built.
        child: The child table, after the condition filter.
        database: The database holding the schemas.
        table_name: The parent table's name.
        relationship: The relationship being aggregated across.
        batch: The aggregation features sharing this condition.
        cutoff_time: The cutoff time, or None.

    Returns:
        The parent table with this condition's columns joined on.
    """
    exprs = []
    for feature in batch:
        built = _build_aggregation(feature, database, relationship, cutoff_time)
        exprs.extend(
            e.alias(n) for e, n in zip(built, feature.output_names, strict=True)
        )

    child = _add_value_count_columns(child, relationship, batch)
    child = _add_comparison_columns(child, relationship, batch)
    grouped = child.group_by(relationship.foreign_key).agg(*exprs)
    table = table.join(
        grouped,
        left_on=database.get_schema(table_name).primary_key,
        right_on=relationship.foreign_key,
        how="left",
    )

    defaults = [
        nw.col(name).fill_null(feature.primitive.default_value).alias(name)
        for feature in batch
        if feature.primitive.default_value is not None
        for name in feature.output_names
    ]
    return table.with_columns(*defaults) if defaults else table


def _build_aggregation(
    feature: AggregationFeature,
    database: Database,
    relationship: Relationship,
    cutoff_time: datetime | None,
) -> tuple[nw.Expr, ...]:
    """Build an aggregation feature's output expressions over the child's columns.

    Args:
        feature: The aggregation feature.
        database: The database, used to find ordering columns.
        relationship: The relationship being aggregated across.
        cutoff_time: The cutoff time, or None.

    Returns:
        One expression per output column.
    """
    primitive = feature.primitive
    if isinstance(primitive, GroupRelativeAggregationPrimitive):
        return primitive.outputs(nw.col(_build_comparison_column_name(feature)))
    inputs = [nw.col(b.name) for b in feature.base_features]
    if isinstance(primitive, OrderedAggregationPrimitive):
        order_by = _order_by(database, relationship.child, primitive.name)
        return primitive.outputs(*inputs, order_by=order_by)
    return _build_expressions(primitive, inputs, cutoff_time)


def _add_value_count_columns(
    child: nw.LazyFrame,
    relationship: Relationship,
    batch: Sequence[AggregationFeature],
) -> nw.LazyFrame:
    """Add the count column that each value-count aggregation reads.

    Args:
        child: The child table, after the condition filter.
        relationship: The relationship being aggregated across.
        batch: The aggregation features sharing this condition.

    Returns:
        The child table with one column per value-count feature.
    """
    counts = [
        _build_value_count_column(feature, relationship)
        for feature in batch
        if isinstance(feature.primitive, ValueCountAggregationPrimitive)
    ]
    return child.with_columns(*counts) if counts else child


def _build_value_count_column(
    feature: AggregationFeature,
    relationship: Relationship,
) -> nw.Expr:
    """Build the count of how often each row's value occurs in its group.

    Args:
        feature: The value-count aggregation feature.
        relationship: The relationship whose foreign key forms the groups.

    Returns:
        The named count expression. It is null where the value is null.
    """
    value = feature.base_features[0].name
    count = nw.len().over(relationship.foreign_key, value)
    return (
        nw.when(~nw.col(value).is_null())
        .then(count)
        .alias(_build_value_count_column_name(feature))
    )


def _add_comparison_columns(
    child: nw.LazyFrame,
    relationship: Relationship,
    batch: Sequence[AggregationFeature],
) -> nw.LazyFrame:
    """Add the comparison column that each group-relative aggregation reduces.

    Args:
        child: The child table, after the condition filter.
        relationship: The relationship being aggregated across.
        batch: The aggregation features sharing this condition.

    Returns:
        The child table with one column per group-relative feature.
    """
    comparisons = [
        _build_comparison_column(feature, feature.primitive, relationship)
        for feature in batch
        if isinstance(feature.primitive, GroupRelativeAggregationPrimitive)
    ]
    return child.with_columns(*comparisons) if comparisons else child


def _build_comparison_column(
    feature: AggregationFeature,
    primitive: GroupRelativeAggregationPrimitive,
    relationship: Relationship,
) -> nw.Expr:
    """Build a group-relative aggregation's comparison column within each group.

    Args:
        feature: The group-relative aggregation feature.
        primitive: The feature's primitive.
        relationship: The relationship whose foreign key forms the groups.

    Returns:
        The named comparison expression.
    """
    inputs = _select_comparison_inputs(feature)
    comparison = primitive.compare_with_group(*inputs)
    return comparison.over(relationship.foreign_key).alias(
        _build_comparison_column_name(feature),
    )


def _select_comparison_inputs(feature: AggregationFeature) -> list[nw.Expr]:
    """Select the columns that a group-relative aggregation's comparison reads.

    Args:
        feature: The group-relative aggregation feature.

    Returns:
        The feature's input columns, then its count column if it is a
        value-count aggregation.
    """
    inputs = [nw.col(b.name) for b in feature.base_features]
    if isinstance(feature.primitive, ValueCountAggregationPrimitive):
        inputs.append(nw.col(_build_value_count_column_name(feature)))
    return inputs


def _build_comparison_column_name(feature: AggregationFeature) -> str:
    """Name the child column a group-relative aggregation reduces.

    Args:
        feature: The group-relative aggregation feature.

    Returns:
        The column name.
    """
    return f"{feature.name}__comparison"


def _build_value_count_column_name(feature: AggregationFeature) -> str:
    """Name the child column that holds a value-count aggregation's counts.

    Args:
        feature: The value-count aggregation feature.

    Returns:
        The column name.
    """
    return f"{feature.name}__value_count"


def _add_directs(
    table: nw.LazyFrame,
    database: Database,
    relationship: Relationship,
    batch: Sequence[DirectFeature],
    cutoff_time: datetime | None,
) -> nw.LazyFrame:
    """Join one parent table's features onto the child with a single join.

    Args:
        table: The child table being built.
        database: The database holding the tables.
        relationship: The relationship being traversed.
        batch: Every direct feature using that relationship.
        cutoff_time: The cutoff time, or None.

    Returns:
        The child table with the batch's columns joined on.

    Raises:
        SchemaError: If the parent table has no primary key.
    """
    parent_key = database.get_schema(relationship.parent).primary_key
    if parent_key is None:
        raise SchemaError(f"parent table {relationship.parent!r} needs a primary_key")
    parent_needed: set[Feature] = set()
    for feature in batch:
        parent_needed.update(_closure(feature.base_features))
    parent = _read_table(database, relationship.parent, parent_needed, cutoff_time)

    selected = [nw.col(parent_key)]
    for feature in batch:
        selected.append(nw.col(feature.base_feature.name).alias(feature.name))

    return table.join(
        parent.select(*selected),
        left_on=relationship.foreign_key,
        right_on=parent_key,
        how="left",
    )


def _apply(
    table: nw.LazyFrame,
    feature: Feature,
    database: Database,
    cutoff_time: datetime | None,
) -> nw.LazyFrame:
    """Add a row-wise feature's columns to a table.

    This function wraps a groupby transform in ``.over(foreign_key)``. It
    wraps an ordered transform primitive in ``.over(foreign_key,
    order_by=...)``. Neither relies on a table-level sort. On lazy backends,
    a sort is not guaranteed to survive later operations. Narwhals also
    requires ``order_by`` for these expressions in any case.

    Args:
        table: The table to extend.
        feature: The feature to compute.
        database: The database, used to find ordering columns.
        cutoff_time: The cutoff time, passed as a keyword argument to a
            :class:`NeedsCutoffTime` primitive's ``outputs()`` when it builds
            its expression.

    Returns:
        The extended table.

    Raises:
        SchemaError: If the feature type is not handled here.
    """
    if not isinstance(feature, (TransformFeature, GroupByTransformFeature)):
        raise SchemaError(f"cannot compile feature type {type(feature).__name__}")

    inputs = [nw.col(b.name) for b in feature.base_features]
    exprs = list(_build_expressions(feature.primitive, inputs, cutoff_time))

    partition = (
        [feature.relationship.foreign_key]
        if isinstance(feature, GroupByTransformFeature)
        else []
    )
    if isinstance(feature.primitive, OrderedTransformPrimitive):
        order_by = _order_by(database, feature.table, feature.primitive.name)
        exprs = [e.over(*partition, order_by=order_by) for e in exprs]
    elif partition:
        exprs = [e.over(*partition) for e in exprs]

    named = [e.alias(n) for e, n in zip(exprs, feature.output_names, strict=True)]
    return table.with_columns(*named)


def _build_expressions(
    primitive: Primitive,
    inputs: Sequence[nw.Expr],
    cutoff_time: datetime | None,
) -> tuple[nw.Expr, ...]:
    """Build a primitive's output expressions, threading the cutoff time when needed.

    Args:
        primitive: The primitive to build.
        inputs: One expression per declared input.
        cutoff_time: The cutoff time, passed as a keyword argument to a
            :class:`NeedsCutoffTime` primitive's ``outputs()`` when it builds
            its expression.

    Returns:
        One expression per output column.
    """
    if isinstance(primitive, NeedsCutoffTime) and cutoff_time is not None:
        return primitive.outputs(*inputs, cutoff_time=cutoff_time)
    return primitive.outputs(*inputs)


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
    schema = database.get_schema(table)
    if schema.row_creation_time is None:
        raise PrimitiveError(
            f"primitive {primitive_name!r} is order-dependent, so table {table!r} "
            f"needs a row_creation_time",
        )
    if schema.primary_key is None:
        return (schema.row_creation_time,)
    return (schema.row_creation_time, schema.primary_key)
