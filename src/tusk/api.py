"""The public entry points.

[`deep_feature_synthesis`][tusk.api.deep_feature_synthesis] builds feature definitions
and computes them. [`apply_features`][tusk.api.apply_features] computes existing ones.
[`collect_feature_matrix`][tusk.api.collect_feature_matrix] collects the resulting
feature matrix.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any

import narwhals as nw

from tusk.database import Database
from tusk.exceptions import TuskError
from tusk.feature_list import FeatureList
from tusk.features import Feature
from tusk.primitives.base import Primitive
from tusk.synthesis import synthesize


def deep_feature_synthesis(
    database: Database,
    target_table: str,
    agg_primitives: Iterable[str | Primitive] | None = None,
    trans_primitives: Iterable[str | Primitive] | None = None,
    conditional_primitives: Iterable[str | Primitive] | None = None,
    max_depth: int = 2,
    cutoff_time: datetime | None = None,
    features_only: bool = False,
) -> Any:
    """Run deep feature synthesis over a database.

    Synthesis raises [`SchemaError`][tusk.exceptions.SchemaError] if the target
    table is unknown, or if the walk builds no features at all.

    It raises [`PrimitiveError`][tusk.exceptions.PrimitiveError] in three cases:

    - The primitive name is unknown.
    - An order-dependent primitive lands on a table with no
      `row_creation_time`.
    - A primitive is the wrong kind for the argument it is passed to.

    Compilation raises [`SchemaError`][tusk.exceptions.SchemaError] if the target
    table has no `primary_key`. It also raises whatever
    [`apply_features`][tusk.api.apply_features] documents for `cutoff_time`.

    Args:
        database: The tables and relationships to synthesize over.
        target_table: The table to build features for. The result has one
            row per *visible* row of this table, keyed by its
            `primary_key`. With no `cutoff_time`, that is every row.
            With one, the target table is filtered like any other, so the
            feature matrix may have fewer rows.
        agg_primitives: Aggregation primitives, as names or instances. None
            selects the documented defaults.
        trans_primitives: Transform primitives, as names or instances.
            Synthesis applies each group or ordered transform primitive
            within each foreign-key group. It applies every other transform
            primitive to each row. None selects the defaults.
        conditional_primitives: Aggregation primitives to compute over only
            the rows each declared condition keeps, as names or instances.
            This is a separate list from `agg_primitives`. A primitive
            named here yields only the conditional features, so name it in
            both lists for the unconditional ones too. None selects
            `CONDITIONAL_DEFAULTS`. `()` computes no conditional
            features.
        max_depth: Maximum number of stacked primitive applications.
        cutoff_time: Only rows whose `row_creation_time` is at or before
            this value are visible, on the target table as well as its
            relatives. Its tz awareness must match the database's Datetime
            columns'. A table with no `row_creation_time` is timeless and
            keeps every row. Setting `cutoff_time` changes nothing on a
            database where no table declares `row_creation_time`. None
            disables filtering. A column listed in a table's
            `row_update_times` holds the value it had before the update,
            wherever that update happened after the cutoff time.
            `cutoff_time` is ignored entirely when `features_only` is
            true. [`deep_feature_synthesis`][tusk.api.deep_feature_synthesis] does not
            compute the features in that case. A `cutoff_time` replaces the
            database's own cutoff time, if it has one.
        features_only: Return the feature definitions without computing
            them.

    Returns:
        feature_matrix: The features on the caller's backend, as the
            backend's native lazy table, if the backend supports lazy
            tables. It is not returned when `features_only` is true.
        features (FeatureList): The feature definitions, reusable with
            [`apply`][tusk.feature_list.FeatureList.apply] or
            [`apply_features`][tusk.api.apply_features].

    Warns:
        CategoricalDtypeWarning: If a Categorical or Enum column is skipped
            because a requested primitive requires a string input.
        UnmatchedPrimitiveWarning: If a requested primitive matched no column
            of its input dtypes anywhere in the walk.
        OverwrittenCutoffTimeWarning: If `database` has a cutoff time that
            differs from `cutoff_time`.
    """
    features = synthesize(
        database=database,
        target_table=target_table,
        agg_primitives=agg_primitives,
        trans_primitives=trans_primitives,
        conditional_primitives=conditional_primitives,
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
    accepts any sequence of features. A [`FeatureList`][tusk.feature_list.FeatureList]
    can compute itself with [`apply`][tusk.feature_list.FeatureList.apply] instead.

    It raises [`SchemaError`][tusk.exceptions.SchemaError] if any of these are
    true:

    - `features` is empty.
    - `features` spans more than one table.
    - `features` targets a table with no `primary_key`.

    It raises [`PrimitiveError`][tusk.exceptions.PrimitiveError] if an
    order-dependent primitive lands on a table with no
    `row_creation_time`.

    It raises [`ValidationError`][tusk.exceptions.ValidationError] if any of these are
    true:

    - `cutoff_time` differs from the database's Datetime columns in tz
      awareness.
    - The database's Datetime columns disagree with each other in tz
      awareness.
    - A feature's primitive measures against `cutoff_time` and none was
      given.

    It raises `TypeError` if `cutoff_time` is not a `datetime`.

    Args:
        features: Feature definitions, all on the same target table.
        database: The database to compute over.
        cutoff_time: Only rows whose `row_creation_time` is at or before
            this value are visible, on the target table as well as its
            relatives. The feature matrix may then have fewer rows than
            the target. Its tz awareness must match the database's
            Datetime columns'. A table with no `row_creation_time` is
            timeless and keeps every row. Setting `cutoff_time` changes
            nothing on a database where no table declares
            `row_creation_time`. None disables filtering. A column
            listed in a table's `row_update_times` holds the value it
            had before the update, wherever that update happened after
            the cutoff time. A `cutoff_time` replaces the database's own
            cutoff time, if it has one.

    Returns:
        feature_matrix: The features on the caller's backend, as the
            backend's native lazy table, if the backend supports lazy
            tables, with one row per visible target row.

    Warns:
        OverwrittenCutoffTimeWarning: If `database` has a cutoff time that
            differs from `cutoff_time`.
    """
    return FeatureList(features).apply(database, cutoff_time)


def collect_feature_matrix(
    feature_matrix: Any,
    backend: str | None = None,
    batch_size: int | None = None,
) -> Any:
    """Collect a feature matrix.

    Args:
        feature_matrix: A feature matrix from
            [`deep_feature_synthesis`][tusk.api.deep_feature_synthesis] or
            [`apply_features`][tusk.api.apply_features].
        backend: Backend to collect to. None collects to the feature
            matrix's own backend.
        batch_size: Number of feature columns to collect per query. None
            collects all of them in one query.

    Returns:
        feature_matrix: The collected feature matrix, as the backend's native
            eager table. With `batch_size`, the rows are sorted by the
            primary key.

    Raises:
        TuskError: If `backend` needs a package that is not installed.
    """
    feature_matrix = nw.from_native(feature_matrix)
    try:
        if batch_size is None:
            return feature_matrix.lazy().collect(backend=backend).to_native()
        return _collect_batchwise(feature_matrix, backend, batch_size)
    except ModuleNotFoundError as exc:
        raise TuskError(f"{backend=} is not installed.") from exc


def _collect_batchwise(
    feature_matrix: nw.LazyFrame | nw.DataFrame,
    backend: str | None,
    batch_size: int,
) -> Any:
    """Collect `batch_size` columns at a time, sorted by the primary key."""
    if not isinstance(batch_size, int):
        raise TypeError(f"{batch_size=} is not an integer.")
    primary_key, *feature_columns = feature_matrix.collect_schema().names()
    keys = _collect_columns(feature_matrix, primary_key, (), backend)
    batches = []
    for column_index in range(0, len(feature_columns), batch_size):
        columns = feature_columns[column_index : column_index + batch_size]
        batch = _collect_columns(feature_matrix, primary_key, columns, backend)
        batches.append(batch.drop(primary_key))
    return nw.concat([keys, *batches], how="horizontal").to_native()


def _collect_columns(
    feature_matrix: nw.LazyFrame | nw.DataFrame,
    primary_key: str,
    columns: Sequence[str],
    backend: str | None,
) -> nw.DataFrame:
    """Collect the primary key and `columns`, sorted by the primary key."""
    batch = feature_matrix.select(primary_key, *columns).sort(primary_key)
    return batch.lazy().collect(backend=backend)  # ty: ignore[invalid-argument-type]
