"""Build one transform column in tusk and in featuretools.

Both helpers read the shared ``ROWS`` and ``GROUPS`` tables by default, or any
child table keyed by ``id``, ordered by ``occurred_at`` and related through
``group_id`` to a parent keyed by ``id``. Both return values in id order.
Both sides order the rows by ``occurred_at``: tusk as the row creation time,
featuretools as the time index. A group transform runs within each
``group_id`` group.
"""

import datetime as dt

import pandas as pd
import polars as pl
import pytest
from transform_cases import (
    GROUPS,
    ROWS,
    feature_values,
    rows_database,
)

import tusk
from tusk.exceptions import PrimitiveError
from tusk.primitives.base import GroupTransformPrimitive
from tusk.primitives.registry import resolve

featuretools = pytest.importorskip("featuretools")

LOGICAL_TYPES = {
    "value": "Double",
    "flag": "Boolean",
    "maybe_flag": "BooleanNullable",
    "label": "Categorical",
}


def featuretools_values(
    primitive_name,
    feature_name,
    *,
    cutoff_time=None,
    rows=ROWS,
    groups=GROUPS,
):
    """Run one transform primitive through featuretools and read one column.

    A primitive that tusk runs within foreign-key groups goes in featuretools'
    ``groupby_trans_primitives``, any other in its ``trans_primitives``.

    Args:
        primitive_name: The primitive's featuretools name.
        feature_name: The featuretools feature column to read.
        cutoff_time: Passed through to ``featuretools.dfs``. None disables
            filtering, so every row is visible.
        rows: The child table, related to ``groups`` through ``group_id``.
        groups: The parent table.

    Returns:
        The column's values in id order, missing values as featuretools
        stores them.
    """
    entityset = (
        featuretools.EntitySet("rows")
        .add_dataframe(
            dataframe_name="groups",
            dataframe=groups.copy(),
            index="id",
        )
        .add_dataframe(
            dataframe_name="rows",
            # featuretools' woodwork initialization sorts and re-types its
            # input in place; copy so the table stays untouched for every other test.
            dataframe=rows.copy(),
            index="id",
            time_index="occurred_at",
            logical_types={
                column: logical_type
                for column, logical_type in LOGICAL_TYPES.items()
                if column in rows
            },
        )
        .add_relationship("groups", "id", "rows", "group_id")
    )
    grouped = _runs_within_groups_in_tusk(primitive_name)
    matrix, _ = featuretools.dfs(
        entityset=entityset,
        target_dataframe_name="rows",
        agg_primitives=[],
        trans_primitives=[] if grouped else [primitive_name],
        groupby_trans_primitives=[primitive_name] if grouped else [],
        max_depth=1,
        cutoff_time=cutoff_time,
    )
    return matrix.sort_index()[feature_name].tolist()


def _runs_within_groups_in_tusk(primitive_name):
    """Report whether tusk runs a primitive of this name within foreign-key groups.

    Args:
        primitive_name: The primitive's featuretools name.

    Returns:
        True for a tusk group transform primitive; False for any other, and
        for a name tusk does not register under the same spelling.
    """
    try:
        return isinstance(resolve(primitive_name), GroupTransformPrimitive)
    except PrimitiveError:
        return False


def tusk_values(primitive_name, column, *, cutoff_time=None, rows=ROWS, groups=GROUPS):
    """Run one transform primitive through tusk on polars and read one column.

    Args:
        primitive_name: The primitive's tusk name.
        column: The tusk feature column to read.
        cutoff_time: Passed through to ``tusk.deep_feature_synthesis``. None
            disables filtering, so every row is visible.
        rows: The child table, related to ``groups`` through ``group_id``.
        groups: The parent table.

    Returns:
        The column's values in id order, null as None.
    """
    matrix, _ = tusk.deep_feature_synthesis(
        database=rows_database(
            pl.from_pandas(rows).lazy(),
            pl.from_pandas(groups).lazy(),
        ),
        target_table="rows",
        agg_primitives=[],
        max_depth=1,
        cutoff_time=cutoff_time,
        trans_primitives=[primitive_name],
    )
    return feature_values(matrix, column)


def assert_agree(ours, theirs):
    """Assert that tusk and featuretools give the same value on every row.

    A tusk null agrees with any featuretools missing value (NaN, NaT, NA),
    since featuretools stores a missing float as NaN.

    Args:
        ours: tusk's column in id order.
        theirs: featuretools' column in id order.
    """
    assert len(ours) == len(theirs)
    for our, their in zip(ours, theirs, strict=True):
        if pd.isna(their):
            assert our is None or pd.isna(our)
        elif isinstance(their, dt.timedelta):
            # pytest.approx refuses a timedelta without an explicit tolerance
            assert our == their
        else:
            assert our == pytest.approx(their)
