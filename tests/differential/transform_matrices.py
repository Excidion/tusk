"""Build one transform column in tusk and in featuretools.

Both helpers read the shared ``ROWS`` and ``GROUPS`` tables by default, or any
child table ordered by ``occurred_at`` and related to its parent through
``group_id``. Both sides order the rows by ``occurred_at``: tusk as the row
creation time, featuretools as the time index. A group transform runs within
each ``group_id`` group.
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
    transform_arguments,
)

import tusk

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
    grouped=False,
    cutoff_time=None,
    rows=ROWS,
    groups=GROUPS,
):
    """Run one transform primitive through featuretools and read one column.

    Args:
        primitive_name: The primitive's featuretools name.
        feature_name: The featuretools feature column to read.
        grouped: Pass the primitive in ``groupby_trans_primitives`` rather
            than ``trans_primitives``.
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


def tusk_values(primitive_name, column, *, cutoff_time=None, rows=ROWS, groups=GROUPS):
    """Run one transform primitive through tusk on polars and read one column.

    A group transform primitive runs in ``groupby_trans_primitives``, any
    other in ``trans_primitives``.

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
        **transform_arguments(primitive_name),
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
