"""Build one transform column over the shared table in tusk and in featuretools.

Both sides order the rows by ``occurred_at``: tusk as the row creation time,
featuretools as the time index.
"""

import datetime as dt

import pandas as pd
import polars as pl
import pytest
from transform_cases import ROWS, feature_values, rows_database

import tusk

featuretools = pytest.importorskip("featuretools")

LOGICAL_TYPES = {
    "value": "Double",
    "flag": "Boolean",
    "maybe_flag": "BooleanNullable",
    "label": "Categorical",
}


def featuretools_values(primitive_name, feature_name, *, cutoff_time=None):
    """Run one transform primitive through featuretools and read one column.

    Args:
        primitive_name: The primitive's featuretools name.
        feature_name: The featuretools feature column to read.
        cutoff_time: Passed through to ``featuretools.dfs``. None disables
            filtering, so every row is visible.

    Returns:
        The column's values in id order, missing values as featuretools
        stores them.
    """
    entityset = featuretools.EntitySet("rows").add_dataframe(
        dataframe_name="rows",
        # featuretools' woodwork initialization sorts and re-types its input
        # in place; copy so ROWS stays untouched for every other test.
        dataframe=ROWS.copy(),
        index="id",
        time_index="occurred_at",
        logical_types=LOGICAL_TYPES,
    )
    matrix, _ = featuretools.dfs(
        entityset=entityset,
        target_dataframe_name="rows",
        agg_primitives=[],
        trans_primitives=[primitive_name],
        max_depth=1,
        cutoff_time=cutoff_time,
    )
    return matrix.sort_index()[feature_name].tolist()


def tusk_values(primitive_name, column, *, cutoff_time=None):
    """Run one transform primitive through tusk on polars and read one column.

    Args:
        primitive_name: The primitive's tusk name.
        column: The tusk feature column to read.
        cutoff_time: Passed through to ``tusk.deep_feature_synthesis``. None
            disables filtering, so every row is visible.

    Returns:
        The column's values in id order, null as None.
    """
    matrix, _ = tusk.deep_feature_synthesis(
        database=rows_database(pl.from_pandas(ROWS).lazy()),
        target_table="rows",
        agg_primitives=[],
        trans_primitives=[primitive_name],
        max_depth=1,
        cutoff_time=cutoff_time,
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
