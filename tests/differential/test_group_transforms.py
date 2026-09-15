"""Cross-check that group and ordered transforms stay within their group.

The shared table in ``tests/transform_cases.py`` puts every row in one group,
so it cannot tell a grouped result from a whole-table one. Here two groups
interleave in ``occurred_at``, so a result that ignored ``group_id`` would
differ from featuretools' ``groupby_trans_primitives``.

Run with: uv run --group validation pytest -m differential

Verified against featuretools 1.31.0.
"""

import datetime as dt

import pandas as pd
import polars as pl
import pytest
from transform_cases import feature_values

import tusk
from differential.transform_matrices import assert_agree

featuretools = pytest.importorskip("featuretools")

pytestmark = pytest.mark.differential

GROUPS = pd.DataFrame({"id": [1, 2]})

INTERLEAVED_ROWS = pd.DataFrame(
    {
        "id": [1, 2, 3, 4, 5, 6],
        "group_id": [1, 2, 1, 2, 1, 2],
        "occurred_at": pd.to_datetime(
            [dt.datetime(2024, 1, day) for day in (1, 2, 3, 4, 5, 6)],
        ),
        "value": [3.0, 10.0, None, 10.0, 1.0, 5.0],
    },
)

CUTOFF = dt.datetime(2024, 1, 4)


@pytest.mark.parametrize(
    ("name", "cutoff_time", "expected"),
    [
        ("percentile", None, [1.0, 5 / 6, None, 5 / 6, 0.5, 1 / 3]),
        ("cum_sum", None, [3.0, 10.0, None, 20.0, 4.0, 25.0]),
        ("percentile", CUTOFF, [1.0, 0.75, None, 0.75]),
        ("cum_sum", CUTOFF, [3.0, 10.0, None, 20.0]),
    ],
)
def test_group_transforms_stay_within_their_group(name, cutoff_time, expected):
    """Group 1 holds 3, null, 1 and group 2 holds 10, 10, 5, in that order.

    Over the whole table ``cum_sum`` of id 2 would be 13 and ``percentile``
    of id 1 would be 0.4, so ``expected`` pins the grouped answer.
    """
    ours = _tusk_values(name, cutoff_time)
    theirs = _featuretools_values(name, cutoff_time)
    assert_agree(ours, expected)
    assert_agree(ours, theirs)


def _tusk_values(name, cutoff_time):
    """Run one group transform through tusk on polars.

    Args:
        name: The primitive's name.
        cutoff_time: Passed through to ``tusk.deep_feature_synthesis``.

    Returns:
        The grouped column's values in id order, null as None.
    """
    database = (
        tusk.Database("interleaved")
        .add_table("groups", pl.from_pandas(GROUPS).lazy(), primary_key="id")
        .add_table(
            "rows",
            pl.from_pandas(INTERLEAVED_ROWS).lazy(),
            primary_key="id",
            row_creation_time="occurred_at",
        )
        .add_relationship(parent="groups", child="rows", foreign_key="group_id")
    )
    matrix, _ = tusk.deep_feature_synthesis(
        database=database,
        target_table="rows",
        agg_primitives=[],
        trans_primitives=[],
        groupby_trans_primitives=[name],
        max_depth=1,
        cutoff_time=cutoff_time,
    )
    return feature_values(matrix, f"{name.upper()}__value__by__group_id")


def _featuretools_values(name, cutoff_time):
    """Run one group transform through featuretools' groupby_trans_primitives.

    Args:
        name: The primitive's name.
        cutoff_time: Passed through to ``featuretools.dfs``.

    Returns:
        The grouped column's values in id order, missing values as NaN.
    """
    entityset = (
        featuretools.EntitySet("interleaved")
        .add_dataframe(dataframe_name="groups", dataframe=GROUPS.copy(), index="id")
        .add_dataframe(
            dataframe_name="rows",
            dataframe=INTERLEAVED_ROWS.copy(),
            index="id",
            time_index="occurred_at",
            logical_types={"value": "Double"},
        )
        .add_relationship("groups", "id", "rows", "group_id")
    )
    matrix, _ = featuretools.dfs(
        entityset=entityset,
        target_dataframe_name="rows",
        agg_primitives=[],
        trans_primitives=[],
        groupby_trans_primitives=[name],
        max_depth=1,
        cutoff_time=cutoff_time,
    )
    return matrix.sort_index()[f"{name.upper()}(value) by group_id"].tolist()
