"""Cross-check tusk's general transform primitives against featuretools.

Each test builds one primitive on both sides over the shared table in
``tests/transform_cases.py`` and compares the one resulting column.

Run with: uv run --group validation pytest -m differential

Verified against featuretools 1.31.0.
"""

import datetime as dt
import math

import pytest

from differential.transform_matrices import (
    assert_agree,
    featuretools_values,
    tusk_values,
)

pytestmark = pytest.mark.differential


@pytest.mark.parametrize(
    ("ours", "column", "theirs", "feature_name"),
    [
        ("is_null", "IS_NULL__value", "is_null", "IS_NULL(value)"),
        ("negate", "NEGATE__value", "negate", "-(value)"),
        ("sine", "SINE__value", "sine", "SINE(value)"),
        ("cosine", "COSINE__value", "cosine", "COSINE(value)"),
        ("absolute", "ABSOLUTE__value", "absolute", "ABSOLUTE(value)"),
    ],
)
def test_general_transforms_match_featuretools(ours, column, theirs, feature_name):
    """Covers a null, zero, a negative number and ties in ``value``."""
    assert_agree(tusk_values(ours, column), featuretools_values(theirs, feature_name))


def test_percentile_matches_featuretools_within_the_group():
    """Every row shares one group; the tied values in ``value`` share a rank."""
    ours = tusk_values("percentile", "PERCENTILE__value__by__group_id")
    theirs = featuretools_values(
        "percentile", "PERCENTILE(value) by group_id", grouped=True
    )
    assert_agree(ours, theirs)


def test_percentile_ranks_the_same_rows_on_both_sides_under_a_cutoff():
    """A cutoff bounds featuretools' grouped percentile as it bounds tusk's.

    At cutoff 2024-03-06, ids 1, 2, 3, 5, 8 are the rows visible (occurred_at
    on or before it); their known values are 0, 4, -1 and 0. featuretools'
    ``Percentile`` sets ``uses_full_dataframe = True``, but featuretools still
    bounds the full-dataframe query by the cutoff before ranking, so both
    sides rank among the same four known values within the one group.
    """
    cutoff = dt.datetime(2024, 3, 6)
    ours = tusk_values(
        "percentile", "PERCENTILE__value__by__group_id", cutoff_time=cutoff
    )
    theirs = featuretools_values(
        "percentile",
        "PERCENTILE(value) by group_id",
        grouped=True,
        cutoff_time=cutoff,
    )
    assert len(ours) == len(theirs) == 5
    # id 3 (value 4) and id 5 (value -1) would rank 6/7 and 1/7 among the
    # seven known values if either side ranked rows created after the cutoff.
    assert_agree(ours, [None, 0.625, 1.0, 0.25, 0.625])
    assert_agree(ours, theirs)


@pytest.mark.parametrize(
    ("ours", "column", "theirs", "feature_name"),
    [
        ("square_root", "SQUARE_ROOT__value", "square_root", "SQUARE_ROOT(value)"),
        (
            "natural_log",
            "NATURAL_LOG__value",
            "natural_logarithm",
            "NATURAL_LOGARITHM(value)",
        ),
    ],
)
def test_square_root_and_natural_log_give_null_where_featuretools_gives_nan(
    ours,
    column,
    theirs,
    feature_name,
):
    """Row id 5 holds -1: tusk answers null, featuretools NaN.

    Every other row agrees.
    """
    ours_values = tusk_values(ours, column)
    theirs_values = featuretools_values(theirs, feature_name)
    negative_row = 4
    assert ours_values[negative_row] is None
    assert math.isnan(theirs_values[negative_row])
    del ours_values[negative_row]
    del theirs_values[negative_row]
    assert_agree(ours_values, theirs_values)
