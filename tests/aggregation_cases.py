"""One parent/child fixture and the value every standalone aggregation gives on it.

Shared by the polars suite, the duckdb suite and the differential suite, so
all three check the same groups. Parent 1 mixes known values with a null,
parent 2 is constant, parent 3 is all null, parent 4 varies, parent 5 has no
children, parent 6 has a single known row, parent 7 has a single null row,
and parent 8 has one known row beside a null.

``created_at`` orders each group against its ids, so ``first`` and ``last``
cannot pass by reading rows in id order: parent 1 starts on its null value,
and parent 4's latest two rows share a time that only the id breaks.
"""

import datetime as dt

import narwhals as nw
import pandas as pd
import pytest

ROW_CREATION_TIME = "created_at"

PARENTS = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6, 7, 8]})

CHILDREN = pd.DataFrame(
    {
        "id": range(1, 18),
        "parent_id": [1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 6, 7, 8, 8],
        "value": [1.0, 2.0, 6.0, None, 4.0, 4.0, None, None, None, 1.0, 5.0, 2.0, 8.0]
        + [3.0, None, 7.0, None],
        "flag": pd.array(
            [True, False, True, None, True, True, None, None, None]
            + [False, False, True, False]
            + [True, None, False, None],
            dtype="boolean",
        ),
        "label": pd.array(
            ["a", "a", "b", None, None, None, None, None, None, "x", "y", "z", "w"]
            + ["q", None, "q", None],
            dtype="string",
        ),
        "seen_at": pd.to_datetime(
            [
                dt.datetime(2024, 1, 1, 5),
                dt.datetime(2024, 1, 1, 9),
                dt.datetime(2024, 3, 2),
                None,
                dt.datetime(2023, 3, 2),
                dt.datetime(2024, 3, 2),
                None,
                None,
                None,
                dt.datetime(2024, 1, 31),
                dt.datetime(2024, 2, 1),
                dt.datetime(2025, 2, 1),
                dt.datetime(2024, 2, 29),
                dt.datetime(2024, 5, 5),
                None,
                dt.datetime(2024, 6, 1),
                None,
            ],
        ),
        ROW_CREATION_TIME: pd.to_datetime(
            [
                dt.datetime(2024, 1, 3),
                dt.datetime(2024, 1, 4),
                dt.datetime(2024, 1, 2),
                dt.datetime(2024, 1, 1),
                dt.datetime(2024, 1, 1),
                dt.datetime(2024, 1, 2),
                dt.datetime(2024, 1, 1),
                dt.datetime(2024, 1, 2),
                dt.datetime(2024, 1, 3),
                dt.datetime(2024, 1, 5),
                dt.datetime(2024, 1, 5),
                dt.datetime(2024, 1, 1),
                dt.datetime(2024, 1, 3),
                dt.datetime(2024, 1, 1),
                dt.datetime(2024, 1, 1),
                dt.datetime(2024, 1, 2),
                dt.datetime(2024, 1, 1),
            ],
        ),
    },
)

EXPECTED = {
    "all_true": (
        "ALL_TRUE__children__flag",
        nw.Boolean,
        [False, True, True, False, None, True, True, False],
    ),
    "any_true": (
        "ANY_TRUE__children__flag",
        nw.Boolean,
        [True, True, False, True, False, True, False, False],
    ),
    "is_unique": (
        "IS_UNIQUE__children__label",
        nw.Boolean,
        [False, False, False, True, None, True, True, True],
    ),
    "kurtosis": (
        "KURTOSIS__children__value",
        nw.Float64,
        [-1.5, None, None, -1.4266666666666667, None, None, None, None],
    ),
    "max_min_delta": (
        "MAX_MIN_DELTA__children__value",
        nw.Float64,
        [5.0, 0.0, None, 7.0, None, 0.0, None, 0.0],
    ),
    "n_true": ("N_TRUE__children__flag", nw.Int64, [2, 2, 0, 1, 0, 1, 0, 0]),
    "percent_unique": (
        "PERCENT_UNIQUE__children__label",
        nw.Float64,
        [0.75, 0.5, 0.3333333333333333, 1.0, None, 1.0, 1.0, 1.0],
    ),
    "skew": (
        "SKEW__children__value",
        nw.Float64,
        [0.5951700641394972, None, None, 0.3651483716701108, None, None, None, None],
    ),
    "variance": (
        "VARIANCE__children__value",
        nw.Float64,
        [7.0, 0.0, None, 10.0, None, None, None, None],
    ),
    "first_last_time_delta": (
        "FIRST_LAST_TIME_DELTA__children__seen_at",
        nw.Duration,
        [
            dt.timedelta(days=60, hours=19),
            dt.timedelta(days=366),
            None,
            dt.timedelta(days=367),
            None,
            dt.timedelta(0),
            None,
            dt.timedelta(0),
        ],
    ),
    "n_unique_days": (
        "N_UNIQUE_DAYS__children__seen_at",
        nw.Int64,
        [3, 2, 1, 4, 0, 1, 1, 2],
    ),
    "n_unique_days_of_calendar_year": (
        "N_UNIQUE_DAYS_OF_CALENDAR_YEAR__children__seen_at",
        nw.Int64,
        [3, 1, 1, 3, 0, 1, 1, 2],
    ),
    "n_unique_days_of_month": (
        "N_UNIQUE_DAYS_OF_MONTH__children__seen_at",
        nw.Int64,
        [3, 1, 1, 3, 0, 1, 1, 2],
    ),
    "n_unique_months": (
        "N_UNIQUE_MONTHS__children__seen_at",
        nw.Int64,
        [3, 2, 1, 3, 0, 1, 1, 2],
    ),
    "first": (
        "FIRST__children__value",
        nw.Float64,
        [None, 4.0, None, 2.0, None, 3.0, None, None],
    ),
    "last": (
        "LAST__children__value",
        nw.Float64,
        [2.0, 4.0, None, 5.0, None, 3.0, None, 7.0],
    ),
    "count_above_mean": (
        "COUNT_ABOVE_MEAN__children__value",
        nw.Int64,
        [1, 0, 0, 2, 0, 0, 0, 0],
    ),
    "count_below_mean": (
        "COUNT_BELOW_MEAN__children__value",
        nw.Int64,
        [2, 0, 0, 2, 0, 0, 0, 0],
    ),
    "count_inside_nth_std": (
        "COUNT_INSIDE_1_STD__children__value",
        nw.Int64,
        [2, 2, 0, 2, 0, 1, 0, 1],
    ),
    "count_outside_nth_std": (
        "COUNT_OUTSIDE_1_STD__children__value",
        nw.Int64,
        [1, 0, 0, 2, 0, 0, 0, 0],
    ),
}


def assert_values_match(got, expected):
    """Compare one feature column against its expected values, parent by parent.

    Args:
        got: The column as a list, in parent id order.
        expected: The expected values; a float is compared approximately.
    """
    assert len(got) == len(expected)
    for actual, wanted in zip(_normalized(got), expected, strict=True):
        if isinstance(wanted, float):
            assert actual == pytest.approx(wanted)
        else:
            assert actual == wanted


def _normalized(values):
    """Make a materialized column comparable with plain Python values.

    Args:
        values: A materialized column.

    Returns:
        The values, with every backend's missing value as None and every
        duration as a timedelta.
    """
    return [_normalized_value(value) for value in values]


def _normalized_value(value):
    """Make one materialized value comparable with a plain Python one.

    Args:
        value: A value from a pandas column.

    Returns:
        None for any missing value, a timedelta for a duration, else the value.
    """
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timedelta):
        return value.to_pytimedelta()
    return value
