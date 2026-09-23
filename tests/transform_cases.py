"""One table and the value every standalone, group and ordered transform gives on it.

Shared by the polars suite, the duckdb suite and the differential suite, so
all three check the same rows. ``occurred_at`` is the row creation time and
orders the rows 3, 1, 5, 2, 8, 4, 7, 6, which differs from id order. Every row
belongs to the one row of ``GROUPS``, so a group transform sees all eight.
``due_on`` holds the calendar dates of ``due_at``.

``EXPECTED`` maps each feature column to the primitive that builds it, as a
name or an instance, its dtype and its values in id order. ``HOLIDAYS`` puts
those dates on a holiday, before the first, after the last, and exactly
halfway between two.
"""

import datetime as dt
import math

import narwhals as nw
import pandas as pd
import pyarrow as pa
import pytest

import tusk
from tusk.primitives import (
    DaysSinceHoliday,
    DaysToHoliday,
    DaysUntilHoliday,
    HolidayName,
    IsHoliday,
)

GROUPS = pd.DataFrame({"id": [1]})

ROWS = pd.DataFrame(
    {
        "id": range(1, 9),
        "group_id": [1] * 8,
        "occurred_at": pd.to_datetime(
            [
                dt.datetime(2024, 2, 29, 13, 45, 30),
                dt.datetime(2024, 3, 2, 8, 0, 5),
                dt.datetime(2023, 12, 31, 23, 59, 59),
                dt.datetime(2024, 3, 10),
                dt.datetime(2024, 3, 1, 6, 30),
                dt.datetime(2024, 3, 20, 18, 15, 45),
                dt.datetime(2024, 3, 15, 12),
                dt.datetime(2024, 3, 5, 9, 9, 9),
            ],
        ),
        "value": [None, 0.0, 4.0, 9.0, -1.0, 2.0, 2.0, 0.0],
        "flag": [True, False, False, False, False, True, False, True],
        "maybe_flag": pd.array(
            [None, True, True, False, False, True, False, None],
            dtype="boolean",
        ),
        "due_at": pd.to_datetime(
            [
                dt.datetime(2024, 2, 29, 13, 45, 30),
                None,
                dt.datetime(1900, 3, 1, 23, 59, 59),
                dt.datetime(2000, 12, 31),
                dt.datetime(2023, 6, 15, 7, 8, 9),
                dt.datetime(2100, 1, 1, 12, 30),
                dt.datetime(2024, 12, 31, 23, 0, 1),
                dt.datetime(2019, 1, 1),
            ],
        ),
        "due_on": [
            dt.date(2024, 2, 29),
            None,
            dt.date(1900, 3, 1),
            dt.date(2000, 12, 31),
            dt.date(2023, 6, 15),
            dt.date(2100, 1, 1),
            dt.date(2024, 12, 31),
            dt.date(2019, 1, 1),
        ],
        "label": pd.array(
            ["a", None, "b", "a", "b", "a", None, "b"],
            dtype="string",
        ),
        "text": pd.array(
            [
                "Hello, world!",
                None,
                "Dog dog cat",
                "   ",
                "a-b [c]",
                "the quick brown fox",
                "!!!",
                "word, word",
            ],
            dtype="string",
        ),
        "url": pd.array(
            [
                "https://play.google.com/store?x=1",
                None,
                "http://www.google.co.in:8080/path",
                "www.facebook.com",
                "https://user:pass@mysite.company/",
                "ftp://files.example.org",
                "http://:80",
                "www.",
            ],
            dtype="string",
        ),
        "email": pd.array(
            [
                "name@gmail.com",
                None,
                "  padded@example.org  ",
                "no-at-sign",
                "a@b@c.com",
                "trailing@",
                "UPPER@Example.COM",
                "",
            ],
            dtype="string",
        ),
    },
)

HOLIDAYS = {
    dt.date(2000, 12, 25): "Christmas",
    dt.date(2019, 1, 1): "New Year's Day",
    dt.date(2024, 2, 29): "Leap Day",
    dt.date(2024, 12, 25): "Christmas",
    dt.date(2025, 1, 6): "Epiphany",
}

HOLIDAY_EXPECTED = {
    IsHoliday(holidays=HOLIDAYS): (
        nw.Boolean,
        [True, None, False, False, False, False, False, True],
    ),
    HolidayName(holidays=HOLIDAYS): (
        nw.String,
        ["Leap Day", None, None, None, None, None, None, "New Year's Day"],
    ),
    DaysUntilHoliday(holidays=HOLIDAYS): (
        nw.Int64,
        [0, None, 36824, 6575, 259, None, 6, 0],
    ),
    DaysSinceHoliday(holidays=HOLIDAYS): (
        nw.Int64,
        [0, None, None, 6, 1626, 27388, 6, 0],
    ),
    DaysToHoliday(holidays=HOLIDAYS): (
        nw.Int64,
        [0, None, 36824, -6, 259, -27388, -6, 0],
    ),
}

EXPECTED = {
    "N_CHARACTERS__text": (
        "n_characters",
        nw.Int64,
        [13, None, 11, 3, 7, 19, 3, 10],
    ),
    "N_WORDS__text": (
        "n_words",
        nw.Int64,
        [2, None, 3, 0, 2, 4, 0, 2],
    ),
    "N_UNIQUE_WORDS__text": (
        "n_unique_words",
        nw.Int64,
        [2, None, 2, 0, 2, 4, 0, 1],
    ),
    "DOMAIN__email": (
        "domain",
        nw.String,
        [
            "gmail.com",
            None,
            "example.org",
            "no-at-sign",
            "c.com",
            None,
            "Example.COM",
            None,
        ],
    ),
    "DOMAIN__url": (
        "domain",
        nw.String,
        [
            "play.google.com",
            None,
            "google.co.in",
            "facebook.com",
            "mysite.company",
            "files.example.org",
            None,
            None,
        ],
    ),
    "TOP_LEVEL_DOMAIN__email": (
        "top_level_domain",
        nw.String,
        ["com", None, "org", None, "com", None, "COM", None],
    ),
    "TOP_LEVEL_DOMAIN__url": (
        "top_level_domain",
        nw.String,
        ["com", None, "in", "com", "company", "org", None, None],
    ),
    "URL_TO_PROTOCOL__url": (
        "url_to_protocol",
        nw.String,
        ["https", None, "http", None, "https", None, "http", None],
    ),
    "IS_NULL__value": (
        "is_null",
        nw.Boolean,
        [True, False, False, False, False, False, False, False],
    ),
    "NEGATE__value": (
        "negate",
        nw.Float64,
        [None, -0.0, -4.0, -9.0, 1.0, -2.0, -2.0, -0.0],
    ),
    "SQUARE_ROOT__value": (
        "square_root",
        nw.Float64,
        [None, 0.0, 2.0, 3.0, None, math.sqrt(2), math.sqrt(2), 0.0],
    ),
    "NATURAL_LOG__value": (
        "natural_log",
        nw.Float64,
        [
            None,
            -math.inf,
            math.log(4),
            math.log(9),
            None,
            math.log(2),
            math.log(2),
            -math.inf,
        ],
    ),
    "SINE__value": (
        "sine",
        nw.Float64,
        [
            None,
            0.0,
            math.sin(4),
            math.sin(9),
            math.sin(-1),
            math.sin(2),
            math.sin(2),
            0.0,
        ],
    ),
    "COSINE__value": (
        "cosine",
        nw.Float64,
        [
            None,
            1.0,
            math.cos(4),
            math.cos(9),
            math.cos(-1),
            math.cos(2),
            math.cos(2),
            1.0,
        ],
    ),
    "MINUTE__due_at": ("minute", nw.Int8, [45, None, 59, 0, 8, 30, 0, 0]),
    "SECOND__due_at": ("second", nw.Int8, [30, None, 59, 0, 9, 0, 1, 0]),
    "DAY_OF_YEAR__due_at": (
        "day_of_year",
        nw.Int16,
        [60, None, 60, 366, 166, 1, 366, 1],
    ),
    "IS_LEAP_YEAR__due_at": (
        "is_leap_year",
        nw.Boolean,
        [True, None, False, True, False, False, True, False],
    ),
    "PERCENTILE__value__by__group_id": (
        "percentile",
        nw.Float64,
        [None, 5 / 14, 12 / 14, 1.0, 2 / 14, 9 / 14, 9 / 14, 5 / 14],
    ),
    "CUM_MEAN__value__by__group_id": (
        "cum_mean",
        nw.Float64,
        [None, 1.0, 4.0, 2.4, 1.5, 16 / 7, 7 / 3, 0.75],
    ),
    "SAME_AS_PREVIOUS__value__by__group_id": (
        "same_as_previous",
        nw.Boolean,
        [None, False, None, False, None, True, False, True],
    ),
    "ABSOLUTE_DIFF__value__by__group_id": (
        "absolute_diff",
        nw.Float64,
        [None, 1.0, None, 9.0, None, 0.0, 7.0, 0.0],
    ),
    "PERCENT_CHANGE__value__by__group_id": (
        "percent_change",
        nw.Float64,
        [None, -1.0, None, math.inf, None, 0.0, -7 / 9, math.nan],
    ),
    "CUMULATIVE_TIME_SINCE_LAST_TRUE__occurred_at__flag__by__group_id": (
        "cumulative_time_since_last_true",
        nw.Duration,
        [
            dt.timedelta(0),
            dt.timedelta(days=1, seconds=65675),
            None,
            dt.timedelta(days=4, seconds=53451),
            dt.timedelta(seconds=60270),
            dt.timedelta(0),
            dt.timedelta(days=10, seconds=10251),
            dt.timedelta(0),
        ],
    ),
    "CUMULATIVE_TIME_SINCE_LAST_FALSE__occurred_at__flag__by__group_id": (
        "cumulative_time_since_last_false",
        nw.Duration,
        [
            dt.timedelta(days=59, seconds=49531),
            dt.timedelta(0),
            dt.timedelta(0),
            dt.timedelta(0),
            dt.timedelta(0),
            dt.timedelta(days=5, seconds=22545),
            dt.timedelta(0),
            dt.timedelta(days=3, seconds=4144),
        ],
    ),
    "CUMULATIVE_TIME_SINCE_LAST_TRUE__occurred_at__maybe_flag__by__group_id": (
        "cumulative_time_since_last_true",
        nw.Duration,
        [
            dt.timedelta(days=59, seconds=49531),
            dt.timedelta(0),
            dt.timedelta(0),
            dt.timedelta(days=7, seconds=57595),
            dt.timedelta(days=60, seconds=23401),
            dt.timedelta(0),
            dt.timedelta(days=13, seconds=14395),
            dt.timedelta(days=3, seconds=4144),
        ],
    ),
    "CUMULATIVE_TIME_SINCE_LAST_FALSE__occurred_at__maybe_flag__by__group_id": (
        "cumulative_time_since_last_false",
        nw.Duration,
        [
            None,
            dt.timedelta(days=1, seconds=5405),
            None,
            dt.timedelta(0),
            dt.timedelta(0),
            dt.timedelta(days=5, seconds=22545),
            dt.timedelta(0),
            dt.timedelta(days=4, seconds=9549),
        ],
    ),
    "CUMULATIVE_TIME_SINCE_LAST_FALSE__due_at__flag__by__group_id": (
        "cumulative_time_since_last_false",
        nw.Duration,
        [
            dt.datetime(2024, 2, 29, 13, 45, 30) - dt.datetime(1900, 3, 1, 23, 59, 59),
            None,
            dt.timedelta(0),
            dt.timedelta(0),
            dt.timedelta(0),
            dt.datetime(2100, 1, 1, 12, 30) - dt.datetime(2024, 12, 31, 23, 0, 1),
            dt.timedelta(0),
            dt.datetime(2019, 1, 1) - dt.datetime(2023, 6, 15, 7, 8, 9),
        ],
    ),
    **{
        f"{primitive.stem}__{column}": (primitive, dtype, values)
        for primitive, (dtype, values) in HOLIDAY_EXPECTED.items()
        for column in ("due_at", "due_on")
    },
}


def rows_database(table, groups):
    """Wrap a child table and its parent in a database.

    Args:
        table: A child table keyed by ``id``, with ``occurred_at`` and
            ``group_id``, such as ``ROWS``, as a native lazy frame of any
            backend.
        groups: Its parent table keyed by ``id``, such as ``GROUPS``, as a
            native lazy frame of the same backend.

    Returns:
        A database that always names the child table ``rows`` and the parent
        ``groups``; ``rows`` is ordered by ``occurred_at`` and belongs to
        ``groups`` through ``group_id``.
    """
    return (
        tusk.Database("rows")
        .add_table("groups", groups, primary_key="id")
        .add_table(
            "rows",
            table,
            primary_key="id",
            row_creation_time="occurred_at",
        )
        .add_relationship(parent="groups", child="rows", foreign_key="group_id")
    )


def feature_values(matrix, column):
    """Materialize one feature column in id order.

    Args:
        matrix: A native feature matrix from ``deep_feature_synthesis``.
        column: The feature column to read.

    Returns:
        The column's values as plain Python objects, null as None.
    """
    return nw.from_native(matrix).collect().sort("id")[column].to_list()


def assert_values_match(got, expected):
    """Compare one feature column against its expected values, row by row.

    A null and a NaN are different values here: ``square_root`` of a negative
    number must be null, not NaN.

    Args:
        got: The column as a list, in id order.
        expected: The expected values; a float is compared approximately.
    """
    assert len(got) == len(expected)
    for actual, wanted in zip(got, expected, strict=True):
        actual = _as_timedelta(actual)
        if wanted is None:
            assert actual is None
        elif isinstance(wanted, float) and math.isnan(wanted):
            assert isinstance(actual, float) and math.isnan(actual)
        elif isinstance(wanted, float):
            assert actual == pytest.approx(wanted)
        else:
            assert actual == wanted


def _as_timedelta(value):
    """Convert duckdb's interval value into a timedelta.

    Args:
        value: One materialized value.

    Returns:
        A timedelta for an interval, else the value unchanged.
    """
    if not isinstance(value, pa.MonthDayNano):
        return value
    assert value.months == 0
    return dt.timedelta(days=value.days, microseconds=value.nanoseconds // 1000)
