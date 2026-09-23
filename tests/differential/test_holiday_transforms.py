"""Cross-check tusk's holiday transform primitives against featuretools.

featuretools reads the US federal holidays of 1950 to 2074 from the
``holidays`` package. tusk is given that same calendar, taken from
featuretools' own ``HolidayUtil``, so both sides read identical holidays.
Each test builds one primitive on both sides over the column ``happened_at``
and compares the one resulting column.

Run with: uv run --group validation pytest -m differential

Verified against featuretools 1.31.0.
"""

import datetime as dt

import pandas as pd
import pytest

from differential.transform_matrices import (
    assert_agree,
    featuretools_values,
    tusk_values,
)
from tusk.primitives import DaysToHoliday, HolidayName, IsHoliday

featuretools = pytest.importorskip("featuretools")

pytestmark = pytest.mark.differential

NEW_YEARS_DAY = "New Year's Day"
HALFWAY_BETWEEN_NEW_YEARS_DAYS = dt.datetime(2024, 7, 2)

AROUND_FEDERAL_HOLIDAYS = [
    dt.datetime(2019, 7, 4, 10, 0, 30),
    dt.datetime(2019, 2, 26),
    None,
    dt.datetime(2021, 12, 24),
    dt.datetime(2024, 11, 28, 23, 59, 59),
    dt.datetime(1949, 12, 25),
    dt.datetime(2100, 12, 25),
    dt.datetime(2023, 1, 2),
]

AROUND_NEW_YEARS_DAY = [
    dt.datetime(2024, 1, 1, 12),
    dt.datetime(2023, 12, 31),
    None,
    HALFWAY_BETWEEN_NEW_YEARS_DAYS,
    dt.datetime(2024, 7, 3),
    dt.datetime(2019, 12, 25, 6),
    dt.datetime(2023, 6, 1),
    dt.datetime(2023, 1, 2),
]


@pytest.fixture(scope="module")
def federal_holidays():
    """The US federal holidays featuretools reads, as tusk takes them."""
    holiday_util = (
        featuretools.primitives.standard.transform.datetime.utils.HolidayUtil("US")
    )
    return dict(holiday_util.federal_holidays)


@pytest.fixture(scope="module")
def new_years_days(federal_holidays):
    """Only the New Year's Days of ``federal_holidays``, not their observed days."""
    return {
        date: name for date, name in federal_holidays.items() if name == NEW_YEARS_DAY
    }


def test_is_holiday_matches_is_federal_holiday(federal_holidays):
    """Covers a time of day, a null, an observed holiday and years outside
    featuretools' calendar.
    """
    ours = tusk_values(
        IsHoliday(holidays=federal_holidays),
        "IS_HOLIDAY__happened_at",
        rows=_rows(AROUND_FEDERAL_HOLIDAYS),
    )
    theirs = featuretools_values(
        "is_federal_holiday",
        "IS_FEDERAL_HOLIDAY(happened_at)",
        rows=_rows(AROUND_FEDERAL_HOLIDAYS),
    )
    assert_agree(ours, theirs)


def test_holiday_name_matches_date_to_holiday(federal_holidays):
    """Covers a time of day, a null, an observed holiday and years outside
    featuretools' calendar.
    """
    ours = tusk_values(
        HolidayName(holidays=federal_holidays),
        "HOLIDAY_NAME__happened_at",
        rows=_rows(AROUND_FEDERAL_HOLIDAYS),
    )
    theirs = featuretools_values(
        "date_to_holiday",
        "DATE_TO_HOLIDAY(happened_at)",
        rows=_rows(AROUND_FEDERAL_HOLIDAYS),
    )
    assert_agree(ours, theirs)


def test_days_to_holiday_matches_distance_to_holiday_within_a_year(new_years_days):
    """tusk is given only New Year's Day. Covers a time of day, a null, the
    exact tie halfway between two New Year's Days, where both sides take the
    past one, and an observed New Year's Day, which is not the holiday itself.
    """
    ours = tusk_values(
        DaysToHoliday(holidays=new_years_days),
        "DAYS_TO_HOLIDAY__happened_at",
        rows=_rows(AROUND_NEW_YEARS_DAY),
    )
    theirs = featuretools_values(
        featuretools.primitives.DistanceToHoliday(holiday=NEW_YEARS_DAY),
        "DISTANCE_TO_HOLIDAY(happened_at)",
        rows=_rows(AROUND_NEW_YEARS_DAY),
    )
    assert_agree(ours, theirs)
    assert ours[AROUND_NEW_YEARS_DAY.index(HALFWAY_BETWEEN_NEW_YEARS_DAYS)] == -183


def test_days_to_holiday_counts_beyond_the_year_where_distance_to_holiday_gives_up(
    new_years_days,
):
    """featuretools answers NaN when the nearest New Year's Day is more than 365
    days away; tusk counts to it however far it is.
    """
    far_from_the_calendar = [dt.datetime(1900, 6, 1), dt.datetime(2100, 6, 1)]
    ours = tusk_values(
        DaysToHoliday(holidays=new_years_days),
        "DAYS_TO_HOLIDAY__happened_at",
        rows=_rows(far_from_the_calendar),
    )
    theirs = featuretools_values(
        featuretools.primitives.DistanceToHoliday(holiday=NEW_YEARS_DAY),
        "DISTANCE_TO_HOLIDAY(happened_at)",
        rows=_rows(far_from_the_calendar),
    )
    assert ours == [
        (min(new_years_days) - dt.date(1900, 6, 1)).days,
        (max(new_years_days) - dt.date(2100, 6, 1)).days,
    ]
    assert all(pd.isna(their) for their in theirs)


def _rows(happened_at):
    """Build a child table of the shape the differential helpers read.

    Args:
        happened_at: One datetime or None per row.

    Returns:
        A table keyed by ``id``, ordered by ``occurred_at``, in the one group
        ``1``, with the given datetimes as ``happened_at``.
    """
    return pd.DataFrame(
        {
            "id": range(1, len(happened_at) + 1),
            "group_id": [1] * len(happened_at),
            "occurred_at": pd.date_range("2024-01-01", periods=len(happened_at)),
            "happened_at": pd.to_datetime(happened_at),
        },
    )
