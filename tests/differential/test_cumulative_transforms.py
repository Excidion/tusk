"""Cross-check tusk's cumulative transform primitives against featuretools.

Both sides order the shared table by ``occurred_at``: ids 3, 1, 5, 2, 8, 4, 7,
6. In that order ``value`` is 4, null, -1, 0, 0, 9, 2, 2. featuretools
forward-fills a null before ``same_as_previous``, ``absolute_diff`` and
``percent_change``; tusk does not, which is where those three diverge.

Run with: uv run --group validation pytest -m differential

Verified against featuretools 1.31.0.
"""

import math

import pytest
from transform_cases import ROWS

from differential.transform_matrices import (
    assert_agree,
    featuretools_values,
    tusk_values,
)

featuretools = pytest.importorskip("featuretools")

pytestmark = pytest.mark.differential


@pytest.mark.parametrize("name", ["cum_sum", "cum_min", "cum_max", "diff"])
def test_cumulative_transforms_match_featuretools(name):
    """Both sides skip a null in the running value; diff is null next to a null."""
    assert_agree(
        tusk_values(name, f"{name.upper()}__value"),
        featuretools_values(name, f"{name.upper()}(value)"),
    )


@pytest.mark.parametrize(
    "name",
    ["cumulative_time_since_last_true", "cumulative_time_since_last_false"],
)
def test_cumulative_time_since_matches_featuretools_in_seconds(name):
    """tusk returns a Duration, featuretools float seconds; the amounts agree."""
    ours = tusk_values(name, f"{name.upper()}__occurred_at__flag")
    ours_in_seconds = [
        None if value is None else value.total_seconds() for value in ours
    ]
    assert_agree(
        ours_in_seconds, featuretools_values(name, f"{name.upper()}(occurred_at, flag)")
    )


def test_cum_mean_divides_by_known_values_where_featuretools_divides_by_rows():
    ours = tusk_values("cum_mean", "CUM_MEAN__value")
    theirs = featuretools_values("cum_mean", "CUM_MEAN(value)")
    assert ours[0] is None
    assert_agree(ours, [None, 1.0, 4.0, 2.4, 1.5, 16 / 7, 7 / 3, 0.75])
    assert_agree(theirs, [None, 0.75, 4.0, 2.0, 1.0, 2.0, 2.0, 0.6])


def test_same_as_previous_does_not_forward_fill():
    """The first row (id 3), the null (id 1) and the row after it (id 5) diverge."""
    ours = tusk_values("same_as_previous", "SAME_AS_PREVIOUS__value")
    theirs = featuretools_values("same_as_previous", "SAME_AS_PREVIOUS(value)")
    assert ours == [None, False, None, False, None, True, False, True]
    assert theirs == [True, False, False, False, False, True, False, True]


def test_absolute_diff_does_not_forward_fill():
    """The null (id 1) and the row after it (id 5) diverge."""
    ours = tusk_values("absolute_diff", "ABSOLUTE_DIFF__value")
    theirs = featuretools_values("absolute_diff", "ABSOLUTE_DIFF(value)")
    assert [ours[0], ours[2], ours[4]] == [None, None, None]
    assert_agree(ours, [None, 1.0, None, 9.0, None, 0.0, 7.0, 0.0])
    assert theirs[0] == 0.0
    assert theirs[4] == 5.0
    assert_agree(theirs, [0.0, 1.0, None, 9.0, 5.0, 0.0, 7.0, 0.0])


def test_percent_change_does_not_forward_fill():
    """The null (id 1) and the row after it (id 5) diverge.

    Zero after zero (id 8) is NaN on both sides.
    """
    ours = tusk_values("percent_change", "PERCENT_CHANGE__value")
    theirs = featuretools_values("percent_change", "PERCENT_CHANGE(value)")
    assert ours[0] is None
    assert ours[4] is None
    assert theirs[0] == 0.0
    assert theirs[4] == -1.25
    assert math.isnan(ours[7])
    assert math.isnan(theirs[7])
    assert_agree(ours, [None, -1.0, None, math.inf, None, 0.0, -7 / 9, math.nan])
    assert_agree(theirs, [0.0, -1.0, None, math.inf, -1.25, 0.0, -7 / 9, math.nan])


def test_cum_count_counts_known_values_where_featuretools_counts_rows():
    """``label`` is null on ids 2 and 7; featuretools still counts those rows."""
    ours = tusk_values("cum_count", "CUM_COUNT__label")
    theirs = featuretools_values("cum_count", "CUM_COUNT(label)")
    assert ours == [2, 3, 1, 5, 3, 6, 5, 4]
    assert theirs == [2, 4, 1, 6, 3, 8, 7, 5]


def test_diff_datetime_is_time_since_previous():
    """featuretools' DFS builds no diff_datetime here, so it is called directly.

    Both lists are in time order.
    """
    ordered = ROWS.sort_values("occurred_at")
    theirs = featuretools.primitives.DiffDatetime()(ordered["occurred_at"]).tolist()
    ours_by_id = dict(
        zip(
            sorted(ROWS["id"]),
            tusk_values("time_since_previous", "TIME_SINCE_PREVIOUS__occurred_at"),
            strict=True,
        ),
    )
    ours = [ours_by_id[row_id] for row_id in ordered["id"]]
    assert_agree(ours, theirs)
