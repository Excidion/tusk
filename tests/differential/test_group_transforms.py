"""Cross-check that group and ordered transforms stay within their group.

The shared table in ``tests/transform_cases.py`` puts every row in one group,
so it cannot tell a grouped result from a whole-table one. Here two groups
interleave in ``occurred_at``, so a result that ignored ``group_id`` would
differ from featuretools' grouped result. Within group 1,
``occurred_at`` orders the rows 5, 3, 1, against id order, so a result
ordered by id would differ too.

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

pytestmark = pytest.mark.differential

GROUPS = pd.DataFrame({"id": [1, 2]})

INTERLEAVED_ROWS = pd.DataFrame(
    {
        "id": [1, 2, 3, 4, 5, 6],
        "group_id": [1, 2, 1, 2, 1, 2],
        "occurred_at": pd.to_datetime(
            [dt.datetime(2024, 1, day) for day in (5, 2, 3, 4, 1, 6)],
        ),
        "value": [3.0, 10.0, None, 10.0, 1.0, 5.0],
    },
)

CUTOFF = dt.datetime(2024, 1, 4)


@pytest.mark.parametrize(
    ("name", "cutoff_time", "expected"),
    [
        ("percentile", None, [1.0, 5 / 6, None, 5 / 6, 0.5, 1 / 3]),
        ("cum_sum", None, [4.0, 10.0, None, 20.0, 1.0, 25.0]),
        ("percentile", CUTOFF, [0.75, None, 0.75, 1.0]),
        ("cum_sum", CUTOFF, [10.0, None, 20.0, 1.0]),
    ],
)
def test_group_transforms_stay_within_their_group(name, cutoff_time, expected):
    """In time order group 1 holds 1, null, 3 and group 2 holds 10, 10, 5.

    The cutoff hides ids 1 and 6. Over the whole table ``cum_sum`` of id 2
    would be 11 and ``percentile`` of id 1 would be 0.4; ordered by id,
    ``cum_sum`` of id 1 would be 3. ``expected`` pins the grouped answer.
    """
    column = f"{name.upper()}__value__by__group_id"
    ours = tusk_values(
        name, column, cutoff_time=cutoff_time, rows=INTERLEAVED_ROWS, groups=GROUPS
    )
    theirs = featuretools_values(
        name,
        f"{name.upper()}(value) by group_id",
        cutoff_time=cutoff_time,
        rows=INTERLEAVED_ROWS,
        groups=GROUPS,
    )
    assert_agree(ours, expected)
    assert_agree(ours, theirs)
