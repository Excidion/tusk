import datetime as dt

import narwhals as nw
import polars as pl
import pytest

import tusk
from tusk.dtypes import DtypeFamily
from tusk.exceptions import ValidationError
from tusk.primitives.base import NeedsCutoffTime, TransformPrimitive
from tusk.primitives.registry import resolve
from tusk.primitives.transform import TRANS_DEFAULTS, TimeSince


@pytest.fixture
def lf():
    # Rows shuffled so t is NOT monotonic in row order.
    # Row 0: t=3/9, Row 1: t=3/4, Row 2: t=3/10
    # When order_by="t", processes in order: Row 1 (3/4), Row 0 (3/9), Row 2 (3/10)
    return nw.from_native(
        pl.LazyFrame(
            {
                "g": [1, 1, 1],
                "t": [
                    dt.datetime(2024, 3, 9, 6),
                    dt.datetime(2024, 3, 4, 5),
                    dt.datetime(2024, 3, 10, 7),
                ],
                "v": [3.0, -2.0, 4.0],
                "w": [1.0, 1.0, 2.0],
            },
        ),
    )


def _apply(lf, name, *columns):
    primitive = resolve(name)
    expr = primitive.outputs(*[nw.col(c) for c in columns])[0]
    if isinstance(primitive, TransformPrimitive) and primitive.order_dependent:
        expr = expr.over(order_by="t")
    return lf.with_columns(expr.alias("o")).collect().to_native()["o"].to_list()


@pytest.mark.parametrize(
    ("name", "columns", "expected"),
    [
        ("year", ("t",), [2024, 2024, 2024]),
        ("month", ("t",), [3, 3, 3]),
        ("day", ("t",), [9, 4, 10]),
        ("hour", ("t",), [6, 5, 7]),
        ("weekday", ("t",), [6, 1, 7]),
        ("is_weekend", ("t",), [True, False, True]),
        ("absolute", ("v",), [3.0, 2.0, 4.0]),
        ("add_numeric", ("v", "w"), [4.0, -1.0, 6.0]),
        ("subtract_numeric", ("v", "w"), [2.0, -3.0, 2.0]),
        ("multiply_numeric", ("v", "w"), [3.0, -2.0, 8.0]),
        ("divide_numeric", ("v", "w"), [3.0, -2.0, 2.0]),
    ],
)
def test_row_wise_transforms(lf, name, columns, expected):
    assert _apply(lf, name, *columns) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # v in t-order (Row 1, Row 0, Row 2): [-2.0, 3.0, 4.0]
        # cum_sum in t-order: [-2.0, 1.0, 5.0]; row order: [1.0, -2.0, 5.0]
        ("cum_sum", [1.0, -2.0, 5.0]),
        # cum_max in t-order: [-2.0, 3.0, 4.0]; row order: [3.0, -2.0, 4.0]
        ("cum_max", [3.0, -2.0, 4.0]),
        # cum_min in t-order: [-2.0, -2.0, -2.0]; row order: [-2.0, -2.0, -2.0]
        ("cum_min", [-2.0, -2.0, -2.0]),
        # diff in t-order: [None, 5.0, 1.0]; row order: [5.0, None, 1.0]
        ("diff", [5.0, None, 1.0]),
    ],
)
def test_order_dependent_transforms(lf, name, expected):
    assert _apply(lf, name, "v") == expected


def test_order_dependent_primitives_are_flagged():
    cum_sum = resolve("cum_sum")
    assert isinstance(cum_sum, TransformPrimitive)
    assert cum_sum.order_dependent is True
    month = resolve("month")
    assert isinstance(month, TransformPrimitive)
    assert month.order_dependent is False


def test_time_since_previous_is_seconds(lf):
    # t in t-order: 3/4, 3/9, 3/10
    # Row 1 (3/4) is first: None
    # Row 0 (3/9): diff from Row 1 (3/4) = 5 days + 1 hour = 432000 + 3600
    # Row 2 (3/10): diff from Row 0 (3/9) = 1 day + 1 hour = 86400 + 3600
    got = _apply(lf, "time_since_previous", "t")
    assert got[1] is None
    assert got[0] == pytest.approx(5 * 86400 + 3600)
    assert got[2] == pytest.approx(86400 + 3600)


def test_arithmetic_commutativity_flags():
    add = resolve("add_numeric")
    assert add.commutative is True
    subtract = resolve("subtract_numeric")
    assert subtract.commutative is False


def test_defaults_exclude_arithmetic():
    assert TRANS_DEFAULTS == ("year", "month", "weekday")


def test_time_since_measures_from_the_cutoff_time():
    """A past timestamp gives a positive duration, a future one negative."""
    frame = nw.from_native(
        pl.LazyFrame(
            {"t": [dt.datetime(2024, 1, 1), dt.datetime(2024, 6, 1), None]},
        ),
    )
    primitive = TimeSince(cutoff_time=dt.datetime(2024, 3, 1))
    got = frame.select(primitive.outputs(nw.col("t"))[0].alias("o")).collect()
    assert got.to_native()["o"].to_list() == [
        dt.timedelta(days=60),
        dt.timedelta(days=-92),
        None,
    ]


def test_time_since_needs_a_datetime_input():
    assert TimeSince().input_dtypes == (DtypeFamily.DATETIME,)
    assert isinstance(TimeSince(), NeedsCutoffTime)


def test_deep_feature_synthesis_rejects_time_since_without_a_cutoff_time(db):
    """The clock is the cutoff time, so there is no answer without one."""
    with pytest.raises(ValidationError, match="time_since needs a cutoff_time"):
        tusk.deep_feature_synthesis(
            database=db,
            target_table="customers",
            agg_primitives=[],
            trans_primitives=["time_since"],
            max_depth=1,
        )


def test_deep_feature_synthesis_computes_time_since_with_a_cutoff_time(db):
    matrix, _ = tusk.deep_feature_synthesis(
        database=db,
        target_table="customers",
        agg_primitives=[],
        trans_primitives=["time_since"],
        max_depth=1,
        cutoff_time=dt.datetime(2024, 3, 1),
    )
    got = matrix.collect().sort("id")
    # customers.signed_up_at is 2024-01-01 for all three rows (tests/conftest.py)
    assert got["TIME_SINCE__signed_up_at"].to_list() == [dt.timedelta(days=60)] * 3
