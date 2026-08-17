import datetime as dt

import narwhals as nw
import polars as pl
import pytest

from tusk.primitives.registry import resolve
from tusk.primitives.transform import TRANS_DEFAULTS


@pytest.fixture
def lf():
    return nw.from_native(
        pl.LazyFrame(
            {
                "g": [1, 1, 1],
                "t": [
                    dt.datetime(2024, 3, 4, 5),
                    dt.datetime(2024, 3, 9, 6),
                    dt.datetime(2024, 3, 10, 7),
                ],
                "v": [-2.0, 3.0, 4.0],
                "w": [1.0, 1.0, 2.0],
            }
        )
    )


def _apply(lf, name, *columns):
    primitive = resolve(name)
    expr = primitive.outputs(*[nw.col(c) for c in columns])[0]
    if getattr(primitive, "order_dependent", False):
        expr = expr.over(order_by="t")
    return lf.with_columns(expr.alias("o")).collect().to_native()["o"].to_list()


@pytest.mark.parametrize(
    ("name", "columns", "expected"),
    [
        ("year", ("t",), [2024, 2024, 2024]),
        ("month", ("t",), [3, 3, 3]),
        ("day", ("t",), [4, 9, 10]),
        ("hour", ("t",), [5, 6, 7]),
        ("weekday", ("t",), [1, 6, 7]),
        ("is_weekend", ("t",), [False, True, True]),
        ("absolute", ("v",), [2.0, 3.0, 4.0]),
        ("add_numeric", ("v", "w"), [-1.0, 4.0, 6.0]),
        ("subtract_numeric", ("v", "w"), [-3.0, 2.0, 2.0]),
        ("multiply_numeric", ("v", "w"), [-2.0, 3.0, 8.0]),
        ("divide_numeric", ("v", "w"), [-2.0, 3.0, 2.0]),
    ],
)
def test_row_wise_transforms(lf, name, columns, expected):
    assert _apply(lf, name, *columns) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("cum_sum", [-2.0, 1.0, 5.0]),
        ("cum_max", [-2.0, 3.0, 4.0]),
        ("cum_min", [-2.0, -2.0, -2.0]),
        ("diff", [None, 5.0, 1.0]),
    ],
)
def test_order_dependent_transforms(lf, name, expected):
    assert _apply(lf, name, "v") == expected


def test_order_dependent_primitives_are_flagged():
    assert getattr(resolve("cum_sum"), "order_dependent", False) is True
    assert getattr(resolve("month"), "order_dependent", False) is False


def test_time_since_previous_is_seconds(lf):
    got = _apply(lf, "time_since_previous", "t")
    assert got[0] is None
    assert got[1] == pytest.approx(5 * 86400 + 3600)


def test_arithmetic_commutativity_flags():
    assert getattr(resolve("add_numeric"), "commutative", False) is True
    assert getattr(resolve("subtract_numeric"), "commutative", False) is False


def test_defaults_exclude_arithmetic():
    assert TRANS_DEFAULTS == ("year", "month", "weekday")
