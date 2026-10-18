import datetime as dt
from decimal import Decimal

import narwhals as nw
import polars as pl
import pytest

from tusk.exceptions import UnsupportedDtypeWarning

pytest.importorskip("sdm")
pytestmark = pytest.mark.sdm

from tusk.sdm._columns import prepare_table  # noqa: E402


def _table(**columns):
    return nw.from_native(pl.DataFrame(columns, strict=False))


def test_numeric_dtypes_stay_numerical():
    table = _table(
        a=pl.Series([1, 2], dtype=pl.Int64),
        b=pl.Series([1, 2], dtype=pl.UInt8),
        c=pl.Series([1.5, None], dtype=pl.Float32),
    )
    result, types = prepare_table("t", table, frozenset())
    assert types == dict.fromkeys("abc", "numerical")
    assert result.columns == ["a", "b", "c"]
    assert result["c"].to_list() == [1.5, None]


def test_decimal_becomes_float64_because_sdm_rejects_it():
    table = _table(x=pl.Series([Decimal("1.5"), None], dtype=pl.Decimal(10, 1)))
    result, types = prepare_table("t", table, frozenset())
    assert types == {"x": "numerical"}
    assert result["x"].to_list() == [1.5, None]
    assert result.schema["x"] == nw.Float64


def test_boolean_becomes_zero_and_one():
    table = _table(x=[True, False, None])
    result, types = prepare_table("t", table, frozenset())
    assert types == {"x": "numerical"}
    assert result["x"].to_list() == [1.0, 0.0, None]
    assert result.schema["x"] == nw.Float64


def test_enum_becomes_its_category_position():
    table = _table(x=pl.Series(["b", "a", None], dtype=pl.Enum(["a", "b"])))
    result, types = prepare_table("t", table, frozenset())
    assert types == {"x": "numerical"}
    assert result["x"].to_list() == [1.0, 0.0, None]


def test_duration_becomes_seconds_as_float():
    table = _table(
        x=pl.Series([dt.timedelta(seconds=1, milliseconds=500), None]),
    )
    result, types = prepare_table("t", table, frozenset())
    assert types == {"x": "numerical"}
    assert result["x"].to_list() == [1.5, None]
    assert result.schema["x"] == nw.Float64


def test_string_and_categorical_stay_categorical():
    table = _table(
        s=["a", "b"],
        c=pl.Series(["a", "b"], dtype=pl.Categorical),
    )
    result, types = prepare_table("t", table, frozenset())
    assert types == {"s": "categorical", "c": "categorical"}
    assert result["s"].to_list() == ["a", "b"]


def test_date_and_datetime_are_datetime():
    table = _table(
        d=[dt.date(2024, 1, 1)],
        t=[dt.datetime(2024, 1, 1, 12)],
    )
    result, types = prepare_table("t", table, frozenset())
    assert types == {"d": "datetime", "t": "datetime"}


def test_columns_without_a_semantic_type_are_left_out_with_one_warning():
    table = _table(
        keep=[1],
        a=[dt.time(1)],
        b=[b"x"],
        c=pl.Series([[1]]),
        d=pl.Series([{"k": 1}]),
    )
    with pytest.warns(UnsupportedDtypeWarning) as caught:
        result, types = prepare_table("orders", table, frozenset())
    assert result.columns == ["keep"]
    assert types == {"keep": "numerical"}
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "'orders'" in message
    for name in "abcd":
        assert message.count(f"'{name}'") == 1


def test_key_columns_are_id_whatever_their_dtype():
    table = _table(
        pk=["a", "b"],
        fk=[dt.time(1), dt.time(2)],
        other=[1, 2],
    )
    result, types = prepare_table("t", table, frozenset({"pk", "fk"}))
    assert types == {"pk": "id", "fk": "id", "other": "numerical"}
    assert result.columns == ["pk", "fk", "other"]


def test_casting_to_string_first_keeps_a_boolean_categorical():
    table = _table(x=[True, False]).with_columns(nw.col("x").cast(nw.String))
    result, types = prepare_table("t", table, frozenset())
    assert types == {"x": "categorical"}


def test_the_input_columns_keep_their_order():
    table = _table(b=[True], a=[1], c=["x"])
    result, _ = prepare_table("t", table, frozenset())
    assert result.columns == ["b", "a", "c"]


def _backend_tables():
    polars = pl.DataFrame(
        {
            "flag": [True, False],
            "span": [dt.timedelta(seconds=2), dt.timedelta(milliseconds=250)],
            "at": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)],
            "name": ["a", "b"],
            "n": [1, 2],
        },
    )
    duckdb = pytest.importorskip("duckdb")
    pytest.importorskip("pandas")
    return {
        "polars": nw.from_native(polars),
        "pandas": nw.from_native(polars.to_pandas()),
        "duckdb": nw.from_native(duckdb.from_arrow(polars.to_arrow())).collect(),
    }


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_backends_give_the_same_types_and_values(backend):
    result, types = prepare_table("t", _backend_tables()[backend], frozenset())
    assert types == {
        "flag": "numerical",
        "span": "numerical",
        "at": "datetime",
        "name": "categorical",
        "n": "numerical",
    }
    assert result["flag"].to_list() == [1.0, 0.0]
    assert result["span"].to_list() == [2.0, 0.25]


def test_duckdb_reads_an_interval_as_unknown_so_the_column_is_left_out():
    table = _backend_tables()["duckdb"]
    with pytest.warns(UnsupportedDtypeWarning, match="'span'"):
        result, types = prepare_table("t", table, frozenset())
    assert types == {
        "flag": "numerical",
        "at": "datetime",
        "name": "categorical",
        "n": "numerical",
    }
    assert result["flag"].to_list() == [1.0, 0.0]
