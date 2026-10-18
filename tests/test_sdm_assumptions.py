"""What sdm accepts, which the export relies on."""

import datetime as dt
import decimal

import pyarrow as pa
import pytest

pytest.importorskip("sdm")
pytestmark = pytest.mark.sdm

from sdm import RelationalData, TableTensor  # noqa: E402
from sdm.relational import RelationalSampler  # noqa: E402


def _tensor(column, stype):
    return TableTensor.from_arrow(pa.table({"x": column}), {"x": stype})


def test_decimal_is_rejected_as_numerical():
    column = pa.array([decimal.Decimal("1.5")], pa.decimal128(10, 1))
    with pytest.raises(TypeError):
        _tensor(column, "numerical")


@pytest.mark.parametrize(
    "column",
    [
        pa.array([1.5, None]),
        pa.array([1, 2], pa.int8()),
        pa.array([1, 2], pa.uint64()),
    ],
)
def test_numbers_are_accepted_as_numerical(column):
    assert _tensor(column, "numerical").size(0) == len(column)


@pytest.mark.parametrize(
    "column",
    [
        pa.array([dt.date(2024, 1, 1), None]),
        pa.array([dt.datetime(2024, 1, 1), None]),
        pa.array([dt.datetime(2024, 1, 1)], pa.timestamp("us", tz="UTC")),
    ],
)
def test_dates_and_timestamps_are_accepted_as_datetime(column):
    assert _tensor(column, "datetime").size(0) == len(column)


@pytest.mark.parametrize(
    "column",
    [
        pa.array([1, 2], pa.int64()),
        pa.array([1, 2], pa.uint64()),
        pa.array(["a", "b"]),
        pa.array([1, None]),
    ],
)
def test_integers_and_strings_are_accepted_as_id(column):
    assert _tensor(column, "id").size(0) == len(column)


def test_a_semantic_type_for_an_absent_column_is_an_error():
    table = pa.table({"x": [1]})
    with pytest.raises(KeyError):
        TableTensor.from_arrow(table, {"x": "numerical", "y": "numerical"})


@pytest.mark.parametrize(
    "time",
    [
        pa.array([dt.date(2024, 1, 1), dt.date(2024, 1, 2)]),
        pa.array([dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)]),
    ],
)
def test_the_sampler_accepts_a_date_or_a_timestamp_as_a_time_column(time):
    pytest.importorskip("pyg_lib")
    parent = TableTensor.from_arrow(
        pa.table({"id": [1, 2], "t": time}),
        {"id": "id", "t": "datetime"},
    )
    child = TableTensor.from_arrow(pa.table({"pid": [1, 2]}), {"pid": "id"})
    relationship = {
        "left_table": "child",
        "left_column": "pid",
        "right_table": "parent",
        "right_column": "id",
    }
    data = RelationalData({"parent": parent, "child": child}, [relationship])
    assert RelationalSampler(data, {"parent": "t"}).time_columns == {"parent": "t"}
