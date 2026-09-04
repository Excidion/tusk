import narwhals as nw
import pytest

from tusk.plotting import render_column_name, render_dtype, render_table_name


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (nw.Int64(), "Int64"),
        (nw.String(), "String"),
        (nw.Float64(), "Float64"),
        (nw.Date(), "Date"),
        (nw.Datetime(time_unit="us"), "Datetime[us]"),
        (nw.Datetime(time_unit="ns", time_zone="UTC"), "Datetime[ns-UTC]"),
        (
            nw.Datetime(time_unit="ns", time_zone="America/New_York"),
            "Datetime[ns-America_New_York]",
        ),
        (nw.Duration(time_unit="ms"), "Duration[ms]"),
        (nw.List(nw.Int64()), "List[Int64]"),
        (nw.List(nw.List(nw.Int64())), "List[List[Int64]]"),
        (nw.Enum(["a", "b", "c"]), "Enum[3]"),
        (nw.Struct({"x": nw.Int64(), "y": nw.Int64()}), "Struct[2]"),
    ],
)
def test_dtype_renders_as_a_mermaid_safe_token(dtype, expected):
    assert render_dtype(dtype) == expected


def test_timezone_punctuation_is_replaced():
    # Slashes, plus signs and colons are all parse errors in Mermaid's type
    # slot, so every character outside the safe set collapses to underscore.
    rendered = render_dtype(nw.Datetime(time_unit="ns", time_zone="UTC+02:00"))
    assert rendered == "Datetime[ns-UTC_02_00]"


FIGURE_SPACE = " "


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("amount", "amount"),
        ("total_2024", "total_2024"),
        ("n-items", "n-items"),
        ("a.b", "a.b"),
        ("straße", "straße"),
        ("unit price", f"unit{FIGURE_SPACE}price"),
        ("2024_total", "_2024_total"),
        ("2024 total", f"_2024{FIGURE_SPACE}total"),
    ],
)
def test_column_name_is_made_parseable(name, expected):
    assert render_column_name(name) == expected


def test_table_name_is_quoted():
    # Quoting is what lets a table name contain a space, which an attribute
    # name cannot.
    assert render_table_name("order items") == '"order items"'
