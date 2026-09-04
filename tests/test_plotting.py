import narwhals as nw
import pytest

from tusk.plotting import render_dtype


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
