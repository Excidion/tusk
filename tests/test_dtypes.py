import narwhals as nw
import pytest

from tusk.dtypes import DtypeFamily, matches

F = DtypeFamily


@pytest.mark.parametrize(
    ("dtype", "family", "expected"),
    [
        (nw.Int64(), F.NUMERIC, True),
        (nw.Float64(), F.NUMERIC, True),
        (nw.Boolean(), F.NUMERIC, False),
        (nw.String(), F.NUMERIC, False),
        (nw.Datetime(), F.TEMPORAL, True),
        (nw.Date(), F.TEMPORAL, True),
        (nw.Int64(), F.TEMPORAL, False),
        (nw.String(), F.STRING, True),
        (nw.Int64(), F.STRING, False),
        (nw.Boolean(), F.BOOLEAN, True),
        (nw.Int64(), F.BOOLEAN, False),
        (nw.String(), F.ANY, True),
        (nw.Datetime(), F.ANY, True),
    ],
)
def test_matches(dtype, family, expected):
    assert matches(dtype, family) is expected
