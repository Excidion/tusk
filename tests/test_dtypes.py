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


def test_matches_unknown_family():
    """Test that ValueError is raised for an unrecognized family."""
    with pytest.raises(ValueError, match="Unrecognized DtypeFamily"):
        matches(nw.Int64(), object())  # type: ignore


def test_categorical_family_matches_categorical_and_enum():
    import polars as pl

    frame = nw.from_native(
        pl.DataFrame(
            {
                "s": ["t"],
                "c": pl.Series(["x"]).cast(pl.Categorical),
                "e": pl.Series(["a"]).cast(pl.Enum(["a", "b"])),
            },
        ),
        eager_only=True,
    )
    schema = frame.schema
    assert matches(schema["c"], DtypeFamily.CATEGORICAL)
    assert matches(schema["e"], DtypeFamily.CATEGORICAL)
    assert not matches(schema["s"], DtypeFamily.CATEGORICAL)


def test_string_family_still_excludes_categorical_and_enum():
    import polars as pl

    frame = nw.from_native(
        pl.DataFrame(
            {
                "s": ["t"],
                "c": pl.Series(["x"]).cast(pl.Categorical),
                "e": pl.Series(["a"]).cast(pl.Enum(["a", "b"])),
            },
        ),
        eager_only=True,
    )
    schema = frame.schema
    assert matches(schema["s"], DtypeFamily.STRING)
    assert not matches(schema["c"], DtypeFamily.STRING)
    assert not matches(schema["e"], DtypeFamily.STRING)
