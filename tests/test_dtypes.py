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


def test_duration_is_temporal_but_not_has_date():
    """A Duration is temporal, yet must not match the calendar family.

    Calendar primitives declare HAS_DATE. Before the split they declared
    TEMPORAL, which a Duration satisfies, so DFS generated YEAR(duration)
    and polars raised InvalidOperationError.
    """
    duration = nw.Duration()
    assert matches(duration, DtypeFamily.TEMPORAL)
    assert matches(duration, DtypeFamily.DURATION)
    assert not matches(duration, DtypeFamily.HAS_DATE)


def test_datetime_and_date_have_date_but_not_duration():
    for dtype in (nw.Datetime(), nw.Date()):
        assert matches(dtype, DtypeFamily.TEMPORAL)
        assert matches(dtype, DtypeFamily.HAS_DATE)
        assert not matches(dtype, DtypeFamily.DURATION)


def test_has_time_matches_datetime_and_time_but_not_date_or_duration():
    """A Date carries no time of day, so it must not match HAS_TIME.

    Hour declares HAS_TIME. Before this family existed it declared
    DATETIME, which a Date satisfies, so DFS generated HOUR(date) and polars
    raised InvalidOperationError.
    """
    assert matches(nw.Datetime(), DtypeFamily.HAS_TIME)
    assert matches(nw.Time(), DtypeFamily.HAS_TIME)
    assert not matches(nw.Date(), DtypeFamily.HAS_TIME)
    assert not matches(nw.Duration(), DtypeFamily.HAS_TIME)


def test_has_date_excludes_time():
    """A bare time of day carries no calendar date to read year/month from."""
    assert not matches(nw.Time(), DtypeFamily.HAS_DATE)
