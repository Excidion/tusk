import narwhals as nw
import polars as pl
import pytest

from tusk.primitives.aggregation import AGG_DEFAULTS, Count, Quantiles
from tusk.primitives.registry import resolve


@pytest.fixture
def lf():
    return nw.from_native(
        pl.LazyFrame(
            {
                "g": [1, 1, 1, 2],
                "v": [1.0, 2.0, 6.0, 4.0],
                "b": [True, False, True, True],
                "s": ["a", "a", "b", "c"],
            },
        ),
    )


def _agg(lf, primitive, column):
    exprs = primitive.outputs(nw.col(column))
    named = [e.alias(f"o{i}") for i, e in enumerate(exprs)]
    got = lf.group_by("g").agg(*named).sort("g").collect().to_native()
    return got


@pytest.mark.parametrize(
    ("name", "column", "expected"),
    [
        ("sum", "v", 9.0),
        ("mean", "v", 3.0),
        ("min", "v", 1.0),
        ("max", "v", 6.0),
        ("median", "v", 2.0),
        ("n_unique", "s", 2),
        ("percent_true", "b", pytest.approx(2 / 3)),
    ],
)
def test_aggregations_over_group_one(lf, name, column, expected):
    got = _agg(lf, resolve(name), column)
    assert got["o0"][0] == expected


def test_count_takes_no_column_input(lf):
    assert Count().input_dtypes == ()
    got = (
        lf.group_by("g")
        .agg(Count().outputs()[0].alias("n"))
        .sort("g")
        .collect()
        .to_native()
    )
    assert got["n"].to_list() == [3, 1]


def test_count_defaults_empty_groups_to_zero():
    assert Count().default_value == 0
    assert Count().stack_on_self is False


def test_count_and_n_unique_declare_int64(lf):
    got = (
        lf.group_by("g")
        .agg(
            Count().outputs()[0].alias("n"),
            resolve("n_unique").outputs(nw.col("s"))[0].alias("u"),
        )
        .collect()
    )
    assert got.collect_schema()["n"] == nw.Int64
    assert got.collect_schema()["u"] == nw.Int64


def test_quantiles_is_multi_output(lf):
    q = Quantiles(qs=(0.0, 0.5, 1.0))
    assert q.number_of_outputs == 3
    assert q.output_names("QUANTILES__v") == (
        "QUANTILES__v__0",
        "QUANTILES__v__1",
        "QUANTILES__v__2",
    )
    got = _agg(lf, q, "v")
    assert [got["o0"][0], got["o1"][0], got["o2"][0]] == [1.0, 2.0, 6.0]


def test_defaults_are_the_documented_set():
    assert AGG_DEFAULTS == ("count", "sum", "mean", "min", "max", "std", "n_unique")
