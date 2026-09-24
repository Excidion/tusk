from datetime import date, datetime, timedelta

import narwhals as nw
import polars as pl
import pytest
from aggregation_cases import (
    CHILDREN,
    EXPECTED,
    PARENTS,
    ROW_CREATION_TIME,
    assert_values_match,
)

import tusk
from tusk.exceptions import PrimitiveError
from tusk.primitives.aggregation import (
    AGG_DEFAULTS,
    Count,
    CountInsideNthStd,
    CountOutsideNthStd,
    Quantiles,
    TimeSinceLast,
)
from tusk.primitives.base import NeedsCutoffTime
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


def test_percent_true_counts_null_as_false():
    lf = nw.from_native(
        pl.LazyFrame({"g": [1, 1, 1, 1, 1], "b": [True, False, True, True, None]}),
    )
    got = _agg(lf, resolve("percent_true"), "b")
    assert got["o0"][0] == pytest.approx(0.6)


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


def test_quantiles_defaults_to_the_quartiles():
    # Quantiles interpolates linearly between the two nearest sorted values.
    # For four sorted values, quantile q sits at zero-indexed position 3*q,
    # so the default quartiles (0.25, 0.5, 0.75) each land strictly between
    # two of them: 10 + 0.75*(20-10) = 17.5, 20 + 0.5*(30-20) = 25.0, and
    # 30 + 0.25*(40-30) = 32.5. "lower"/"higher"/"nearest" interpolation
    # would instead return one of the surrounding data points, so these
    # values pin the "linear" behaviour, not just the default qs.
    lf = nw.from_native(
        pl.LazyFrame({"g": [1, 1, 1, 1], "v": [10.0, 20.0, 30.0, 40.0]}),
    )
    got = _agg(lf, Quantiles(), "v")
    assert [got["o0"][0], got["o1"][0], got["o2"][0]] == [17.5, 25.0, 32.5]


def test_defaults_are_the_documented_set():
    assert AGG_DEFAULTS == (
        "count",
        "sum",
        "mean",
        "min",
        "max",
        "std",
        "n_unique",
        "skew",
        "percent_true",
        "mode",
    )


@pytest.fixture
def timed_lf():
    return nw.from_native(
        pl.LazyFrame(
            {
                "g": [1, 1, 1, 2],
                "t": [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 5),
                    datetime(2024, 1, 9),
                    datetime(2024, 2, 1),
                ],
                "b": [True, False, None, False],
            },
        ),
    )


CUTOFF = datetime(2024, 1, 10)


def _agg_against_cutoff(lf, primitive, *columns):
    exprs = primitive.outputs(*[nw.col(c) for c in columns], cutoff_time=CUTOFF)
    named = [e.alias(f"o{i}") for i, e in enumerate(exprs)]
    return lf.group_by("g").agg(*named).sort("g").collect().to_native()


@pytest.mark.parametrize(
    ("name", "columns", "expected"),
    [
        ("time_since_first", ("t",), timedelta(days=9)),
        ("time_since_last", ("t",), timedelta(days=1)),
        ("time_since_last_true", ("t", "b"), timedelta(days=9)),
        ("time_since_last_false", ("t", "b"), timedelta(days=5)),
    ],
)
def test_time_since_aggregations_measure_against_the_cutoff(
    timed_lf,
    name,
    columns,
    expected,
):
    got = _agg_against_cutoff(timed_lf, resolve(name), *columns)
    assert got["o0"][0] == expected


@pytest.mark.parametrize(
    ("name", "columns"),
    [
        ("time_since_last_true", ("t", "b")),
        ("time_since_last_false", ("t", "b")),
    ],
)
def test_time_since_last_flag_is_null_without_a_matching_row(name, columns):
    lf = nw.from_native(
        pl.LazyFrame(
            {"g": [1, 1], "t": [datetime(2024, 1, 1)] * 2, "b": [None, None]},
            schema_overrides={"b": pl.Boolean},
        ),
    )
    got = _agg_against_cutoff(lf, resolve(name), *columns)
    assert got["o0"][0] is None


@pytest.mark.parametrize(
    "name",
    ["time_since_first", "time_since_last", "time_since_last_true"],
)
def test_time_since_aggregations_need_a_cutoff_time(name):
    assert isinstance(resolve(name), NeedsCutoffTime)


def test_time_since_aggregations_return_a_duration(timed_lf):
    got = timed_lf.group_by("g").agg(
        TimeSinceLast().outputs(nw.col("t"), cutoff_time=CUTOFF)[0].alias("d"),
    )
    assert got.collect_schema()["d"] == nw.Duration


@pytest.mark.parametrize("primitive_name", sorted(EXPECTED))
def test_standalone_aggregations_on_every_kind_of_group(primitive_name):
    column, dtype, expected = EXPECTED[primitive_name]
    database = (
        tusk.Database("cases")
        .add_table("parents", pl.from_pandas(PARENTS).lazy(), primary_key="id")
        .add_table(
            "children",
            pl.from_pandas(CHILDREN).lazy(),
            primary_key="id",
            row_creation_time=ROW_CREATION_TIME,
        )
        .add_relationship(parent="parents", child="children", foreign_key="parent_id")
    )
    matrix, _ = tusk.deep_feature_synthesis(
        database=database,
        target_table="parents",
        agg_primitives=[primitive_name],
        trans_primitives=[],
        max_depth=1,
    )
    assert nw.from_native(matrix).collect_schema()[column] == dtype
    got = matrix.collect().sort("id").to_pandas()[column].tolist()
    assert_values_match(got, expected)


def test_max_min_delta_preserves_input_dtype_on_integer_column():
    lf = nw.from_native(pl.LazyFrame({"g": [1, 1], "n": [3, 10]}))

    prim = resolve("max_min_delta")
    assert prim.return_dtype((nw.Int64,)) == nw.Int64

    got = _agg(lf, prim, "n")
    values = got["o0"].to_list()
    assert values == [7]


def test_first_last_time_delta_of_dates_is_a_duration():
    lf = nw.from_native(
        pl.LazyFrame({"g": [1, 1], "d": [date(2024, 1, 1), date(2024, 3, 2)]}),
    )
    got = _agg(lf, resolve("first_last_time_delta"), "d")
    assert got["o0"][0] == timedelta(days=61)


def _parents_and_children():
    return (
        tusk.Database("cases")
        .add_table("parents", pl.from_pandas(PARENTS).lazy(), primary_key="id")
        .add_table(
            "children",
            pl.from_pandas(CHILDREN).lazy(),
            primary_key="id",
            row_creation_time=ROW_CREATION_TIME,
            where={"small": nw.col("value") < 6.0},
        )
        .add_relationship(parent="parents", child="children", foreign_key="parent_id")
    )


def _parent_values(database, column, **dfs_arguments):
    matrix, _ = tusk.deep_feature_synthesis(
        database=database,
        target_table="parents",
        trans_primitives=[],
        max_depth=1,
        **dfs_arguments,
    )
    return matrix.collect().sort("id")[column].to_list()


@pytest.mark.parametrize("primitive", [CountInsideNthStd, CountOutsideNthStd])
def test_a_band_of_negative_width_is_rejected(primitive):
    with pytest.raises(PrimitiveError, match="n >= 0"):
        primitive(n=-1)


def test_a_wider_band_takes_in_values_the_default_leaves_out():
    """Parent 1's 6.0 lies 1.39 population deviations from its mean of 3."""
    matrix_values = [
        _parent_values(
            _parents_and_children(),
            f"{primitive.stem}__children__value",
            agg_primitives=[primitive],
        )
        for primitive in (CountInsideNthStd(n=3), CountOutsideNthStd(n=3))
    ]
    assert matrix_values == [[3, 2, 0, 4, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0]]


@pytest.mark.parametrize(
    ("primitive", "stem"),
    [
        (CountInsideNthStd(), "COUNT_INSIDE_1_STD"),
        (CountInsideNthStd(n=1.5), "COUNT_INSIDE_1_5_STD"),
        (CountOutsideNthStd(n=2.0), "COUNT_OUTSIDE_2_STD"),
    ],
)
def test_the_band_width_is_spelled_into_the_feature_name(primitive, stem):
    assert primitive.generate_name(["children", "value"]) == f"{stem}__children__value"
    assert (
        primitive.generate_display_name(["children.value"]) == f"{stem}(children.value)"
    )


def test_two_band_widths_give_two_features():
    matrix, _ = tusk.deep_feature_synthesis(
        database=_parents_and_children(),
        target_table="parents",
        agg_primitives=[CountInsideNthStd(n=1), CountInsideNthStd(n=2)],
        trans_primitives=[],
        max_depth=1,
    )
    assert {
        "COUNT_INSIDE_1_STD__children__value",
        "COUNT_INSIDE_2_STD__children__value",
    } <= set(matrix.collect_schema().names())


def test_a_conditioned_count_measures_against_the_mean_of_the_kept_rows():
    """Without its 6.0, parent 1's mean is 1.5, so only the 2.0 lies above it."""
    above = _parent_values(
        _parents_and_children(),
        "COUNT_ABOVE_MEAN__children__value__WHERE__small",
        agg_primitives=[],
        conditional_primitives=["count_above_mean"],
    )
    assert above == [1, 0, 0, 1, 0, 0, 0, 0]
