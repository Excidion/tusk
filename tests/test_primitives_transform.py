import datetime as dt
from zoneinfo import ZoneInfo

import narwhals as nw
import polars as pl
import pytest

import tusk
from tusk.dtypes import DtypeFamily
from tusk.exceptions import ValidationError
from tusk.primitives.base import NeedsCutoffTime, TransformPrimitive
from tusk.primitives.registry import resolve
from tusk.primitives.transform import TRANS_DEFAULTS, TimeSince, TimeSincePrevious
from tusk.synthesis import synthesize


@pytest.fixture
def lf():
    # Rows shuffled so t is NOT monotonic in row order.
    # Row 0: t=3/9, Row 1: t=3/4, Row 2: t=3/10
    # When order_by="t", processes in order: Row 1 (3/4), Row 0 (3/9), Row 2 (3/10)
    return nw.from_native(
        pl.LazyFrame(
            {
                "g": [1, 1, 1],
                "t": [
                    dt.datetime(2024, 3, 9, 6),
                    dt.datetime(2024, 3, 4, 5),
                    dt.datetime(2024, 3, 10, 7),
                ],
                "v": [3.0, -2.0, 4.0],
                "w": [1.0, 1.0, 2.0],
            },
        ),
    )


def _apply(lf, name, *columns):
    primitive = resolve(name)
    expr = primitive.outputs(*[nw.col(c) for c in columns])[0]
    if isinstance(primitive, TransformPrimitive) and primitive.order_dependent:
        expr = expr.over(order_by="t")
    return lf.with_columns(expr.alias("o")).collect().to_native()["o"].to_list()


@pytest.mark.parametrize(
    ("name", "columns", "expected"),
    [
        ("year", ("t",), [2024, 2024, 2024]),
        ("month", ("t",), [3, 3, 3]),
        ("day", ("t",), [9, 4, 10]),
        ("hour", ("t",), [6, 5, 7]),
        ("weekday", ("t",), [6, 1, 7]),
        ("is_weekend", ("t",), [True, False, True]),
        ("absolute", ("v",), [3.0, 2.0, 4.0]),
        ("add_numeric", ("v", "w"), [4.0, -1.0, 6.0]),
        ("subtract_numeric", ("v", "w"), [2.0, -3.0, 2.0]),
        ("multiply_numeric", ("v", "w"), [3.0, -2.0, 8.0]),
        ("divide_numeric", ("v", "w"), [3.0, -2.0, 2.0]),
    ],
)
def test_row_wise_transforms(lf, name, columns, expected):
    assert _apply(lf, name, *columns) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # v in t-order (Row 1, Row 0, Row 2): [-2.0, 3.0, 4.0]
        # cum_sum in t-order: [-2.0, 1.0, 5.0]; row order: [1.0, -2.0, 5.0]
        ("cum_sum", [1.0, -2.0, 5.0]),
        # cum_max in t-order: [-2.0, 3.0, 4.0]; row order: [3.0, -2.0, 4.0]
        ("cum_max", [3.0, -2.0, 4.0]),
        # cum_min in t-order: [-2.0, -2.0, -2.0]; row order: [-2.0, -2.0, -2.0]
        ("cum_min", [-2.0, -2.0, -2.0]),
        # diff in t-order: [None, 5.0, 1.0]; row order: [5.0, None, 1.0]
        ("diff", [5.0, None, 1.0]),
    ],
)
def test_order_dependent_transforms(lf, name, expected):
    assert _apply(lf, name, "v") == expected


def test_order_dependent_primitives_are_flagged():
    cum_sum = resolve("cum_sum")
    assert isinstance(cum_sum, TransformPrimitive)
    assert cum_sum.order_dependent is True
    month = resolve("month")
    assert isinstance(month, TransformPrimitive)
    assert month.order_dependent is False


def test_time_since_previous_is_a_timedelta(lf):
    # t in t-order: 3/4, 3/9, 3/10
    # Row 1 (3/4) is first: None
    # Row 0 (3/9): diff from Row 1 (3/4) = 5 days + 1 hour
    # Row 2 (3/10): diff from Row 0 (3/9) = 1 day + 1 hour
    got = _apply(lf, "time_since_previous", "t")
    assert got[1] is None
    assert got[0] == dt.timedelta(days=5, hours=1)
    assert got[2] == dt.timedelta(days=1, hours=1)


def test_time_since_previous_returns_a_duration():
    """One type for every elapsed-time primitive in tusk.

    featuretools returns float seconds here; tusk returns a Duration, the
    same type time_since returns, and leaves the choice of unit to the
    caller.
    """
    assert TimeSincePrevious().output_dtype == nw.Duration


def test_arithmetic_commutativity_flags():
    add = resolve("add_numeric")
    assert add.commutative is True
    subtract = resolve("subtract_numeric")
    assert subtract.commutative is False


def test_defaults_exclude_arithmetic():
    assert TRANS_DEFAULTS == ("year", "month", "weekday")


def test_time_since_measures_from_the_cutoff_time():
    """A past timestamp gives a positive duration, a future one negative."""
    frame = nw.from_native(
        pl.LazyFrame(
            {"t": [dt.datetime(2024, 1, 1), dt.datetime(2024, 6, 1), None]},
        ),
    )
    primitive = TimeSince()
    expr = primitive.outputs(nw.col("t"), cutoff_time=dt.datetime(2024, 3, 1))[0]
    got = frame.select(expr.alias("o")).collect()
    assert got.to_native()["o"].to_list() == [
        dt.timedelta(days=60),
        dt.timedelta(days=-92),
        None,
    ]


def test_time_since_needs_a_has_date_input():
    assert TimeSince().input_dtypes == (DtypeFamily.HAS_DATE,)
    assert isinstance(TimeSince(), NeedsCutoffTime)


def test_time_since_cannot_be_built_without_a_cutoff_time():
    """There is no unbound state to guard against: the argument is required."""
    with pytest.raises(TypeError, match="cutoff_time"):
        TimeSince().outputs(nw.col("t"))  # ty: ignore[missing-argument]


def test_deep_feature_synthesis_rejects_time_since_without_a_cutoff_time(db):
    """A cutoff_time is required, so there is no answer without one."""
    with pytest.raises(ValidationError, match="time_since needs a cutoff_time"):
        tusk.deep_feature_synthesis(
            database=db,
            target_table="customers",
            agg_primitives=[],
            trans_primitives=["time_since"],
            max_depth=1,
        )


def test_cutoff_time_is_not_a_constructor_argument():
    """A primitive never stores the cutoff time, so there is nothing to set.

    This is what lets one FeatureList be applied at several cutoff times: the
    value reaches the expression at build time and is never written down.
    """
    # Passed as **kwargs because a type checker rejects the literal call
    # before it can ever run.
    with pytest.raises(TypeError, match="cutoff_time"):
        TimeSince(**{"cutoff_time": dt.datetime(2024, 3, 1)})


def test_time_since_rejects_a_mismatched_cutoff_with_no_row_creation_time():
    """time_since measures against cutoff_time even with no row_creation_time
    declared, so the tz mismatch must surface as ValidationError -- not as a
    raw backend error naming a narwhals expression.
    """
    db = tusk.Database("x").add_table(
        "events",
        pl.LazyFrame(
            {"id": [1], "at": [dt.datetime(2024, 1, 1, tzinfo=ZoneInfo("UTC"))]},
        ),
        primary_key="id",
    )
    with pytest.raises(ValidationError, match="tz-naive"):
        matrix, _ = tusk.deep_feature_synthesis(
            database=db,
            target_table="events",
            agg_primitives=[],
            trans_primitives=["time_since"],
            max_depth=1,
            cutoff_time=dt.datetime(2026, 1, 1),
        )
        matrix.collect()


def test_deep_feature_synthesis_computes_time_since_with_a_cutoff_time(db):
    matrix, _ = tusk.deep_feature_synthesis(
        database=db,
        target_table="customers",
        agg_primitives=[],
        trans_primitives=["time_since"],
        max_depth=1,
        cutoff_time=dt.datetime(2024, 3, 1),
    )
    got = matrix.collect().sort("id")
    # customers.signed_up_at is 2024-01-01 for all three rows (tests/conftest.py)
    assert got["TIME_SINCE__signed_up_at"].to_list() == [dt.timedelta(days=60)] * 3


@pytest.fixture
def booleans():
    """Every pairing of True, False and null, so the truth table is complete."""
    return nw.from_native(
        pl.LazyFrame(
            {
                "a": [True, True, True, False, False, False, None, None, None],
                "b": [True, False, None, True, False, None, True, False, None],
            },
        ),
    )


def test_and_follows_three_valued_logic(booleans):
    """A null is unknown, not a third value: False AND unknown is still False.

    featuretools propagates the null here; every SQL backend tusk pushes down
    to does not.
    """
    assert _apply(booleans, "and", "a", "b") == [
        True,
        False,
        None,
        False,
        False,
        False,
        None,
        False,
        None,
    ]


def test_or_follows_three_valued_logic(booleans):
    """True OR unknown is True, whatever the unknown turns out to be."""
    assert _apply(booleans, "or", "a", "b") == [
        True,
        True,
        True,
        True,
        False,
        None,
        True,
        None,
        None,
    ]


def test_not_leaves_a_null_null(booleans):
    assert _apply(booleans, "not", "a") == [
        False,
        False,
        False,
        True,
        True,
        True,
        None,
        None,
        None,
    ]


def test_boolean_primitives_take_and_return_booleans():
    assert resolve("and").input_dtypes == (DtypeFamily.BOOLEAN, DtypeFamily.BOOLEAN)
    assert resolve("or").input_dtypes == (DtypeFamily.BOOLEAN, DtypeFamily.BOOLEAN)
    assert resolve("not").input_dtypes == (DtypeFamily.BOOLEAN,)
    for name in ("and", "or", "not"):
        assert resolve(name).output_dtype == nw.Boolean


def test_and_and_or_are_commutative_but_not_is_not_stacked_on_itself():
    """NOT(NOT(x)) is x, so synthesizing it would only cost a column."""
    assert resolve("and").commutative is True
    assert resolve("or").commutative is True
    assert resolve("not").stack_on_self is False


def test_deep_feature_synthesis_builds_the_boolean_transforms():
    """AND and OR are commutative, so each pair of columns is generated once."""
    db = tusk.Database("flags").add_table(
        "accounts",
        pl.LazyFrame(
            {
                "id": [1, 2, 3],
                "is_active": [True, False, None],
                "is_verified": [True, True, False],
            },
        ),
        primary_key="id",
    )
    matrix, _ = tusk.deep_feature_synthesis(
        database=db,
        target_table="accounts",
        agg_primitives=[],
        trans_primitives=["and", "or", "not"],
        max_depth=1,
    )
    got = matrix.collect().sort("id")
    assert got["AND__is_active__is_verified"].to_list() == [True, False, False]
    assert got["OR__is_active__is_verified"].to_list() == [True, True, None]
    assert got["NOT__is_active"].to_list() == [False, True, None]
    assert "AND__is_verified__is_active" not in got.columns


@pytest.fixture
def comparable():
    return nw.from_native(
        pl.LazyFrame(
            {
                "left": [1.0, 2.0, 3.0, None],
                "right": [3.0, 2.0, 1.0, 1.0],
                "earlier": [
                    dt.datetime(2024, 1, 1),
                    dt.datetime(2024, 1, 2),
                    dt.datetime(2024, 1, 3),
                    None,
                ],
                "later": [
                    dt.datetime(2024, 1, 2),
                    dt.datetime(2024, 1, 2),
                    dt.datetime(2024, 1, 1),
                    dt.datetime(2024, 1, 1),
                ],
                "word_left": ["a", "b", "a", None],
                "word_right": ["b", "b", "a", "a"],
                "flag_left": [True, True, False, None],
                "flag_right": [False, True, False, True],
            },
        ),
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("greater_than", [False, False, True, None]),
        ("greater_than_equal_to", [False, True, True, None]),
        ("less_than", [True, False, False, None]),
        ("less_than_equal_to", [True, True, False, None]),
        ("equal", [False, True, False, None]),
        ("not_equal", [True, False, True, None]),
    ],
)
def test_comparisons_compare_numbers(comparable, name, expected):
    assert _apply(comparable, name, "left", "right") == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("greater_than", [False, False, True, None]),
        ("less_than", [True, False, False, None]),
        ("equal", [False, True, False, None]),
    ],
)
def test_comparisons_compare_datetimes(comparable, name, expected):
    assert _apply(comparable, name, "earlier", "later") == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("equal", [False, True, True, None]),
        ("not_equal", [True, False, False, None]),
    ],
)
def test_equal_and_not_equal_compare_strings(comparable, name, expected):
    assert _apply(comparable, name, "word_left", "word_right") == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("equal", [False, True, True, None]),
        ("not_equal", [True, False, False, None]),
    ],
)
def test_equal_and_not_equal_compare_booleans(comparable, name, expected):
    assert _apply(comparable, name, "flag_left", "flag_right") == expected


def test_comparisons_take_numeric_or_datetime_pairs():
    for name in (
        "greater_than",
        "greater_than_equal_to",
        "less_than",
        "less_than_equal_to",
    ):
        assert resolve(name).signatures == (
            (DtypeFamily.NUMERIC, DtypeFamily.NUMERIC),
            (DtypeFamily.HAS_DATE, DtypeFamily.HAS_DATE),
        )
        assert resolve(name).output_dtype == nw.Boolean
        assert resolve(name).commutative is False


def test_equality_also_takes_booleans_and_strings():
    """Equality is meaningful for every type that compares on both backends."""
    for name in ("equal", "not_equal"):
        assert resolve(name).signatures == (
            (DtypeFamily.NUMERIC, DtypeFamily.NUMERIC),
            (DtypeFamily.HAS_DATE, DtypeFamily.HAS_DATE),
            (DtypeFamily.BOOLEAN, DtypeFamily.BOOLEAN),
            (DtypeFamily.STRING, DtypeFamily.STRING),
        )
        assert resolve(name).commutative is True


def test_comparing_a_number_with_a_string_is_never_synthesized():
    """greater_than has no (NUMERIC, STRING) shape, so amount and label never pair.

    A schema needs two numeric columns for this to be a meaningful check:
    with only one, ``greater_than`` would generate nothing at all, and the
    negative assertion would hold vacuously regardless of whether pairing
    across families were actually prevented. ``quantity`` gives it a
    numeric partner so a GREATER_THAN feature is proven to exist before its
    absence with ``label`` is proven.

    A plain ``"label" in name`` check would also match the identity feature
    named ``label`` itself, which synthesize always includes regardless of
    the requested transforms -- so the negative assertion targets
    GREATER_THAN features specifically.
    """
    db = tusk.Database("mixed").add_table(
        "events",
        pl.LazyFrame(
            {"id": [1], "amount": [1.0], "quantity": [2.0], "label": ["x"]},
        ),
        primary_key="id",
    )
    names = {
        f.name
        for f in synthesize(
            database=db,
            target_table="events",
            agg_primitives=[],
            trans_primitives=["greater_than"],
            max_depth=1,
        )
    }
    assert "GREATER_THAN__amount__quantity" in names
    assert not any("GREATER_THAN" in name and "label" in name for name in names)


@pytest.fixture
def signed():
    return nw.from_native(
        pl.LazyFrame(
            {
                "v": [7.0, -7.0, 7.0, -7.0, 1.0],
                "w": [2.0, 2.0, -2.0, -2.0, 0.0],
                "flag": [True, False, None, True, False],
            },
        ),
    )


def test_modulo_floors_rather_than_truncating(signed):
    """Python's rule, not C's: the result takes the divisor's sign.

    polars floors and duckdb truncates, so a plain % would mean two different
    things depending on the backend.
    """
    assert _apply(signed, "modulo_numeric", "v", "w")[:4] == [1.0, 1.0, -1.0, -1.0]


def test_modulo_by_zero_is_a_polars_float_artefact(signed):
    """`% 0` has no floored answer, and the backends do not even agree on
    dtype: duckdb nulls it for both ints and floats, polars nulls it for
    ints but produces NaN for floats. This only pins the float case, which
    is what `signed` supplies; see `ModuloNumeric`'s docstring for the rest.
    """
    result = _apply(signed, "modulo_numeric", "v", "w")[4]
    assert result != result  # NaN


@pytest.fixture
def narrow_integers():
    return nw.from_native(
        pl.LazyFrame(
            {"v": [20000], "w": [30000]},
            schema={"v": pl.Int16, "w": pl.Int16},
        ),
    )


def test_modulo_does_not_overflow_the_input_width(narrow_integers):
    """20000 % 30000 is 20000 either way, but 20000 + 30000 overflows Int16.

    A correction that runs even when the remainder already agrees with the
    divisor's sign would force this overflow for no reason.
    """
    assert _apply(narrow_integers, "modulo_numeric", "v", "w") == [20000]


def test_multiply_numeric_boolean_masks_the_number(signed):
    assert _apply(signed, "multiply_numeric_boolean", "v", "flag") == [
        7.0,
        -0.0,
        None,
        -7.0,
        0.0,
    ]


def test_modulo_and_masking_declare_their_inputs():
    assert resolve("modulo_numeric").signatures == (
        (DtypeFamily.NUMERIC, DtypeFamily.NUMERIC),
    )
    assert resolve("multiply_numeric_boolean").signatures == (
        (DtypeFamily.NUMERIC, DtypeFamily.BOOLEAN),
    )


@pytest.fixture
def labels():
    """Two Enum columns whose member lists differ.

    That difference is the case the cast exists for: polars raises
    ``SchemaError: Enum mismatch`` on a direct comparison of these two.
    """
    return nw.from_native(
        pl.LazyFrame(
            {
                "status": pl.Series(
                    ["open", "closed", "open"],
                    dtype=pl.Enum(["open", "closed", "pending"]),
                ),
                "tier": pl.Series(
                    ["open", "open", None],
                    dtype=pl.Enum(["open", "closed"]),
                ),
            },
        ),
    )


def test_categorical_equality_compares_enums_with_different_members(labels):
    assert _apply(labels, "equal_categorical", "status", "tier") == [
        True,
        False,
        None,
    ]


def test_categorical_inequality_compares_enums_with_different_members(labels):
    assert _apply(labels, "not_equal_categorical", "status", "tier") == [
        False,
        True,
        None,
    ]


def test_comparing_the_enums_without_the_cast_would_raise(labels):
    """Guards the reason the primitive casts at all."""
    with pytest.raises(Exception, match="Enum"):
        labels.with_columns(
            (nw.col("status") == nw.col("tier")).alias("o"),
        ).collect()


def test_categorical_equality_takes_categorical_pairs():
    for name in ("equal_categorical", "not_equal_categorical"):
        assert resolve(name).signatures == (
            (DtypeFamily.CATEGORICAL, DtypeFamily.CATEGORICAL),
        )
        assert resolve(name).output_dtype == nw.Boolean
        assert resolve(name).commutative is True
