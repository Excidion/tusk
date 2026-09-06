"""Cross-check tusk's binary transform primitives against featuretools.

The arithmetic four agree everywhere. The comparisons agree wherever both
inputs are known, and also on the two rows where one input is null: measured
directly (not assumed), both sides answer null there for all six comparison
primitives, the same "a null gives a null" contract SQL uses. That agreement
is asserted explicitly by
``test_comparisons_agree_where_an_input_is_null`` rather than taken on faith.

``modulo_numeric`` is built floored rather than as a plain ``%`` (polars
floors, duckdb truncates), which is also what featuretools' pandas
implementation does, so the two agree on negative operands. A zero divisor is
excluded from that agreement and asserted separately: it is backend-defined
per the primitive's own docstring, and tusk's float NaN and featuretools'
concrete 0 are two more entries in that same backend-defined set, not a bug
on either side. featuretools reaches 0 because ``pandas.arrays.IntegerArray``
does not raise or null out ``%`` by zero the way it does for ``/``.

featuretools names every one of these features with an infix operator
between the base feature names (``left > right``, ``left % right``), not
with ``PRIMITIVE_NAME(left, right)`` as the other differential files'
aggregation and unary features are named; ``_featuretools_column`` reads that
convention off ``theirs.columns`` rather than assuming it. This file does not
consume ``differential._as_tusk``, unlike the other differential files: that
helper translates featuretools' ``NAME(a, b)`` form, and these primitives are
named infix on the featuretools side, so there is no parenthesized name for
it to translate.

``multiply_numeric_boolean`` is exercised too, over its own fixture, since it
takes ``(NUMERIC, BOOLEAN)`` rather than two numeric columns.
``equal_categorical`` and ``not_equal_categorical`` are not: featuretools has
no counterpart for either, so there is nothing to differ against.

Run with: uv run --group validation pytest -m differential

Verified against featuretools 1.31.0.
"""

import pytest

import tusk

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pl = pytest.importorskip("polars")
featuretools = pytest.importorskip("featuretools")

pytestmark = pytest.mark.differential

ARITHMETIC = ["add_numeric", "subtract_numeric", "multiply_numeric", "divide_numeric"]
COMPARISONS = [
    "greater_than",
    "greater_than_equal_to",
    "less_than",
    "less_than_equal_to",
    "equal",
    "not_equal",
]

# featuretools spells every one of these as an infix operator between the
# base feature names ("left > right"), not as `PRIMITIVE_NAME(left, right)` --
# read off ``theirs.columns`` rather than assumed from the brief.
_FEATURETOOLS_OPERATOR = {
    "add_numeric": "+",
    "subtract_numeric": "-",
    "multiply_numeric": "*",
    "divide_numeric": "/",
    "greater_than": ">",
    "greater_than_equal_to": ">=",
    "less_than": "<",
    "less_than_equal_to": "<=",
    "equal": "=",
    "not_equal": "!=",
    "modulo_numeric": "%",
}


@pytest.fixture
def rows():
    """Numbers covering both signs, a zero divisor, equality, and nulls."""
    left = [7.0, -7.0, 2.0, 2.0, 1.0, None, 3.0]
    right = [2.0, 2.0, 2.0, -2.0, 0.0, 4.0, None]
    frame = pd.DataFrame(
        {
            "id": np.arange(1, len(left) + 1),
            "left": left,
            "right": right,
        },
    )
    _assert_rows_invariants(frame)
    return frame


def _assert_rows_invariants(frame):
    """Guard the cases ``rows`` is built to cover.

    Losing any one of these would leave a test passing over nothing.

    Args:
        frame: The table built by ``rows``.
    """
    both_known = frame["left"].notna() & frame["right"].notna()
    assert both_known.any()
    assert (~both_known).any()
    assert (frame["left"] < 0).any()
    assert (frame["right"] == 0).any()
    assert (frame["left"] == frame["right"]).any()


@pytest.mark.parametrize("primitive_name", ARITHMETIC)
def test_arithmetic_transforms_match_featuretools(rows, primitive_name):
    """The four that shipped untested, now held to featuretools' values."""
    ours, theirs = _both_matrices(rows, primitive_name)
    assert _numbers(ours[_tusk_column(primitive_name)]) == _numbers(
        theirs[_featuretools_column(primitive_name)],
    )


@pytest.mark.parametrize("primitive_name", COMPARISONS)
def test_comparisons_match_featuretools_where_nothing_is_null(rows, primitive_name):
    """Wherever both inputs are known, tusk and featuretools agree."""
    ours, theirs = _both_matrices(rows, primitive_name)
    known = rows.set_index("id")[["left", "right"]].notna().all(axis=1)
    assert _nullable(ours[_tusk_column(primitive_name)][known]) == _nullable(
        theirs[_featuretools_column(primitive_name)][known],
    )


@pytest.mark.parametrize("primitive_name", COMPARISONS)
def test_comparisons_agree_where_an_input_is_null(rows, primitive_name):
    """A null operand gives a null answer on both sides, for every comparison."""
    ours, theirs = _both_matrices(rows, primitive_name)
    unknown = ~rows.set_index("id")[["left", "right"]].notna().all(axis=1)
    assert unknown.any()
    assert ours[_tusk_column(primitive_name)][unknown].isna().all()
    assert theirs[_featuretools_column(primitive_name)][unknown].isna().all()


def test_modulo_matches_featuretools_on_negative_operands(rows):
    """Both floor, so -7 % 2 is 1 and 7 % -2 is -1 on each side.

    The zero-divisor row is excluded here and asserted on its own in
    ``test_modulo_diverges_on_a_zero_divisor`` instead; the null-operand row
    is excluded too, since it has no operand sign to compare.
    """
    ours, theirs = _both_matrices(rows, "modulo_numeric")
    operands = rows.set_index("id")[["left", "right"]]
    known_nonzero_divisor = operands.notna().all(axis=1) & (operands["right"] != 0)
    assert (~known_nonzero_divisor).any()
    assert _numbers(
        ours[_tusk_column("modulo_numeric")][known_nonzero_divisor],
    ) == _numbers(
        theirs[_featuretools_column("modulo_numeric")][known_nonzero_divisor],
    )


@pytest.fixture
def numeric_and_boolean_rows():
    """A numeric column and a boolean column, each with a null of its own.

    ``multiply_numeric_boolean`` takes ``(NUMERIC, BOOLEAN)``, a shape the
    ``rows`` fixture above cannot cover since both its columns are numeric.
    """
    value = [2.0, 1.0, 2.0, None, 5.0]
    flag = pd.array([True, True, False, True, None], dtype="boolean")
    frame = pd.DataFrame(
        {
            "id": np.arange(1, len(value) + 1),
            "value": value,
            "flag": flag,
        },
    )
    _assert_numeric_and_boolean_rows_invariants(frame)
    return frame


def _assert_numeric_and_boolean_rows_invariants(frame):
    """Guard the cases ``numeric_and_boolean_rows`` is built to cover.

    Args:
        frame: The table built by ``numeric_and_boolean_rows``.
    """
    assert (frame["flag"] == True).any()  # noqa: E712
    assert (frame["flag"] == False).any()  # noqa: E712
    assert frame["value"].isna().any()
    assert frame["flag"].isna().any()


def test_multiply_numeric_boolean_matches_featuretools(numeric_and_boolean_rows):
    """A flag masks the number the same way on both sides, null included."""
    ours, theirs = _both_matrices(numeric_and_boolean_rows, "multiply_numeric_boolean")
    theirs_column = _featuretools_column_for_multiply_numeric_boolean(theirs)
    assert _numbers(ours["MULTIPLY_NUMERIC_BOOLEAN__value__flag"]) == _numbers(
        theirs[theirs_column],
    )


def _featuretools_column_for_multiply_numeric_boolean(theirs):
    """Find featuretools' output column, whichever operand order it chose.

    featuretools names this feature after the alphabetical order of the base
    feature names rather than the primitive's declared argument order, so
    ``theirs.columns`` rather than the ``(value, flag)`` order tusk uses.

    Args:
        theirs: The featuretools matrix built for this primitive.

    Returns:
        The one column spelling the product of ``value`` and ``flag``.
    """
    candidates = [
        name for name in theirs.columns if name in {"value * flag", "flag * value"}
    ]
    assert len(candidates) == 1
    return candidates[0]


def test_modulo_diverges_on_a_zero_divisor(rows):
    """A zero divisor is backend-defined, per ``ModuloNumeric``'s own docstring.

    tusk floors a float remainder of NaN back to NaN, the value plain ``%``
    already gives on polars; featuretools' pandas nullable-Int64 columns
    give 0 instead, because ``IntegerArray.__mod__`` does not propagate a
    zero divisor into a null the way ``__truediv__`` does. Neither engine is
    wrong; they are simply two more entries in the same backend-defined set.
    """
    ours, theirs = _both_matrices(rows, "modulo_numeric")
    zero_divisor = rows.set_index("id")["right"] == 0
    assert zero_divisor.any()
    assert ours[_tusk_column("modulo_numeric")][zero_divisor].isna().all()
    assert (theirs[_featuretools_column("modulo_numeric")][zero_divisor] == 0).all()


def _tusk_column(primitive_name):
    """Name tusk's left-right output column for one binary primitive.

    Args:
        primitive_name: The primitive's name.

    Returns:
        The column name tusk gives the ``(left, right)`` feature.
    """
    return f"{primitive_name.upper()}__left__right"


def _featuretools_column(primitive_name):
    """Name featuretools' left-right output column for one binary primitive.

    Args:
        primitive_name: The primitive's name.

    Returns:
        The column name featuretools gives the ``(left, right)`` feature.
    """
    return f"left {_FEATURETOOLS_OPERATOR[primitive_name]} right"


def _numbers(series):
    """Render a numeric column as a comparable list.

    NaN is mapped to None so that a NaN on one side equals a NaN on the
    other; ``float('nan') != float('nan')`` would otherwise fail the
    comparison against itself.

    Args:
        series: A numeric column from either matrix.

    Returns:
        One float or None per row.
    """
    return [None if pd.isna(value) else float(value) for value in series]


def _nullable(series):
    """Render a boolean column as a comparable list, with nulls as None.

    Args:
        series: A boolean column from either matrix.

    Returns:
        One True, False or None per row.
    """
    return [None if pd.isna(value) else bool(value) for value in series]


def _both_matrices(frame, primitive_name):
    """Build the same one-primitive matrix on both sides.

    Args:
        frame: The table to build features over.
        primitive_name: The primitive's name, spelled the same way by tusk
            and featuretools.

    Returns:
        A tuple of the tusk matrix and the featuretools matrix, both indexed
        by the table's primary key.
    """
    return (
        _tusk_matrix(frame, primitive_name),
        _featuretools_matrix(frame, primitive_name),
    )


def _featuretools_matrix(frame, primitive_name):
    """Run one transform primitive through featuretools and return its matrix.

    Args:
        frame: The table to build features over.
        primitive_name: The primitive's featuretools name, the only entry in
            ``trans_primitives`` so exactly one feature is built.

    Returns:
        The feature matrix, sorted by the table's index.
    """
    es = featuretools.EntitySet("rows")
    es = es.add_dataframe(dataframe_name="rows", dataframe=frame, index="id")
    matrix, _ = featuretools.dfs(
        entityset=es,
        target_dataframe_name="rows",
        agg_primitives=[],
        trans_primitives=[primitive_name],
        max_depth=1,
    )
    return matrix.sort_index()


def _tusk_matrix(frame, primitive_name):
    """Run one transform primitive through tusk and return its matrix.

    Args:
        frame: The table to build features over.
        primitive_name: The primitive's name, the only entry in
            ``trans_primitives`` so exactly one feature is built.

    Returns:
        The feature matrix, collected and sorted by the table's primary key.
    """
    db = tusk.Database("rows_db").add_table(
        "rows",
        pl.from_pandas(frame).lazy(),
        primary_key="id",
    )
    matrix, _ = tusk.deep_feature_synthesis(
        database=db,
        target_table="rows",
        agg_primitives=[],
        trans_primitives=[primitive_name],
        max_depth=1,
    )
    return matrix.collect().sort("id").to_pandas().set_index("id")
