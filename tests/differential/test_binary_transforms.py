"""Cross-check tusk's binary transform primitives against featuretools.

The arithmetic four agree everywhere. Over two Numeric columns, the
comparisons agree wherever both inputs are known, and also on the rows where
one input is null: both sides answer null there for all six comparison
primitives, the same "a null gives a null" contract SQL uses. That agreement
is asserted explicitly by
``test_comparisons_agree_where_an_input_is_null`` rather than taken on faith.

Over two Datetime columns -- the ``(HAS_DATE, HAS_DATE)`` signature the
comparisons also declare -- the six agree wherever both inputs are known, but
diverge where one is null: tusk still answers null, while featuretools'
pandas implementation compares ``NaT`` with plain ``datetime64[ns]``
semantics, where every ordering and equality comparison is false and only
``not_equal`` is true. ``test_datetime_comparisons_diverge_where_an_operand_is_null``
asserts both sides of that divergence rather than hiding either one.

``equal`` and ``not_equal`` alone declare two more signatures, ``(BOOLEAN,
BOOLEAN)`` and ``(STRING, STRING)``, each cross-checked over its own fixture.
Both agree with featuretools on a null operand, the same "a null gives a
null" contract the Numeric fixture confirms: featuretools' woodwork layer
keeps a Boolean pair on pandas' nullable ``boolean`` dtype and a String pair
(kept to enough distinct words to stay off low-cardinality inference) on
``Unknown`` (pandas nullable ``string``), and both dtypes' ``eq``/``ne``
propagate a null operand rather than settling the comparison the way plain
``datetime64[ns]`` or a ``pd.CategoricalDtype`` pair does.

``modulo_numeric`` is built floored rather than as a plain ``%`` (polars
floors, duckdb truncates), which is also what featuretools' pandas
implementation does, so the two agree on negative operands -- on both a
negative dividend (``-7 % 2``) and a negative divisor (``7 % -2``), the two
directions the sign-asymmetric floor correction distinguishes. A zero divisor
is excluded from that agreement and asserted separately: it is
backend-defined per the primitive's own docstring, and tusk's float NaN and
featuretools' concrete 0 are two more entries in that same backend-defined
set, not a bug on either side. featuretools reaches 0 because woodwork infers
the whole-number ``left``/``right`` columns as ``IntegerNullable``, so its
``%`` runs on pandas nullable ``Int64`` columns, where a zero divisor gives 0
rather than a null; tusk keeps the columns float64 the whole way, where ``%``
by zero is NaN.

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
takes ``(NUMERIC, BOOLEAN)`` rather than two numeric columns. tusk has no
``multiply_boolean`` of its own -- SQL has one conjunction, not two -- but
featuretools' ``multiply_boolean`` is cross-checked against tusk's ``and``
anyway, since it is the primitive that motivates the coverage-table claim
that the two agree on every row, nulls included (unlike featuretools' own
``and``; see ``tests/differential/test_boolean_transforms.py`` for that
divergence).

``equal_categorical`` and ``not_equal_categorical`` are cross-checked too,
against featuretools' generic ``equal``/``not_equal``: featuretools has no
primitive of the same name, but its unconstrained equality primitives accept
a pair of Categorical columns as readily as any other, so those are the
counterpart to differ against.

Run with: uv run --group validation pytest -m differential

Verified against featuretools 1.31.0.
"""

import datetime as dt

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
    "multiply_boolean": "*",
}


@pytest.fixture
def rows():
    """Numbers covering both signs, a zero divisor, equality, and nulls."""
    left = [7.0, -7.0, 2.0, 2.0, 1.0, None, 3.0, 7.0]
    right = [2.0, 2.0, 2.0, -2.0, 0.0, 4.0, None, -2.0]
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

    Losing any one of these would leave a test passing over nothing. The
    negative-divisor and nonzero-remainder pair guards against a fixture
    where every negative-divisor row happens to floor to a remainder of 0
    (as the pre-existing ``(2.0, -2.0)`` row does) — floor and truncation
    agree on 0, so such a fixture would discriminate nothing about
    ``ModuloNumeric``'s sign-asymmetric floor correction.

    Args:
        frame: The table built by ``rows``.
    """
    both_known = frame["left"].notna() & frame["right"].notna()
    assert both_known.any()
    assert (~both_known).any()
    assert (frame["left"] < 0).any()
    assert (frame["right"] == 0).any()
    assert (frame["right"] < 0).any()
    assert (frame["left"] == frame["right"]).any()
    remainder = frame["left"] % frame["right"]
    assert (remainder[frame["right"] < 0] != 0).any()


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


@pytest.fixture
def datetime_rows():
    """Two Datetime columns, ordered both ways, equal once, null once each."""
    left = pd.to_datetime(
        [
            dt.datetime(2024, 1, 1),
            dt.datetime(2024, 1, 5),
            dt.datetime(2024, 1, 3),
            None,
            dt.datetime(2024, 1, 2),
        ],
    )
    right = pd.to_datetime(
        [
            dt.datetime(2024, 1, 2),
            dt.datetime(2024, 1, 5),
            dt.datetime(2024, 1, 1),
            dt.datetime(2024, 1, 6),
            None,
        ],
    )
    frame = pd.DataFrame(
        {
            "id": np.arange(1, len(left) + 1),
            "left": left,
            "right": right,
        },
    )
    _assert_datetime_rows_invariants(frame)
    return frame


def _assert_datetime_rows_invariants(frame):
    """Guard the cases ``datetime_rows`` is built to cover.

    Args:
        frame: The table built by ``datetime_rows``.
    """
    both_known = frame["left"].notna() & frame["right"].notna()
    assert both_known.any()
    assert (~both_known).any()
    assert (frame["left"] > frame["right"]).any()
    assert (frame["left"] < frame["right"]).any()
    assert (frame["left"] == frame["right"]).any()


@pytest.mark.parametrize("primitive_name", COMPARISONS)
def test_datetime_comparisons_match_featuretools_where_nothing_is_null(
    datetime_rows,
    primitive_name,
):
    """The (HAS_DATE, HAS_DATE) signature agrees too, wherever both are known."""
    ours, theirs = _both_matrices(datetime_rows, primitive_name)
    known = datetime_rows.set_index("id")[["left", "right"]].notna().all(axis=1)
    assert _nullable(ours[_tusk_column(primitive_name)][known]) == _nullable(
        theirs[_featuretools_column(primitive_name)][known],
    )


@pytest.mark.parametrize("primitive_name", COMPARISONS)
def test_datetime_comparisons_diverge_where_an_operand_is_null(
    datetime_rows,
    primitive_name,
):
    """A null Datetime operand is answered differently on each side.

    tusk still gives a null, the same "a null gives a null" contract the
    Numeric fixture confirms. featuretools instead runs its comparisons over
    plain ``datetime64[ns]`` values, where pandas treats ``NaT`` as ordered
    nowhere and unequal everywhere: every ordering and equality comparison
    answers false, and only ``not_equal`` answers true.
    """
    ours, theirs = _both_matrices(datetime_rows, primitive_name)
    unknown = ~datetime_rows.set_index("id")[["left", "right"]].notna().all(axis=1)
    assert unknown.any()
    assert ours[_tusk_column(primitive_name)][unknown].isna().all()
    featuretools_answer_on_a_null_operand = primitive_name == "not_equal"
    assert (
        theirs[_featuretools_column(primitive_name)][unknown]
        == featuretools_answer_on_a_null_operand
    ).all()


EQUALITY = ["equal", "not_equal"]


@pytest.fixture
def boolean_pair_rows():
    """Two Boolean columns, each with a null of its own.

    ``equal``/``not_equal`` also declare a ``(BOOLEAN, BOOLEAN)`` signature,
    a shape neither ``rows`` nor ``datetime_rows`` above covers.
    """
    left = pd.array([True, True, False, False, None], dtype="boolean")
    right = pd.array([True, False, False, None, True], dtype="boolean")
    frame = pd.DataFrame(
        {
            "id": np.arange(1, len(left) + 1),
            "left": left,
            "right": right,
        },
    )
    _assert_boolean_pair_rows_invariants(frame)
    return frame


def _assert_boolean_pair_rows_invariants(frame):
    """Guard the cases ``boolean_pair_rows`` is built to cover.

    Args:
        frame: The table built by ``boolean_pair_rows``.
    """
    known = frame["left"].notna() & frame["right"].notna()
    assert (frame["left"][known] == frame["right"][known]).any()
    assert (frame["left"][known] != frame["right"][known]).any()
    assert frame["left"].isna().any()
    assert frame["right"].isna().any()


@pytest.mark.parametrize("primitive_name", EQUALITY)
def test_boolean_equality_matches_featuretools_where_nothing_is_null(
    boolean_pair_rows,
    primitive_name,
):
    """The (BOOLEAN, BOOLEAN) signature agrees wherever both flags are known."""
    ours, theirs = _both_matrices(boolean_pair_rows, primitive_name)
    known = boolean_pair_rows.set_index("id")[["left", "right"]].notna().all(axis=1)
    assert _nullable(ours[_tusk_column(primitive_name)][known]) == _nullable(
        theirs[_featuretools_column(primitive_name)][known],
    )


@pytest.mark.parametrize("primitive_name", EQUALITY)
def test_boolean_equality_agrees_with_featuretools_where_an_operand_is_null(
    boolean_pair_rows,
    primitive_name,
):
    """A null Boolean operand gives a null answer on both sides, unlike Datetime.

    featuretools' woodwork layer keeps this pair on pandas' nullable
    ``boolean`` dtype, whose ``eq``/``ne`` propagate a null operand instead
    of settling the comparison the way plain ``datetime64[ns]`` does.
    """
    ours, theirs = _both_matrices(boolean_pair_rows, primitive_name)
    unknown = ~boolean_pair_rows.set_index("id")[["left", "right"]].notna().all(axis=1)
    assert unknown.any()
    assert ours[_tusk_column(primitive_name)][unknown].isna().all()
    assert theirs[_featuretools_column(primitive_name)][unknown].isna().all()


@pytest.fixture
def string_pair_rows():
    """Two String columns, each with a null of its own.

    Eight distinct words across eight rows keep woodwork's inference on
    ``Unknown`` (pandas nullable ``string``) rather than ``Categorical`` --
    see ``test_string_equality_agrees_with_featuretools_where_an_operand_is_null``
    for why that distinction matters here. ``equal``/``not_equal`` also
    declare a ``(STRING, STRING)`` signature, a shape none of the fixtures
    above cover.
    """
    left = ["apple", "banana", "apple", "cherry", None, "banana", "date", None]
    right = ["apple", "grape", "apple", "cherry", "kiwi", None, "date", "fig"]
    frame = pd.DataFrame(
        {
            "id": np.arange(1, len(left) + 1),
            "left": left,
            "right": right,
        },
    )
    _assert_string_pair_rows_invariants(frame)
    return frame


def _assert_string_pair_rows_invariants(frame):
    """Guard the cases ``string_pair_rows`` is built to cover.

    Args:
        frame: The table built by ``string_pair_rows``.
    """
    known = frame["left"].notna() & frame["right"].notna()
    assert (frame["left"][known] == frame["right"][known]).any()
    assert (frame["left"][known] != frame["right"][known]).any()
    assert frame["left"].isna().any()
    assert frame["right"].isna().any()


@pytest.mark.parametrize("primitive_name", EQUALITY)
def test_string_equality_matches_featuretools_where_nothing_is_null(
    string_pair_rows,
    primitive_name,
):
    """The (STRING, STRING) signature agrees wherever both words are known."""
    ours, theirs = _both_matrices(string_pair_rows, primitive_name)
    known = string_pair_rows.set_index("id")[["left", "right"]].notna().all(axis=1)
    assert _nullable(ours[_tusk_column(primitive_name)][known]) == _nullable(
        theirs[_featuretools_column(primitive_name)][known],
    )


@pytest.mark.parametrize("primitive_name", EQUALITY)
def test_string_equality_agrees_with_featuretools_where_an_operand_is_null(
    string_pair_rows,
    primitive_name,
):
    """A null String operand gives a null answer on both sides too.

    woodwork infers this pair as ``Unknown``, so featuretools runs its
    generic ``equal``/``not_equal`` on pandas' nullable ``string`` dtype
    rather than routing through the ``pd.CategoricalDtype`` branch that
    unions category sets -- the branch that makes ``equal_categorical``'s
    comparison diverge on a null label (see ``labelled_rows`` above).
    Nullable-string ``eq``/``ne`` instead propagate the null, the same
    contract SQL and tusk use. The dtype assertion below pins that
    inference so a future fixture edit (e.g. lower cardinality, tipping
    woodwork toward ``Categorical``) fails here instead of silently
    invalidating this explanation.
    """
    assert _featuretools_input_dtype(string_pair_rows, "left") == pd.StringDtype()
    ours, theirs = _both_matrices(string_pair_rows, primitive_name)
    unknown = ~string_pair_rows.set_index("id")[["left", "right"]].notna().all(axis=1)
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
    nonzero_divisor = operands.notna().all(axis=1) & (operands["right"] != 0)
    assert nonzero_divisor.any()
    assert _numbers(
        ours[_tusk_column("modulo_numeric")][nonzero_divisor],
    ) == _numbers(
        theirs[_featuretools_column("modulo_numeric")][nonzero_divisor],
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


@pytest.fixture
def boolean_rows():
    """Every pairing of True, False and null across two boolean columns.

    Used to cross-check tusk's ``and`` against featuretools' differently
    named ``multiply_boolean``, a shape neither ``rows`` nor
    ``numeric_and_boolean_rows`` above covers.
    """
    left = [True, True, True, False, False, False, None, None, None]
    right = [True, False, None, True, False, None, True, False, None]
    frame = pd.DataFrame(
        {
            "id": np.arange(1, len(left) + 1),
            "left": pd.array(left, dtype="boolean"),
            "right": pd.array(right, dtype="boolean"),
        },
    )
    _assert_boolean_rows_invariants(frame)
    return frame


def _assert_boolean_rows_invariants(frame):
    """Guard the cases ``boolean_rows`` is built to cover.

    Args:
        frame: The table built by ``boolean_rows``.
    """
    known = frame["left"].notna() & frame["right"].notna()
    assert (frame["left"][known] & frame["right"][known]).any()
    assert (~frame["left"][known] & ~frame["right"][known]).any()
    assert frame["left"].isna().any()
    assert frame["right"].isna().any()


def test_and_matches_featuretools_multiply_boolean(boolean_rows):
    """tusk has no ``multiply_boolean``: its ``and`` already answers this way.

    featuretools has two boolean-conjunction primitives that disagree with
    each other on a null operand -- ``and`` (``np.logical_and``, which
    propagates the null; see ``tests/differential/test_boolean_transforms.py``
    for that divergence) and ``multiply_boolean`` (``np.bitwise_and``, which
    settles a null operand against a known ``False`` instead). tusk's ``and``
    agrees with the latter on every row, nulls included, because SQL has one
    conjunction, not two.
    """
    ours = _tusk_matrix(boolean_rows, "and")
    theirs = _featuretools_matrix(boolean_rows, "multiply_boolean")
    assert _nullable(ours[_tusk_column("and")]) == _nullable(
        theirs[_featuretools_column("multiply_boolean")],
    )


@pytest.fixture
def labelled_rows():
    """Two Categorical columns with different category sets, one null label.

    ``equal_categorical`` and ``not_equal_categorical`` take
    ``(CATEGORICAL, CATEGORICAL)``, a shape none of the fixtures above cover.
    featuretools has no primitive of the same name, but its generic
    ``equal``/``not_equal`` accept any column pair, so this fixture is run
    through those instead.
    """
    status = pd.Categorical(
        ["open", "closed", "open", "closed"],
        categories=["open", "closed", "pending"],
    )
    tier = pd.Categorical(
        ["open", "open", None, "closed"],
        categories=["open", "closed"],
    )
    frame = pd.DataFrame(
        {
            "id": np.arange(1, len(status) + 1),
            "status": status,
            "tier": tier,
        },
    )
    _assert_labelled_rows_invariants(frame)
    return frame


def _assert_labelled_rows_invariants(frame):
    """Guard the cases ``labelled_rows`` is built to cover.

    Args:
        frame: The table built by ``labelled_rows``.
    """
    assert set(frame["status"].cat.categories) != set(frame["tier"].cat.categories)
    known = frame["status"].notna() & frame["tier"].notna()
    labels_match = frame["status"].astype(str) == frame["tier"].astype(str)
    assert (labels_match & known).any()
    assert (~labels_match & known).any()
    assert frame["tier"].isna().any()


@pytest.mark.parametrize(
    ("tusk_primitive_name", "featuretools_primitive_name"),
    [("equal_categorical", "equal"), ("not_equal_categorical", "not_equal")],
)
def test_categorical_equality_matches_featuretools_where_nothing_is_null(
    labelled_rows,
    tusk_primitive_name,
    featuretools_primitive_name,
):
    """Wherever both labels are known, comparing by label agrees either way."""
    ours = _tusk_matrix(labelled_rows, tusk_primitive_name)
    theirs = _featuretools_matrix(labelled_rows, featuretools_primitive_name)
    known = labelled_rows.set_index("id")[["status", "tier"]].notna().all(axis=1)
    assert _nullable(
        ours[_tusk_column(tusk_primitive_name, "status", "tier")][known],
    ) == _nullable(
        theirs[_featuretools_column(featuretools_primitive_name, "status", "tier")][
            known
        ],
    )


@pytest.mark.parametrize(
    ("tusk_primitive_name", "featuretools_primitive_name", "featuretools_answer"),
    [
        ("equal_categorical", "equal", False),
        ("not_equal_categorical", "not_equal", True),
    ],
)
def test_categorical_equality_diverges_where_a_label_is_null(
    labelled_rows,
    tusk_primitive_name,
    featuretools_primitive_name,
    featuretools_answer,
):
    """A null label is answered differently on each side.

    tusk gives a null: an unknown label cannot be shown equal or unequal to
    anything. featuretools' generic equality primitives instead compare the
    raw pandas values, where a null label settles the comparison like any
    other mismatch -- false for equality, true for inequality.
    """
    ours = _tusk_matrix(labelled_rows, tusk_primitive_name)
    theirs = _featuretools_matrix(labelled_rows, featuretools_primitive_name)
    unknown = ~labelled_rows.set_index("id")[["status", "tier"]].notna().all(axis=1)
    assert unknown.any()
    assert (
        ours[_tusk_column(tusk_primitive_name, "status", "tier")][unknown].isna().all()
    )
    assert (
        theirs[_featuretools_column(featuretools_primitive_name, "status", "tier")][
            unknown
        ]
        == featuretools_answer
    ).all()


def test_modulo_diverges_on_a_zero_divisor(rows):
    """A zero divisor is backend-defined, per ``ModuloNumeric``'s own docstring.

    tusk keeps ``left``/``right`` as float64 the whole way, where ``%`` by
    zero is NaN. featuretools' woodwork layer instead infers a whole-number
    float column with nulls as ``IntegerNullable``, so its ``%`` runs on
    pandas nullable ``Int64`` columns, where a zero divisor gives 0 rather
    than a null (matching numpy's own int-mod-by-zero, `0`, not the
    divide-by-zero `inf`/`nan` a true division would give). Neither engine is
    wrong; they are simply two more entries in the same backend-defined set.
    The dtype assertion below pins that inference so a future fixture edit
    that stops it (e.g. a fractional value) fails here instead of silently
    invalidating this explanation.
    """
    ours, theirs = _both_matrices(rows, "modulo_numeric")
    assert pd.api.types.is_integer_dtype(_featuretools_input_dtype(rows, "right"))
    zero_divisor = rows.set_index("id")["right"] == 0
    assert zero_divisor.any()
    assert ours[_tusk_column("modulo_numeric")][zero_divisor].isna().all()
    assert (theirs[_featuretools_column("modulo_numeric")][zero_divisor] == 0).all()


def _featuretools_input_dtype(frame, column_name):
    """Read the pandas dtype woodwork assigns one input column.

    Args:
        frame: The table featuretools will run features over.
        column_name: The column to inspect.

    Returns:
        The dtype of that column once featuretools' woodwork layer has
        inferred its logical type.
    """
    es = featuretools.EntitySet("rows")
    es = es.add_dataframe(dataframe_name="rows", dataframe=frame.copy(), index="id")
    return es["rows"][column_name].dtype


def _tusk_column(primitive_name, left_name="left", right_name="right"):
    """Name tusk's output column for one binary primitive.

    Args:
        primitive_name: The primitive's name.
        left_name: The first input column's name.
        right_name: The second input column's name.

    Returns:
        The column name tusk gives the ``(left_name, right_name)`` feature.
    """
    return f"{primitive_name.upper()}__{left_name}__{right_name}"


def _featuretools_column(primitive_name, left_name="left", right_name="right"):
    """Name featuretools' output column for one binary primitive.

    Args:
        primitive_name: The primitive's name.
        left_name: The first input column's name.
        right_name: The second input column's name.

    Returns:
        The column name featuretools gives the ``(left_name, right_name)``
        feature.
    """
    return f"{left_name} {_FEATURETOOLS_OPERATOR[primitive_name]} {right_name}"


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
