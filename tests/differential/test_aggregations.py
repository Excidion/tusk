"""Cross-check tusk's aggregation primitives against featuretools, one at a time.

Each test builds one primitive on one side and the same primitive on the
other, over a single parent/child pair, and compares the one resulting
column. Model: ``tests/differential/test_vs_featuretools.py``; this file
builds its own matrices rather than importing them, sharing only
``_as_tusk`` from :mod:`differential`.

Run with: uv run --group validation pytest -m differential

Verified against featuretools 1.31.0.
"""

import datetime as dt
import math

import pytest
from aggregation_cases import CHILDREN, PARENTS, assert_values_match

import tusk
from differential import _as_tusk

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pl = pytest.importorskip("polars")
featuretools = pytest.importorskip("featuretools")

pytestmark = pytest.mark.differential


@pytest.fixture
def parent_and_child():
    """A parent/child pair covering the cases aggregations can diverge on.

    Parent 9 and 10 are childless. The ``value`` column has nulls, including
    at least one group holding both a null and a real value. The
    ``is_active`` column has nulls too, including a group with both a null
    and known true/false values; parent group 5 (``all_null_group_id``) is
    forced entirely null so an all-null group is also covered.
    """
    rng = np.random.default_rng(0)
    parents = pd.DataFrame({"id": np.arange(1, 11)})
    child_values = rng.normal(size=40)
    child_values[rng.random(40) < 0.2] = np.nan
    children = pd.DataFrame(
        {
            "id": np.arange(1, 41),
            "parent_id": rng.integers(1, 9, size=40),  # parents 9 and 10 get none
            "value": child_values,
        },
    )
    is_active = pd.array(rng.random(40) < 0.5, dtype="boolean")
    is_active[rng.random(40) < 0.2] = pd.NA
    children["is_active"] = is_active
    all_null_group_id = 5
    children.loc[children["parent_id"] == all_null_group_id, "is_active"] = pd.NA

    _assert_parent_and_child_invariants(parents, children)
    return parents, children


def _assert_parent_and_child_invariants(parents, children):
    """Guard the cases ``parent_and_child`` is built to cover.

    Without these, a change to the random draws could silently drop the one
    case a differential test relies on to catch a divergence.

    Args:
        parents: The parent table built by ``parent_and_child``.
        children: The child table built by ``parent_and_child``.
    """
    childless_parents = set(parents["id"]) - set(children["parent_id"])
    assert childless_parents

    groups_with_a_null_and_a_real_value = children.groupby("parent_id")["value"].apply(
        lambda values: values.isna().any() and values.notna().any(),
    )
    assert groups_with_a_null_and_a_real_value.any()

    def has_a_null_and_both_true_and_false(flags):
        known_flags = flags.dropna()
        return flags.isna().any() and known_flags.any() and (~known_flags).any()

    booleans_by_group = children.groupby("parent_id")["is_active"]
    assert booleans_by_group.apply(has_a_null_and_both_true_and_false).any()
    assert booleans_by_group.apply(lambda flags: flags.isna().all()).any()


def test_median_matches_featuretools_on_every_parent_row(parent_and_child):
    """tusk's MEDIAN(children.value) equals featuretools' on every parent row.

    Covers both cases where median could diverge: a null-containing group,
    where each side must skip the nulls rather than propagate them, and an
    empty group, where median is undefined and both sides must report it as
    such rather than pick a default.
    """
    parents, children = parent_and_child
    ours = _tusk_matrix(parents, children, "median")
    theirs = _featuretools_matrix(parents, children, "median")
    pd.testing.assert_series_equal(
        ours[_as_tusk("MEDIAN(children.value)")].reset_index(drop=True).astype(float),
        theirs["MEDIAN(children.value)"].reset_index(drop=True).astype(float),
        check_names=False,
    )


def test_percent_true_matches_featuretools_on_every_parent_row(parent_and_child):
    """tusk's PERCENT_TRUE(children.is_active) equals featuretools' on every row.

    Covers the three cases where the two could diverge: a group mixing true,
    false, and null values, where a null must count as false in the
    denominator rather than being skipped; a group that is entirely null,
    where the result is 0.0 rather than null; and an empty group, where both
    sides report the same missing value rather than 0.0.
    """
    parents, children = parent_and_child
    ours = _tusk_matrix(parents, children, "percent_true")
    theirs = _featuretools_matrix(parents, children, "percent_true")
    pd.testing.assert_series_equal(
        ours[_as_tusk("PERCENT_TRUE(children.is_active)")]
        .reset_index(drop=True)
        .astype(float),
        theirs["PERCENT_TRUE(children.is_active)"].reset_index(drop=True).astype(float),
        check_names=False,
    )


CUTOFF_TIME = dt.datetime(2024, 3, 1)
TIME_INDEX = "occurred_at"


@pytest.fixture
def timed_parent_and_child():
    """A parent/child pair covering the cases a cutoff aggregation can diverge on.

    Parent 5 is childless. Parent 2's flags are all null and parent 3 has no
    true flag, so both lack a row for TIME_SINCE_LAST_TRUE to measure from.
    Parent 4's latest child row lies after ``CUTOFF_TIME``, so a side that
    fails to filter it out reports a negative elapsed time. Every timestamp
    lands on a whole day so the seconds conversion in the tests is exact.
    """
    children = pd.DataFrame(
        {
            "id": np.arange(1, 10),
            "parent_id": [1, 1, 2, 2, 3, 3, 4, 4, 4],
            TIME_INDEX: pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-02-01",
                    "2024-01-15",
                    "2024-02-15",
                    "2024-01-20",
                    "2024-02-20",
                    "2024-01-25",
                    "2024-02-25",
                    "2024-04-01",
                ],
            ),
            "is_active": pd.array(
                [True, False, None, None, False, False, True, True, True],
                dtype="boolean",
            ),
        },
    )
    parents = pd.DataFrame({"id": np.arange(1, 6)})
    _assert_timed_parent_and_child_invariants(parents, children)
    return parents, children


def _assert_timed_parent_and_child_invariants(parents, children):
    """Guard the cases ``timed_parent_and_child`` is built to cover.

    Args:
        parents: The parent table built by ``timed_parent_and_child``.
        children: The child table built by ``timed_parent_and_child``.
    """
    assert set(parents["id"]) - set(children["parent_id"])
    assert (children[TIME_INDEX] > CUTOFF_TIME).any()
    flags_by_group = children.groupby("parent_id")["is_active"]
    assert flags_by_group.apply(lambda flags: flags.isna().all()).any()
    assert flags_by_group.apply(lambda flags: not flags.fillna(False).any()).any()
    assert flags_by_group.apply(lambda flags: flags.fillna(False).all()).any()


@pytest.mark.parametrize(
    ("primitive_name", "column"),
    [
        ("time_since_first", f"TIME_SINCE_FIRST(children.{TIME_INDEX})"),
        ("time_since_last", f"TIME_SINCE_LAST(children.{TIME_INDEX})"),
    ],
)
def test_time_since_first_and_last_match_featuretools_on_every_parent_row(
    timed_parent_and_child,
    primitive_name,
    column,
):
    """tusk's elapsed time, converted to seconds, equals featuretools' on every row.

    Covers a childless parent, where the elapsed time is undefined on both
    sides, and a group whose latest row lies after the cutoff, where both
    sides must measure from the latest row that already existed.
    """
    parents, children = timed_parent_and_child
    ours = _seconds(_our_column(parents, children, primitive_name, _as_tusk(column)))
    theirs = _theirs_column(parents, children, primitive_name, column)
    pd.testing.assert_series_equal(ours, theirs, check_names=False)


@pytest.mark.parametrize(
    ("primitive_name", "column"),
    [
        (
            "time_since_last_true",
            f"TIME_SINCE_LAST_TRUE(children.{TIME_INDEX}, is_active)",
        ),
        (
            "time_since_last_false",
            f"TIME_SINCE_LAST_FALSE(children.{TIME_INDEX}, is_active)",
        ),
    ],
)
def test_time_since_last_flag_matches_featuretools_where_a_flagged_row_exists(
    timed_parent_and_child,
    primitive_name,
    column,
):
    """tusk agrees with featuretools on every group that has a row to measure from.

    The groups without one are the deliberate divergence and are covered by
    ``test_time_since_last_flag_is_null_rather_than_zero_without_a_flagged_row``.
    """
    parents, children = timed_parent_and_child
    ours = _seconds(
        _our_column(
            parents, children, primitive_name, _as_tusk(_tusk_spelling(column))
        ),
    )
    theirs = _theirs_column(parents, children, primitive_name, column)
    measurable = ours.notna()
    assert measurable.any()
    pd.testing.assert_series_equal(
        ours[measurable],
        theirs[measurable],
        check_names=False,
    )


@pytest.mark.parametrize(
    ("primitive_name", "column"),
    [
        (
            "time_since_last_true",
            f"TIME_SINCE_LAST_TRUE(children.{TIME_INDEX}, is_active)",
        ),
        (
            "time_since_last_false",
            f"TIME_SINCE_LAST_FALSE(children.{TIME_INDEX}, is_active)",
        ),
    ],
)
def test_time_since_last_flag_is_null_rather_than_zero_without_a_flagged_row(
    timed_parent_and_child,
    primitive_name,
    column,
):
    """A group with no flagged row is null in tusk and zero in featuretools.

    Zero would say the flagged row happened exactly at the cutoff, which is a
    different claim from never having happened at all.
    """
    parents, children = timed_parent_and_child
    ours = _seconds(
        _our_column(
            parents, children, primitive_name, _as_tusk(_tusk_spelling(column))
        ),
    )
    theirs = _theirs_column(parents, children, primitive_name, column)
    unmeasurable = ours.isna()
    assert unmeasurable.any()
    assert (theirs[unmeasurable] == 0.0).all()


def _tusk_spelling(column):
    """Qualify a two-input feature's second argument with its table.

    featuretools names the second argument bare (``is_active``); tusk names
    every input by the table it came from.

    Args:
        column: The featuretools feature name.

    Returns:
        The same name with the second argument qualified.

    """
    return column.replace(", is_active", ", children.is_active")


def _our_column(parents, children, primitive_name, column):
    """Compute one tusk feature at ``CUTOFF_TIME``.

    Args:
        parents: The parent table.
        children: The child table.
        primitive_name: The primitive to build.
        column: The tusk column name to pull out.

    Returns:
        The column, indexed from zero.
    """
    matrix = _tusk_matrix(
        parents,
        children,
        primitive_name,
        time_index=TIME_INDEX,
        cutoff_time=CUTOFF_TIME,
    )
    return matrix[column].reset_index(drop=True)


def _theirs_column(parents, children, primitive_name, column):
    """Compute one featuretools feature at ``CUTOFF_TIME``.

    Args:
        parents: The parent table.
        children: The child table.
        primitive_name: The primitive to build.
        column: The featuretools column name to pull out.

    Returns:
        The column in seconds, indexed from zero.
    """
    matrix = _featuretools_matrix(
        parents,
        children,
        primitive_name,
        time_index=TIME_INDEX,
        cutoff_time=CUTOFF_TIME,
    )
    return matrix[column].reset_index(drop=True).astype(float)


def _seconds(elapsed):
    """Convert a tusk duration column to the float seconds featuretools reports.

    Args:
        elapsed: A column of durations.

    Returns:
        The same column in seconds.
    """
    return elapsed.dt.total_seconds().astype(float)


def _featuretools_matrix(
    parents,
    children,
    primitive_name,
    *,
    time_index=None,
    cutoff_time=None,
    logical_types=None,
):
    """Run one aggregation primitive through featuretools and return its matrix.

    Args:
        parents: The parent table.
        children: The child table.
        primitive_name: The aggregation primitive's featuretools name, the
            only entry in ``agg_primitives`` so exactly one feature is built.
        time_index: The child column featuretools filters against
            ``cutoff_time``, or None to keep every child row.
        cutoff_time: The moment the primitive measures against, or None.
        logical_types: Woodwork logical types for the child's columns, or
            None to let woodwork infer them.

    Returns:
        The feature matrix, sorted by the parent's index.
    """
    es = featuretools.EntitySet("aggregations")
    es = es.add_dataframe(dataframe_name="parents", dataframe=parents, index="id")
    es = es.add_dataframe(
        dataframe_name="children",
        dataframe=children,
        index="id",
        time_index=time_index,
        logical_types=logical_types,
    )
    es = es.add_relationship("parents", "id", "children", "parent_id")
    matrix, _ = featuretools.dfs(
        entityset=es,
        target_dataframe_name="parents",
        agg_primitives=[primitive_name],
        trans_primitives=[],
        max_depth=1,
        cutoff_time=cutoff_time,
    )
    return matrix.sort_index()


def _tusk_matrix(
    parents,
    children,
    primitive_name,
    *,
    time_index=None,
    cutoff_time=None,
):
    """Run one aggregation primitive through tusk and return its matrix.

    Args:
        parents: The parent table.
        children: The child table.
        primitive_name: The aggregation primitive's name, the only entry in
            ``agg_primitives`` so exactly one feature is built.
        time_index: The child column tusk filters against ``cutoff_time``,
            named ``row_creation_time`` here, or None to keep every child row.
        cutoff_time: The moment the primitive measures against, or None.

    Returns:
        The feature matrix, collected and sorted by the parent's primary key.
    """
    db = (
        tusk.Database("aggregations")
        .add_table("parents", pl.from_pandas(parents).lazy(), primary_key="id")
        .add_table(
            "children",
            pl.from_pandas(children).lazy(),
            primary_key="id",
            row_creation_time=time_index,
        )
        .add_relationship(parent="parents", child="children", foreign_key="parent_id")
    )
    matrix, _ = tusk.deep_feature_synthesis(
        database=db,
        target_table="parents",
        agg_primitives=[primitive_name],
        trans_primitives=[],
        max_depth=1,
        cutoff_time=cutoff_time,
    )
    return matrix.collect().sort("id").to_pandas().set_index("id")


FEATURETOOLS_LOGICAL_TYPES = {
    "value": "Double",
    "flag": "BooleanNullable",
    "label": "Categorical",
}


def _ours_and_theirs(
    tusk_name,
    featuretools_name,
    featuretools_column,
    *,
    children=CHILDREN,
    logical_types=FEATURETOOLS_LOGICAL_TYPES,
    featuretools_time_index=None,
):
    """Compute one aggregation on both sides, as plain values per parent id.

    Woodwork infers whole-number floats as ``IntegerNullable``, which
    ``kurtosis`` rejects, so the child's columns are declared explicitly.
    featuretools' ``first_last_time_delta`` only reads its time index.

    Args:
        tusk_name: tusk's name for the primitive.
        featuretools_name: featuretools' name for the primitive.
        featuretools_column: featuretools' name for the feature column.
        children: The child table.
        logical_types: Woodwork logical types for the child's columns.
        featuretools_time_index: The child column featuretools treats as its
            time index, or None.

    Returns:
        tusk's and featuretools' values, each a dict from parent id to a
        value, with durations in seconds and missing values as None.
    """
    tusk_column = _as_tusk(featuretools_column).replace(
        featuretools_name.upper(),
        tusk_name.upper(),
        1,
    )
    ours = _tusk_matrix(PARENTS, children, tusk_name)[tusk_column]
    theirs = _featuretools_matrix(
        PARENTS,
        children,
        featuretools_name,
        logical_types=logical_types,
        time_index=featuretools_time_index,
    )[featuretools_column]
    return _plain_by_parent(ours), _plain_by_parent(theirs)


def _plain_by_parent(column):
    """Key a feature column by parent id with plain Python values.

    Args:
        column: A feature column indexed by parent id.

    Returns:
        A dict from parent id to a value; durations become seconds and every
        missing value becomes None.
    """
    return {parent: _plain(value) for parent, value in column.items()}


def _plain(value):
    """Turn one materialized value into a plain Python one.

    Args:
        value: A value from a pandas column.

    Returns:
        None for any missing value, seconds for a duration, and the Python
        scalar for a numpy one, so ``is False`` works on a boolean.
    """
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    if isinstance(value, np.generic):
        return value.item()
    return value


@pytest.mark.parametrize(
    ("tusk_name", "featuretools_name", "featuretools_column"),
    [
        ("all_true", "all", "ALL(children.flag)"),
        ("n_true", "num_true", "NUM_TRUE(children.flag)"),
        ("variance", "variance", "VARIANCE(children.value)"),
    ],
)
def test_standalone_aggregations_match_featuretools_on_every_parent_row(
    tusk_name,
    featuretools_name,
    featuretools_column,
):
    """tusk and featuretools agree on a mixed, constant, all-null and empty group."""
    ours, theirs = _ours_and_theirs(tusk_name, featuretools_name, featuretools_column)
    assert_values_match(list(ours.values()), list(theirs.values()))


@pytest.mark.parametrize(
    ("tusk_name", "featuretools_name", "featuretools_column"),
    [
        ("n_unique_days", "n_unique_days", "N_UNIQUE_DAYS(children.seen_at)"),
        (
            "n_unique_days_of_calendar_year",
            "n_unique_days_of_calendar_year",
            "N_UNIQUE_DAYS_OF_CALENDAR_YEAR(children.seen_at)",
        ),
        (
            "n_unique_days_of_month",
            "n_unique_days_of_month",
            "N_UNIQUE_DAYS_OF_MONTH(children.seen_at)",
        ),
        ("n_unique_months", "n_unique_months", "N_UNIQUE_MONTHS(children.seen_at)"),
    ],
)
def test_distinct_date_counts_count_a_null_as_a_value(
    tusk_name,
    featuretools_name,
    featuretools_column,
):
    """tusk counts a null ``seen_at`` as one distinct value; featuretools drops it.

    Parents 1 and 3 each hold a null ``seen_at``, so tusk's count runs one
    higher there. Parents 2, 4 and 5 hold no null, so the two sides agree.
    """
    ours, theirs = _ours_and_theirs(tusk_name, featuretools_name, featuretools_column)
    for parent in (2, 4, 5):
        assert ours[parent] == theirs[parent]
    for parent in (1, 3):
        assert ours[parent] == theirs[parent] + 1


def test_any_true_of_an_empty_group_is_false_rather_than_null():
    """No row is true, so tusk says False; featuretools reports a missing value."""
    ours, theirs = _ours_and_theirs("any_true", "any", "ANY(children.flag)")
    assert_values_match(
        [ours[p] for p in (1, 2, 3, 4)], [theirs[p] for p in (1, 2, 3, 4)]
    )
    assert ours[5] is False
    assert theirs[5] is None


def test_skew_differs_from_featuretools_only_by_its_bias_correction():
    """featuretools' skew is tusk's scaled by pandas' sample correction.

    The correction for n known values is sqrt(n * (n - 1)) / (n - 2).
    Groups too small or too constant to have a skew are missing on both sides.
    """
    ours, theirs = _ours_and_theirs("skew", "skew", "SKEW(children.value)")
    known_counts = CHILDREN.groupby("parent_id")["value"].count()
    for parent in (1, 4):
        n = known_counts[parent]
        correction = math.sqrt(n * (n - 1)) / (n - 2)
        assert theirs[parent] == pytest.approx(ours[parent] * correction)
        assert theirs[parent] != pytest.approx(ours[parent])
    for parent in (2, 3, 5):
        assert ours[parent] is None
        assert theirs[parent] is None


def test_kurtosis_matches_featuretools_only_on_a_fully_known_group():
    """featuretools answers 0 wherever tusk skips a null or finds no spread."""
    ours, theirs = _ours_and_theirs("kurtosis", "kurtosis", "KURTOSIS(children.value)")
    assert ours[4] == pytest.approx(theirs[4])
    assert ours[1] == pytest.approx(-1.5)
    assert theirs[1] == 0.0
    for parent in (2, 3, 5):
        assert ours[parent] is None
        assert theirs[parent] == 0.0


@pytest.mark.parametrize(
    ("tusk_name", "featuretools_column", "children", "featuretools_time_index"),
    [
        ("max_min_delta", "MAX_MIN_DELTA(children.value)", CHILDREN, None),
        (
            "first_last_time_delta",
            "FIRST_LAST_TIME_DELTA(children.seen_at)",
            CHILDREN[CHILDREN["seen_at"].notna()],
            "seen_at",
        ),
    ],
)
def test_a_delta_without_a_known_value_is_null_rather_than_zero(
    tusk_name,
    featuretools_column,
    children,
    featuretools_time_index,
):
    """A group with nothing to subtract is null in tusk and 0 in featuretools.

    featuretools only reads ``first_last_time_delta`` from its time index,
    which may not hold nulls, so that case drops the null datetimes; parent 3
    then has no children at all.
    """
    ours, theirs = _ours_and_theirs(
        tusk_name,
        tusk_name,
        featuretools_column,
        children=children,
        featuretools_time_index=featuretools_time_index,
    )
    assert_values_match([ours[p] for p in (1, 2, 4)], [theirs[p] for p in (1, 2, 4)])
    for parent in (3, 5):
        assert ours[parent] is None
        assert theirs[parent] == 0.0


def test_is_unique_of_an_empty_group_is_null_rather_than_false():
    """No rows to compare, so tusk answers null; featuretools answers False."""
    ours, theirs = _ours_and_theirs(
        "is_unique", "is_unique", "IS_UNIQUE(children.label)"
    )
    assert_values_match(
        [ours[p] for p in (1, 2, 3, 4)], [theirs[p] for p in (1, 2, 3, 4)]
    )
    assert ours[5] is None
    assert theirs[5] is False


def test_is_unique_counts_repeated_nulls_where_has_no_duplicates_drops_them():
    """tusk ships no separate ``has_no_duplicates``: ``is_unique`` answers it too.

    tusk's IS_UNIQUE counts a repeated null as a repeat, agreeing with
    featuretools' HAS_NO_DUPLICATES only where a group holds no null.
    featuretools drops nulls before comparing, so a null-only or repeated-null
    group looks unique to it but not to tusk.
    """
    ours = _plain_by_parent(
        _tusk_matrix(PARENTS, CHILDREN, "is_unique")["IS_UNIQUE__children__label"],
    )
    theirs = _plain_by_parent(
        _featuretools_matrix(
            PARENTS,
            CHILDREN,
            "has_no_duplicates",
            logical_types=FEATURETOOLS_LOGICAL_TYPES,
        )["HAS_NO_DUPLICATES(children.label)"],
    )
    assert_values_match([ours[p] for p in (1, 4)], [theirs[p] for p in (1, 4)])
    for parent in (2, 3):
        assert ours[parent] is False
        assert theirs[parent] is True
    assert ours[5] is None
    assert theirs[5] is True


def test_percent_unique_of_an_empty_group_is_null_rather_than_zero():
    """percent_unique diverges from featuretools wherever ``label`` holds a null.

    Parent 4's labels hold no null, so the two sides agree there. Parents 1-3
    mix or consist of nulls, so tusk's fraction counts them and runs higher
    than featuretools', which drops them. Parent 5 has no children: a
    fraction of no rows is undefined in tusk, while featuretools reports 0.
    """
    ours, theirs = _ours_and_theirs(
        "percent_unique",
        "percent_unique",
        "PERCENT_UNIQUE(children.label)",
    )
    assert ours[4] == theirs[4]
    assert_values_match([ours[p] for p in (1, 2, 3)], [0.75, 0.5, 0.3333333333333333])
    assert_values_match([theirs[p] for p in (1, 2, 3)], [0.5, 0.0, 0.0])
    assert ours[5] is None
    assert theirs[5] == 0.0
