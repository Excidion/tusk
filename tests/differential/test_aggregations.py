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

import pytest

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
            ``agg_primitives`` so exactly one feature is built. tusk and
            featuretools spell aggregation primitive names the same way.
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
