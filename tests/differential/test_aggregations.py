"""Cross-check tusk's aggregation primitives against featuretools, one at a time.

Each test builds one primitive on one side and the same primitive on the
other, over a single parent/child pair, and compares the one resulting
column. Model: ``tests/differential/test_vs_featuretools.py``; this file is
self-contained and does not import from it.

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


def _featuretools_matrix(parents, children, primitive_name):
    """Run one aggregation primitive through featuretools and return its matrix.

    Args:
        parents: The parent table.
        children: The child table.
        primitive_name: The aggregation primitive's featuretools name, the
            only entry in ``agg_primitives`` so exactly one feature is built.

    Returns:
        The feature matrix, sorted by the parent's index.
    """
    es = featuretools.EntitySet("aggregations")
    es = es.add_dataframe(dataframe_name="parents", dataframe=parents, index="id")
    es = es.add_dataframe(dataframe_name="children", dataframe=children, index="id")
    es = es.add_relationship("parents", "id", "children", "parent_id")
    matrix, _ = featuretools.dfs(
        entityset=es,
        target_dataframe_name="parents",
        agg_primitives=[primitive_name],
        trans_primitives=[],
        max_depth=1,
    )
    return matrix.sort_index()


def _tusk_matrix(parents, children, primitive_name):
    """Run one aggregation primitive through tusk and return its matrix.

    Args:
        parents: The parent table.
        children: The child table.
        primitive_name: The aggregation primitive's name, the only entry in
            ``agg_primitives`` so exactly one feature is built. tusk and
            featuretools spell aggregation primitive names the same way.

    Returns:
        The feature matrix, collected and sorted by the parent's primary key.
    """
    db = (
        tusk.Database("aggregations")
        .add_table("parents", pl.from_pandas(parents).lazy(), primary_key="id")
        .add_table("children", pl.from_pandas(children).lazy(), primary_key="id")
        .add_relationship(parent="parents", child="children", foreign_key="parent_id")
    )
    matrix, _ = tusk.deep_feature_synthesis(
        database=db,
        target_table="parents",
        agg_primitives=[primitive_name],
        trans_primitives=[],
        max_depth=1,
    )
    return matrix.collect().sort("id").to_pandas().set_index("id")


def _as_tusk(name: str) -> str:
    """Translate a featuretools feature name into tusk's column name.

    Every construct featuretools spells with punctuation -- application,
    argument separator, path step, multi-output index, groupby suffix --
    tusk spells with ``__``.

    Args:
        name: A featuretools feature name, e.g. ``MEDIAN(children.value)``.

    Returns:
        The equivalent tusk column name, e.g. ``MEDIAN__children__value``.
    """
    for old, new in (
        (" by ", "__by__"),
        (", ", "__"),
        ("(", "__"),
        (")", ""),
        ("[", "__"),
        ("]", ""),
        (".", "__"),
    ):
        name = name.replace(old, new)
    return name
