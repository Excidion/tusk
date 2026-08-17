"""Cross-check tusk's values against featuretools on synthetic data.

Temporary scaffolding: this tier validates the algorithm during development and
is removed once the differential suite has passed unchanged across two
consecutive releases that added primitives. See the spec, section 11.

Run with: uv run --group validation pytest -m differential

Verified against featuretools 1.31.0. That version's ``EntitySet.add_dataframe``
takes ``dataframe_name=``, ``dataframe=``, ``index=`` keywords, and
``add_relationship`` takes four positional arguments:
``(parent_dataframe_name, parent_column_name, child_dataframe_name,
child_column_name)``. Both match what this file assumed.

On synthetic data with empty groups (customers with no sessions), tusk and
featuretools *disagree* on SUM: tusk fills empty aggregations with the
primitive's declared ``default_value`` (``None``/null for SUM, per
``tusk.primitives.aggregation.Sum``), while featuretools fills empty SUM
groups with ``0.0``. COUNT agrees (both use 0) because tusk's Count primitive
declares ``default_value = 0``. This is a deliberate, spec-pinned semantic
choice in tusk (see ``tests/test_compiler_aggregation.py``), not a bug -- the
tests below assert both sides' actual behavior explicitly rather than papering
over the difference.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

import tusk

featuretools = pytest.importorskip("featuretools")

pytestmark = pytest.mark.differential

PRIMITIVES = {
    "count": "count",
    "sum": "sum",
    "mean": "mean",
    "min": "min",
    "max": "max",
}


@pytest.fixture
def synthetic():
    """A two-table dataset with empty groups and nulls."""
    rng = np.random.default_rng(0)
    customers = pd.DataFrame({"id": np.arange(1, 21)})
    sessions = pd.DataFrame(
        {
            "id": np.arange(1, 61),
            "customer_id": rng.integers(1, 18, size=60),  # customers 18-20 get none
            "value": rng.normal(size=60),
        }
    )
    return customers, sessions


def _featuretools_matrix(customers, sessions):
    es = featuretools.EntitySet("s")
    es = es.add_dataframe(dataframe_name="customers", dataframe=customers, index="id")
    es = es.add_dataframe(dataframe_name="sessions", dataframe=sessions, index="id")
    es = es.add_relationship("customers", "id", "sessions", "customer_id")
    matrix, _ = featuretools.dfs(
        entityset=es,
        target_dataframe_name="customers",
        agg_primitives=list(PRIMITIVES),
        trans_primitives=[],
        max_depth=1,
    )
    return matrix.sort_index()


def _tusk_matrix(customers, sessions):
    es = (
        tusk.EntitySet("s")
        .add_dataframe("customers", pl.from_pandas(customers).lazy(), primary_key="id")
        .add_dataframe("sessions", pl.from_pandas(sessions).lazy(), primary_key="id")
        .add_relationship(
            parent="customers", child="sessions", foreign_key="customer_id"
        )
    )
    matrix, _ = tusk.dfs(
        entityset=es,
        target_dataframe_name="customers",
        agg_primitives=list(PRIMITIVES),
        trans_primitives=[],
        max_depth=1,
    )
    return matrix.collect().sort("id").to_pandas().set_index("id")


@pytest.fixture
def matrices(synthetic):
    """Both tools' feature matrices over the same synthetic tables."""
    customers, sessions = synthetic
    return _tusk_matrix(customers, sessions), _featuretools_matrix(customers, sessions)


@pytest.mark.parametrize(
    ("tusk_name", "featuretools_name"),
    [
        ("COUNT(sessions)", "COUNT(sessions)"),
        ("MEAN(sessions.value)", "MEAN(sessions.value)"),
        ("MIN(sessions.value)", "MIN(sessions.value)"),
        ("MAX(sessions.value)", "MAX(sessions.value)"),
    ],
)
def test_values_match_featuretools(matrices, tusk_name, featuretools_name):
    """These primitives agree on every row, including empty groups.

    COUNT's empty-group default is 0 on both sides. MEAN/MIN/MAX are
    undefined on an empty group and both tools represent that as NaN/null,
    which ``assert_series_equal`` treats as equal.
    """
    ours, theirs = matrices
    pd.testing.assert_series_equal(
        ours[tusk_name].reset_index(drop=True).astype(float),
        theirs[featuretools_name].reset_index(drop=True).astype(float),
        check_names=False,
    )


def test_sum_matches_on_populated_groups(matrices, synthetic):
    """SUM agrees with featuretools everywhere a group actually has rows.

    The empty-group case is where the two tools diverge; see
    ``test_empty_groups_diverge_on_sum`` below.
    """
    customers, sessions = synthetic
    ours, theirs = matrices
    populated = sorted(set(sessions["customer_id"]) & set(customers["id"]))
    assert populated
    ours_sum = ours.loc[populated, "SUM(sessions.value)"].reset_index(drop=True)
    theirs_sum = theirs.loc[populated, "SUM(sessions.value)"].reset_index(drop=True)
    pd.testing.assert_series_equal(
        ours_sum.astype(float), theirs_sum.astype(float), check_names=False
    )


def test_empty_groups_agree_on_count(matrices, synthetic):
    """Both tools default COUNT to 0 for a customer with no sessions."""
    customers, sessions = synthetic
    ours, theirs = matrices
    childless = sorted(set(customers["id"]) - set(sessions["customer_id"]))
    assert childless
    assert (ours.loc[childless, "COUNT(sessions)"] == 0).all()
    assert (theirs.loc[childless, "COUNT(sessions)"] == 0).all()


def test_empty_groups_diverge_on_sum(matrices, synthetic):
    """Documented, expected disagreement: tusk gives null, featuretools gives 0.0.

    tusk's ``Sum`` primitive has no declared ``default_value``, so the
    compiler leaves empty groups null (see ``tusk/primitives/aggregation.py``
    and ``tests/test_compiler_aggregation.py::test_empty_group_gets_null_for_mean``,
    which pins the same behavior for MEAN). featuretools instead fills empty
    SUM aggregations with 0.0. Both are internally consistent semantic
    choices; tusk's is intentional per spec and is not adjusted to match
    featuretools here.
    """
    customers, sessions = synthetic
    ours, theirs = matrices
    childless = sorted(set(customers["id"]) - set(sessions["customer_id"]))
    assert childless
    assert ours.loc[childless, "SUM(sessions.value)"].isna().all()
    assert (theirs.loc[childless, "SUM(sessions.value)"] == 0.0).all()
