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

On synthetic data with empty groups (customers with no sessions), this suite
originally caught tusk and featuretools *disagreeing* on SUM: tusk left an
empty group's SUM null while featuretools filled it with 0.0. That was
adjudicated: sum over an empty set is 0, the additive identity, and tusk
already reports COUNT = 0 for the same rows -- asserting a known-zero count
alongside an unknown total was internally inconsistent. ``Sum`` now declares
``default_value = 0`` (see ``tusk/primitives/aggregation.py``), so SUM agrees
with featuretools on empty groups too; that agreement is now a guarded
invariant below rather than a documented divergence. MEAN/MIN/MAX remain
null on empty groups on both sides, since those are genuinely undefined over
an empty set (0/0, and min/max of nothing).
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
        ("SUM(sessions.value)", "SUM(sessions.value)"),
        ("MEAN(sessions.value)", "MEAN(sessions.value)"),
        ("MIN(sessions.value)", "MIN(sessions.value)"),
        ("MAX(sessions.value)", "MAX(sessions.value)"),
    ],
)
def test_values_match_featuretools(matrices, tusk_name, featuretools_name):
    """These primitives agree on every row, including empty groups.

    COUNT and SUM both default to 0 for an empty group, on both sides.
    MEAN/MIN/MAX are undefined on an empty group and both tools represent
    that as NaN/null, which ``assert_series_equal`` treats as equal.
    """
    ours, theirs = matrices
    pd.testing.assert_series_equal(
        ours[tusk_name].reset_index(drop=True).astype(float),
        theirs[featuretools_name].reset_index(drop=True).astype(float),
        check_names=False,
    )


def test_empty_groups_agree_on_count_and_sum(matrices, synthetic):
    """Both tools default COUNT and SUM to 0 for a customer with no sessions.

    This is the guarded invariant that replaced a real, adjudicated
    disagreement: tusk's ``Sum`` primitive previously had no declared
    ``default_value`` and left empty groups null, while featuretools filled
    them with 0.0. tusk now declares ``default_value = 0`` on ``Sum`` to
    match -- sum over an empty set is 0, the additive identity, and it would
    be inconsistent to report a known-zero COUNT alongside an unknown SUM.
    """
    customers, sessions = synthetic
    ours, theirs = matrices
    childless = sorted(set(customers["id"]) - set(sessions["customer_id"]))
    assert childless
    assert (ours.loc[childless, "COUNT(sessions)"] == 0).all()
    assert (theirs.loc[childless, "COUNT(sessions)"] == 0).all()
    assert (ours.loc[childless, "SUM(sessions.value)"] == 0).all()
    assert (theirs.loc[childless, "SUM(sessions.value)"] == 0).all()


def test_empty_groups_stay_null_for_mean_min_max(matrices, synthetic):
    """MEAN/MIN/MAX are genuinely undefined over an empty set on both sides."""
    customers, sessions = synthetic
    ours, theirs = matrices
    childless = sorted(set(customers["id"]) - set(sessions["customer_id"]))
    assert childless
    for column in (
        "MEAN(sessions.value)",
        "MIN(sessions.value)",
        "MAX(sessions.value)",
    ):
        assert ours.loc[childless, column].isna().all()
        assert theirs.loc[childless, column].isna().all()
