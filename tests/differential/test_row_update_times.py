"""Cross-check row update times against featuretools' secondary time index.

featuretools nulls a secondary-time-index column whose timestamp is after the
cutoff, no matter what value the column held before. tusk instead serves the
value declared in ``row_update_times``, so the two agree only where every
declared pre-update value is null -- which is the case this file pins.

Run with: uv run --group validation pytest -m differential

Verified against featuretools 1.31.0.
"""

import datetime as dt

import pytest

import tusk
from differential import _as_tusk

pd = pytest.importorskip("pandas")
pl = pytest.importorskip("polars")
featuretools = pytest.importorskip("featuretools")

pytestmark = pytest.mark.differential

CUTOFF_TIME = dt.datetime(2024, 6, 1)


@pytest.fixture
def customers_and_orders():
    """A parent/child pair covering the cases a secondary time index can diverge on.

    Order 10 belongs to customer 1 and was updated after the cutoff, so its
    amount is unknown at the cutoff. Order 11 also belongs to customer 1 but
    was updated before the cutoff, so customer 1's group mixes an updated and
    an unupdated amount. Order 12 is customer 2's only order and was updated
    after the cutoff, so customer 2's group is entirely unknown -- the empty
    case where MEAN is undefined.
    """
    customers = pd.DataFrame(
        {"id": [1, 2], "signed_up_at": [dt.datetime(2024, 1, 1)] * 2},
    )
    orders = pd.DataFrame(
        {
            "id": [10, 11, 12],
            "customer_id": [1, 1, 2],
            "amount": [1.0, 2.0, 4.0],
            "placed_at": [dt.datetime(2024, 3, 1)] * 3,
            "updated_at": [
                dt.datetime(2024, 9, 1),
                dt.datetime(2024, 4, 1),
                dt.datetime(2024, 9, 1),
            ],
        },
    )
    _assert_invariants(orders)
    return customers, orders


def _assert_invariants(orders):
    """Guard the cases ``customers_and_orders`` is built to cover.

    Without these, a change to the fixture could silently drop the mixed
    group or the entirely-updated group this test relies on to catch a
    divergence.

    Args:
        orders: The order table built by ``customers_and_orders``.
    """
    updated_after_cutoff_by_customer = orders.assign(
        updated_after_cutoff=orders["updated_at"] > CUTOFF_TIME,
    ).groupby("customer_id")["updated_after_cutoff"]
    assert updated_after_cutoff_by_customer.apply(
        lambda updated: updated.any() and not updated.all(),
    ).any()
    assert updated_after_cutoff_by_customer.all().any()


def test_a_null_pre_update_value_matches_a_secondary_time_index(customers_and_orders):
    """tusk matches featuretools' MEAN when every pre-update value is null.

    Without the update rewind, customer 1's mean would be 1.5 (both orders'
    current amounts) rather than 2.0, and customer 2's would be 4.0 rather
    than null -- so a real behavioural difference is pinned here, not a
    tautology.
    """
    customers, orders = customers_and_orders
    ours = _tusk_matrix(customers, orders)[_as_tusk("MEAN(orders.amount)")]
    theirs = _featuretools_matrix(customers, orders)["MEAN(orders.amount)"]
    pd.testing.assert_series_equal(
        ours.reset_index(drop=True).astype(float),
        theirs.reset_index(drop=True).astype(float),
        check_names=False,
    )


def _featuretools_matrix(customers, orders):
    """Run MEAN(orders.amount) through featuretools' secondary time index.

    Args:
        customers: The parent table.
        orders: The child table, with ``updated_at`` declared as a secondary
            time index over ``amount``.

    Returns:
        The feature matrix, sorted by the parent's index.
    """
    entityset = featuretools.EntitySet("shop")
    entityset.add_dataframe(
        dataframe=customers,
        dataframe_name="customers",
        index="id",
        time_index="signed_up_at",
    )
    entityset.add_dataframe(
        dataframe=orders,
        dataframe_name="orders",
        index="id",
        time_index="placed_at",
        secondary_time_index={"updated_at": ["amount"]},
    )
    entityset.add_relationship("customers", "id", "orders", "customer_id")
    matrix, _ = featuretools.dfs(
        entityset=entityset,
        target_dataframe_name="customers",
        agg_primitives=["mean"],
        trans_primitives=[],
        max_depth=1,
        cutoff_time=CUTOFF_TIME,
    )
    return matrix.sort_index()


def _tusk_matrix(customers, orders):
    """Run MEAN(orders.amount) through tusk's row update times.

    Args:
        customers: The parent table.
        orders: The child table, declaring ``updated_at`` as a row update
            time over ``amount`` with a null pre-update value.

    Returns:
        The feature matrix, collected and sorted by the parent's primary key.
    """
    database = (
        tusk.Database("shop")
        .add_table(
            "customers",
            pl.from_pandas(customers).lazy(),
            primary_key="id",
            row_creation_time="signed_up_at",
        )
        .add_table(
            "orders",
            pl.from_pandas(orders).lazy(),
            primary_key="id",
            row_creation_time="placed_at",
            row_update_times={"updated_at": {"amount": None, "updated_at": None}},
        )
        .add_relationship(parent="customers", child="orders", foreign_key="customer_id")
    )
    matrix, _ = tusk.deep_feature_synthesis(
        database=database,
        target_table="customers",
        agg_primitives=["mean"],
        trans_primitives=[],
        max_depth=1,
        cutoff_time=CUTOFF_TIME,
    )
    return matrix.collect().sort("id").to_pandas().set_index("id")
