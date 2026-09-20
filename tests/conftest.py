"""Shared fixtures. The backend list is the portability knob: adding a backend
here is intended to be a one-line change."""

import datetime as dt

import narwhals as nw
import polars as pl
import pytest

import tusk

BACKENDS = ["polars"]


@pytest.fixture(params=BACKENDS)
def backend(request):
    """Name of the dataframe backend under test."""
    return request.param


def _frames():
    """Three related tables with hand-checkable values.

    customers 1 and 2 have sessions; customer 3 has none (empty-group case).
    session 30 has no transactions (nested empty-group case).
    """
    customers = pl.LazyFrame(
        {
            "id": [1, 2, 3],
            "age": [30, 40, 50],
            "signed_up_at": [dt.datetime(2024, 1, 1)] * 3,
        },
    )
    sessions = pl.LazyFrame(
        {
            "id": [10, 20, 30],
            "customer_id": [1, 1, 2],
            "started_at": [
                dt.datetime(2024, 3, 4),  # Monday
                dt.datetime(2024, 3, 5),
                dt.datetime(2024, 3, 6),
            ],
        },
    )
    transactions = pl.LazyFrame(
        {
            "id": [100, 101, 102, 103],
            "session_id": [10, 10, 20, 20],
            "amount": [1.0, 3.0, 10.0, 20.0],
            "verified": [True, False, True, False],
            "occurred_at": [
                dt.datetime(2024, 3, 4, 1),
                dt.datetime(2024, 3, 4, 2),
                dt.datetime(2024, 3, 5, 1),
                dt.datetime(2024, 3, 5, 2),
            ],
        },
    )
    return customers, sessions, transactions


@pytest.fixture
def db():
    """A three-table retail database."""
    customers, sessions, transactions = _frames()
    return (
        tusk.Database("retail")
        .add_table(
            "customers",
            customers,
            primary_key="id",
            row_creation_time="signed_up_at",
        )
        .add_table(
            "sessions",
            sessions,
            primary_key="id",
            row_creation_time="started_at",
        )
        .add_table(
            "transactions",
            transactions,
            primary_key="id",
            row_creation_time="occurred_at",
        )
        .add_relationship(
            parent="customers",
            child="sessions",
            foreign_key="customer_id",
        )
        .add_relationship(
            parent="sessions",
            child="transactions",
            foreign_key="session_id",
        )
    )


@pytest.fixture
def updating_db():
    """A shop whose orders have a status and a parent that get rewritten.

    Against a cutoff of 2024-06-01: order 10 was updated after it, order 11
    never, order 12 before it.
    """
    customers = pl.LazyFrame(
        {
            "id": [1, 2],
            "signed_up_at": [dt.datetime(2024, 1, 1)] * 2,
        },
    )
    orders = pl.LazyFrame(
        {
            "id": [10, 11, 12],
            "customer_id": [1, 1, 2],
            "amount": [1.0, 2.0, 4.0],
            "placed_at": [dt.datetime(2024, 3, 1)] * 3,
            "status": ["delivered", "pending", "delivered"],
            "updated_at": [
                dt.datetime(2024, 9, 1),
                None,
                dt.datetime(2024, 4, 1),
            ],
        },
    )
    return (
        tusk.Database("shop")
        .add_table(
            "customers",
            customers,
            primary_key="id",
            row_creation_time="signed_up_at",
        )
        .add_table(
            "orders",
            orders,
            primary_key="id",
            row_creation_time="placed_at",
            row_update_times={
                "updated_at": {"status": "pending", "updated_at": None},
            },
        )
        .add_relationship(parent="customers", child="orders", foreign_key="customer_id")
    )


def clause_database(*, large_threshold: float = 10.0, declare_clauses: bool = True):
    """A shop database whose orders carry declared where/when clauses.

    Customer 1 has orders of 10.0 and 20.0 (both large) and one of 1.0;
    customer 2 has a single order of 2.0. Order 10 closes 2024-06-01, so it
    is open at a 2024-05-01 cutoff and closed at 2024-08-01. Products 1 and 2
    are priced 3.0 and 4.0 and are joined to orders by ``product_id``.

    Args:
        large_threshold: The amount an order must meet to satisfy the
            ``"large"`` where clause.
        declare_clauses: When False, ``orders`` is added with no ``where``
            and no ``when`` at all, for tests of the undeclared-clause path.

    Returns:
        A database with ``customers``, ``orders`` and ``products`` tables,
        related by ``customer_id`` and ``product_id``.
    """
    customers = pl.LazyFrame(
        {"id": [1, 2], "signed_up_at": [dt.datetime(2024, 1, 1)] * 2}
    )
    orders = pl.LazyFrame(
        {
            "id": [10, 11, 12, 13],
            "customer_id": [1, 1, 1, 2],
            "product_id": [1, 2, 1, 2],
            "amount": [10.0, 20.0, 1.0, 2.0],
            "placed_at": [dt.datetime(2024, 3, 1)] * 4,
            "closed_at": [
                dt.datetime(2024, 6, 1),
                None,
                dt.datetime(2024, 4, 1),
                None,
            ],
        },
    )
    products = pl.LazyFrame({"id": [1, 2], "price": [3.0, 4.0]})

    where = (
        {
            "large": nw.col("amount") >= large_threshold,
            "impossible": nw.col("amount") < 0.0,
        }
        if declare_clauses
        else None
    )
    when = (
        {
            "open": lambda cutoff: (
                nw.col("closed_at").is_null() | (nw.col("closed_at") > cutoff)
            ),
        }
        if declare_clauses
        else None
    )

    return (
        tusk.Database("shop")
        .add_table(
            "customers",
            customers,
            primary_key="id",
            row_creation_time="signed_up_at",
        )
        .add_table(
            "orders",
            orders,
            primary_key="id",
            row_creation_time="placed_at",
            where=where,
            when=when,
        )
        .add_table("products", products, primary_key="id")
        .add_relationship(parent="customers", child="orders", foreign_key="customer_id")
        .add_relationship(parent="products", child="orders", foreign_key="product_id")
    )
