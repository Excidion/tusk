"""Shared fixtures. The backend list is the portability knob: adding a backend
here is intended to be a one-line change."""

import datetime as dt

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
