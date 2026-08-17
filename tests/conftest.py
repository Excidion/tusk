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
        }
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
        }
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
        }
    )
    return customers, sessions, transactions


@pytest.fixture
def es():
    """A three-table retail entity set."""
    customers, sessions, transactions = _frames()
    return (
        tusk.EntitySet("retail")
        .add_dataframe(
            "customers", customers, primary_key="id", row_creation_time="signed_up_at"
        )
        .add_dataframe(
            "sessions", sessions, primary_key="id", row_creation_time="started_at"
        )
        .add_dataframe(
            "transactions",
            transactions,
            primary_key="id",
            row_creation_time="occurred_at",
        )
        .add_relationship(
            parent="customers", child="sessions", foreign_key="customer_id"
        )
        .add_relationship(
            parent="sessions", child="transactions", foreign_key="session_id"
        )
    )
