import polars as pl
import pytest

import tusk

pytest.importorskip("sdm")
pytestmark = pytest.mark.sdm

from tusk.sdm._graph import (  # noqa: E402
    key_columns,
    relationship_mappings,
    time_columns,
)


def test_a_relationship_has_the_child_on_the_left(db):
    assert relationship_mappings(db) == [
        {
            "left_table": "sessions",
            "left_column": "customer_id",
            "right_table": "customers",
            "right_column": "id",
        },
        {
            "left_table": "transactions",
            "left_column": "session_id",
            "right_table": "sessions",
            "right_column": "id",
        },
    ]


def test_time_columns_hold_the_declared_row_creation_times(db):
    assert time_columns(db) == {
        "customers": "signed_up_at",
        "sessions": "started_at",
        "transactions": "occurred_at",
    }


def test_time_columns_omit_a_table_without_a_row_creation_time():
    database = tusk.Database("d").add_table(
        "a",
        pl.LazyFrame({"id": [1]}),
        primary_key="id",
    )
    assert time_columns(database) == {}


def test_key_columns_are_the_primary_key_and_the_foreign_keys(db):
    assert key_columns(db, "customers") == frozenset({"id"})
    assert key_columns(db, "sessions") == frozenset({"id", "customer_id"})
    assert key_columns(db, "transactions") == frozenset({"id", "session_id"})


@pytest.mark.filterwarnings("ignore::tusk.exceptions.MissingPrimaryKeyWarning")
def test_a_child_without_a_primary_key_has_only_its_foreign_keys():
    database = (
        tusk.Database("d")
        .add_table("p", pl.LazyFrame({"id": [1]}), primary_key="id")
        .add_table("c", pl.LazyFrame({"p_id": [1], "x": [1]}))
        .add_relationship(parent="p", child="c", foreign_key="p_id")
    )
    assert key_columns(database, "c") == frozenset({"p_id"})
