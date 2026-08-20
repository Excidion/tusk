"""Portability checks against a second, SQL-generating backend.

The core suite runs on polars, which accepts any string as a column name.
A backend that generates SQL does not: an unquoted ``MEAN(transactions.amount)``
parses as a function call on table ``transactions``, and ``customers.age``
as a column of table ``customers``. Feature names are therefore plain SQL
identifiers (spec section 6), and these tests are what holds that line --
without them the naming is only asserted, never verified.

duckdb is the cheapest such backend to run in-process, so it stands in for
the whole class here.
"""

import narwhals as nw
import pytest

import tusk

duckdb = pytest.importorskip("duckdb")


@pytest.fixture
def duck_db():
    """The three-table retail database, backed by duckdb relations.

    Mirrors the shape of the polars ``db`` fixture in ``conftest``: customer 1
    has two sessions, customer 2 has one session with no transactions, and
    customer 3 has none at all.

    Returns:
        A tuple of the Database and the duckdb connection backing it.
    """
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE customers AS SELECT * FROM (VALUES "
        "(1, 30, TIMESTAMP '2024-01-01'), (2, 40, TIMESTAMP '2024-01-01'), "
        "(3, 50, TIMESTAMP '2024-01-01')) t(id, age, signed_up_at)"
    )
    con.execute(
        "CREATE TABLE sessions AS SELECT * FROM (VALUES "
        "(10, 1, TIMESTAMP '2024-03-04'), (20, 1, TIMESTAMP '2024-03-05'), "
        "(30, 2, TIMESTAMP '2024-03-06')) t(id, customer_id, started_at)"
    )
    con.execute(
        "CREATE TABLE transactions AS SELECT * FROM (VALUES "
        "(100, 10, 1.0, TIMESTAMP '2024-03-04 01:00'), "
        "(101, 10, 3.0, TIMESTAMP '2024-03-04 02:00'), "
        "(102, 20, 10.0, TIMESTAMP '2024-03-05 01:00'), "
        "(103, 20, 20.0, TIMESTAMP '2024-03-05 02:00')) "
        "t(id, session_id, amount, occurred_at)"
    )
    database = (
        tusk.Database("retail")
        .add_table(
            "customers",
            nw.from_native(con.sql("SELECT * FROM customers")),
            primary_key="id",
            row_creation_time="signed_up_at",
        )
        .add_table(
            "sessions",
            nw.from_native(con.sql("SELECT * FROM sessions")),
            primary_key="id",
            row_creation_time="started_at",
        )
        .add_table(
            "transactions",
            nw.from_native(con.sql("SELECT * FROM transactions")),
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
    return database, con


@pytest.mark.parametrize("target", ["customers", "sessions", "transactions"])
def test_every_generated_name_is_a_sql_identifier(duck_db, target):
    """No feature name carries a character SQL would parse as syntax.

    Every target is covered because the feature kinds are not evenly spread:
    ``customers`` is the root and so has no direct features at all, and it is
    direct features that carry the parent's name into the column.

    Args:
        duck_db: The duckdb-backed database.
        target: Table to synthesize features for.
    """
    database, _ = duck_db
    features = tusk.deep_feature_synthesis(
        database=database,
        target_table=target,
        max_depth=2,
        features_only=True,
    )
    assert features
    for feature in features:
        for name in feature.output_names:
            assert name.replace("_", "").isalnum(), name


def test_depth_two_matrix_computes_on_duckdb(duck_db):
    """A stacked aggregation holds the right value on a SQL backend.

    Customer 1's sessions average 2.0 and 15.0, so the mean of those means is
    8.5. Customer 2's only session has no transactions and customer 3 has no
    sessions, so both are null.

    Args:
        duck_db: The duckdb-backed database.
    """
    database, _ = duck_db
    features = tusk.deep_feature_synthesis(
        database=database,
        target_table="customers",
        max_depth=2,
        agg_primitives=["mean"],
        trans_primitives=[],
        features_only=True,
    )
    matrix = tusk.apply_features(features, database).pl()
    row = {r["id"]: r for r in matrix.to_dicts()}
    stacked = "MEAN__sessions__MEAN__transactions__amount"
    assert stacked in matrix.columns
    assert row[1][stacked] == pytest.approx(8.5)
    assert row[2][stacked] is None
    assert row[3][stacked] is None


def test_direct_feature_crosses_a_join_on_duckdb(duck_db):
    """A parent column copied down keeps its value through a SQL join.

    Args:
        duck_db: The duckdb-backed database.
    """
    database, _ = duck_db
    features = tusk.deep_feature_synthesis(
        database=database,
        target_table="sessions",
        max_depth=1,
        agg_primitives=[],
        trans_primitives=[],
        features_only=True,
    )
    matrix = tusk.apply_features(features, database).pl()
    ages = {r["id"]: r["customers__age"] for r in matrix.to_dicts()}
    assert ages == {10: 30, 20: 30, 30: 40}
