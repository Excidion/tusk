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

import datetime as dt

import narwhals as nw
import pytest

import tusk

duckdb = pytest.importorskip("duckdb")


@pytest.fixture
def duck_db():
    """The three-table retail database, backed by duckdb relations.

    Mirrors the shape of the polars ``db`` fixture in ``conftest``: customer 1
    has two sessions, customer 2 has one session with no transactions, and
    customer 3 has none at all. ``transactions.is_completed`` is boolean with
    nulls: session 10's two transactions are one true and one null, and
    session 20's two are both null.

    Returns:
        A tuple of the Database and the duckdb connection backing it.
    """
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE customers AS SELECT * FROM (VALUES "
        "(1, 30, TIMESTAMP '2024-01-01'), (2, 40, TIMESTAMP '2024-01-01'), "
        "(3, 50, TIMESTAMP '2024-01-01')) t(id, age, signed_up_at)",
    )
    con.execute(
        "CREATE TABLE sessions AS SELECT * FROM (VALUES "
        "(10, 1, TIMESTAMP '2024-03-04'), (20, 1, TIMESTAMP '2024-03-05'), "
        "(30, 2, TIMESTAMP '2024-03-06')) t(id, customer_id, started_at)",
    )
    con.execute(
        "CREATE TABLE transactions AS SELECT * FROM (VALUES "
        "(100, 10, 1.0, TIMESTAMP '2024-03-04 01:00', TRUE), "
        "(101, 10, 3.0, TIMESTAMP '2024-03-04 02:00', NULL), "
        "(102, 20, 10.0, TIMESTAMP '2024-03-05 01:00', NULL), "
        "(103, 20, 20.0, TIMESTAMP '2024-03-05 02:00', NULL)) "
        "t(id, session_id, amount, occurred_at, is_completed)",
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


def test_percent_true_holds_the_null_rule_on_duckdb(duck_db):
    """PERCENT_TRUE's null-counts-as-false rule survives translation to SQL.

    Session 10 has one true and one null transaction, so a null lowered its
    fraction to 0.5 rather than being skipped. Session 20's transactions are
    both null, so its fraction is 0.0, not null. Session 30 has no
    transactions at all, so it is null.

    Args:
        duck_db: The duckdb-backed database.
    """
    database, _ = duck_db
    features = tusk.deep_feature_synthesis(
        database=database,
        target_table="sessions",
        max_depth=1,
        agg_primitives=["percent_true"],
        trans_primitives=[],
        features_only=True,
    )
    matrix = tusk.apply_features(features, database).pl()
    row = {r["id"]: r for r in matrix.to_dicts()}
    column = "PERCENT_TRUE__transactions__is_completed"
    assert column in matrix.columns
    assert row[10][column] == pytest.approx(0.5)
    assert row[20][column] == pytest.approx(0.0)
    assert row[30][column] is None


def test_time_since_holds_the_elapsed_time_on_duckdb(duck_db):
    """TIME_SINCE survives translation to SQL: a datetime literal minus a column.

    ``occurred_at`` is ``transactions``' own ``row_creation_time``, so
    ``cutoff_time`` is chosen after every transaction -- an earlier cutoff
    would filter the very rows this test needs to see, since a target row is
    invisible at a cutoff before it was created.

    Materialized with ``.df()`` rather than the ``.pl()`` every neighbouring
    test uses: duckdb's own ``DuckDBPyRelation.pl()`` raises
    ``polars.exceptions.ComputeError: could not import from
    `month_day_nano_interval` type`` for an INTERVAL column, a pyarrow/polars
    limitation on this stack (duckdb 1.5.5, pyarrow 25.0.1, polars 1.43.2)
    reached only now that a duckdb-backed test produces a Duration column.
    duckdb itself computes the right interval -- ``.df()`` and ``.fetchall()``
    both return it correctly -- so this is not a narwhals SQL translation bug.

    Args:
        duck_db: The duckdb-backed database.
    """
    database, _ = duck_db
    cutoff_time = dt.datetime(2024, 3, 6)
    features = tusk.deep_feature_synthesis(
        database=database,
        target_table="transactions",
        max_depth=1,
        agg_primitives=[],
        trans_primitives=["time_since"],
        features_only=True,
    )
    matrix = tusk.apply_features(features, database, cutoff_time=cutoff_time).df()
    rows_by_id = matrix.set_index("id")
    column = "TIME_SINCE__occurred_at"
    assert column in matrix.columns
    assert rows_by_id.loc[100, column] == dt.timedelta(days=1, hours=23)
    assert rows_by_id.loc[101, column] == dt.timedelta(days=1, hours=22)
    assert rows_by_id.loc[102, column] == dt.timedelta(hours=23)
    assert rows_by_id.loc[103, column] == dt.timedelta(hours=22)


def test_quantiles_interpolates_linearly_on_duckdb(duck_db):
    """QUANTILES' linear interpolation survives translation to SQL.

    Session 10's amounts are 1.0 and 3.0, so the default quartiles split that
    gap into quarters: 1.5, 2.0, 2.5. Session 20's are 10.0 and 20.0, giving
    12.5, 15.0, 17.5. Session 30 has no transactions, so all three outputs
    are null.

    Args:
        duck_db: The duckdb-backed database.
    """
    database, _ = duck_db
    features = tusk.deep_feature_synthesis(
        database=database,
        target_table="sessions",
        max_depth=1,
        agg_primitives=["quantiles"],
        trans_primitives=[],
        features_only=True,
    )
    matrix = tusk.apply_features(features, database).pl()
    row = {r["id"]: r for r in matrix.to_dicts()}
    columns = [f"QUANTILES__transactions__amount__{i}" for i in range(3)]
    assert all(column in matrix.columns for column in columns)
    assert [row[10][column] for column in columns] == pytest.approx([1.5, 2.0, 2.5])
    assert [row[20][column] for column in columns] == pytest.approx(
        [12.5, 15.0, 17.5],
    )
    assert [row[30][column] for column in columns] == [None, None, None]


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


def test_uniqueness_check_runs_on_duckdb(duck_db):
    db, con = duck_db
    assert db.validate() is db

    con.execute("CREATE TABLE dupes AS SELECT * FROM (VALUES (1),(1),(2)) t(id)")
    with pytest.raises(tusk.exceptions.ValidationError, match="not unique"):
        db.add_table("dupes", con.table("dupes"), primary_key="id", validate=True)
    assert "dupes" not in db.table_names


def test_empty_relation_passes_uniqueness_on_duckdb(duck_db):
    # narwhals lowers n_unique on SQL backends to
    # count_distinct(x) + max(x IS NULL). MAX() over zero rows is SQL NULL,
    # so an empty duckdb relation used to make the distinct count come back
    # as None, and 0 == None is False -- a false ValidationError on data with
    # no rows to contradict the declaration. polars cannot reproduce this: it
    # returns 0 distinct values for the same empty frame, not None.
    db, con = duck_db
    con.execute("CREATE TABLE empty (id INTEGER)")
    db.add_table("empty", con.table("empty"), primary_key="id", validate=True)
    assert "empty" in db.table_names


def test_null_primary_key_is_caught_on_duckdb(duck_db):
    db, con = duck_db
    con.execute("CREATE TABLE nulls AS SELECT * FROM (VALUES (1),(NULL)) t(id)")
    with pytest.raises(tusk.exceptions.ValidationError, match="null"):
        db.add_table("nulls", con.table("nulls"), primary_key="id", validate=True)
    assert "nulls" not in db.table_names
