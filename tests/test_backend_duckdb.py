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
import math

import narwhals as nw
import pytest
from aggregation_cases import (
    CHILDREN,
    EXPECTED,
    PARENTS,
    ROW_CREATION_TIME,
    assert_values_match,
)
from transform_cases import (
    EXPECTED as TRANSFORM_EXPECTED,
)
from transform_cases import (
    GROUPS,
    ROWS,
    feature_values,
    rows_database,
)
from transform_cases import (
    assert_values_match as assert_transform_values_match,
)

import tusk
from tusk.database import Relationship
from tusk.feature_list import FeatureList
from tusk.features import AggregationFeature, IdentityFeature
from tusk.primitives import Negate, resolve

duckdb = pytest.importorskip("duckdb")
pd = pytest.importorskip("pandas")


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


@pytest.fixture
def duckdb_database():
    """A shop database whose orders carry an always-false ``where`` condition.

    Mirrors ``duck_db``'s shape: two customers, each owning one order, so
    without the condition the aggregation would see rows -- the point is that
    ``impossible`` empties both groups anyway.

    Returns:
        A duckdb-backed database with customers and orders, related by
        ``customer_id``.
    """
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE customers AS SELECT * FROM (VALUES "
        "(1, TIMESTAMP '2024-01-01'), (2, TIMESTAMP '2024-01-01')) "
        "t(id, signed_up_at)",
    )
    con.execute(
        "CREATE TABLE orders AS SELECT * FROM (VALUES "
        "(10, 1, 5.0, TIMESTAMP '2024-03-01'), (11, 2, 7.0, TIMESTAMP '2024-03-01')) "
        "t(id, customer_id, amount, placed_at)",
    )
    return (
        tusk.Database("shop")
        .add_table(
            "customers",
            nw.from_native(con.sql("SELECT * FROM customers")),
            primary_key="id",
            row_creation_time="signed_up_at",
        )
        .add_table(
            "orders",
            nw.from_native(con.sql("SELECT * FROM orders")),
            primary_key="id",
            row_creation_time="placed_at",
            where={"impossible": nw.col("amount") < 0.0},
        )
        .add_relationship(parent="customers", child="orders", foreign_key="customer_id")
    )


def test_empty_mask_falls_back_to_the_primitive_default(duckdb_database):
    """No matching rows gives the same value as no rows at all."""
    features = FeatureList(
        [
            AggregationFeature(
                resolve(name),
                (IdentityFeature("orders", "amount", nw.Float64()),),
                Relationship("customers", "orders", "customer_id"),
                condition=("where", "impossible"),
            )
            for name in ("sum", "mean")
        ],
    )
    matrix = nw.from_native(features.apply(duckdb_database)).lazy().collect()
    assert matrix["SUM__orders__amount__WHERE__impossible"].to_list() == [0.0, 0.0]
    assert matrix["MEAN__orders__amount__WHERE__impossible"].to_list() == [None, None]


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


def test_time_since_aggregations_hold_the_elapsed_time_on_duckdb(duck_db):
    """The cutoff-measuring aggregations survive translation to SQL.

    TIME_SINCE_LAST_TRUE is the one that constrains the implementation: a SQL
    backend rejects an aggregate nested inside another aggregate, so the
    selected row has to be picked with a blanking ``when`` rather than by
    comparing against the group's own maximum.

    ``occurred_at`` is ``transactions``' own ``row_creation_time``, so
    ``cutoff_time`` is chosen after every transaction; see
    ``test_time_since_holds_the_elapsed_time_on_duckdb`` for why the matrix is
    materialized with ``.df()``.

    Args:
        duck_db: The duckdb-backed database.
    """
    database, _ = duck_db
    cutoff_time = dt.datetime(2024, 3, 6)
    features = tusk.deep_feature_synthesis(
        database=database,
        target_table="sessions",
        max_depth=1,
        agg_primitives=["time_since_last", "time_since_last_true"],
        trans_primitives=[],
        features_only=True,
    )
    matrix = tusk.apply_features(features, database, cutoff_time=cutoff_time).df()
    rows_by_id = matrix.set_index("id")
    latest = "TIME_SINCE_LAST__transactions__occurred_at"
    latest_true = (
        "TIME_SINCE_LAST_TRUE__transactions__occurred_at__transactions__is_completed"
    )
    assert rows_by_id.loc[10, latest] == dt.timedelta(days=1, hours=22)
    assert rows_by_id.loc[20, latest] == dt.timedelta(hours=22)
    assert rows_by_id.loc[30, latest] is pd.NaT
    assert rows_by_id.loc[10, latest_true] == dt.timedelta(days=1, hours=23)
    assert rows_by_id.loc[20, latest_true] is pd.NaT


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


def test_boolean_transforms_hold_three_valued_logic_on_duckdb(duck_db):
    """AND, OR and NOT translate to the SQL operators, nulls and all.

    The divergence from featuretools is the whole point of the rows where one
    input is null: SQL answers FALSE AND NULL with FALSE and TRUE OR NULL with
    TRUE, because the unknown cannot change the outcome.

    Args:
        duck_db: The duckdb-backed database.
    """
    _, con = duck_db
    con.execute(
        "CREATE TABLE accounts AS SELECT * FROM (VALUES "
        "(1, TRUE, TRUE), (2, FALSE, NULL), (3, TRUE, NULL), (4, NULL, NULL)) "
        "t(id, is_active, is_verified)",
    )
    database = tusk.Database("flags").add_table(
        "accounts",
        con.table("accounts"),
        primary_key="id",
    )
    matrix = tusk.deep_feature_synthesis(
        database=database,
        target_table="accounts",
        max_depth=1,
        agg_primitives=[],
        trans_primitives=["and", "or", "not"],
    )[0].pl()
    row = {r["id"]: r for r in matrix.to_dicts()}
    assert row[1]["AND__is_active__is_verified"] is True
    assert row[2]["AND__is_active__is_verified"] is False
    assert row[3]["AND__is_active__is_verified"] is None
    assert row[4]["AND__is_active__is_verified"] is None
    assert row[1]["OR__is_active__is_verified"] is True
    assert row[2]["OR__is_active__is_verified"] is None
    assert row[3]["OR__is_active__is_verified"] is True
    assert row[4]["OR__is_active__is_verified"] is None
    assert row[4]["NOT__is_active"] is None


def test_binary_transforms_translate_to_sql(duck_db):
    """Comparisons, floored modulo and masking survive the trip to SQL.

    Modulo is the reason this test exists: duckdb's ``%`` truncates toward
    zero, so a plain remainder would answer -1 here where polars answers 1.
    Row 5's zero divisor pins duckdb's half of ``ModuloNumeric``'s
    backend-defined rule, over both a float column (``v``/``w``) and an
    integer one (``iv``/``iw``) -- the polars half is pinned in
    ``tests/test_primitives_transform.py``.

    Args:
        duck_db: The duckdb-backed database.
    """
    _, con = duck_db
    con.execute(
        "CREATE TABLE readings AS SELECT * FROM (VALUES "
        "(1, -7.0, 2.0, TRUE, -7, 2), (2, 7.0, -2.0, FALSE, 7, -2), "
        "(3, 5.0, NULL, NULL, 5, NULL), (4, 3.0, 3.0, TRUE, 3, 3), "
        "(5, 5.0, 0.0, TRUE, 5, 0)) "
        "t(id, v, w, flag, iv, iw)",
    )
    database = tusk.Database("sensors").add_table(
        "readings",
        con.table("readings"),
        primary_key="id",
    )
    matrix = tusk.deep_feature_synthesis(
        database=database,
        target_table="readings",
        max_depth=1,
        agg_primitives=[],
        trans_primitives=[
            "greater_than",
            "equal",
            "modulo_numeric",
            "multiply_numeric_boolean",
        ],
    )[0].pl()
    row = {r["id"]: r for r in matrix.to_dicts()}
    assert row[1]["MODULO_NUMERIC__v__w"] == 1.0
    assert row[2]["MODULO_NUMERIC__v__w"] == -1.0
    assert row[3]["MODULO_NUMERIC__v__w"] is None
    assert row[5]["MODULO_NUMERIC__v__w"] is None
    assert row[5]["MODULO_NUMERIC__iv__iw"] is None
    assert row[1]["GREATER_THAN__v__w"] is False
    assert row[2]["GREATER_THAN__v__w"] is True
    assert row[3]["GREATER_THAN__v__w"] is None
    assert row[1]["EQUAL__v__w"] is False
    assert row[2]["EQUAL__v__w"] is False
    assert row[3]["EQUAL__v__w"] is None
    assert row[4]["EQUAL__v__w"] is True
    assert row[1]["MULTIPLY_NUMERIC_BOOLEAN__v__flag"] == -7.0
    assert row[2]["MULTIPLY_NUMERIC_BOOLEAN__v__flag"] == 0.0
    assert row[3]["MULTIPLY_NUMERIC_BOOLEAN__v__flag"] is None


def test_date_and_datetime_pair_compares_cleanly_on_duckdb(duck_db):
    """A Date operand casts up to midnight, agreeing with a plain Datetime pair.

    Args:
        duck_db: The duckdb-backed database.
    """
    _, con = duck_db
    con.execute(
        "CREATE TABLE readings AS SELECT * FROM (VALUES "
        "(1, DATE '2024-01-01', TIMESTAMP '2024-01-01 00:00:00'), "
        "(2, DATE '2024-01-02', TIMESTAMP '2024-01-02 12:00:00'), "
        "(3, DATE '2024-01-03', TIMESTAMP '2024-01-03 00:00:00')) "
        "t(id, d, ts)",
    )
    database = tusk.Database("sensors").add_table(
        "readings",
        con.table("readings"),
        primary_key="id",
    )
    matrix = tusk.deep_feature_synthesis(
        database=database,
        target_table="readings",
        max_depth=1,
        agg_primitives=[],
        trans_primitives=["equal"],
    )[0].pl()
    row = {r["id"]: r for r in matrix.to_dicts()}
    assert row[1]["EQUAL__d__ts"] is True
    assert row[2]["EQUAL__d__ts"] is False
    assert row[3]["EQUAL__d__ts"] is True


def test_date_aggregations_hold_the_elapsed_time_and_count_on_duckdb(duck_db):
    """FIRST_LAST_TIME_DELTA and N_UNIQUE_DAYS survive a native duckdb DATE column.

    Every other duckdb test's temporal columns are TIMESTAMP, so none of them
    exercises the ``cast(nw.Datetime)`` in ``FirstLastTimeDelta.build``, which
    exists because duckdb subtracts two DATEs into a day count rather than an
    interval. Customer 1 has two visits 61 days apart on 2 distinct days,
    customer 2 has a single visit (0 days apart, 1 distinct day), and
    customer 3 has none (both null).

    Args:
        duck_db: The duckdb-backed database.
    """
    _, con = duck_db
    con.execute(
        "CREATE TABLE date_customers AS SELECT * FROM (VALUES (1), (2), (3)) t(id)",
    )
    con.execute(
        "CREATE TABLE visits AS SELECT * FROM (VALUES "
        "(1, 1, DATE '2024-01-01'), (2, 1, DATE '2024-03-02'), "
        "(3, 2, DATE '2024-06-01')) "
        "t(id, customer_id, visited_on)",
    )
    database = (
        tusk.Database("clinic")
        .add_table("date_customers", con.table("date_customers"), primary_key="id")
        .add_table("visits", con.table("visits"), primary_key="id")
        .add_relationship(
            parent="date_customers",
            child="visits",
            foreign_key="customer_id",
        )
    )
    features = tusk.deep_feature_synthesis(
        database=database,
        target_table="date_customers",
        max_depth=1,
        agg_primitives=["first_last_time_delta", "n_unique_days"],
        trans_primitives=[],
        features_only=True,
    )
    matrix = tusk.apply_features(features, database).df()
    rows_by_id = matrix.set_index("id")
    delta = "FIRST_LAST_TIME_DELTA__visits__visited_on"
    unique_days = "N_UNIQUE_DAYS__visits__visited_on"
    assert rows_by_id.loc[1, delta] == dt.timedelta(days=61)
    assert rows_by_id.loc[2, delta] == dt.timedelta(days=0)
    assert rows_by_id.loc[3, delta] is pd.NaT
    assert rows_by_id.loc[1, unique_days] == 2
    assert rows_by_id.loc[2, unique_days] == 1
    assert rows_by_id.loc[3, unique_days] == 0


@pytest.mark.parametrize("primitive_name", sorted(EXPECTED))
def test_standalone_aggregations_give_the_polars_values_on_duckdb(primitive_name):
    """Every standalone aggregation survives translation to SQL, group by group.

    The constant group pins the skew and kurtosis guard, which duckdb would
    otherwise answer with 0.0 or null where polars answers NaN. ``n_true``
    pins the cast after SUM, and ``first_last_time_delta`` the interval
    subtraction. Materialized with ``.df()``; see
    ``test_time_since_holds_the_elapsed_time_on_duckdb`` for why.

    Args:
        primitive_name: The aggregation under test.
    """
    column, _, expected = EXPECTED[primitive_name]
    con = duckdb.connect()
    con.register("parents_frame", PARENTS)
    con.register("children_frame", CHILDREN)
    database = (
        tusk.Database("cases")
        .add_table(
            "parents",
            nw.from_native(con.sql("SELECT * FROM parents_frame")),
            primary_key="id",
        )
        .add_table(
            "children",
            nw.from_native(con.sql("SELECT * FROM children_frame")),
            primary_key="id",
            row_creation_time=ROW_CREATION_TIME,
        )
        .add_relationship(parent="parents", child="children", foreign_key="parent_id")
    )
    matrix, _ = tusk.deep_feature_synthesis(
        database=database,
        target_table="parents",
        agg_primitives=[primitive_name],
        trans_primitives=[],
        max_depth=1,
    )
    got = matrix.df().sort_values("id")[column].tolist()
    assert_values_match(got, expected)


@pytest.mark.parametrize("column", sorted(TRANSFORM_EXPECTED))
def test_transforms_give_the_polars_values_on_duckdb(column):
    """Every standalone, group and ordered transform survives translation to SQL.

    ``square_root`` and ``natural_log`` pin the negative-input guard, which
    polars would otherwise answer with NaN, and the ordered primitives pin
    the ordering by ``occurred_at``.

    Args:
        column: The feature column under test.
    """
    primitive, _, expected = TRANSFORM_EXPECTED[column]
    con = duckdb.connect()
    con.register("rows_frame", ROWS)
    con.register("groups_frame", GROUPS)
    matrix, _ = tusk.deep_feature_synthesis(
        database=rows_database(
            con.sql("SELECT * FROM rows_frame"),
            con.sql("SELECT * FROM groups_frame"),
        ),
        target_table="rows",
        agg_primitives=[],
        max_depth=1,
        trans_primitives=[primitive],
    )
    assert_transform_values_match(feature_values(matrix, column), expected)


def test_percentile_ranks_within_each_group_on_duckdb():
    """The grouped percentile partitions both its rank and its count window.

    narwhals pushes the grouped ``percentile`` into two separate windows on
    duckdb: one for ``rank("average")``, one for ``count()``. If only one
    carried the ``parent_id`` partition, ranks would divide by the whole
    table's count instead of the group's.
    """
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE percentile_parents AS SELECT * FROM (VALUES (1), (2)) t(id)",
    )
    con.execute(
        "CREATE TABLE percentile_children AS SELECT * FROM (VALUES "
        "(1, 1, 1.0), (2, 1, 3.0), (3, 1, 3.0), (4, 2, 5.0), (5, 2, NULL)) "
        "t(id, parent_id, amount)",
    )
    database = (
        tusk.Database("groups")
        .add_table(
            "parents",
            con.table("percentile_parents"),
            primary_key="id",
        )
        .add_table(
            "children",
            con.table("percentile_children"),
            primary_key="id",
        )
        .add_relationship(parent="parents", child="children", foreign_key="parent_id")
    )
    matrix, _ = tusk.deep_feature_synthesis(
        database=database,
        target_table="children",
        agg_primitives=[],
        trans_primitives=["percentile"],
        max_depth=1,
    )
    got = feature_values(matrix, "PERCENTILE__amount__by__parent_id")
    assert_transform_values_match(got, [1 / 3, 2.5 / 3, 2.5 / 3, 1.0, None])


def test_cumulative_time_since_stays_within_each_group_on_duckdb():
    """Parent 2's first row must not see parent 1's match."""
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE cumulative_parents AS SELECT * FROM (VALUES (1), (2)) t(id)",
    )
    con.execute(
        "CREATE TABLE cumulative_children AS SELECT * FROM (VALUES "
        "(1, 1, TIMESTAMP '2024-01-01', TRUE), "
        "(2, 1, TIMESTAMP '2024-01-02', FALSE), "
        "(3, 2, TIMESTAMP '2024-01-03', FALSE), "
        "(4, 2, TIMESTAMP '2024-01-04', TRUE)) "
        't(id, parent_id, "at", flag)',
    )
    database = (
        tusk.Database("groups")
        .add_table(
            "parents",
            con.table("cumulative_parents"),
            primary_key="id",
        )
        .add_table(
            "children",
            con.table("cumulative_children"),
            primary_key="id",
            row_creation_time="at",
        )
        .add_relationship(parent="parents", child="children", foreign_key="parent_id")
    )
    matrix, _ = tusk.deep_feature_synthesis(
        database=database,
        target_table="children",
        agg_primitives=[],
        trans_primitives=["cumulative_time_since_last_true"],
        max_depth=1,
    )
    got = feature_values(
        matrix, "CUMULATIVE_TIME_SINCE_LAST_TRUE__at__flag__by__parent_id"
    )
    assert_transform_values_match(
        got,
        [dt.timedelta(0), dt.timedelta(days=1), None, dt.timedelta(0)],
    )


def test_negate_does_not_overflow_an_integer_dtype_on_duckdb():
    """duckdb raises on negating TINYINT's minimum and wraps an unsigned value."""
    con = duckdb.connect()
    frame = nw.from_native(
        con.sql(
            "SELECT * FROM (VALUES (CAST(-128 AS TINYINT), 4294967295::UINTEGER)) "
            "t(small, unsigned)",
        ),
    )
    negate = Negate()
    got = frame.select(
        negate.outputs(nw.col("small"))[0].alias("small"),
        negate.outputs(nw.col("unsigned"))[0].alias("unsigned"),
    ).collect()
    assert got["small"].to_list() == [128.0]
    assert got["unsigned"].to_list() == [-4294967295.0]


def test_absolute_diff_does_not_overflow_an_integer_dtype_on_duckdb():
    """duckdb raises subtracting TINYINT's minimum from its maximum, and on a
    UTINYINT decrease."""
    con = duckdb.connect()
    frame = nw.from_native(
        con.sql(
            "SELECT * FROM (VALUES "
            "(TIMESTAMP '2024-01-01', CAST(-128 AS TINYINT), CAST(200 AS UTINYINT)), "
            "(TIMESTAMP '2024-01-02', CAST(127 AS TINYINT), CAST(1 AS UTINYINT))) "
            # "at" quoted: unquoted it collides with duckdb's AT TIME ZONE keyword.
            't("at", small, unsigned)',
        ),
    )
    absolute_diff = resolve("absolute_diff")
    got = frame.select(
        absolute_diff.outputs(nw.col("small"))[0].over(order_by="at").alias("small"),
        absolute_diff.outputs(nw.col("unsigned"))[0]
        .over(order_by="at")
        .alias("unsigned"),
    ).collect()
    assert got["small"].to_list() == [None, 255.0]
    assert got["unsigned"].to_list() == [None, 199.0]


def test_percent_change_gives_negative_infinity_over_a_zero_previous_value_on_duckdb():
    """A negative value after a zero previous one; polars and duckdb must agree."""
    con = duckdb.connect()
    frame = nw.from_native(
        con.sql(
            "SELECT * FROM (VALUES "
            "(TIMESTAMP '2024-01-01', CAST(0.0 AS DOUBLE)), "
            "(TIMESTAMP '2024-01-02', CAST(-1.0 AS DOUBLE))) "
            # "at" quoted: unquoted it collides with duckdb's AT TIME ZONE keyword.
            't("at", v)',
        ),
    )
    percent_change = resolve("percent_change")
    got = (
        frame.select(
            percent_change.outputs(nw.col("v"))[0].over(order_by="at").alias("v"),
        )
        .collect()["v"]
        .to_list()
    )
    assert got[0] is None
    assert got[1] == -math.inf


@pytest.mark.parametrize(
    ("name", "expected"), [("minute", [2, None]), ("second", [3, None])]
)
def test_minute_and_second_read_a_time_column_on_duckdb(name, expected):
    con = duckdb.connect()
    frame = nw.from_native(
        # "at" quoted: unquoted it collides with duckdb's AT TIME ZONE keyword.
        con.sql("SELECT * FROM (VALUES (1, TIME '01:02:03'), (2, NULL)) t(id, \"at\")"),
    )
    primitive = resolve(name)
    got = frame.select("id", primitive.outputs(nw.col("at"))[0].alias("o"))
    assert got.collect().sort("id")["o"].to_list() == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "cumulative_time_since_last_true",
            [dt.timedelta(0), dt.timedelta(days=4), dt.timedelta(0)],
        ),
        (
            "cumulative_time_since_last_false",
            [None, dt.timedelta(0), dt.timedelta(days=4)],
        ),
    ],
)
def test_cumulative_time_since_measures_a_date_column_on_duckdb(name, expected):
    """duckdb subtracts two DATEs into a day count, so the primitive casts first.

    Args:
        name: The primitive under test.
        expected: The elapsed time per row, in id order.
    """
    con = duckdb.connect()
    frame = nw.from_native(
        con.sql(
            "SELECT * FROM (VALUES (1, DATE '2024-01-01', TRUE), "
            "(2, DATE '2024-01-05', FALSE), (3, DATE '2024-01-09', TRUE)) "
            "t(id, d, f)",
        ),
    )
    primitive = resolve(name)
    elapsed = primitive.outputs(nw.col("d"), nw.col("f"))[0].over(order_by="id")
    got = frame.select("id", elapsed.alias("o")).collect().sort("id")["o"].to_list()
    assert_transform_values_match(got, expected)
