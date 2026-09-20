from dataclasses import dataclass, replace
from datetime import datetime, timedelta

import narwhals as nw
import polars as pl
import pytest
from conftest import clause_database

import tusk
from tusk.compiler import compile_features
from tusk.database import Relationship
from tusk.dtypes import DtypeFamily as F
from tusk.feature_list import FeatureList
from tusk.features import AggregationFeature, DirectFeature, IdentityFeature
from tusk.primitives.aggregation import Count, Mean, NUnique, Quantiles, Sum
from tusk.primitives.base import AggregationPrimitive, NeedsCutoffTime
from tusk.primitives.registry import resolve

CUSTOMER_SESSION = Relationship("customers", "sessions", "customer_id")
SESSION_TX = Relationship("sessions", "transactions", "session_id")

AMOUNT = IdentityFeature("transactions", "amount", nw.Float64())
OCCURRED_AT = IdentityFeature("transactions", "occurred_at", nw.Datetime())


@dataclass(frozen=True)
class CutoffAggregation(NeedsCutoffTime, AggregationPrimitive):
    """An aggregation that measures against the cutoff time.

    Time since the most recent child row, i.e. the smallest of the per-row
    gaps to the cutoff -- a stand-in for the shape of primitive tusk does not
    ship yet: an aggregation over a per-row cutoff-relative expression.
    """

    name = "cutoff_aggregation_direct"
    input_dtypes = (F.HAS_DATE,)

    def build(self, *inputs, cutoff_time):
        return (nw.lit(cutoff_time) - inputs[0]).min()


def collect(features, db, cutoff_time=None):
    return (
        compile_features(FeatureList(features), db, cutoff_time)
        .collect()
        .to_native()
        .sort("id")
    )


def test_count_of_children(db):
    # customer 1 has sessions 10 and 20; customer 2 has session 30; customer 3 has none.
    got = collect([AggregationFeature(Count(), (), CUSTOMER_SESSION)], db)
    assert got["COUNT__sessions"].to_list() == [2, 1, 0]


def test_empty_group_gets_count_default_of_zero(db):
    got = collect([AggregationFeature(Count(), (), CUSTOMER_SESSION)], db)
    assert got["COUNT__sessions"][2] == 0


def test_empty_group_gets_null_for_mean(db):
    sessions_mean = AggregationFeature(Mean(), (AMOUNT,), SESSION_TX)
    got = collect([AggregationFeature(Mean(), (sessions_mean,), CUSTOMER_SESSION)], db)
    assert got["MEAN__sessions__MEAN__transactions__amount"][2] is None


def test_depth_two_aggregation_values(db):
    # session 10 -> mean(1, 3) = 2; session 20 -> mean(10, 20) = 15; session 30 -> null.
    # customer 1 -> mean(2, 15) = 8.5; customer 2 -> null; customer 3 -> null.
    sessions_mean = AggregationFeature(Mean(), (AMOUNT,), SESSION_TX)
    feature = AggregationFeature(Mean(), (sessions_mean,), CUSTOMER_SESSION)
    got = collect([feature], db)
    assert got[feature.name].to_list() == [8.5, None, None]


def test_n_unique_counts_a_null_as_one_distinct_value(db):
    """A group whose only value is null has 1 distinct value, not 0.

    Customer 2's only session is 30, which has no transactions, so
    MEAN(transactions.amount) is null there; that single null counts as one
    distinct value.

    Customer 1 has sessions 10 (mean 2.0) and 20 (mean 15.0): two distinct
    non-null values. Customer 3 has no sessions at all, so the left join
    leaves null and default_value fills 0 -- an empty group, not an all-null
    one.
    """
    sessions_mean = AggregationFeature(Mean(), (AMOUNT,), SESSION_TX)
    feature = AggregationFeature(NUnique(), (sessions_mean,), CUSTOMER_SESSION)
    got = collect([feature], db)
    assert got[feature.name].to_list() == [2, 1, 0]


def test_multi_output_aggregation_produces_one_column_per_output(db):
    feature = AggregationFeature(Quantiles(qs=(0.0, 1.0)), (AMOUNT,), SESSION_TX)
    got = compile_features(FeatureList([feature]), db).collect().to_native().sort("id")
    assert got[feature.output_names[0]].to_list() == [1.0, 10.0, None]
    assert got[feature.output_names[1]].to_list() == [3.0, 20.0, None]


def test_many_aggregations_from_one_child_produce_one_join(db):
    features = [
        AggregationFeature(Count(), (), SESSION_TX),
        AggregationFeature(Mean(), (AMOUNT,), SESSION_TX),
        AggregationFeature(Sum(), (AMOUNT,), SESSION_TX),
    ]
    plan = compile_features(FeatureList(features), db).to_native().explain()
    assert plan.count("LEFT JOIN:") == 1
    assert plan.count("AGGREGATE") == 1


def test_a_hand_built_cutoff_time_aggregation_computes(db):
    """resolve() never runs on a hand-built AggregationFeature.

    deep_feature_synthesis routes every primitive through
    tusk.primitives.registry.resolve(), but constructing an AggregationFeature
    directly and compiling it -- a supported path, used throughout this file
    -- skips resolve() entirely. _add_aggregations now threads cutoff_time
    into the primitive's outputs() exactly as _apply does for row-wise
    features, so this computes rather than raising.

    The cutoff sits after every transaction. Session 10's latest transaction
    is 2024-03-04 02:00, session 20's is 2024-03-05 02:00, so the minimum
    cutoff-relative gap in each group is the distance to that row. Session 30
    has no transactions, so its group never appears in the aggregate and the
    left join leaves it null.
    """
    feature = AggregationFeature(CutoffAggregation(), (OCCURRED_AT,), SESSION_TX)
    cutoff_time = datetime(2024, 3, 10)
    got = collect([feature], db, cutoff_time=cutoff_time)
    assert got[feature.name].to_list() == [
        timedelta(days=5, hours=22),
        timedelta(days=4, hours=22),
        None,
    ]


def test_a_dfs_synthesized_cutoff_time_aggregation_computes(db):
    """A cutoff-measuring aggregation computes when reached through DFS.

    test_a_hand_built_cutoff_time_aggregation_computes above pins
    _add_aggregations in isolation, by constructing an AggregationFeature
    directly -- which skips resolve() and synthesize() entirely. This drives
    the same primitive through deep_feature_synthesis so resolve(),
    synthesize() and compile_features() are all exercised end to end, the way
    a caller actually reaches _add_aggregations. Expected values match
    test_a_hand_built_cutoff_time_aggregation_computes.
    """
    feature_matrix, features = tusk.deep_feature_synthesis(
        database=db,
        target_table="sessions",
        agg_primitives=[CutoffAggregation()],
        trans_primitives=[],
        max_depth=1,
        cutoff_time=datetime(2024, 3, 10),
    )
    (feature,) = [f for f in features if isinstance(f, AggregationFeature)]
    got = feature_matrix.collect().sort("id")
    assert got[feature.name].to_list() == [
        timedelta(days=5, hours=22),
        timedelta(days=4, hours=22),
        None,
    ]


def test_a_dfs_requested_cutoff_time_transform_used_as_an_aggregation_fails(db):
    """TimeSince is NeedsCutoffTime, TransformPrimitive -- not an AggregationPrimitive.

    synthesize() now checks that every agg_primitives entry is actually an
    AggregationPrimitive, so requesting time_since by name through
    agg_primitives is rejected before it ever reaches the compiler: it is a
    user error, naming the wrong argument for a transform primitive, rather
    than a tusk limitation for narwhals' group_by().agg() to discover on its
    own terms.
    """
    with pytest.raises(tusk.exceptions.PrimitiveError, match="'time_since'"):
        tusk.deep_feature_synthesis(
            database=db,
            target_table="customers",
            agg_primitives=["time_since"],
            trans_primitives=[],
            max_depth=2,
            cutoff_time=datetime(2024, 3, 1),
        )


def test_an_update_survives_an_aggregation(updating_db):
    # Count is zero-arity in this codebase, so a base column cannot feed it;
    # n_unique is the aggregation primitive that both takes a column input
    # and is sensitive to the value being counted.
    status = IdentityFeature("orders", "status", nw.String())
    distinct_statuses = AggregationFeature(
        resolve("n_unique"),
        (status,),
        Relationship("customers", "orders", "customer_id"),
    )
    got = (
        compile_features(
            FeatureList([distinct_statuses]),
            updating_db,
            cutoff_time=datetime(2024, 6, 1),
        )
        .collect()
        .to_native()
        .sort("id")
    )
    # Without masking, customer 1's orders show 'delivered' and 'pending' (2
    # distinct values); with order 10's status rewound to 'pending', both of
    # customer 1's orders read 'pending' (1 distinct value). Customer 2's
    # single order is unaffected either way.
    assert got["N_UNIQUE__orders__status"].to_list() == [1, 1]


def test_an_updated_foreign_key_stops_a_child_reaching_its_parent():
    customers = pl.LazyFrame(
        {"id": [1, 2], "signed_up_at": [datetime(2024, 1, 1)] * 2},
    )
    orders = pl.LazyFrame(
        {
            "id": [10, 11],
            "customer_id": [1, 2],
            "amount": [1.0, 2.0],
            "placed_at": [datetime(2024, 3, 1)] * 2,
            "moved_at": [datetime(2024, 9, 1), None],
        },
    )
    db = (
        tusk.Database("shop")
        .add_table(
            "customers", customers, primary_key="id", row_creation_time="signed_up_at"
        )
        .add_table(
            "orders",
            orders,
            primary_key="id",
            row_creation_time="placed_at",
            row_update_times={"moved_at": {"customer_id": None, "moved_at": None}},
        )
        .add_relationship(parent="customers", child="orders", foreign_key="customer_id")
    )
    amount = IdentityFeature("orders", "amount", nw.Float64())
    total = AggregationFeature(
        resolve("sum"), (amount,), Relationship("customers", "orders", "customer_id")
    )
    got = (
        compile_features(FeatureList([total]), db, cutoff_time=datetime(2024, 6, 1))
        .collect()
        .to_native()
        .sort("id")
    )
    # Order 10 was moved to customer 1 after the cutoff, so at the cutoff it
    # belonged to nobody. Sum's default_value of 0 covers customer 1's
    # resulting empty group, per the empty-group convention in this file.
    assert got["SUM__orders__amount"].to_list() == [0.0, 2.0]


def test_where_clause_masks_the_aggregated_rows():
    """Only rows passing the clause reach the aggregation."""
    database = clause_database()
    features = FeatureList(
        [
            AggregationFeature(
                resolve("sum"),
                (IdentityFeature("orders", "amount", nw.Float64()),),
                Relationship("customers", "orders", "customer_id"),
                clause=("where", "large"),
            ),
        ],
    )
    matrix = nw.from_native(features.apply(database)).lazy().collect()
    values = dict(
        zip(matrix["id"], matrix["SUM__orders__amount__WHERE__large"], strict=True),
    )
    assert values == {1: 30.0, 2: 0.0}


def test_clause_reaches_a_direct_feature_from_a_third_table():
    """A mask on the child's rows masks values joined onto that child."""
    database = clause_database()
    price = DirectFeature(
        IdentityFeature("products", "price", nw.Float64()),
        Relationship("products", "orders", "product_id"),
    )
    features = FeatureList(
        [
            AggregationFeature(
                resolve("sum"),
                (price,),
                Relationship("customers", "orders", "customer_id"),
                clause=("where", "large"),
            ),
        ],
    )
    matrix = nw.from_native(features.apply(database)).lazy().collect()
    column = [c for c in matrix.columns if c.startswith("SUM__orders")][0]
    assert dict(zip(matrix["id"], matrix[column], strict=True)) == {1: 7.0, 2: 0.0}


def test_when_clause_receives_the_cutoff_time():
    """The same feature gives different values at two cutoffs."""
    database = clause_database()
    features = FeatureList(
        [
            AggregationFeature(
                resolve("count"),
                (),
                Relationship("customers", "orders", "customer_id"),
                clause=("when", "open"),
            ),
        ],
    )
    early = (
        nw.from_native(
            features.apply(database, cutoff_time=datetime(2024, 5, 1)),
        )
        .lazy()
        .collect()
    )
    late = (
        nw.from_native(
            features.apply(database, cutoff_time=datetime(2024, 8, 1)),
        )
        .lazy()
        .collect()
    )
    name = "COUNT__orders__WHEN__open"
    assert dict(zip(early["id"], early[name], strict=True)) == {1: 2, 2: 1}
    assert dict(zip(late["id"], late[name], strict=True)) == {1: 1, 2: 1}


def test_when_clause_without_a_cutoff_time_is_rejected():
    """A cutoff-measuring clause names itself in the error."""
    from tusk.exceptions import ValidationError

    database = clause_database()
    features = FeatureList(
        [
            AggregationFeature(
                resolve("count"),
                (),
                Relationship("customers", "orders", "customer_id"),
                clause=("when", "open"),
            ),
        ],
    )
    with pytest.raises(ValidationError, match="open"):
        features.apply(database)


def test_where_clause_without_a_cutoff_time_is_allowed():
    """A static clause measures nothing, so it needs no cutoff."""
    database = clause_database()
    features = FeatureList(
        [
            AggregationFeature(
                resolve("count"),
                (),
                Relationship("customers", "orders", "customer_id"),
                clause=("where", "large"),
            ),
        ],
    )
    matrix = nw.from_native(features.apply(database)).lazy().collect()
    assert dict(
        zip(matrix["id"], matrix["COUNT__orders__WHERE__large"], strict=True),
    ) == {1: 2, 2: 0}


def test_empty_mask_falls_back_to_the_primitive_default_on_polars():
    """No matching rows gives the same value as no rows at all.

    The duckdb twin of this test is
    ``test_empty_mask_falls_back_to_the_primitive_default`` in
    ``tests/test_backend_duckdb.py``, against identical expected values.
    """
    database = clause_database()
    features = FeatureList(
        [
            AggregationFeature(
                resolve(name),
                (IdentityFeature("orders", "amount", nw.Float64()),),
                Relationship("customers", "orders", "customer_id"),
                clause=("where", "impossible"),
            )
            for name in ("sum", "mean")
        ],
    )
    matrix = nw.from_native(features.apply(database)).lazy().collect()
    assert matrix["SUM__orders__amount__WHERE__impossible"].to_list() == [0.0, 0.0]
    assert matrix["MEAN__orders__amount__WHERE__impossible"].to_list() == [None, None]


def test_clause_sees_pre_update_values(updating_db):
    """A clause on an updated column reads the value restored by the cutoff."""
    database = updating_db
    schema = database.schema("orders")
    database._schemas["orders"] = replace(
        schema,
        where={"delivered": nw.col("status") == "delivered"},
    )
    features = FeatureList(
        [
            AggregationFeature(
                resolve("count"),
                (),
                Relationship("customers", "orders", "customer_id"),
                clause=("where", "delivered"),
            ),
        ],
    )
    matrix = (
        nw.from_native(
            features.apply(database, cutoff_time=datetime(2024, 6, 1)),
        )
        .lazy()
        .collect()
    )
    assert dict(
        zip(matrix["id"], matrix["COUNT__orders__WHERE__delivered"], strict=True),
    ) == {1: 0, 2: 1}


def test_a_clause_does_not_scope_the_tables_below_it():
    """A clause masks its own table's rows, never its children's.

    Car 1 was owned by customer 1 and is now owned by customer 2, and was
    repaired twice under each owner. A current clause on cars selects the
    car for customer 2 only, and that car brings its whole repair history
    with it -- all four repairs, not the two from customer 2's era.
    """
    customers = pl.LazyFrame(
        {"id": [1, 2], "signed_up_at": [datetime(2024, 1, 1)] * 2},
    )
    # Car 1 carries every repair (see ``repairs`` below), so it is the row
    # that must currently belong to customer 2 for that history to reach
    # them; car 2 is a decoy, sold, so customer 1's "current" group is
    # masked to empty rather than simply absent.
    cars = pl.LazyFrame(
        {
            "id": [1, 2],
            "customer_id": [2, 1],
            "bought_at": [datetime(2024, 1, 1)] * 2,
            "sold_at": [None, datetime(2024, 6, 1)],
        },
    )
    repairs = pl.LazyFrame(
        {
            "id": [100, 101, 102, 103],
            "car_id": [1, 1, 1, 1],
            "cost": [1.0, 2.0, 4.0, 8.0],
            "repaired_at": [
                datetime(2024, 3, 1),
                datetime(2024, 4, 1),
                datetime(2024, 8, 1),
                datetime(2024, 9, 1),
            ],
        },
    )
    database = (
        tusk.Database("garage")
        .add_table(
            "customers",
            customers,
            primary_key="id",
            row_creation_time="signed_up_at",
        )
        .add_table(
            "cars",
            cars,
            primary_key="id",
            row_creation_time="bought_at",
            when={
                "current": lambda cutoff: (
                    nw.col("sold_at").is_null() | (nw.col("sold_at") > cutoff)
                ),
            },
        )
        .add_table(
            "repairs",
            repairs,
            primary_key="id",
            row_creation_time="repaired_at",
        )
        .add_relationship(parent="customers", child="cars", foreign_key="customer_id")
        .add_relationship(parent="cars", child="repairs", foreign_key="car_id")
    )

    repair_count = AggregationFeature(
        resolve("count"),
        (),
        Relationship("cars", "repairs", "car_id"),
    )
    feature = AggregationFeature(
        resolve("sum"),
        (repair_count,),
        Relationship("customers", "cars", "customer_id"),
        clause=("when", "current"),
    )
    features = FeatureList([feature])
    matrix = (
        nw.from_native(
            features.apply(database, cutoff_time=datetime(2024, 12, 1)),
        )
        .lazy()
        .collect()
    )

    values = dict(zip(matrix["id"], matrix[feature.name], strict=True))
    assert values == {1: 0, 2: 4}
