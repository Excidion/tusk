from dataclasses import dataclass
from datetime import datetime, timedelta

import narwhals as nw
import pytest

import tusk
from tusk.compiler import compile_features
from tusk.database import Relationship
from tusk.feature_list import FeatureList
from tusk.features import AggregationFeature, IdentityFeature
from tusk.primitives.aggregation import Count, Mean, NUnique, Quantiles, Sum
from tusk.primitives.base import AggregationPrimitive, NeedsCutoffTime

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


def test_n_unique_does_not_count_null_as_a_distinct_value(db):
    """A group whose only values are null has 0 known distinct values, not 1.

    Customer 2's only session is 30, which has no transactions, so
    MEAN(transactions.amount) is null there. Polars counts that null as a
    distinct value, which would contradict NUnique's own default_value of 0
    and diverge from featuretools' NUM_UNIQUE.

    Customer 1 has sessions 10 (mean 2.0) and 20 (mean 15.0): two distinct
    non-null values. Customer 3 has no sessions at all, so the left join
    leaves null and default_value fills 0.
    """
    sessions_mean = AggregationFeature(Mean(), (AMOUNT,), SESSION_TX)
    feature = AggregationFeature(NUnique(), (sessions_mean,), CUSTOMER_SESSION)
    got = collect([feature], db)
    assert got[feature.name].to_list() == [2, 0, 0]


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
