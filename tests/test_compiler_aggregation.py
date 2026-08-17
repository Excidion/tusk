import narwhals as nw

from tusk.compiler import compile_features
from tusk.entityset import Relationship
from tusk.features import AggregationFeature, IdentityFeature
from tusk.primitives.aggregation import Count, Mean, Quantiles, Sum

CUSTOMER_SESSION = Relationship("customers", "sessions", "customer_id")
SESSION_TX = Relationship("sessions", "transactions", "session_id")

AMOUNT = IdentityFeature("transactions", "amount", nw.Float64())


def collect(features, es):
    return compile_features(features, es).collect().to_native().sort("id")


def test_count_of_children(es):
    # customer 1 has sessions 10 and 20; customer 2 has session 30; customer 3 has none.
    got = collect([AggregationFeature(Count(), (), CUSTOMER_SESSION)], es)
    assert got["COUNT(sessions)"].to_list() == [2, 1, 0]


def test_empty_group_gets_count_default_of_zero(es):
    got = collect([AggregationFeature(Count(), (), CUSTOMER_SESSION)], es)
    assert got["COUNT(sessions)"][2] == 0


def test_empty_group_gets_null_for_mean(es):
    sessions_mean = AggregationFeature(Mean(), (AMOUNT,), SESSION_TX)
    got = collect([AggregationFeature(Mean(), (sessions_mean,), CUSTOMER_SESSION)], es)
    assert got["MEAN(sessions.MEAN(transactions.amount))"][2] is None


def test_depth_two_aggregation_values(es):
    # session 10 -> mean(1, 3) = 2; session 20 -> mean(10, 20) = 15; session 30 -> null.
    # customer 1 -> mean(2, 15) = 8.5; customer 2 -> null; customer 3 -> null.
    sessions_mean = AggregationFeature(Mean(), (AMOUNT,), SESSION_TX)
    feature = AggregationFeature(Mean(), (sessions_mean,), CUSTOMER_SESSION)
    got = collect([feature], es)
    assert got[feature.name].to_list() == [8.5, None, None]


def test_multi_output_aggregation_produces_one_column_per_output(es):
    feature = AggregationFeature(Quantiles(qs=(0.0, 1.0)), (AMOUNT,), SESSION_TX)
    got = compile_features([feature], es).collect().to_native().sort("id")
    assert got[feature.output_names[0]].to_list() == [1.0, 10.0, None]
    assert got[feature.output_names[1]].to_list() == [3.0, 20.0, None]


def test_many_aggregations_from_one_child_produce_one_join(es):
    features = [
        AggregationFeature(Count(), (), SESSION_TX),
        AggregationFeature(Mean(), (AMOUNT,), SESSION_TX),
        AggregationFeature(Sum(), (AMOUNT,), SESSION_TX),
    ]
    plan = compile_features(features, es).to_native().explain()
    assert plan.count("LEFT JOIN:") == 1
    assert plan.count("AGGREGATE") == 1
