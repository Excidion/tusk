import narwhals as nw

from tusk.entityset import Relationship
from tusk.features import (
    AggregationFeature,
    DirectFeature,
    GroupByTransformFeature,
    IdentityFeature,
    TransformFeature,
)
from tusk.primitives.aggregation import Count, Mean, Quantiles
from tusk.primitives.registry import resolve

CUSTOMER_SESSION = Relationship("customers", "sessions", "customer_id")
SESSION_TX = Relationship("sessions", "transactions", "session_id")

amount = IdentityFeature("transactions", "amount", nw.Float64())


def test_identity_feature():
    assert amount.name == "amount"
    assert amount.depth == 0
    assert amount.table == "transactions"
    assert amount.output_names == ("amount",)


def test_aggregation_feature_names_and_depth():
    feature = AggregationFeature(Mean(), (amount,), SESSION_TX)
    assert feature.name == "MEAN__transactions__amount"
    assert feature.table == "sessions"
    assert feature.depth == 1
    assert feature.dtype == nw.Float64


def test_zero_arity_aggregation_names_the_table():
    feature = AggregationFeature(Count(), (), SESSION_TX)
    assert feature.name == "COUNT__transactions"
    assert feature.depth == 1


def test_stacked_aggregation_reaches_depth_two():
    inner = AggregationFeature(Mean(), (amount,), SESSION_TX)
    outer = AggregationFeature(resolve("sum"), (inner,), CUSTOMER_SESSION)
    assert outer.name == "SUM__sessions__MEAN__transactions__amount"
    assert outer.depth == 2
    assert outer.table == "customers"


def test_direct_feature():
    age = IdentityFeature("customers", "age", nw.Int64())
    feature = DirectFeature(age, CUSTOMER_SESSION)
    assert feature.name == "customers__age"
    assert feature.table == "sessions"
    assert feature.depth == 1
    assert feature.dtype == nw.Int64()
    assert feature.base_features == (age,)


def test_transform_feature():
    started = IdentityFeature("sessions", "started_at", nw.Datetime())
    feature = TransformFeature(resolve("month"), (started,))
    assert feature.name == "MONTH__started_at"
    assert feature.table == "sessions"
    assert feature.depth == 1


def test_groupby_transform_feature_names_the_group():
    feature = GroupByTransformFeature(resolve("cum_sum"), (amount,), SESSION_TX)
    assert feature.name == "CUM_SUM__amount__by__session_id"
    assert feature.table == "transactions"
    assert feature.depth == 1


def test_multi_output_feature_expands_names():
    feature = AggregationFeature(Quantiles(qs=(0.5, 0.9)), (amount,), SESSION_TX)
    assert feature.output_names == (
        "QUANTILES__transactions__amount__0",
        "QUANTILES__transactions__amount__1",
    )


def test_features_deduplicate_by_structural_equality():
    a = AggregationFeature(Mean(), (amount,), SESSION_TX)
    b = AggregationFeature(Mean(), (amount,), SESSION_TX)
    assert a == b
    assert len({a, b}) == 1
