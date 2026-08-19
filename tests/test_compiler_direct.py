import narwhals as nw

from tusk.compiler import compile_features
from tusk.database import Relationship
from tusk.features import AggregationFeature, DirectFeature, IdentityFeature
from tusk.primitives.aggregation import Count

CUSTOMER_SESSION = Relationship("customers", "sessions", "customer_id")
SESSION_TX = Relationship("sessions", "transactions", "session_id")

AGE = IdentityFeature("customers", "age", nw.Int64())
AMOUNT = IdentityFeature("transactions", "amount", nw.Float64())


def test_direct_feature_copies_parent_column_down(db):
    feature = DirectFeature(AGE, CUSTOMER_SESSION)
    got = compile_features([feature], db).collect().to_native().sort("id")
    # sessions 10, 20 belong to customer 1 (age 30); session 30 to customer 2 (age 40).
    assert got["customers__age"].to_list() == [30, 30, 40]


def test_direct_feature_of_a_derived_parent_feature(db):
    parent_count = AggregationFeature(Count(), (), CUSTOMER_SESSION)
    feature = DirectFeature(parent_count, CUSTOMER_SESSION)
    got = compile_features([feature], db).collect().to_native().sort("id")
    assert got["customers__COUNT__sessions"].to_list() == [2, 2, 1]


def test_direct_features_from_one_parent_share_a_join(db):
    count_feature = AggregationFeature(Count(), (), CUSTOMER_SESSION)
    features = [
        DirectFeature(AGE, CUSTOMER_SESSION),
        DirectFeature(count_feature, CUSTOMER_SESSION),
    ]
    plan = compile_features(features, db).to_native().explain()
    # One join for the direct features, one inside the parent for its own aggregate.
    assert plan.count("LEFT JOIN:") == 2


def test_transform_can_stack_on_a_direct_feature(db):
    from tusk.features import TransformFeature
    from tusk.primitives.registry import resolve

    direct = DirectFeature(AGE, CUSTOMER_SESSION)
    feature = TransformFeature(resolve("absolute"), (direct,))
    got = compile_features([feature], db).collect().to_native().sort("id")
    assert got["ABSOLUTE__customers__age"].to_list() == [30, 30, 40]
