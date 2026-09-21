from dataclasses import dataclass

import narwhals as nw
import pytest

from tusk.database import Relationship
from tusk.dtypes import DtypeFamily as F
from tusk.exceptions import PrimitiveError
from tusk.features import (
    AggregationFeature,
    DirectFeature,
    GroupByTransformFeature,
    IdentityFeature,
    TransformFeature,
)
from tusk.primitives.aggregation import Count, Mean, Quantiles
from tusk.primitives.base import Primitive
from tusk.primitives.registry import resolve

CUSTOMER_SESSION = Relationship("customers", "sessions", "customer_id")
SESSION_TX = Relationship("sessions", "transactions", "session_id")

amount = IdentityFeature("transactions", "amount", nw.Float64())


@dataclass(frozen=True)
class Neither(Primitive):
    """A primitive that is neither an aggregation nor a transform primitive.

    Stands in for a custom primitive that subclasses only
    :class:`~tusk.primitives.base.Primitive`, to exercise the case
    :func:`~tusk.features._require_kind` cannot name a rejected
    primitive's actual kind for.
    """

    name = "neither"
    input_dtypes = (F.NUMERIC,)

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Pass the input through unchanged."""
        return expr


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


@pytest.mark.parametrize("name", ["cum_sum", "percentile"])
def test_transform_feature_rejects_a_group_transform_primitive(name):
    with pytest.raises(PrimitiveError, match="runs within foreign-key groups"):
        TransformFeature(resolve(name), (amount,))


def test_groupby_transform_feature_rejects_a_row_wise_primitive():
    with pytest.raises(PrimitiveError, match="'absolute'.*GroupTransformPrimitive"):
        GroupByTransformFeature(resolve("absolute"), (amount,), SESSION_TX)


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


def test_transform_feature_rejects_an_aggregation_primitive():
    with pytest.raises(PrimitiveError, match="'mean'.*transform"):
        TransformFeature(Mean(), (amount,))


def test_aggregation_feature_rejects_a_transform_primitive():
    with pytest.raises(PrimitiveError, match="'month'.*aggregation"):
        AggregationFeature(resolve("month"), (amount,), SESSION_TX)


def test_transform_feature_rejects_a_primitive_of_neither_kind():
    with pytest.raises(
        PrimitiveError,
        match=(
            "primitive mismatch: 'neither' in a transform feature is not "
            "TransformPrimitive"
        ),
    ):
        TransformFeature(Neither(), (amount,))


def test_condition_appears_in_aggregation_name():
    """A where condition becomes a trailing name part."""
    feature = AggregationFeature(
        resolve("count"),
        (),
        Relationship("customers", "orders", "customer_id"),
        condition=("where", "enterprise"),
    )
    assert feature.name == "COUNT__orders__WHERE__enterprise"
    assert feature.display_name == "COUNT(orders WHERE enterprise)"


def test_when_condition_uses_the_when_token():
    """A when condition is distinguishable from a where condition of the same key."""
    relationship = Relationship("customers", "orders", "customer_id")
    where = AggregationFeature(
        resolve("count"),
        (),
        relationship,
        condition=("where", "current"),
    )
    when = AggregationFeature(
        resolve("count"),
        (),
        relationship,
        condition=("when", "current"),
    )
    assert where.name == "COUNT__orders__WHERE__current"
    assert when.name == "COUNT__orders__WHEN__current"
    assert where != when


def test_condition_renders_after_the_last_argument():
    """A condition on a one-input aggregation sits inside the parentheses."""
    feature = AggregationFeature(
        resolve("sum"),
        (IdentityFeature("orders", "amount", nw.Float64()),),
        Relationship("customers", "orders", "customer_id"),
        condition=("when", "current"),
    )
    assert feature.name == "SUM__orders__amount__WHEN__current"
    assert feature.display_name == "SUM(orders.amount WHEN current)"


def test_multi_output_condition_feature_keeps_the_output_index_last():
    """A condition's name parts are baked into the base name, so __0/__1 stays last."""
    feature = AggregationFeature(
        Quantiles(qs=(0.5, 0.9)),
        (IdentityFeature("orders", "amount", nw.Float64()),),
        Relationship("customers", "orders", "customer_id"),
        condition=("where", "large"),
    )
    assert feature.output_names == (
        "QUANTILES__orders__amount__WHERE__large__0",
        "QUANTILES__orders__amount__WHERE__large__1",
    )


def test_unconditioned_aggregation_name_is_unchanged():
    """The existing path keeps its exact name."""
    feature = AggregationFeature(
        resolve("count"),
        (),
        Relationship("customers", "orders", "customer_id"),
    )
    assert feature.name == "COUNT__orders"
    assert feature.display_name == "COUNT(orders)"
