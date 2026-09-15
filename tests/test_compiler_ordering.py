from dataclasses import dataclass
from datetime import datetime

import narwhals as nw
import polars as pl
import pytest

import tusk
from tusk.compiler import compile_features
from tusk.database import Relationship
from tusk.dtypes import DtypeFamily as F
from tusk.exceptions import PrimitiveError
from tusk.feature_list import FeatureList
from tusk.features import GroupByTransformFeature, IdentityFeature
from tusk.primitives.base import GroupTransformPrimitive
from tusk.primitives.registry import register, resolve

SESSION_TX = Relationship("sessions", "transactions", "session_id")
AMOUNT = IdentityFeature("transactions", "amount", nw.Float64())
CREATION_TIMES = [datetime(2024, 1, 3), datetime(2024, 1, 1), datetime(2024, 1, 2)]


@register
@dataclass(frozen=True)
class ShareOfGroupTotal(GroupTransformPrimitive):
    """Test-only primitive: value divided by its group's total.

    Covers the grouped path of `_apply` without an ordering
    (`.over(partition)` with no `order_by`) end to end for a user-defined
    group transform primitive, the compiler's extension point: `x / x.sum()`
    combines a per-row value with a group aggregate, exactly the shape
    `.over(partition)` exists for.
    """

    name = "share_of_group_total"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64

    def build(self, expr):
        return expr / expr.sum()


def test_groupby_cum_sum_restarts_per_group(db):
    feature = GroupByTransformFeature(resolve("cum_sum"), (AMOUNT,), SESSION_TX)
    got = compile_features(FeatureList([feature]), db).collect().to_native().sort("id")
    # session 10: 1, 1+3; session 20: 10, 10+20
    assert got[feature.name].to_list() == [1.0, 4.0, 10.0, 30.0]


def test_groupby_unordered_transform_applies_over_partition(db):
    feature = GroupByTransformFeature(
        resolve("share_of_group_total"),
        (AMOUNT,),
        SESSION_TX,
    )
    got = compile_features(FeatureList([feature]), db).collect().to_native().sort("id")
    # session 10 totals 4 (1+3): shares 0.25, 0.75
    # session 20 totals 30 (10+20): shares 1/3, 2/3
    got_values = got[feature.name].to_list()
    expected = [0.25, 0.75, 10.0 / 30.0, 20.0 / 30.0]
    assert got_values == pytest.approx(expected)


def test_ordering_uses_row_creation_time_not_frame_order():
    frame = pl.LazyFrame(
        {
            "id": [1, 2, 3],
            "g": [1, 1, 1],
            "v": [100.0, 1.0, 10.0],
            "t": CREATION_TIMES,  # deliberately not row order
        },
    )
    parent = pl.LazyFrame({"id": [1]})
    db = (
        tusk.Database("x")
        .add_table("p", parent, primary_key="id")
        .add_table("c", frame, primary_key="id", row_creation_time="t")
        .add_relationship(parent="p", child="c", foreign_key="g")
    )
    feature = GroupByTransformFeature(
        resolve("cum_sum"),
        (IdentityFeature("c", "v", nw.Float64()),),
        Relationship("p", "c", "g"),
    )
    got = compile_features(FeatureList([feature]), db).collect().to_native().sort("id")
    # ordered by t: 1.0, then 10.0, then 100.0 -> cumulative 111.0, 1.0, 11.0 by id
    assert got[feature.name].to_list() == [111.0, 1.0, 11.0]


def test_ordered_primitive_without_row_creation_time_raises():
    db = (
        tusk.Database("x")
        .add_table("p", pl.LazyFrame({"id": [1]}), primary_key="id")
        .add_table(
            "t", pl.LazyFrame({"id": [1], "g": [1], "v": [1.0]}), primary_key="id"
        )
        .add_relationship(parent="p", child="t", foreign_key="g")
    )
    feature = GroupByTransformFeature(
        resolve("cum_sum"),
        (IdentityFeature("t", "v", nw.Float64()),),
        Relationship("p", "t", "g"),
    )
    with pytest.raises(PrimitiveError, match="row_creation_time"):
        compile_features(FeatureList([feature]), db)
