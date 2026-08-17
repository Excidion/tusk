from dataclasses import dataclass

import narwhals as nw
import polars as pl
import pytest

import tusk
from tusk.compiler import compile_features
from tusk.dtypes import DtypeFamily as F
from tusk.entityset import Relationship
from tusk.exceptions import PrimitiveError
from tusk.features import GroupByTransformFeature, IdentityFeature, TransformFeature
from tusk.primitives.base import TransformPrimitive
from tusk.primitives.registry import register, resolve

SESSION_TX = Relationship("sessions", "transactions", "session_id")
AMOUNT = IdentityFeature("transactions", "amount", nw.Float64())


@register
@dataclass(frozen=True)
class ShareOfGroupTotal(TransformPrimitive):
    """Test-only primitive: value divided by its group's total.

    Exists to cover the grouped, non-order-dependent path of `_apply`
    (`.over(partition)` with no `order_by`) end to end. No *built-in*
    primitive can exercise that branch: every registered
    non-order-dependent TransformPrimitive (absolute, add_numeric, year,
    ...) is elementwise, and narwhals forbids `.over()` on a purely
    elementwise expression (verified: wrapping e.g. `abs()` in `.over()`
    raises `InvalidOperationError: Cannot apply over to elementwise
    expression`, regardless of order_by). That doesn't make the branch
    dead -- it is the compiler's extension point for user-defined,
    group-aware transform primitives, and `x / x.sum()` (share of a
    group's total) is a canonical example of one: it combines a per-row
    value with a group aggregate, exactly the shape `.over(partition)`
    exists for.
    """

    name = "share_of_group_total"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64

    def build(self, expr):
        return expr / expr.sum()


def test_groupby_cum_sum_restarts_per_group(es):
    feature = GroupByTransformFeature(resolve("cum_sum"), (AMOUNT,), SESSION_TX)
    got = compile_features([feature], es).collect().to_native().sort("id")
    # session 10: 1, 1+3; session 20: 10, 10+20
    assert got[feature.name].to_list() == [1.0, 4.0, 10.0, 30.0]


def test_ungrouped_order_dependent_transform(es):
    feature = TransformFeature(resolve("cum_sum"), (AMOUNT,))
    got = compile_features([feature], es).collect().to_native().sort("id")
    assert got["CUM_SUM(amount)"].to_list() == [1.0, 4.0, 14.0, 34.0]


def test_ungrouped_order_dependent_transform_uses_row_creation_time_not_frame_order():
    frame = pl.LazyFrame(
        {
            "id": [1, 2, 3],
            "v": [100.0, 1.0, 10.0],
            "t": [3, 1, 2],  # deliberately not row order
        }
    )
    es = tusk.EntitySet("x").add_dataframe(
        "c", frame, primary_key="id", row_creation_time="t"
    )
    feature = TransformFeature(
        resolve("cum_sum"), (IdentityFeature("c", "v", nw.Float64()),)
    )
    got = compile_features([feature], es).collect().to_native().sort("id")
    # ordered by t: 1.0, then 10.0, then 100.0 -> cumulative 111.0, 1.0, 11.0 by id
    assert got[feature.name].to_list() == [111.0, 1.0, 11.0]


def test_groupby_non_order_dependent_transform_applies_over_partition(es):
    feature = GroupByTransformFeature(
        resolve("share_of_group_total"), (AMOUNT,), SESSION_TX
    )
    got = compile_features([feature], es).collect().to_native().sort("id")
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
            "t": [3, 1, 2],  # deliberately not row order
        }
    )
    parent = pl.LazyFrame({"id": [1]})
    es = (
        tusk.EntitySet("x")
        .add_dataframe("p", parent, primary_key="id")
        .add_dataframe("c", frame, primary_key="id", row_creation_time="t")
        .add_relationship(parent="p", child="c", foreign_key="g")
    )
    feature = GroupByTransformFeature(
        resolve("cum_sum"),
        (IdentityFeature("c", "v", nw.Float64()),),
        Relationship("p", "c", "g"),
    )
    got = compile_features([feature], es).collect().to_native().sort("id")
    # ordered by t: 1.0, then 10.0, then 100.0 -> cumulative 111.0, 1.0, 11.0 by id
    assert got[feature.name].to_list() == [111.0, 1.0, 11.0]


def test_order_dependent_primitive_without_row_creation_time_raises():
    es = tusk.EntitySet("x").add_dataframe(
        "t", pl.LazyFrame({"id": [1], "v": [1.0]}), primary_key="id"
    )
    feature = TransformFeature(
        resolve("cum_sum"), (IdentityFeature("t", "v", nw.Float64()),)
    )
    with pytest.raises(PrimitiveError, match="row_creation_time"):
        compile_features([feature], es)
