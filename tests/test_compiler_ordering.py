import narwhals as nw
import polars as pl
import pytest

import tusk
from tusk.compiler import compile_features
from tusk.entityset import Relationship
from tusk.exceptions import PrimitiveError
from tusk.features import GroupByTransformFeature, IdentityFeature, TransformFeature
from tusk.primitives.registry import resolve

SESSION_TX = Relationship("sessions", "transactions", "session_id")
AMOUNT = IdentityFeature("transactions", "amount", nw.Float64())


def test_groupby_cum_sum_restarts_per_group(es):
    feature = GroupByTransformFeature(resolve("cum_sum"), (AMOUNT,), SESSION_TX)
    got = compile_features([feature], es).collect().to_native().sort("id")
    # session 10: 1, 1+3; session 20: 10, 10+20
    assert got[feature.name].to_list() == [1.0, 4.0, 10.0, 30.0]


def test_ungrouped_order_dependent_transform(es):
    feature = TransformFeature(resolve("cum_sum"), (AMOUNT,))
    got = compile_features([feature], es).collect().to_native().sort("id")
    assert got["CUM_SUM(amount)"].to_list() == [1.0, 4.0, 14.0, 34.0]


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
