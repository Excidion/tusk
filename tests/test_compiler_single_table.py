import datetime as dt

import narwhals as nw

from tusk.compiler import compile_features
from tusk.features import IdentityFeature, TransformFeature
from tusk.primitives.registry import resolve


def test_identity_features_round_trip(es):
    age = IdentityFeature("customers", "age", nw.Int64())
    got = compile_features([age], es).collect().to_native().sort("id")
    assert got.columns == ["id", "age"]
    assert got["age"].to_list() == [30, 40, 50]


def test_transform_feature_is_computed(es):
    started = IdentityFeature("sessions", "started_at", nw.Datetime())
    feature = TransformFeature(resolve("day"), (started,))
    got = compile_features([feature], es).collect().to_native().sort("id")
    assert got["DAY(started_at)"].to_list() == [4, 5, 6]


def test_stacked_transform_is_computed(es):
    started = IdentityFeature("sessions", "started_at", nw.Datetime())
    day = TransformFeature(resolve("day"), (started,))
    doubled = TransformFeature(resolve("add_numeric"), (day, day))
    got = compile_features([day, doubled], es).collect().to_native().sort("id")
    assert got["ADD_NUMERIC(DAY(started_at), DAY(started_at))"].to_list() == [8, 10, 12]


def test_cutoff_filters_rows(es):
    started = IdentityFeature("sessions", "started_at", nw.Datetime())
    feature = TransformFeature(resolve("day"), (started,))
    compiled = compile_features([feature], es, cutoff_time=dt.datetime(2024, 3, 5))
    got = compiled.collect().to_native()
    assert got["id"].to_list() == [10, 20]


def test_result_stays_lazy(es):
    age = IdentityFeature("customers", "age", nw.Int64())
    assert isinstance(compile_features([age], es), nw.LazyFrame)
