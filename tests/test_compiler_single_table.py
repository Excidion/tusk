import datetime as dt

import narwhals as nw
import polars as pl
import pytest

import tusk
from tusk.compiler import compile_features
from tusk.exceptions import MissingPrimaryKeyWarning, SchemaError
from tusk.features import IdentityFeature, TransformFeature
from tusk.primitives.registry import resolve


def test_identity_features_round_trip(db):
    age = IdentityFeature("customers", "age", nw.Int64())
    got = compile_features([age], db).collect().to_native().sort("id")
    assert got.columns == ["id", "age"]
    assert got["age"].to_list() == [30, 40, 50]


def test_transform_feature_is_computed(db):
    started = IdentityFeature("sessions", "started_at", nw.Datetime())
    feature = TransformFeature(resolve("day"), (started,))
    got = compile_features([feature], db).collect().to_native().sort("id")
    assert got["DAY__started_at"].to_list() == [4, 5, 6]


def test_stacked_transform_is_computed(db):
    started = IdentityFeature("sessions", "started_at", nw.Datetime())
    day = TransformFeature(resolve("day"), (started,))
    doubled = TransformFeature(resolve("add_numeric"), (day, day))
    got = compile_features([day, doubled], db).collect().to_native().sort("id")
    assert got["ADD_NUMERIC__DAY__started_at__DAY__started_at"].to_list() == [8, 10, 12]


def test_cutoff_filters_rows(db):
    started = IdentityFeature("sessions", "started_at", nw.Datetime())
    feature = TransformFeature(resolve("day"), (started,))
    compiled = compile_features([feature], db, cutoff_time=dt.datetime(2024, 3, 5))
    got = compiled.collect().to_native()
    assert got["id"].to_list() == [10, 20]


def test_cutoff_filters_the_target_table_too(db):
    """The matrix can have fewer rows than the target table. This is intended.

    A target row that did not exist yet at the cutoff has no features to
    compute, which is also what featuretools does.
    """
    age = IdentityFeature("customers", "age", nw.Int64())
    before_any_signup = dt.datetime(2023, 12, 31)
    got = compile_features([age], db, cutoff_time=before_any_signup).collect()
    assert got.to_native().height == 0


def test_cutoff_is_a_no_op_without_a_row_creation_time():
    """A table with no row_creation_time is timeless and passes through.

    Documented rather than warned (spec section 8), so a database that
    declares no row_creation_time anywhere silently ignores a cutoff. This is
    the branch a user is most likely to be surprised by, so it is pinned.
    """
    timeless = tusk.Database("t").add_table(
        "events",
        pl.LazyFrame({"id": [1, 2, 3], "n": [1.0, 2.0, 3.0]}),
        primary_key="id",
    )
    feature = IdentityFeature("events", "n", nw.Float64())
    got = compile_features(
        [feature], timeless, cutoff_time=dt.datetime(1970, 1, 1)
    ).collect()
    assert got.to_native()["n"].to_list() == [1.0, 2.0, 3.0]


def test_empty_feature_list_raises(db):
    with pytest.raises(SchemaError, match="no features"):
        compile_features([], db)


def test_features_spanning_multiple_tables_raise(db):
    age = IdentityFeature("customers", "age", nw.Int64())
    started = IdentityFeature("sessions", "started_at", nw.Datetime())
    with pytest.raises(SchemaError, match="customers.*sessions"):
        compile_features([age, started], db)


def test_target_without_a_primary_key_raises():
    with pytest.warns(MissingPrimaryKeyWarning):
        db = tusk.Database("t").add_table(
            "events", pl.LazyFrame({"id": [1], "n": [1.0]})
        )
    feature = IdentityFeature("events", "n", nw.Float64())
    with pytest.raises(SchemaError, match="primary_key"):
        compile_features([feature], db)


def test_result_stays_lazy(db):
    age = IdentityFeature("customers", "age", nw.Int64())
    assert isinstance(compile_features([age], db), nw.LazyFrame)
