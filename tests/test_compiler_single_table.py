import datetime as dt

import narwhals as nw
import polars as pl
import pytest

import tusk
from tusk.compiler import compile_features
from tusk.exceptions import MissingPrimaryKeyWarning, SchemaError
from tusk.feature_list import FeatureList
from tusk.features import IdentityFeature, TransformFeature
from tusk.primitives.registry import resolve


def test_identity_features_round_trip(db):
    age = IdentityFeature("customers", "age", nw.Int64())
    got = compile_features(FeatureList([age]), db).collect().to_native().sort("id")
    assert got.columns == ["id", "age"]
    assert got["age"].to_list() == [30, 40, 50]


def test_transform_feature_is_computed(db):
    started = IdentityFeature("sessions", "started_at", nw.Datetime())
    feature = TransformFeature(resolve("day"), (started,))
    got = compile_features(FeatureList([feature]), db).collect().to_native().sort("id")
    assert got["DAY__started_at"].to_list() == [4, 5, 6]


def test_stacked_transform_is_computed(db):
    started = IdentityFeature("sessions", "started_at", nw.Datetime())
    day = TransformFeature(resolve("day"), (started,))
    doubled = TransformFeature(resolve("add_numeric"), (day, day))
    got = (
        compile_features(FeatureList([day, doubled]), db)
        .collect()
        .to_native()
        .sort("id")
    )
    assert got["ADD_NUMERIC__DAY__started_at__DAY__started_at"].to_list() == [8, 10, 12]


def test_cutoff_filters_rows(db):
    started = IdentityFeature("sessions", "started_at", nw.Datetime())
    feature = TransformFeature(resolve("day"), (started,))
    compiled = compile_features(
        FeatureList([feature]), db, cutoff_time=dt.datetime(2024, 3, 5)
    )
    got = compiled.collect().to_native()
    assert got["id"].to_list() == [10, 20]


def test_cutoff_filters_the_target_table_too(db):
    """The matrix can have fewer rows than the target table. This is intended.

    A target row that did not exist yet at the cutoff has no features to
    compute, which is also what featuretools does.
    """
    age = IdentityFeature("customers", "age", nw.Int64())
    before_any_signup = dt.datetime(2023, 12, 31)
    got = compile_features(
        FeatureList([age]), db, cutoff_time=before_any_signup
    ).collect()
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
        FeatureList([feature]),
        timeless,
        cutoff_time=dt.datetime(1970, 1, 1),
    ).collect()
    assert got.to_native()["n"].to_list() == [1.0, 2.0, 3.0]


def test_target_without_a_primary_key_raises():
    with pytest.warns(MissingPrimaryKeyWarning):
        db = tusk.Database("t").add_table(
            "events",
            pl.LazyFrame({"id": [1], "n": [1.0]}),
        )
    feature = IdentityFeature("events", "n", nw.Float64())
    with pytest.raises(SchemaError, match="primary_key"):
        compile_features(FeatureList([feature]), db)


def test_result_stays_lazy(db):
    age = IdentityFeature("customers", "age", nw.Int64())
    assert isinstance(compile_features(FeatureList([age]), db), nw.LazyFrame)


def test_a_column_updated_after_the_cutoff_serves_its_earlier_value(updating_db):
    status = IdentityFeature("orders", "status", nw.String())
    got = (
        compile_features(
            FeatureList([status]), updating_db, cutoff_time=dt.datetime(2024, 6, 1)
        )
        .collect()
        .to_native()
        .sort("id")
    )
    # 10 was updated after the cutoff, 11 never, 12 before it.
    assert got["status"].to_list() == ["pending", "pending", "delivered"]


def test_a_null_update_time_keeps_its_value():
    # Order 11's status is 'pending' either way, so prove the null path on a
    # row whose two answers differ: give it a delivered status and no update.
    orders = pl.LazyFrame(
        {
            "id": [11],
            "placed_at": [dt.datetime(2024, 3, 1)],
            "status": ["delivered"],
            "updated_at": [None],
        },
        schema={
            "id": pl.Int64,
            "placed_at": pl.Datetime,
            "status": pl.String,
            "updated_at": pl.Datetime,
        },
    )
    db = tusk.Database("shop").add_table(
        "orders",
        orders,
        primary_key="id",
        row_creation_time="placed_at",
        row_update_times={"updated_at": {"status": "pending", "updated_at": None}},
    )
    status = IdentityFeature("orders", "status", nw.String())
    got = (
        compile_features(FeatureList([status]), db, cutoff_time=dt.datetime(2024, 6, 1))
        .collect()
        .to_native()
    )
    assert got["status"].to_list() == ["delivered"]


def test_an_update_time_updates_itself(updating_db):
    updated = IdentityFeature("orders", "updated_at", nw.Datetime())
    got = (
        compile_features(
            FeatureList([updated]), updating_db, cutoff_time=dt.datetime(2024, 6, 1)
        )
        .collect()
        .to_native()
        .sort("id")
    )
    assert got["updated_at"].to_list() == [None, None, dt.datetime(2024, 4, 1)]


def test_an_update_time_updating_itself_does_not_disturb_the_other_columns(
    updating_db,
):
    """Every mask reads the frame as it was, so order within the call is moot.

    If the masks were applied one after another, masking ``updated_at`` first
    would null it and every later condition would then read null as
    'never updated' and keep the post-cutoff status.
    """
    status = IdentityFeature("orders", "status", nw.String())
    updated = IdentityFeature("orders", "updated_at", nw.Datetime())
    got = (
        compile_features(
            FeatureList([status, updated]),
            updating_db,
            cutoff_time=dt.datetime(2024, 6, 1),
        )
        .collect()
        .to_native()
        .sort("id")
    )
    assert got["status"].to_list() == ["pending", "pending", "delivered"]
    assert got["updated_at"].to_list() == [None, None, dt.datetime(2024, 4, 1)]


def test_no_cutoff_updates_nothing(updating_db):
    status = IdentityFeature("orders", "status", nw.String())
    got = (
        compile_features(FeatureList([status]), updating_db)
        .collect()
        .to_native()
        .sort("id")
    )
    assert got["status"].to_list() == ["delivered", "pending", "delivered"]


def test_an_update_survives_a_stacked_transform(updating_db):
    """A rewound column feeds a transform of a transform, not just a read."""
    updated = IdentityFeature("orders", "updated_at", nw.Datetime())
    month = TransformFeature(resolve("month"), (updated,))
    doubled = TransformFeature(resolve("add_numeric"), (month, month))
    got = (
        compile_features(
            FeatureList([doubled]),
            updating_db,
            cutoff_time=dt.datetime(2024, 6, 1),
        )
        .collect()
        .to_native()
        .sort("id")
    )
    # Orders 10 and 11 have no known update at the cutoff; order 12's is April.
    assert got["ADD_NUMERIC__MONTH__updated_at__MONTH__updated_at"].to_list() == [
        None,
        None,
        8,
    ]


def test_a_timeless_table_survives_the_masking_step(db):
    """with_columns() rejects an empty expression list, so the no-op path
    must return the frame untouched rather than call it with nothing."""
    age = IdentityFeature("customers", "age", nw.Int64())
    got = (
        compile_features(FeatureList([age]), db, cutoff_time=dt.datetime(2024, 6, 1))
        .collect()
        .to_native()
        .sort("id")
    )
    assert got["age"].to_list() == [30, 40, 50]
