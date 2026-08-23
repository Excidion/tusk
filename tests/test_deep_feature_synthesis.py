from datetime import datetime

import polars as pl
import pytest

import tusk
from tusk.primitives import Quantiles


def test_readme_example_compiles_and_collects(db):
    """The README's headline call, verbatim, against the README's schema.

    A multi-output primitive at max_depth=2 previously emitted a phantom
    un-indexed column and raised ColumnNotFoundError here.
    """
    feature_matrix, features = tusk.deep_feature_synthesis(
        database=db,
        target_table="customers",
        agg_primitives=["mean", "count", Quantiles(qs=(0.25, 0.5, 0.75))],
        trans_primitives=["month", "weekday"],
        max_depth=2,
        cutoff_time=datetime(2026, 1, 1),
    )
    got = feature_matrix.collect()
    expected = {name for f in features for name in f.output_names}
    assert expected <= set(got.columns)
    assert any(c.startswith("QUANTILES__") for c in got.columns)


def test_deep_feature_synthesis_end_to_end(db):
    matrix, features = tusk.deep_feature_synthesis(
        database=db,
        target_table="customers",
        agg_primitives=["count", "mean"],
        trans_primitives=[],
        max_depth=2,
    )
    assert isinstance(matrix, pl.LazyFrame)
    got = matrix.collect().sort("id")
    assert got["COUNT__sessions"].to_list() == [2, 1, 0]
    assert {f.name for f in features} <= set(got.columns)


def test_features_only_returns_definitions_alone(db):
    features = tusk.deep_feature_synthesis(
        database=db,
        target_table="customers",
        agg_primitives=["count"],
        trans_primitives=[],
        max_depth=1,
        features_only=True,
    )
    assert isinstance(features, list)
    assert {f.name for f in features} == {"age", "COUNT__sessions"}


def test_defaults_are_applied(db):
    features = tusk.deep_feature_synthesis(
        database=db,
        target_table="customers",
        features_only=True,
    )
    names = {f.name for f in features}
    assert "COUNT__sessions" in names
    assert not any(n.startswith("ADD_NUMERIC") for n in names)


def test_zero_config_generates_transform_features(db):
    """The defaults are temporal-only transforms, so they need a temporal input.

    Every non-key column on this schema that a default transform can read is a
    row_creation_time. Excluding those from primitive inputs left a zero-config
    run with aggregations and passthrough columns but not one transform.
    """
    features = tusk.deep_feature_synthesis(
        database=db,
        target_table="customers",
        features_only=True,
    )
    names = {f.name for f in features}
    assert {"YEAR__signed_up_at", "MONTH__signed_up_at", "WEEKDAY__signed_up_at"} <= (
        names
    )
    # Stacked over the child's own row_creation_time as well.
    assert "MEAN__sessions__MONTH__started_at" in names
    # The raw time index is still not a feature.
    assert "signed_up_at" not in names


def test_raw_row_creation_time_does_not_leak_through_direct_features(db):
    """A parent's raw time index must never cross a join as a DirectFeature.

    Regression test for the leak introduced when ``key_columns()`` split into
    ``input_excluded_columns`` / ``output_excluded_columns``: ``_directs``
    gated candidate bases through the *input* set, so a parent's raw
    ``row_creation_time`` -- legitimately eligible as a primitive input --
    also became eligible as a passthrough column carried onto the child.
    Derived features over the same column, such as ``MONTH(signed_up_at)``,
    must still be reachable and still cross the join.

    Covers both one hop (``sessions`` <- ``customers``) and two hops
    (``transactions`` <- ``sessions`` <- ``customers``), since the leak
    recurses: a blocked raw column at the first hop can never become a nested
    base at the second.
    """
    matrix, features = tusk.deep_feature_synthesis(
        database=db,
        target_table="sessions",
        agg_primitives=[],
        trans_primitives=["month"],
        max_depth=2,
    )
    names = {f.name for f in features}
    assert "customers__signed_up_at" not in names
    assert "customers__MONTH__signed_up_at" in names
    got = matrix.collect()
    assert "customers__signed_up_at" not in got.columns
    assert "customers__MONTH__signed_up_at" in got.columns

    matrix2, features2 = tusk.deep_feature_synthesis(
        database=db,
        target_table="transactions",
        agg_primitives=[],
        trans_primitives=["month"],
        max_depth=3,
    )
    names2 = {f.name for f in features2}
    assert "sessions__started_at" not in names2
    assert "sessions__customers__signed_up_at" not in names2
    assert "sessions__customers__MONTH__signed_up_at" in names2
    got2 = matrix2.collect()
    assert "sessions__started_at" not in got2.columns
    assert "sessions__customers__signed_up_at" not in got2.columns
    assert "sessions__customers__MONTH__signed_up_at" in got2.columns


def test_apply_features_reapplies_definitions(db):
    features = tusk.deep_feature_synthesis(
        database=db,
        target_table="customers",
        agg_primitives=["count"],
        trans_primitives=[],
        max_depth=1,
        features_only=True,
    )
    matrix = tusk.apply_features(features, db)
    assert "COUNT__sessions" in matrix.collect().columns


def test_eager_input_still_returns_a_lazy_plan():
    customers = pl.DataFrame({"id": [1, 2], "age": [30, 40]})
    sessions = pl.DataFrame({"id": [10, 11], "customer_id": [1, 1]})
    db = (
        tusk.Database("x")
        .add_table("customers", customers, primary_key="id")
        .add_table("sessions", sessions, primary_key="id")
        .add_relationship(
            parent="customers",
            child="sessions",
            foreign_key="customer_id",
        )
    )
    matrix, _ = tusk.deep_feature_synthesis(
        database=db,
        target_table="customers",
        agg_primitives=["count"],
        trans_primitives=[],
        max_depth=1,
    )
    assert isinstance(matrix, pl.LazyFrame)
    assert matrix.collect().sort("id")["COUNT__sessions"].to_list() == [2, 0]


def test_mixing_eager_and_lazy_frames_gives_the_same_plan_either_way():
    """Eagerness of any one table must not decide the whole database's output.

    The order the tables go in used to pick the return type, so an eager frame
    added first silently collected a lazy scan added second.
    """
    eager_customers = pl.DataFrame({"id": [1, 2], "age": [30, 40]})
    lazy_sessions = pl.LazyFrame({"id": [10, 11], "customer_id": [1, 1]})

    matrices = []
    for customers, sessions in (
        (eager_customers, lazy_sessions),
        (eager_customers.lazy(), lazy_sessions.collect()),
    ):
        db = (
            tusk.Database("x")
            .add_table("customers", customers, primary_key="id")
            .add_table("sessions", sessions, primary_key="id")
            .add_relationship(
                parent="customers",
                child="sessions",
                foreign_key="customer_id",
            )
        )
        matrix, _ = tusk.deep_feature_synthesis(
            database=db,
            target_table="customers",
            agg_primitives=["count"],
            trans_primitives=[],
            max_depth=1,
        )
        assert isinstance(matrix, pl.LazyFrame)
        matrices.append(matrix.collect().sort("id"))

    assert matrices[0].equals(matrices[1])
    assert matrices[0]["COUNT__sessions"].to_list() == [2, 0]


def test_deep_feature_synthesis_never_materializes_for_lazy_input(tmp_path):
    """Scan a real file, delete it, then build features: only collect() may fail."""
    path = tmp_path / "sessions.parquet"
    pl.DataFrame({"id": [1, 2], "customer_id": [1, 1]}).write_parquet(path)
    db = (
        tusk.Database("x")
        .add_table("customers", pl.LazyFrame({"id": [1]}), primary_key="id")
        .add_table("sessions", pl.scan_parquet(path), primary_key="id")
        .add_relationship(
            parent="customers",
            child="sessions",
            foreign_key="customer_id",
        )
    )
    path.unlink()

    matrix, _ = tusk.deep_feature_synthesis(
        database=db,
        target_table="customers",
        agg_primitives=["count"],
        trans_primitives=[],
        max_depth=1,
    )

    with pytest.raises(FileNotFoundError):
        matrix.collect()


def test_unknown_primitive_fails_before_any_query(db):
    with pytest.raises(tusk.exceptions.PrimitiveError, match="dubbled"):
        tusk.deep_feature_synthesis(
            database=db,
            target_table="customers",
            trans_primitives=["dubbled"],
            features_only=True,
        )
