from datetime import datetime

import polars as pl
import pytest

import tusk
from tusk.primitives import Quantiles


def test_readme_example_compiles_and_collects(es):
    """The README's headline call, verbatim, against the README's schema.

    A multi-output primitive at max_depth=2 previously emitted a phantom
    un-indexed column and raised ColumnNotFoundError here.
    """
    feature_matrix, features = tusk.dfs(
        entityset=es,
        target_dataframe_name="customers",
        agg_primitives=["mean", "count", Quantiles(qs=(0.25, 0.5, 0.75))],
        trans_primitives=["month", "weekday"],
        max_depth=2,
        cutoff_time=datetime(2026, 1, 1),
    )
    got = feature_matrix.collect()
    expected = {name for f in features for name in f.output_names}
    assert expected <= set(got.columns)
    assert any(c.startswith("QUANTILES(") for c in got.columns)


def test_dfs_end_to_end(es):
    matrix, features = tusk.dfs(
        entityset=es,
        target_dataframe_name="customers",
        agg_primitives=["count", "mean"],
        trans_primitives=[],
        max_depth=2,
    )
    assert isinstance(matrix, pl.LazyFrame)
    got = matrix.collect().sort("id")
    assert got["COUNT(sessions)"].to_list() == [2, 1, 0]
    assert {f.name for f in features} <= set(got.columns)


def test_features_only_returns_definitions_alone(es):
    features = tusk.dfs(
        entityset=es,
        target_dataframe_name="customers",
        agg_primitives=["count"],
        trans_primitives=[],
        max_depth=1,
        features_only=True,
    )
    assert isinstance(features, list)
    assert {f.name for f in features} == {"age", "COUNT(sessions)"}


def test_defaults_are_applied(es):
    features = tusk.dfs(
        entityset=es, target_dataframe_name="customers", features_only=True
    )
    names = {f.name for f in features}
    assert "COUNT(sessions)" in names
    assert not any(n.startswith("ADD_NUMERIC") for n in names)


def test_calculate_feature_matrix_reapplies_definitions(es):
    features = tusk.dfs(
        entityset=es,
        target_dataframe_name="customers",
        agg_primitives=["count"],
        trans_primitives=[],
        max_depth=1,
        features_only=True,
    )
    matrix = tusk.calculate_feature_matrix(features, es)
    assert "COUNT(sessions)" in matrix.collect().columns


def test_eager_input_round_trips_to_eager_output():
    customers = pl.DataFrame({"id": [1, 2], "age": [30, 40]})
    sessions = pl.DataFrame({"id": [10, 11], "customer_id": [1, 1]})
    es = (
        tusk.EntitySet("x")
        .add_dataframe("customers", customers, primary_key="id")
        .add_dataframe("sessions", sessions, primary_key="id")
        .add_relationship(
            parent="customers", child="sessions", foreign_key="customer_id"
        )
    )
    matrix, _ = tusk.dfs(
        entityset=es,
        target_dataframe_name="customers",
        agg_primitives=["count"],
        trans_primitives=[],
        max_depth=1,
    )
    assert isinstance(matrix, pl.DataFrame)
    assert matrix.sort("id")["COUNT(sessions)"].to_list() == [2, 0]


def test_dfs_never_materializes_for_lazy_input(tmp_path):
    """Scan a real file, delete it, then build features: only collect() may fail."""
    path = tmp_path / "sessions.parquet"
    pl.DataFrame({"id": [1, 2], "customer_id": [1, 1]}).write_parquet(path)
    es = (
        tusk.EntitySet("x")
        .add_dataframe("customers", pl.LazyFrame({"id": [1]}), primary_key="id")
        .add_dataframe("sessions", pl.scan_parquet(path), primary_key="id")
        .add_relationship(
            parent="customers", child="sessions", foreign_key="customer_id"
        )
    )
    path.unlink()

    matrix, _ = tusk.dfs(
        entityset=es,
        target_dataframe_name="customers",
        agg_primitives=["count"],
        trans_primitives=[],
        max_depth=1,
    )

    with pytest.raises(FileNotFoundError):
        matrix.collect()


def test_unknown_primitive_fails_before_any_query(es):
    with pytest.raises(tusk.exceptions.PrimitiveError, match="dubbled"):
        tusk.dfs(
            entityset=es,
            target_dataframe_name="customers",
            trans_primitives=["dubbled"],
            features_only=True,
        )
