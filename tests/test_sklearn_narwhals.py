import warnings

import duckdb
import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

import tusk
from tusk.exceptions import EncoderError, TuskError
from tusk.sklearn import DFSTransformer, NarwhalsEncoder, NarwhalsMixin, _narwhals

NUMBERS = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 9.0]})


class NarwhalsScaler(NarwhalsMixin, StandardScaler):
    pass


class PandasTargetEncoder(NarwhalsMixin, TargetEncoder):
    convert_to = "pandas"


class TypeRecorder(TransformerMixin, BaseEstimator):
    """Records the type of what it is fitted on."""

    def fit(self, X, y=None):
        self.received_ = type(X)
        return self

    def transform(self, X):
        return X


def as_duckdb(frame):
    connection = duckdb.connect()
    connection.register("t", frame.to_arrow())
    return connection.sql("select * from t")


@pytest.mark.parametrize("backend", ["polars", "pandas", "pyarrow", "duckdb"])
def test_the_mixin_accepts_any_backend(backend):
    native = {
        "polars": lambda: NUMBERS,
        "pandas": NUMBERS.to_pandas,
        "pyarrow": NUMBERS.to_arrow,
        "duckdb": lambda: as_duckdb(NUMBERS),
    }[backend]()
    scaler = NarwhalsScaler().fit(native)
    assert list(scaler.get_feature_names_out()) == ["a", "b"]
    np.testing.assert_allclose(
        scaler.transform(native).mean(axis=0), [0.0, 0.0], atol=1e-12
    )


@pytest.mark.parametrize(
    ("convert_to", "expected"),
    [
        ("numpy", np.ndarray),
        ("pandas", pd.DataFrame),
        ("polars", pl.DataFrame),
        ("narwhals", nw.LazyFrame),
    ],
)
def test_convert_to_sets_what_the_estimator_receives(convert_to, expected):
    recorder = type("Recorder", (NarwhalsMixin, TypeRecorder), {})
    recorder.convert_to = convert_to
    assert recorder().fit(as_duckdb(NUMBERS)).received_ is expected


def test_a_numpy_fit_then_transform_issues_no_feature_name_warning():
    scaler = NarwhalsScaler().fit(NUMBERS)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scaler.transform(NUMBERS)


def test_an_estimator_with_its_own_fit_transform_receives_the_conversion():
    frame = pl.DataFrame({"c": ["x", "y"] * 10})
    y = [0, 1] * 10
    assert PandasTargetEncoder().fit_transform(frame, y).shape == (20, 1)
    fitted = PandasTargetEncoder().fit(frame, y)
    assert list(fitted.get_feature_names_out()) == ["c"]


def test_set_output_pandas_names_the_columns():
    out = NarwhalsScaler().set_output(transform="pandas").fit_transform(NUMBERS)
    assert list(out.columns) == ["a", "b"]


def test_clone_keeps_convert_to():
    assert clone(PandasTargetEncoder()).convert_to == "pandas"


def test_an_absent_package_names_the_install_command(monkeypatch):
    monkeypatch.setattr(_narwhals, "find_spec", lambda name: None)
    with pytest.raises(TuskError, match="uv add pandas"):
        PandasTargetEncoder().fit(pl.DataFrame({"c": ["x", "y"]}), [0, 1])


def test_the_encoder_fits_a_lazy_table_without_collecting(monkeypatch):
    relation = as_duckdb(NUMBERS)

    def refuse(*args, **kwargs):
        raise AssertionError("fit collected the table")

    monkeypatch.setattr(nw.LazyFrame, "collect", refuse)
    encoder = NarwhalsEncoder().fit(relation)
    assert list(encoder.feature_names_in_) == ["a", "b"]


def test_the_encoder_returns_what_set_output_asks_for():
    categories = pl.DataFrame({"c": pl.Series(["x", "y"]).cast(pl.Categorical)})
    assert isinstance(NarwhalsEncoder().fit_transform(categories), np.ndarray)
    as_pandas = NarwhalsEncoder().set_output(transform="pandas")
    assert str(as_pandas.fit_transform(categories)["c"].dtype) == "category"
    as_polars = NarwhalsEncoder().set_output(transform="polars")
    assert isinstance(as_polars.fit_transform(as_duckdb(NUMBERS)), pl.DataFrame)


def test_clone_keeps_the_output_setting():
    encoder = NarwhalsEncoder().set_output(transform="polars").fit(NUMBERS)
    assert isinstance(clone(encoder).fit_transform(NUMBERS), pl.DataFrame)


def test_set_output_rejects_an_unsupported_value():
    with pytest.raises(ValueError, match="choose from"):
        NarwhalsEncoder().set_output(transform="pyarrow")


def test_the_encoder_rejects_other_columns_or_dtypes():
    encoder = NarwhalsEncoder().fit(NUMBERS)
    with pytest.raises(EncoderError, match="absent"):
        encoder.transform(NUMBERS.select("a"))
    with pytest.raises(EncoderError, match="'a'"):
        encoder.transform(NUMBERS.with_columns(pl.col("a").cast(pl.Int64)))


def test_the_encoder_is_a_convert_step_between_dfs_and_a_model():
    customers = duckdb.sql(
        "select * from (values "
        "(1, 30.0, timestamp '2024-01-01'), "
        "(2, 40.0, timestamp '2024-01-01'), "
        "(3, 50.0, timestamp '2024-01-01'), "
        "(4, 60.0, timestamp '2024-01-01')) t(id, age, signed_up_at)",
    )
    database = tusk.Database("shop").add_table(
        "customers", customers, primary_key="id", row_creation_time="signed_up_at"
    )
    received = []

    class Model(LogisticRegression):
        def fit(self, X, y, **params):
            received.append(X)
            return super().fit(X, y, **params)

    with sklearn.config_context(enable_metadata_routing=True):
        pipeline = Pipeline(
            [
                ("dfs", DFSTransformer(target_table="customers")),
                ("convert", NarwhalsEncoder().set_output(transform="pandas")),
                ("model", Model()),
            ]
        )
        pipeline.fit([1, 2, 3, 4], [0, 0, 1, 1], database=database)
    assert isinstance(received[0], pd.DataFrame)
    assert received[0]["age"].tolist() == [30.0, 40.0, 50.0, 60.0]
