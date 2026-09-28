import datetime as dt
from decimal import Decimal

import duckdb
import narwhals as nw
import numpy as np
import polars as pl
import pytest
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import tusk
from tusk.sklearn import (
    DFSSelectorTransformer,
    NarwhalsMixin,
    StringEncoder,
    TableEncoder,
)

EVERY_DTYPE = pl.DataFrame(
    {
        "n": [1, 2],
        "x": [Decimal("1.5"), Decimal("2.5")],
        "b": [True, False],
        "s": ["foo", "bar"],
        "c": pl.Series(["x", "y"]).cast(pl.Categorical),
        "e": pl.Series(["a", "b"], dtype=pl.Enum(["b", "a"])),
        "d": [dt.date(2024, 3, 5), dt.date(2024, 4, 6)],
        "t": [dt.time(9, 30), dt.time(10, 0)],
        "w": [dt.datetime(2024, 3, 5, 9, 30), dt.datetime(2024, 3, 5, 9, 30)],
        "u": [dt.timedelta(minutes=90), dt.timedelta(minutes=1)],
        "l": [[1], [2]],
    }
)


class TypeRecorder(TransformerMixin, BaseEstimator):
    """Records the type of what it is fitted on, and passes it through."""

    def fit(self, X, y=None):
        self.received_ = type(X)
        return self

    def transform(self, X):
        return nw.from_native(X).to_numpy() if not isinstance(X, np.ndarray) else X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class PolarsRecorder(NarwhalsMixin, TypeRecorder):
    convert_to = "polars"


def as_duckdb(table):
    connection = duckdb.connect()
    connection.register("t", table.to_arrow())
    return connection.sql("select * from t")


def small_strings():
    return TableEncoder(string=StringEncoder(n_components=2))


def test_every_dtype_reaches_its_group():
    encoder = small_strings().set_output(transform="polars")
    out = encoder.fit_transform(EVERY_DTYPE)
    assert out.columns == list(encoder.get_feature_names_out())
    assert out.columns == [
        "numeric__n",
        "numeric__x",
        "boolean__b",
        "string__s_svd_0",
        "string__s_svd_1",
        "categorical__c_x",
        "categorical__c_y",
        "enum__e_code",
        "date__d_month",
        "date__d_day",
        "time__t_hour",
        "time__t_minute",
        "datetime__w_month",
        "datetime__w_day",
        "datetime__w_hour",
        "datetime__w_minute",
        "duration__u_total_seconds",
    ]


def test_decimal_is_cast_to_float64():
    out = small_strings().set_output(transform="polars").fit_transform(EVERY_DTYPE)
    assert out["numeric__x"].dtype == pl.Float64
    assert out["numeric__x"].to_list() == [1.5, 2.5]


def test_drop_and_passthrough():
    encoder = TableEncoder(numeric="drop", string="passthrough", other="drop")
    out = encoder.set_output(transform="polars").fit_transform(EVERY_DTYPE)
    assert "numeric__n" not in out.columns
    assert out["string__s"].to_list() == ["foo", "bar"]


@pytest.mark.parametrize("backend", ["pandas", "pyarrow"])
def test_other_backends(backend):
    table = EVERY_DTYPE.drop("d", "t", "l")
    native = table.to_pandas() if backend == "pandas" else table.to_arrow()
    encoder = small_strings().fit(native)
    assert encoder.transform(native).shape == (2, len(encoder.get_feature_names_out()))


def test_an_empty_group_is_skipped():
    encoder = TableEncoder()
    encoder.fit(pl.DataFrame({"n": [1.0, 2.0]}))
    assert list(encoder.groups_) == ["numeric"]


def test_a_schema_only_fit_does_not_collect(monkeypatch):
    table = pl.DataFrame({"n": [1.0, 2.0], "w": [dt.datetime(2024, 3, 5, 9, 30)] * 2})
    relation = as_duckdb(table)

    def refuse(*args, **kwargs):
        raise AssertionError("fit collected the table")

    monkeypatch.setattr(nw.LazyFrame, "collect", refuse)
    TableEncoder().fit(relation)


def test_transform_collects_a_lazy_table_once(monkeypatch):
    table = pl.DataFrame(
        {
            "n": [1.0, 2.0],
            "s": ["foo", "bar"],
            "w": [dt.datetime(2024, 3, 5, 9, 30)] * 2,
        }
    )
    relation = as_duckdb(table)
    encoder = small_strings().fit(relation)
    calls = []
    collect = nw.LazyFrame.collect

    def counting(self, *args, **kwargs):
        calls.append(self)
        return collect(self, *args, **kwargs)

    monkeypatch.setattr(nw.LazyFrame, "collect", counting)
    assert encoder.transform(relation).shape == (2, 1 + 2 + 4)
    assert len(calls) == 1


def test_a_bare_estimator_receives_numpy_and_a_mixin_its_convert_to():
    encoder = TableEncoder(numeric=TypeRecorder(), string=PolarsRecorder())
    encoder.fit(EVERY_DTYPE)
    assert encoder.groups_["numeric"][0].received_ is np.ndarray
    assert encoder.groups_["string"][0].received_ is pl.DataFrame


def test_a_value_that_is_not_an_estimator_is_rejected():
    with pytest.raises(ValueError, match="numeric='scale'"):
        TableEncoder(numeric="scale").fit(EVERY_DTYPE)


def test_nested_parameters_do_not_change_the_defaults():
    changed = TableEncoder().set_params(date__components=["year"])
    assert changed.date.components == ["year"]
    assert TableEncoder().date.components == ("month", "day")
    assert clone(changed).date.components == ["year"]


def test_it_encodes_inside_dfs_selection_on_duckdb():
    customers = duckdb.sql(
        "select * from (values "
        "(1, 30, 'gold', timestamp '2024-01-01'), "
        "(2, 40, 'silver', timestamp '2024-01-01'), "
        "(3, 50, 'gold', timestamp '2024-01-01'), "
        "(4, 60, 'silver', timestamp '2024-01-01')) t(id, age, tier, signed_up_at)",
    )
    database = tusk.Database("shop").add_table(
        "customers", customers, primary_key="id", row_creation_time="signed_up_at"
    )
    encoder = TableEncoder(numeric=StandardScaler(), string="drop", datetime="drop")
    selector = DFSSelectorTransformer(
        target_table="customers",
        selection_pipeline=Pipeline(
            [("encode", encoder), ("select", SelectKBest(f_classif, k=1))]
        ),
        trans_primitives=[],
    )
    with pytest.warns(UserWarning, match="fed no encoded column"):
        selector.fit([1, 2, 3, 4], [0, 0, 1, 1], database=database)
    assert [feature.name for feature in selector.features_] == ["age"]
    assert list(selector.get_feature_names_out()) == ["numeric__age"]
    assert selector.transform([1, 2], database=database).shape == (2, 1)
