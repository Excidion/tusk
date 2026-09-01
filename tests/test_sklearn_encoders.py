import datetime as dt

import polars as pl
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tusk.dtypes import DtypeFamily
from tusk.exceptions import EncoderError
from tusk.sklearn import dtype_selector
from tusk.sklearn._encoders import (
    get_encoder_prefix,
    get_last_step,
    validate_selection_pipeline,
)

FRAME = pl.DataFrame(
    {"age": [20.0, 30.0], "cat": ["a", "b"], "cnt": [1.0, 2.0]},
)


def test_dtype_selector_splits_numeric_from_string():
    assert dtype_selector("numeric")(FRAME) == ["age", "cnt"]
    assert dtype_selector("string")(FRAME) == ["cat"]


def test_dtype_selector_keeps_booleans_and_dates_out_of_string():
    frame = pl.DataFrame(
        {
            "n": [1.0],
            "s": ["a"],
            "b": [True],
            "d": [dt.datetime(2024, 1, 1)],
        },
    )
    assert dtype_selector("numeric")(frame) == ["n"]
    assert dtype_selector("string")(frame) == ["s"]
    assert dtype_selector("boolean")(frame) == ["b"]
    assert dtype_selector("temporal")(frame) == ["d"]


def test_dtype_selector_separates_categorical_from_string():
    frame = pl.DataFrame(
        {
            "s": ["t"],
            "c": pl.Series(["x"]).cast(pl.Categorical),
            "e": pl.Series(["a"]).cast(pl.Enum(["a", "b"])),
        },
    )
    assert dtype_selector("string")(frame) == ["s"]
    assert dtype_selector(DtypeFamily.CATEGORICAL)(frame) == ["c", "e"]


def test_dtype_selector_rejects_an_unknown_family():
    with pytest.raises(ValueError):
        dtype_selector("texty")


def test_dtype_selector_separates_has_date_from_duration():
    """The narrow families are the supported way to route a duration column."""
    frame = pl.DataFrame(
        {
            "when": [dt.datetime(2024, 1, 1)],
            "elapsed": [dt.timedelta(hours=3)],
            "n": [1],
        },
    )
    assert dtype_selector("has_date")(frame) == ["when"]
    assert dtype_selector("duration")(frame) == ["elapsed"]
    assert dtype_selector("temporal")(frame) == ["when", "elapsed"]


def test_dtype_selector_has_time_matches_datetime_and_time_only():
    """HAS_TIME routes a bare time-of-day column alongside a full datetime."""
    frame = pl.DataFrame(
        {
            "when": [dt.datetime(2024, 1, 1)],
            "at": [dt.time(9, 30)],
            "on": [dt.date(2024, 1, 1)],
        },
    )
    assert dtype_selector("has_time")(frame) == ["when", "at"]


def test_dtype_selector_reevaluates_on_a_subset():
    assert dtype_selector("numeric")(FRAME.select(["age"])) == ["age"]


# sklearn's feature-name detection falls back to the dataframe interchange
# protocol for any non-pandas frame; polars has deprecated that protocol, so
# fitting a raw polars frame directly (bypassing narwhals, as ColumnTransformer
# and FunctionTransformer do internally) raises a DeprecationWarning that is
# sklearn/polars version noise, not a defect under test here.
_INTERCHANGE_DEPRECATION = (
    "ignore:Support for the dataframe interchange protocol "
    "is deprecated:DeprecationWarning"
)


@pytest.mark.filterwarnings(_INTERCHANGE_DEPRECATION)
def test_dtype_selector_works_inside_a_column_transformer_on_polars():
    encoder = ColumnTransformer(
        [
            ("oh", OneHotEncoder(handle_unknown="ignore"), dtype_selector("string")),
            ("num", StandardScaler(), dtype_selector("numeric")),
        ],
    ).fit(FRAME)
    assert list(encoder.get_feature_names_out()) == [
        "oh__cat_a",
        "oh__cat_b",
        "num__age",
        "num__cnt",
    ]


def test_get_last_step_returns_the_final_step():
    selector = SelectKBest(f_classif, k=1)
    assert (
        get_last_step(Pipeline([("s", StandardScaler()), ("sel", selector)]))
        is selector
    )
    assert get_last_step(selector) is selector


@pytest.mark.filterwarnings(_INTERCHANGE_DEPRECATION)
def test_get_encoder_prefix_is_identity_when_there_is_no_encoder():
    prefix = get_encoder_prefix(SelectKBest(f_classif, k=1))
    fitted = prefix.fit(FRAME)
    assert list(fitted.get_feature_names_out()) == ["age", "cat", "cnt"]


@pytest.mark.filterwarnings(_INTERCHANGE_DEPRECATION)
def test_get_encoder_prefix_of_a_single_step_pipeline_is_identity():
    prefix = get_encoder_prefix(Pipeline([("sel", SelectKBest(f_classif, k=1))]))
    assert list(prefix.fit(FRAME).get_feature_names_out()) == ["age", "cat", "cnt"]


def test_the_selection_pipeline_must_end_in_a_selector():
    with pytest.raises(EncoderError, match="SelectorMixin"):
        validate_selection_pipeline(Pipeline([("s", StandardScaler())]))


def test_explicit_column_lists_are_refused():
    selection_pipeline = Pipeline(
        [
            ("enc", ColumnTransformer([("num", StandardScaler(), ["age", "cnt"])])),
            ("sel", SelectKBest(f_classif, k=1)),
        ],
    )
    with pytest.raises(EncoderError, match="dtype_selector"):
        validate_selection_pipeline(selection_pipeline)


class _Opaque(BaseEstimator, TransformerMixin):
    """A transformer with no get_feature_names_out.

    TransformerMixin does not supply one -- only OneToOneFeatureMixin and
    ClassNamePrefixFeaturesOutMixin do -- so this needs no trickery. It stands
    in for scikit-lego's TypeSelector without taking the dependency.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


def test_a_step_without_feature_names_is_refused():
    selection_pipeline = Pipeline(
        [("opaque", _Opaque()), ("sel", SelectKBest(f_classif, k=1))],
    )
    with pytest.raises(EncoderError, match="get_feature_names_out"):
        validate_selection_pipeline(selection_pipeline)


def test_a_callable_column_list_is_accepted():
    selection_pipeline = Pipeline(
        [
            (
                "enc",
                ColumnTransformer(
                    [("num", StandardScaler(), dtype_selector("numeric"))]
                ),
            ),
            ("sel", SelectKBest(f_classif, k=1)),
        ],
    )
    validate_selection_pipeline(selection_pipeline)
