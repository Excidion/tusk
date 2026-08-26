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
from tusk.sklearn._encoders import encoder_prefix, selector_of, validate_inner

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


def test_selector_of_finds_the_last_step():
    selector = SelectKBest(f_classif, k=1)
    assert (
        selector_of(Pipeline([("s", StandardScaler()), ("sel", selector)])) is selector
    )
    assert selector_of(selector) is selector


@pytest.mark.filterwarnings(_INTERCHANGE_DEPRECATION)
def test_encoder_prefix_is_identity_when_there_is_no_encoder():
    prefix = encoder_prefix(SelectKBest(f_classif, k=1))
    fitted = prefix.fit(FRAME)
    assert list(fitted.get_feature_names_out()) == ["age", "cat", "cnt"]


@pytest.mark.filterwarnings(_INTERCHANGE_DEPRECATION)
def test_encoder_prefix_of_a_single_step_pipeline_is_identity():
    prefix = encoder_prefix(Pipeline([("sel", SelectKBest(f_classif, k=1))]))
    assert list(prefix.fit(FRAME).get_feature_names_out()) == ["age", "cat", "cnt"]


def test_inner_must_end_in_a_selector():
    with pytest.raises(EncoderError, match="SelectorMixin"):
        validate_inner(Pipeline([("s", StandardScaler())]))


def test_explicit_column_lists_are_refused():
    inner = Pipeline(
        [
            ("enc", ColumnTransformer([("num", StandardScaler(), ["age", "cnt"])])),
            ("sel", SelectKBest(f_classif, k=1)),
        ],
    )
    with pytest.raises(EncoderError, match="dtype_selector"):
        validate_inner(inner)


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
    inner = Pipeline(
        [("opaque", _Opaque()), ("sel", SelectKBest(f_classif, k=1))],
    )
    with pytest.raises(EncoderError, match="get_feature_names_out"):
        validate_inner(inner)


def test_a_callable_column_list_is_accepted():
    inner = Pipeline(
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
    validate_inner(inner)
