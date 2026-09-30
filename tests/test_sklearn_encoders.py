import polars as pl
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tusk.exceptions import EncoderError
from tusk.sklearn._encoders import (
    get_encoder_prefix,
    get_last_step,
    validate_selection_pipeline,
)

FRAME = pl.DataFrame(
    {"age": [20.0, 30.0], "cat": ["a", "b"], "cnt": [1.0, 2.0]},
)


# sklearn's feature-name detection falls back to the dataframe interchange
# protocol for any non-pandas frame; polars has deprecated that protocol, so
# fitting a raw polars frame directly (bypassing narwhals, as ColumnTransformer
# and FunctionTransformer do internally) raises a DeprecationWarning that is
# sklearn/polars version noise, not a defect under test here.
_INTERCHANGE_DEPRECATION = (
    "ignore:Support for the dataframe interchange protocol "
    "is deprecated:DeprecationWarning"
)


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
    with pytest.raises(EncoderError, match="TableEncoder"):
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
                    [
                        (
                            "num",
                            StandardScaler(),
                            make_column_selector(dtype_include="number"),
                        )
                    ]
                ),
            ),
            ("sel", SelectKBest(f_classif, k=1)),
        ],
    )
    validate_selection_pipeline(selection_pipeline)
