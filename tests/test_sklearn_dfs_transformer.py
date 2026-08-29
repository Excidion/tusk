import warnings

import narwhals as nw
import numpy as np
import pytest
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline

from tusk.exceptions import SchemaError, UnmatchedPrimitiveWarning
from tusk.sklearn import DFSTransformer

KEYS = [1, 2, 3]
Y = [0, 1, 0]

# Fitting a raw polars frame directly (bypassing narwhals, as scikit-learn's own
# validation does internally) raises a DeprecationWarning that is sklearn/polars
# version noise, not a defect under test here -- see test_sklearn_encoders.py for
# the same precedent.
_INTERCHANGE_DEPRECATION = (
    "ignore:Support for the dataframe interchange protocol "
    "is deprecated:DeprecationWarning"
)


def _transformer():
    return DFSTransformer(target_table="customers", max_depth=2)


def test_fit_reads_no_rows_and_produces_feature_definitions(db):
    fitted = _transformer().fit(KEYS, database=db)
    assert fitted.features_
    assert all(f.table == "customers" for f in fitted.features_)


def test_transform_returns_one_row_per_key_in_key_order(db):
    # age identifies the customer (1->30, 2->40, 3->50), so asserting it pins
    # the row order rather than merely the row count.
    fitted = _transformer().fit(KEYS, database=db)
    out = nw.from_native(fitted.transform([3, 1], database=db))
    assert out["age"].to_list() == [50, 30]


def test_the_primary_key_is_not_a_feature(db):
    fitted = _transformer().fit(KEYS, database=db)
    out = nw.from_native(fitted.transform(KEYS, database=db))
    assert "id" not in out.columns


def test_feature_names_match_the_matrix_columns(db):
    fitted = _transformer().fit(KEYS, database=db)
    out = nw.from_native(fitted.transform(KEYS, database=db))
    assert list(fitted.get_feature_names_out()) == list(out.columns)


def test_an_unknown_key_raises_rather_than_dropping_a_row(db):
    fitted = _transformer().fit(KEYS, database=db)
    with pytest.raises(SchemaError, match="no row"):
        fitted.transform([1, 999], database=db)


def test_transform_falls_back_to_the_database_seen_at_fit(db):
    fitted = _transformer().fit(KEYS, database=db)
    out = nw.from_native(fitted.transform(KEYS))
    assert out.shape[0] == 3


@pytest.mark.filterwarnings(_INTERCHANGE_DEPRECATION)
def test_it_routes_the_database_through_a_pipeline(db):
    # An imputer sits between dfs and clf: customer 3 has no sessions and
    # session 30 has no transactions, so max_depth=2 legitimately produces
    # null MEAN/MIN/MAX/STD features for those nested empty groups (confirmed
    # against tusk.deep_feature_synthesis directly, independent of this
    # wrapper) -- real DFS output, not something LogisticRegression accepts
    # unpreprocessed. What is under test here is metadata routing, not
    # feature-matrix completeness, so the fix belongs in the fixture pipeline
    # rather than in DFSTransformer.
    with sklearn.config_context(enable_metadata_routing=True):
        pipe = Pipeline(
            [
                ("dfs", _transformer()),
                ("impute", SimpleImputer()),
                ("clf", LogisticRegression()),
            ],
        )
        pipe.fit(KEYS, Y, database=db)
        assert len(pipe.predict([1, 2], database=db)) == 2


@pytest.mark.filterwarnings(_INTERCHANGE_DEPRECATION)
def test_cross_val_score_produces_finite_scores(db):
    # Only 3 customers exist, so with Y's one class having a single member,
    # `cv=3` StratifiedKFold refuses to split (a class needs >= n_splits
    # members) and, even with a non-stratified KFold, the fold that holds out
    # that single member leaves LogisticRegression a single-class training
    # set, which it structurally cannot fit -- true for any implementation,
    # not particular to DFSTransformer. KFold plus a classifier indifferent to
    # class count isolates what this test is actually checking: that scores
    # come back at all, i.e. that the routed database reached every fold.
    with sklearn.config_context(enable_metadata_routing=True):
        pipe = Pipeline([("dfs", _transformer()), ("clf", DummyClassifier())])
        scores = cross_val_score(
            pipe,
            KEYS,
            Y,
            cv=KFold(n_splits=3),
            params={"database": db},
        )
    assert np.isfinite(scores).all()


@pytest.mark.filterwarnings(_INTERCHANGE_DEPRECATION)
def test_grid_search_can_tune_a_dfs_parameter(db):
    # Same KFold/DummyClassifier substitution as test_cross_val_score, for the
    # same reason. max_depth=1 additionally has nothing on the immediate
    # `sessions` table for sum/mean/min/max/std to match -- those primitives
    # only find a column two hops away, on `transactions` -- so every fold at
    # that depth raises UnmatchedPrimitiveWarning; test_synthesis.py asserts
    # this exact warning on this exact fixture, so it is expected here too
    # rather than left to leak into the run's warnings summary.
    with sklearn.config_context(enable_metadata_routing=True):
        search = GridSearchCV(
            Pipeline([("dfs", _transformer()), ("clf", DummyClassifier())]),
            {"dfs__max_depth": [1, 2]},
            cv=KFold(n_splits=3),
        )
        with pytest.warns(UnmatchedPrimitiveWarning, match="'sessions'"):
            search.fit(KEYS, Y, database=db)
    assert search.best_params_["dfs__max_depth"] in (1, 2)


def test_fit_transform_forwards_the_database_without_warning(db):
    # TransformerMixin's fit_transform drops metadata on the way to transform,
    # and sklearn warns about exactly that. The database_ fallback would hide a
    # broken override, so the absence of the warning is what we assert.
    with sklearn.config_context(enable_metadata_routing=True):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            _transformer().fit_transform(KEYS, Y, database=db)


def test_importing_tusk_does_not_import_sklearn():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import sys, tusk; print('sklearn' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"
