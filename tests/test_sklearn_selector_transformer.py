import datetime as dt

import narwhals as nw
import numpy as np
import polars as pl
import pytest
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, SelectorMixin, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import tusk
from tusk.exceptions import (
    EncoderError,
    LineageError,
    LineageWarning,
    SchemaError,
    UnencodedFeatureWarning,
)
from tusk.sklearn import DFSSelectorTransformer, DFSTransformer, dtype_selector

# Fitting a raw polars frame directly (bypassing narwhals, as scikit-learn's own
# validation does internally) raises a DeprecationWarning that is sklearn/polars
# version noise, not a defect under test here -- see test_sklearn_encoders.py for
# the same precedent.
_INTERCHANGE_DEPRECATION = (
    "ignore:Support for the dataframe interchange protocol "
    "is deprecated:DeprecationWarning"
)

KEYS = list(range(1, 13))
Y = [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1]

# One transaction count per customer, and one amount per customer. Both vary,
# and every customer has at least two transactions: a selector scoring 31
# features over 12 rows needs every matrix column to be non-null and
# non-constant, or f_classif divides by a zero variance and warns.
_COUNTS = [2, 3, 4, 5, 3, 2, 5, 4, 2, 3, 4, 5]
_AMOUNTS = [3.0, 17.0, 8.5, 41.0, 12.0, 6.25, 99.0, 23.5, 4.0, 55.0, 31.0, 7.75]


@pytest.fixture
def shop():
    """Two tables, wide enough to select over.

    The shared ``db`` fixture cannot serve here: three rows, and empty groups
    that legitimately produce null MEAN/STD columns, which SelectKBest refuses
    outright.
    """
    customers = pl.LazyFrame(
        {
            "id": KEYS,
            "age": [
                23.0,
                31.0,
                45.0,
                52.0,
                28.0,
                39.0,
                61.0,
                34.0,
                47.0,
                26.0,
                55.0,
                42.0,
            ],
            "region": ["north", "south", "east"] * 4,
            "signed_up_at": [
                dt.datetime(2021 + i % 2, 1 + i % 11, 1 + i) for i in range(12)
            ],
        },
    )
    ids, customer_ids, amounts, times = [], [], [], []
    for customer, count in zip(KEYS, _COUNTS, strict=True):
        for i in range(count):
            ids.append(100 + len(ids))
            customer_ids.append(customer)
            amounts.append(_AMOUNTS[(customer + 3 * i) % 12] + i)
            times.append(
                dt.datetime(2023, 1, 1) + dt.timedelta(days=60 * customer + 45 * i)
            )
    transactions = pl.LazyFrame(
        {
            "id": ids,
            "customer_id": customer_ids,
            "amount": amounts,
            "occurred_at": times,
        },
    )
    return (
        tusk.Database("shop")
        .add_table(
            "customers",
            customers,
            primary_key="id",
            row_creation_time="signed_up_at",
        )
        .add_table(
            "transactions",
            transactions,
            primary_key="id",
            row_creation_time="occurred_at",
        )
        .add_relationship(
            parent="customers",
            child="transactions",
            foreign_key="customer_id",
        )
    )


def _encoder():
    return ColumnTransformer(
        [
            ("oh", OneHotEncoder(handle_unknown="ignore"), dtype_selector("string")),
            ("num", StandardScaler(), dtype_selector("numeric")),
        ],
    )


def _inner(k=2):
    return Pipeline([("enc", _encoder()), ("sel", SelectKBest(f_classif, k=k))])


def _transformer(k=2, **kwargs):
    return DFSSelectorTransformer(
        target_table="customers",
        inner=_inner(k),
        max_depth=2,
        **kwargs,
    )


def _matrix(database, **kwargs):
    """The unpruned tusk-space matrix, with its real column names."""
    plain = DFSTransformer(target_table="customers", max_depth=2, **kwargs)
    return plain.fit(KEYS, database=database).transform(KEYS, database=database)


def _encoded_position(matrix, name):
    """Where ``name`` lands in encoded space, read off a real fit.

    Derived rather than hardcoded: it is the encoder, not this test, that
    decides how many columns one-hot expands to and in what order they come.
    """
    names = list(_encoder().fit(matrix).get_feature_names_out())
    return names.index(name)


class KeepPositions(SelectorMixin, BaseEstimator):
    """A selector whose choice is known in advance, unlike SelectKBest's."""

    def __init__(self, positions=()):
        self.positions = positions

    def fit(self, X, y=None):
        frame = nw.from_native(X, eager_only=True, pass_through=True)
        self.n_features_in_ = (
            len(frame.columns)
            if isinstance(frame, nw.DataFrame)
            else np.asarray(X).shape[1]
        )
        return self

    def _get_support_mask(self):
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[list(self.positions)] = True
        return mask


class WidthStamped(TransformerMixin, BaseEstimator):
    """An encoder whose output names change when the matrix narrows.

    Nothing sane does this; it is the cheapest way to reach the post-refit
    invariant, which no correct encoder can trip.
    """

    def fit(self, X, y=None):
        self.names_ = list(nw.from_native(X, eager_only=True).columns)
        return self

    def transform(self, X):
        return nw.from_native(X, eager_only=True).select(self.names_).to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(
            [f"{n}_of{len(self.names_)}" for n in self.names_],
            dtype=object,
        )


@pytest.mark.filterwarnings(_INTERCHANGE_DEPRECATION)
def test_fitting_prunes_the_feature_list(shop):
    pruned = len(_transformer().fit(KEYS, Y, database=shop).features_)
    everything = len(
        DFSTransformer(target_table="customers", max_depth=2)
        .fit(KEYS, database=shop)
        .features_,
    )
    assert 0 < pruned < everything


@pytest.mark.filterwarnings(_INTERCHANGE_DEPRECATION)
def test_transform_width_equals_the_selectors_choice(shop):
    fitted = _transformer(k=2).fit(KEYS, Y, database=shop)
    out = np.asarray(fitted.transform(KEYS, database=shop))
    assert out.shape == (12, 2)


@pytest.mark.filterwarnings(_INTERCHANGE_DEPRECATION)
def test_pruned_features_are_never_computed(shop):
    everything = DFSTransformer(target_table="customers", max_depth=2).fit(
        KEYS,
        database=shop,
    )
    all_columns = set(everything.get_feature_names_out())

    fitted = _transformer().fit(KEYS, Y, database=shop)
    kept = {n for f in fitted.features_ for n in f.output_names}
    dropped = all_columns - kept
    assert dropped, "selection kept every feature, so this proves nothing"

    # The parent's transform computes the tusk-space matrix for whatever
    # features_ currently holds, which after fit is the pruned list.
    computed = set(
        nw.from_native(DFSTransformer.transform(fitted, KEYS, database=shop)).columns,
    )
    assert computed == kept
    assert not computed & dropped


@pytest.mark.filterwarnings(_INTERCHANGE_DEPRECATION)
def test_the_survivors_are_exactly_the_features_feeding_kept_columns(shop):
    matrix = _matrix(shop)
    positions = [
        _encoded_position(matrix, "oh__region_south"),
        _encoded_position(matrix, "num__MAX__transactions__amount"),
    ]
    transformer = DFSSelectorTransformer(
        target_table="customers",
        inner=Pipeline([("enc", _encoder()), ("sel", KeepPositions(positions))]),
        max_depth=2,
    )
    fitted = transformer.fit(KEYS, Y, database=shop)
    assert [f.name for f in fitted.features_] == [
        "region",
        "MAX__transactions__amount",
    ]


@pytest.mark.filterwarnings(_INTERCHANGE_DEPRECATION)
def test_a_multi_output_feature_survives_whole(shop):
    quantiles = {"agg_primitives": ["quantiles"], "trans_primitives": []}
    matrix = _matrix(shop, **quantiles)
    position = _encoded_position(matrix, "num__QUANTILES__transactions__amount__1")
    transformer = DFSSelectorTransformer(
        target_table="customers",
        inner=Pipeline([("enc", _encoder()), ("sel", KeepPositions([position]))]),
        agg_primitives=["quantiles"],
        trans_primitives=[],
    )
    fitted = transformer.fit(KEYS, Y, database=shop)

    assert [f.name for f in fitted.features_] == ["QUANTILES__transactions__amount"]
    # One encoded column was kept, but the feature owning it cannot be computed
    # one output at a time, so all three of its columns are still built.
    computed = nw.from_native(DFSTransformer.transform(fitted, KEYS, database=shop))
    assert len(computed.columns) == 3
    assert np.asarray(fitted.transform(KEYS, database=shop)).shape == (12, 1)


@pytest.mark.filterwarnings(_INTERCHANGE_DEPRECATION)
def test_transform_matches_encoding_the_whole_matrix(shop):
    # Pruning is an optimization: encoding only the survivors must give the
    # same numbers as encoding everything and applying the selector.
    expected = _inner(k=3).fit_transform(_matrix(shop), Y)
    fitted = _transformer(k=3).fit(KEYS, Y, database=shop)
    np.testing.assert_allclose(
        np.asarray(fitted.transform(KEYS, database=shop), dtype=float),
        expected,
    )


@pytest.mark.filterwarnings(_INTERCHANGE_DEPRECATION)
def test_feature_names_out_are_readable_not_sentinels(shop):
    fitted = _transformer().fit(KEYS, Y, database=shop)
    names = list(fitted.get_feature_names_out())
    assert len(names) == 2
    assert all(fitted.sentinels_.prefix not in name for name in names)
    assert any("__transactions__" in name or "region" in name for name in names)


@pytest.mark.filterwarnings(_INTERCHANGE_DEPRECATION)
def test_an_opaque_encoder_keeps_every_feature_and_warns(shop):
    # PCA sees only the numeric columns: it cannot be fitted on a string one
    # at all, and what is under test is its opaque output names, not its dtype
    # handling.
    inner = Pipeline(
        [
            (
                "enc",
                ColumnTransformer(
                    [("pca", PCA(n_components=2), dtype_selector("numeric"))],
                ),
            ),
            ("sel", SelectKBest(f_classif, k=1)),
        ],
    )
    transformer = DFSSelectorTransformer(
        target_table="customers",
        inner=inner,
        agg_primitives=["count", "mean"],
        trans_primitives=[],
    )
    everything = len(
        DFSTransformer(
            target_table="customers",
            agg_primitives=["count", "mean"],
            trans_primitives=[],
        )
        .fit(KEYS, database=shop)
        .features_,
    )
    with pytest.warns(LineageWarning, match="pruned nothing"):
        fitted = transformer.fit(KEYS, Y, database=shop)
    assert len(fitted.features_) == everything
    assert np.asarray(fitted.transform(KEYS, database=shop)).shape == (12, 1)


@pytest.mark.filterwarnings(_INTERCHANGE_DEPRECATION)
def test_a_partial_encoder_warns_about_features_it_never_saw(shop):
    # Only string columns are encoded, and remainder defaults to "drop", so
    # every numeric feature silently feeds nothing and gets pruned.
    inner = Pipeline(
        [
            (
                "enc",
                ColumnTransformer(
                    [
                        (
                            "oh",
                            OneHotEncoder(handle_unknown="ignore"),
                            dtype_selector("string"),
                        ),
                    ],
                ),
            ),
            ("sel", SelectKBest(f_classif, k=1)),
        ],
    )
    transformer = DFSSelectorTransformer(target_table="customers", inner=inner)
    with pytest.warns(UnencodedFeatureWarning, match="fed no encoded column"):
        fitted = transformer.fit(KEYS, Y, database=shop)
    assert [f.name for f in fitted.features_] == ["region"]


@pytest.mark.filterwarnings(_INTERCHANGE_DEPRECATION)
def test_selecting_nothing_raises_rather_than_producing_an_empty_matrix(shop):
    transformer = DFSSelectorTransformer(
        target_table="customers",
        inner=Pipeline([("enc", _encoder()), ("sel", KeepPositions([]))]),
        max_depth=2,
    )
    with pytest.raises(SchemaError, match="eliminated every feature"):
        transformer.fit(KEYS, Y, database=shop)


@pytest.mark.filterwarnings(_INTERCHANGE_DEPRECATION)
def test_a_kept_column_missing_after_the_refit_raises(shop):
    transformer = DFSSelectorTransformer(
        target_table="customers",
        inner=Pipeline([("stamp", WidthStamped()), ("sel", KeepPositions([0, 1]))]),
        max_depth=2,
    )
    with pytest.raises(LineageError, match="vanished"):
        transformer.fit(KEYS, Y, database=shop)


@pytest.mark.filterwarnings(_INTERCHANGE_DEPRECATION)
def test_an_inner_without_an_encoder_selects_the_matrix_itself(shop):
    matrix = nw.from_native(
        _matrix(shop, agg_primitives=["count"], trans_primitives=[])
    )
    position = list(matrix.columns).index("COUNT__transactions")
    transformer = DFSSelectorTransformer(
        target_table="customers",
        inner=KeepPositions([position]),
        agg_primitives=["count"],
        trans_primitives=[],
    )
    fitted = transformer.fit(KEYS, Y, database=shop)
    assert [f.name for f in fitted.features_] == ["COUNT__transactions"]
    assert list(fitted.get_feature_names_out()) == ["COUNT__transactions"]
    out = nw.from_native(fitted.transform(KEYS, database=shop))
    assert out.shape == (12, 1)


def test_inner_is_validated_at_fit_not_construction(shop):
    transformer = DFSSelectorTransformer(
        target_table="customers",
        inner=StandardScaler(),
    )
    with pytest.raises(EncoderError, match="SelectorMixin"):
        transformer.fit(KEYS, Y, database=shop)


@pytest.mark.filterwarnings(_INTERCHANGE_DEPRECATION)
def test_it_works_as_a_pipeline_step(shop):
    with sklearn.config_context(enable_metadata_routing=True):
        pipe = Pipeline([("dfs", _transformer()), ("clf", LogisticRegression())])
        pipe.fit(np.array(KEYS).reshape(-1, 1), Y, database=shop)
        assert len(pipe.predict(np.array([[1], [2]]), database=shop)) == 2


@pytest.mark.filterwarnings(_INTERCHANGE_DEPRECATION)
def test_predicting_against_another_database_uses_the_pruned_features(shop):
    fitted = _transformer().fit(KEYS, Y, database=shop)
    before = list(fitted.features_)
    fitted.transform([1, 2], database=shop)
    assert fitted.features_ == before
