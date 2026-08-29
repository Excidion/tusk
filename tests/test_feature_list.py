"""FeatureList: the validated, self-applying collection synthesis hands back."""

import pickle
from collections.abc import Sequence
from datetime import date, datetime

import polars as pl
import pytest

import tusk
from tusk.exceptions import SchemaError
from tusk.feature_list import FeatureList
from tusk.primitives import Quantiles


@pytest.fixture
def features(db):
    """Feature definitions on ``customers``, including a multi-output one.

    Quantiles needs a numeric input, and the only one below ``customers`` is
    two hops down, so the multi-output case needs ``max_depth=2``.
    """
    return tusk.deep_feature_synthesis(
        database=db,
        target_table="customers",
        agg_primitives=["count", Quantiles(qs=(0.25, 0.75))],
        trans_primitives=[],
        max_depth=2,
        features_only=True,
    )


def test_synthesis_returns_a_feature_list(features):
    assert isinstance(features, FeatureList)


def test_deep_feature_synthesis_returns_one_alongside_the_matrix(db):
    _, features = tusk.deep_feature_synthesis(
        database=db,
        target_table="customers",
        agg_primitives=["count"],
        trans_primitives=[],
        max_depth=1,
    )
    assert isinstance(features, FeatureList)


def test_it_is_a_sequence(features):
    assert isinstance(features, Sequence)
    assert len(features) == len(list(features))
    assert features[0] in features
    assert list(reversed(features))[-1] == features[0]


def test_it_supports_the_comprehensions_a_list_would(features):
    assert [f.name for f in features if f.depth == 0]


def test_slicing_keeps_the_class_so_a_slice_can_be_applied(features):
    sliced = features[:2]
    assert isinstance(sliced, FeatureList)
    assert list(sliced) == list(features)[:2]


def test_an_empty_slice_is_rejected_like_an_empty_list(features):
    with pytest.raises(SchemaError, match="cannot be empty"):
        features[:0]


def test_a_slice_past_the_end_is_rejected_too(features):
    """The documented cost of slices staying a FeatureList.

    Every other Python sequence returns empty here. Pinned so the behaviour
    is a decision on the record rather than a surprise.
    """
    with pytest.raises(SchemaError, match="cannot be empty"):
        features[len(features) + 5 :]


def test_it_rejects_an_empty_collection():
    with pytest.raises(SchemaError, match="cannot be empty"):
        FeatureList([])


def test_synthesis_that_generates_nothing_raises_rather_than_returning_empty(db):
    """A target whose only column is its key has nothing to build from.

    Before ``synthesize`` returned a ``FeatureList`` this came back as ``[]``
    and only failed later, at compile time.
    """
    database = tusk.Database("x").add_table(
        "events",
        pl.LazyFrame({"id": [1, 2]}),
        primary_key="id",
    )
    with pytest.raises(SchemaError, match="no features generated for 'events'"):
        tusk.deep_feature_synthesis(
            database=database,
            target_table="events",
            agg_primitives=[],
            trans_primitives=[],
            features_only=True,
        )


def test_it_rejects_features_spanning_tables(db):
    customers = tusk.deep_feature_synthesis(
        database=db,
        target_table="customers",
        agg_primitives=["count"],
        trans_primitives=[],
        max_depth=1,
        features_only=True,
    )
    sessions = tusk.deep_feature_synthesis(
        database=db,
        target_table="sessions",
        agg_primitives=["count"],
        trans_primitives=[],
        max_depth=1,
        features_only=True,
    )
    with pytest.raises(SchemaError, match="customers.*sessions"):
        FeatureList([*customers, *sessions])


def test_target_table_is_the_shared_table(features):
    assert features.target_table == "customers"


def test_output_names_flattens_multi_output_features(features):
    assert list(features.output_names) == [n for f in features for n in f.output_names]
    # The multi-output primitive contributes more names than it does features.
    assert len(features.output_names) > len(features)


def test_apply_matches_the_free_function(features, db):
    assert (
        features.apply(db)
        .collect()
        .equals(
            tusk.apply_features(features, db).collect(),
        )
    )


def test_apply_honors_the_cutoff_time(features, db):
    matrix = features.apply(db, cutoff_time=datetime(2024, 1, 1))
    assert set(features.output_names) <= set(matrix.collect().columns)


def test_apply_rejects_a_non_datetime_cutoff(features, db):
    with pytest.raises(TypeError, match="must be a datetime"):
        features.apply(db, cutoff_time=date(2024, 1, 1))


def test_apply_features_accepts_a_plain_sequence(features, db):
    matrix = tusk.apply_features(list(features), db)
    assert isinstance(matrix, pl.LazyFrame)


def test_repr_is_a_summary_not_a_dump(features):
    text = repr(features)
    assert text == f"FeatureList({len(features)} features on 'customers')"
    assert "AggregationFeature" not in text


def test_equality_is_structural(features, db):
    same = tusk.deep_feature_synthesis(
        database=db,
        target_table="customers",
        agg_primitives=["count", Quantiles(qs=(0.25, 0.75))],
        trans_primitives=[],
        max_depth=2,
        features_only=True,
    )
    assert features == same
    assert hash(features) == hash(same)


def test_it_is_not_equal_to_a_plain_list(features):
    assert features != list(features)


def test_it_is_exported_from_the_package(features):
    assert isinstance(features, tusk.FeatureList)


def test_a_feature_list_survives_pickle_equal_and_hash_stable(features):
    """The property the frozen-dataclass rule on primitives exists to protect.

    A loaded feature set has to equal the one it was saved from, or it will
    not deduplicate against a freshly synthesized set and the combination
    produces duplicate columns.
    """
    restored = pickle.loads(pickle.dumps(features))
    assert restored == features
    assert hash(restored) == hash(features)
    assert restored.target_table == features.target_table
    assert restored.output_names == features.output_names


def test_a_restored_feature_list_still_computes(features, db):
    restored = pickle.loads(pickle.dumps(features))
    assert restored.apply(db).collect().equals(features.apply(db).collect())


def test_restored_features_deduplicate_against_a_fresh_run(features, db):
    """A loaded set combined with a fresh one must collapse, not double."""
    restored = pickle.loads(pickle.dumps(features))
    combined = tusk.FeatureList(dict.fromkeys([*restored, *features]))
    assert len(combined) == len(features)
