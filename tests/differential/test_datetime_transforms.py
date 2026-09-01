"""Cross-check tusk's datetime transform primitives against featuretools.

Each test builds one primitive on one side and the same primitive on the
other, over a single table, and compares the one resulting column. Model:
``tests/differential/test_aggregations.py``; this file builds its own
matrices rather than importing them, sharing only ``_as_tusk`` from
:mod:`differential`.

Run with: uv run --group validation pytest -m differential

Verified against featuretools 1.31.0.
"""

import datetime as dt

import pytest

import tusk
from differential import _as_tusk

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pl = pytest.importorskip("polars")
featuretools = pytest.importorskip("featuretools")

pytestmark = pytest.mark.differential


@pytest.fixture
def rows():
    """A single table covering the cases TIME_SINCE can diverge on.

    ``occurred_at`` has a null value, which must stay null on both sides, and
    a value after ``cutoff_time``, where the elapsed time is negative rather
    than clipped to zero. Every timestamp lands on a whole second so the
    seconds conversion in the test is exact.
    """
    cutoff_time = dt.datetime(2024, 3, 1)
    occurred_at = pd.to_datetime(
        [
            dt.datetime(2024, 1, 1),
            dt.datetime(2024, 2, 15),
            None,
            dt.datetime(2023, 12, 1),
            dt.datetime(2024, 6, 1),
        ],
    )
    frame = pd.DataFrame(
        {"id": np.arange(1, len(occurred_at) + 1), "occurred_at": occurred_at},
    )
    _assert_rows_invariants(frame, cutoff_time)
    return frame, cutoff_time


def _assert_rows_invariants(frame, cutoff_time):
    """Guard the cases ``rows`` is built to cover.

    Without these, a change to the fixture's timestamps could silently drop
    the one case a differential test relies on to catch a divergence.

    Args:
        frame: The table built by ``rows``.
        cutoff_time: The moment TIME_SINCE measures against.
    """
    assert frame["occurred_at"].isna().any()
    assert (frame["occurred_at"] > cutoff_time).any()


def test_time_since_matches_featuretools_on_every_row(rows):
    """tusk's TIME_SINCE, converted to seconds, equals featuretools' TIME_SINCE.

    Covers a null datetime, which must stay null on both sides, and a
    timestamp after the cutoff time, where the elapsed time is negative.
    """
    frame, cutoff_time = rows
    ours = _tusk_matrix(frame, cutoff_time, "time_since")
    theirs = _featuretools_matrix(frame, cutoff_time, "time_since")
    ours_in_seconds = ours[_as_tusk("TIME_SINCE(occurred_at)")].dt.total_seconds()
    pd.testing.assert_series_equal(
        ours_in_seconds.reset_index(drop=True).astype(float),
        theirs["TIME_SINCE(occurred_at)"].reset_index(drop=True).astype(float),
        check_names=False,
    )


def _featuretools_matrix(frame, cutoff_time, primitive_name):
    """Run one transform primitive through featuretools and return its matrix.

    Args:
        frame: The table to build features over.
        cutoff_time: The moment the primitive measures against, broadcast to
            every row.
        primitive_name: The primitive's featuretools name, the only entry in
            ``trans_primitives`` so exactly one feature is built.

    Returns:
        The feature matrix, sorted by the table's index.
    """
    es = featuretools.EntitySet("rows")
    es = es.add_dataframe(dataframe_name="rows", dataframe=frame, index="id")
    matrix, _ = featuretools.dfs(
        entityset=es,
        target_dataframe_name="rows",
        agg_primitives=[],
        trans_primitives=[primitive_name],
        max_depth=1,
        cutoff_time=cutoff_time,
    )
    return matrix.sort_index()


def _tusk_matrix(frame, cutoff_time, primitive_name):
    """Run one transform primitive through tusk and return its matrix.

    Args:
        frame: The table to build features over.
        cutoff_time: The moment the primitive measures against.
        primitive_name: The primitive's name, the only entry in
            ``trans_primitives`` so exactly one feature is built. tusk and
            featuretools spell this primitive's name the same way.

    Returns:
        The feature matrix, collected and sorted by the table's primary key.
    """
    db = tusk.Database("rows_db").add_table(
        "rows",
        pl.from_pandas(frame).lazy(),
        primary_key="id",
    )
    matrix, _ = tusk.deep_feature_synthesis(
        database=db,
        target_table="rows",
        agg_primitives=[],
        trans_primitives=[primitive_name],
        max_depth=1,
        cutoff_time=cutoff_time,
    )
    return matrix.collect().sort("id").to_pandas().set_index("id")
