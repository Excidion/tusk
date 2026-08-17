"""Performance runs against a real relational dataset.

This is the only tier that puts evidence behind the scale-beyond-memory goal.
Kept until deliberately cut. See the spec, section 11.

Run with: uv run --group benchmark pytest -m benchmark -s

Verified against relbench 2.1.2: that version's ``Table`` exposes ``df``,
``pkey_col``, ``time_col``, and ``fkey_col_to_pkey_table``, mapping directly
onto tusk's ``primary_key``, ``row_creation_time``, and ``add_relationship``
respectively -- no attribute-name translation was needed, matching the
mapping this tier exists to verify.

Memory is read via ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` (whole-
process peak RSS in KB on Linux) rather than ``tracemalloc``. tusk's data
lives in narwhals/polars frames, which are backed by off-heap Rust
allocations that ``tracemalloc`` -- a Python-heap-only tracker -- would not
see, making it misleading for a library whose selling point is exactly that
it does not hold everything on the Python heap. ``ru_maxrss`` is cumulative
since process start rather than resettable per phase, so the two readings
below are "peak RSS so far" checkpoints, not isolated per-phase figures.
"""

import resource
import time

import polars as pl
import pytest

import tusk

pytestmark = pytest.mark.benchmark

relbench_datasets = pytest.importorskip("relbench.datasets")


def _entity_set(db):
    """Map a relbench Database onto a tusk EntitySet.

    Args:
        db: A relbench Database.

    Returns:
        A tusk EntitySet with the same tables and relationships.
    """
    es = tusk.EntitySet("relbench")
    for name, table in db.table_dict.items():
        es.add_dataframe(
            name,
            pl.from_pandas(table.df).lazy(),
            primary_key=table.pkey_col,
            row_creation_time=table.time_col,
        )
    for name, table in db.table_dict.items():
        for foreign_key, parent in table.fkey_col_to_pkey_table.items():
            es.add_relationship(parent=parent, child=name, foreign_key=foreign_key)
    return es


def _peak_rss_mb():
    """Current peak resident set size of this process, in megabytes.

    Returns:
        Peak RSS since process start, in MB (Linux reports ru_maxrss in KB).
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


@pytest.fixture(scope="module")
def db():
    """The rel-f1 database: 9 tables, 13 foreign keys, largest table ~28k rows."""
    return relbench_datasets.get_dataset("rel-f1", download=True).get_db()


def test_dfs_on_relbench(db):
    """Runs synthesis and compute on rel-f1 and reports timing/memory.

    Target table is ``standings`` (the largest by row count in rel-f1, ~28k
    rows), chosen by the same "largest table" rule the brief specifies: it
    maximizes the row count phase 2 has to materialize, giving the most
    meaningful timing and memory signal among the 9 available targets.

    rel-f1's schema is a genuine multi-table shape, not a toy: 9 tables, 13
    foreign-key relationships, and fan-in diamonds (``races``, ``drivers``,
    and ``constructors`` are each referenced by three or more child tables),
    which exercises tusk's join and aggregation planning well beyond the
    hand-built 3-table fixtures used in the core suite.
    """
    n_tables = len(db.table_dict)
    n_relationships = sum(len(t.fkey_col_to_pkey_table) for t in db.table_dict.values())
    es = _entity_set(db)
    target = max(db.table_dict, key=lambda n: len(db.table_dict[n].df))
    target_rows = len(db.table_dict[target].df)

    start = time.perf_counter()
    features = tusk.dfs(
        entityset=es, target_dataframe_name=target, max_depth=2, features_only=True
    )
    synthesis_seconds = time.perf_counter() - start
    peak_after_synthesis_mb = _peak_rss_mb()

    start = time.perf_counter()
    matrix = tusk.calculate_feature_matrix(features, es).collect()
    compute_seconds = time.perf_counter() - start
    peak_after_compute_mb = _peak_rss_mb()

    print(
        f"\ndataset=rel-f1 tables={n_tables} relationships={n_relationships}\n"
        f"target={target} target_rows={target_rows}\n"
        f"matrix_rows={matrix.height} features={len(features)}\n"
        f"synthesis={synthesis_seconds:.2f}s "
        f"peak_rss_after_synthesis={peak_after_synthesis_mb:.1f}MB\n"
        f"compute={compute_seconds:.2f}s "
        f"peak_rss_after_compute={peak_after_compute_mb:.1f}MB"
    )

    assert matrix.height == target_rows
    assert len(features) > 0
