"""Performance runs against a real relational dataset.

This is the only tier that puts evidence behind the scale-beyond-memory goal.
Kept until deliberately cut. See the spec, section 11.

Run with: uv run --group benchmark pytest -m benchmark -s

Verified against relbench 2.1.2: that version's ``Table`` exposes ``df``,
``pkey_col``, ``time_col``, and ``fkey_col_to_pkey_table``, mapping directly
onto tusk's ``primary_key``, ``row_creation_time``, and ``add_relationship``
respectively -- no attribute-name translation was needed, matching the
mapping this tier exists to verify.

Three things about this tier are shaped by rel-ratebeer being large enough to
stress the machine rather than only the code. None applied to rel-f1, whose
largest table held 28k rows.

*The backend is duckdb, not polars.* At this size the distinction stops being
cosmetic. The matrix below is 339 features over 11.8M rows; polars' streaming
engine grows until the kernel kills it rather than spilling, and was measured
doing so at ~21GB across every route tried -- one plan, forced
``engine="streaming"``, feature-batched with the parts joined afterwards, and
pairwise joins sinking between steps. duckdb computes the same matrix in one
call within a fixed budget by spilling to disk. That is the difference the
scale claim rests on, so the benchmark runs on the backend that demonstrates
it.

*Frames are scanned from disk, not held in memory.* relbench hands over eager
pandas frames -- ~14GB for this dataset -- so the fixture writes each to
parquet, drops the ``Database``, and hands tusk ``read_parquet`` relations. A
benchmark that first loaded every row into RAM could not say anything about
computing over data that does not fit there.

*Memory is read as a resettable per-phase peak.* ``VmHWM`` in
``/proc/self/status`` is the kernel's high-water mark of resident set size,
and writing ``5`` to ``/proc/self/clear_refs`` resets it. Reading it after
each phase attributes memory to that phase instead of reporting one
process-lifetime maximum that the pandas staging would dominate no matter what
tusk did. ``tracemalloc`` is not an option: the data lives in off-heap
allocations a Python-heap-only tracker would not see. Both files are
Linux-only, which the CI for this tier is.
"""

import gc
import re
import time
from pathlib import Path

import narwhals as nw
import pytest

import tusk

pytestmark = pytest.mark.benchmark

relbench_datasets = pytest.importorskip("relbench.datasets")
duckdb = pytest.importorskip("duckdb")

DATASET = "rel-ratebeer"
TARGET = "beer_ratings"
MAX_DEPTH = 2

MEMORY_LIMIT = "12GB"
"""Deliberately below what the equivalent polars run demanded before it was
killed, so that finishing at all means spilling rather than merely having
enough RAM.

This bounds duckdb's buffer pool, not the process: measured peak RSS is
around 16GB, since the limit does not cover allocations outside the buffer
manager. The evidence is that the run completes by spilling to disk where
polars grew unboundedly, not that RSS stays under this number."""


def _reset_peak_rss():
    """Reset the kernel's peak-RSS watermark for this process.

    Makes the next :func:`_peak_rss_mb` reading a peak for the phase that
    follows rather than for the process so far.
    """
    Path("/proc/self/clear_refs").write_text("5\n")


def _peak_rss_mb():
    """Peak resident set size since the last reset, in megabytes.

    Returns:
        ``VmHWM`` from ``/proc/self/status``, converted from KB to MB.

    Raises:
        RuntimeError: If the kernel does not report ``VmHWM``, which would
            make every memory figure in this tier meaningless.
    """
    status = Path("/proc/self/status").read_text()
    match = re.search(r"^VmHWM:\s+(\d+) kB", status, re.M)
    if match is None:
        raise RuntimeError("no VmHWM in /proc/self/status")
    return int(match.group(1)) / 1024


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    """rel-ratebeer staged to parquet and mapped onto a duckdb-backed Database.

    Writes every relbench table to parquet through duckdb, releases the pandas
    frames, and builds the Database over ``read_parquet`` relations, so the
    timed phases start from data on disk rather than data in RAM.

    Each frame is dropped as soon as it is staged, and duckdb's memory limit
    is not imposed until they are all gone. relbench materializes ~14GB of
    pandas here; letting that coexist with a 12GB duckdb budget needs 26GB of
    RAM for a step that does no work worth measuring.

    Args:
        tmp_path_factory: pytest's session-scoped temporary directory factory.

    Returns:
        A tuple of the Database, the row count per table, the number of
        foreign-key relationships, and the duckdb connection.
    """
    relbench_db = relbench_datasets.get_dataset(DATASET, download=True).get_db()
    directory = tmp_path_factory.mktemp(DATASET)

    connection = duckdb.connect()
    connection.execute(f"SET temp_directory='{directory / 'spill'}'")

    schemas = {}
    row_counts = {}
    for name in list(relbench_db.table_dict):
        table = relbench_db.table_dict.pop(name)
        path = directory / f"{name}.parquet"
        connection.register("staging", table.df)
        connection.execute(f"COPY staging TO '{path}' (FORMAT parquet)")
        connection.unregister("staging")
        row_counts[name] = len(table.df)
        schemas[name] = (table.pkey_col, table.time_col, table.fkey_col_to_pkey_table)
        del table
        gc.collect()

    del relbench_db
    gc.collect()
    connection.execute(f"SET memory_limit='{MEMORY_LIMIT}'")

    database = tusk.Database("relbench")
    for name, (primary_key, row_creation_time, _) in schemas.items():
        path = directory / f"{name}.parquet"
        database.add_table(
            name,
            nw.from_native(connection.sql(f"SELECT * FROM read_parquet('{path}')")),
            primary_key=primary_key,
            row_creation_time=row_creation_time,
        )
    relationships = 0
    for name, (_, _, foreign_keys) in schemas.items():
        for foreign_key, parent in foreign_keys.items():
            # validate=False: matching_key_dtypes demands the foreign key and
            # the parent's primary key have identical dtypes, because pyarrow
            # will not join Int64 to Int32. duckdb will, and this tier measures
            # DFS throughput on a third-party dataset whose key widths are not
            # ours to police -- a mismatch here would abort the benchmark over
            # a join duckdb performs happily.
            database.add_relationship(
                parent=parent,
                child=name,
                foreign_key=foreign_key,
                validate=False,
            )
            relationships += 1

    return database, row_counts, relationships, connection


def test_deep_feature_synthesis_on_relbench(staged, tmp_path):
    """Runs synthesis and compute on rel-ratebeer and reports timing/memory.

    rel-ratebeer's schema is a genuine multi-table shape, not a toy: 13
    tables, 16 foreign-key relationships, and fan-in diamonds (``beers``,
    ``users``, ``countries`` and ``places`` are each referenced by three or
    more child tables), which exercises tusk's join and aggregation planning
    well beyond the hand-built 3-table fixtures in the core suite.

    ``beer_ratings`` is both the largest table and the semantically load-
    bearing one -- a rating is the thing a model would predict. Nothing
    references its primary key, so every feature reaches it by traversing
    *up* to ``users``, ``beers`` and ``availability``; the depth-2 features
    are aggregations computed on those parents over their own children and
    then carried down, which is the most expensive shape this schema offers.

    Args:
        staged: The Database, row counts, relationship count and connection.
        tmp_path: Directory to write the feature matrix into.
    """
    database, row_counts, relationships, connection = staged
    target_rows = row_counts[TARGET]

    _reset_peak_rss()
    start = time.perf_counter()
    features = tusk.deep_feature_synthesis(
        database=database,
        target_table=TARGET,
        max_depth=MAX_DEPTH,
        features_only=True,
    )
    synthesis_seconds = time.perf_counter() - start
    synthesis_peak_mb = _peak_rss_mb()

    matrix_path = tmp_path / "feature_matrix.parquet"
    _reset_peak_rss()
    start = time.perf_counter()
    tusk.apply_features(features, database).write_parquet(str(matrix_path))
    compute_seconds = time.perf_counter() - start
    compute_peak_mb = _peak_rss_mb()

    matrix_rows = connection.sql(
        f"SELECT count(*) FROM read_parquet('{matrix_path}')",
    ).fetchone()[0]

    print(
        f"\ndataset={DATASET} backend=duckdb tables={len(row_counts)} "
        f"relationships={relationships}\n"
        f"target={TARGET} target_rows={target_rows} max_depth={MAX_DEPTH} "
        f"memory_limit={MEMORY_LIMIT}\n"
        f"matrix_rows={matrix_rows} features={len(features)} "
        f"matrix_mb={matrix_path.stat().st_size / 1e6:.0f}\n"
        f"synthesis={synthesis_seconds:.2f}s peak_rss={synthesis_peak_mb:.0f}MB\n"
        f"compute={compute_seconds:.1f}s peak_rss={compute_peak_mb:.0f}MB",
    )

    assert matrix_rows == target_rows
    assert len(features) > 0
