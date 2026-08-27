import duckdb
import narwhals as nw
import pytest

import tusk
from tusk.sklearn import DFSTransformer

KEYS = [1, 2, 3]


@pytest.fixture
def duckdb_db():
    """A two-table database whose keys are duckdb's default int32."""
    customers = duckdb.sql(
        "select * from (values "
        "(1, 30, timestamp '2024-01-01'), "
        "(2, 40, timestamp '2024-01-01'), "
        "(3, 50, timestamp '2024-01-01')) t(id, age, signed_up_at)",
    )
    sessions = duckdb.sql(
        "select * from (values "
        "(10, 1, timestamp '2024-03-04'), "
        "(20, 1, timestamp '2024-03-05'), "
        "(30, 2, timestamp '2024-03-06')) t(id, customer_id, started_at)",
    )
    return (
        tusk.Database("retail")
        .add_table(
            "customers",
            customers,
            primary_key="id",
            row_creation_time="signed_up_at",
        )
        .add_table(
            "sessions",
            sessions,
            primary_key="id",
            row_creation_time="started_at",
        )
        .add_relationship(
            parent="customers", child="sessions", foreign_key="customer_id"
        )
    )


def test_a_duckdb_database_returns_rows_in_key_order(duckdb_db):
    # duckdb keys these tables as int32 while the requested keys infer as
    # int64; the reorder join fails on that mismatch unless the key frame is
    # built with the collected frame's own dtype.
    fitted = DFSTransformer(target_table="customers").fit(KEYS, database=duckdb_db)
    matrix = nw.from_native(fitted.transform([3, 1], database=duckdb_db))
    assert matrix.shape[0] == 2


def test_collecting_natively_from_duckdb_gives_pyarrow(duckdb_db):
    fitted = DFSTransformer(target_table="customers").fit(KEYS, database=duckdb_db)
    matrix = fitted.transform(KEYS, database=duckdb_db)
    assert type(matrix).__module__.startswith("pyarrow")


def test_output_backend_pandas_gives_a_pandas_frame(duckdb_db):
    pytest.importorskip("pandas")
    fitted = DFSTransformer(
        target_table="customers",
        output_backend="pandas",
    ).fit(KEYS, database=duckdb_db)
    matrix = fitted.transform(KEYS, database=duckdb_db)
    assert type(matrix).__module__.startswith("pandas")


def test_an_unknown_output_backend_lists_the_ones_that_work(duckdb_db):
    # narwhals rejects the name before any import is attempted, so this is a
    # ValueError from narwhals rather than tusk's missing-package TuskError.
    fitted = DFSTransformer(
        target_table="customers",
        output_backend="nosuchbackend",
    ).fit(KEYS, database=duckdb_db)
    with pytest.raises(ValueError, match="pandas.*polars.*pyarrow"):
        fitted.transform(KEYS, database=duckdb_db)
