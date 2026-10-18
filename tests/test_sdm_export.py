import datetime as dt
import warnings

import pytest

import tusk

pytest.importorskip("sdm")
pytestmark = pytest.mark.sdm

from sdm import Stype, TableTensor  # noqa: E402

from tusk.sdm import to_relational_sampler  # noqa: E402


def _types(tensor):
    return {
        str(stype): list(names)
        for stype, names in tensor.columns.items()
        if len(names) > 0
    }


def test_the_tables_hold_the_columns_of_their_semantic_types(db):
    tables = to_relational_sampler(db).data.tables
    assert set(tables) == {"customers", "sessions", "transactions"}
    assert _types(tables["customers"]) == {
        "numerical": ["age"],
        "datetime": ["signed_up_at"],
        "id": ["id"],
    }
    assert _types(tables["transactions"]) == {
        "numerical": ["amount", "verified"],
        "categorical": ["channel"],
        "datetime": ["occurred_at"],
        "id": ["id", "session_id"],
    }


def test_a_relationship_joins_the_child_to_the_parent(db):
    relationships = to_relational_sampler(db).data.relationships
    assert [
        (r.left_table, r.left_columns, r.right_table, r.right_columns)
        for r in relationships
    ] == [
        ("sessions", ("customer_id",), "customers", ("id",)),
        ("transactions", ("session_id",), "sessions", ("id",)),
    ]


def test_the_time_columns_are_the_row_creation_times(db):
    assert to_relational_sampler(db).time_columns == {
        "customers": "signed_up_at",
        "sessions": "started_at",
        "transactions": "occurred_at",
    }


def test_as_of_drops_later_rows(db):
    sampler = to_relational_sampler(db.as_of(dt.datetime(2024, 3, 4, 12)))
    tables = sampler.data.tables
    assert tables["customers"].size(0) == 3
    assert tables["sessions"].size(0) == 1
    assert tables["transactions"].size(0) == 2


def test_as_of_restores_updated_columns(updating_db):
    cutoff = dt.datetime(2024, 6, 1)
    sampler = to_relational_sampler(updating_db.as_of(cutoff))
    orders = sampler.data.tables["orders"].to_pandas()
    assert orders["status"].tolist() == ["pending", "pending", "delivered"]


def _database_on(backend, frames):
    def convert(frame):
        table = frame.collect()
        if backend == "pandas":
            return table.to_pandas()
        if backend == "duckdb":
            import duckdb

            return duckdb.from_arrow(table.to_arrow())
        return table

    primary_keys = {"customers": "id", "sessions": "id", "transactions": "id"}
    database = tusk.Database("d")
    for name, frame in frames.items():
        database.add_table(name, convert(frame), primary_key=primary_keys[name])
    return database.add_relationship(
        parent="customers", child="sessions", foreign_key="customer_id"
    ).add_relationship(
        parent="sessions", child="transactions", foreign_key="session_id"
    )


@pytest.mark.parametrize("backend", ["pandas", "duckdb"])
def test_every_backend_exports_the_same_tables(backend):
    pytest.importorskip("pandas")
    pytest.importorskip("duckdb")
    from conftest import _frames

    customers, sessions, transactions = _frames()
    frames = {
        "customers": customers,
        "sessions": sessions,
        "transactions": transactions,
    }
    expected = to_relational_sampler(_database_on("polars", frames)).data.tables
    actual = to_relational_sampler(_database_on(backend, frames)).data.tables
    for name, tensor in expected.items():
        assert _types(actual[name]) == _types(tensor)
        left = tensor.to_pandas().sort_index(axis=1).to_dict("list")
        right = actual[name].to_pandas().sort_index(axis=1).to_dict("list")
        assert right == left


def test_the_sampler_samples_one_task_row(db):
    pytest.importorskip("pyg_lib")
    sampler = to_relational_sampler(db)
    task_table = TableTensor.from_columns(
        {"customer": [1, 2], "at": [dt.datetime(2024, 3, 6)] * 2},
        {"customer": "id", "at": "datetime"},
    )
    output = sampler(
        task_table,
        {"task_column": "customer", "table": "customers", "table_column": "id"},
        [2, 2],
        task_time_column="at",
    )
    assert set(output.related_tables.tables) <= {
        "customers",
        "sessions",
        "transactions",
    }
    assert output.task_table.size(0) == 2


def test_kumo_relational_runs_on_the_sampled_tables_without_a_stype_warning(db):
    pytest.importorskip("pyg_lib")
    torch = pytest.importorskip("torch")
    from sdm.models import KumoRelational

    sampler = to_relational_sampler(db)

    def sample(ids, labels):
        task_table = TableTensor.from_columns(
            {
                "customer": ids,
                "at": [dt.datetime(2024, 3, 7)] * len(ids),
                "label": labels,
            },
            {"customer": "id", "at": "datetime", "label": "numerical"},
        )
        link = {"task_column": "customer", "table": "customers", "table_column": "id"}
        output = sampler(task_table, link, [4, 4], task_time_column="at")
        return output.task_table, output.related_tables

    context, related_context = sample([1, 2], [1.0, 2.0])
    query, related_query = sample([3], [0.0])
    model = KumoRelational(task="regression", pretrained=False, device="cpu")
    with pytest.warns() as caught:
        warnings.warn("keeps pytest.warns satisfied", stacklevel=1)
        out = model(
            x_context=context.drop_columns("label"),
            y_context=context[:, "label"],
            x_query=query.drop_columns("label"),
            related_context_tables=related_context,
            related_query_tables=related_query,
            num_hops=1,
        )
    assert isinstance(out, torch.Tensor)
    assert not [w for w in caught if "unsupported feature stypes" in str(w.message)]


def test_a_cudf_database_exports_the_same_tables_as_a_pandas_one():
    cudf = pytest.importorskip("cudf")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    from conftest import _frames

    customers, _, _ = _frames()
    frame = customers.collect()
    pandas_db = tusk.Database("d").add_table(
        "customers", frame.to_pandas(), primary_key="id"
    )
    cudf_db = tusk.Database("d").add_table(
        "customers", cudf.DataFrame(frame.to_pandas()), primary_key="id"
    )
    expected = to_relational_sampler(pandas_db).data.tables["customers"]
    actual = to_relational_sampler(cudf_db).data.tables["customers"]
    assert actual.is_cuda
    assert _types(actual) == _types(expected)
    assert Stype.id in actual.active_stypes
