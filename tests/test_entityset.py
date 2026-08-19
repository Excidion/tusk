import narwhals as nw
import polars as pl
import pyarrow as pa
import pytest

import tusk
from tusk.database import Relationship
from tusk.exceptions import MissingPrimaryKeyWarning, SchemaError


def test_schema_is_read_from_the_frame(es):
    schema = es.schema("transactions")
    assert schema.primary_key == "id"
    assert schema.row_creation_time == "occurred_at"
    assert schema.dtypes["amount"] == nw.Float64


def test_relationship_accessors(es):
    assert es.children_of("customers") == [
        Relationship(parent="customers", child="sessions", foreign_key="customer_id")
    ]
    assert es.parents_of("transactions") == [
        Relationship(parent="sessions", child="transactions", foreign_key="session_id")
    ]
    assert es.children_of("transactions") == []


def test_input_exclusions_are_join_keys_only(es):
    # sessions has primary_key "id" and one parent relationship whose foreign
    # key is "customer_id". started_at is the row_creation_time, which is a
    # measurement, not a join key, so it stays available to primitives.
    assert es.input_excluded_columns("sessions") == frozenset({"id", "customer_id"})


def test_output_exclusions_add_the_row_creation_time(es):
    # The matrix drops the time index as a raw passthrough column, on top of
    # the join keys. Features derived from it are unaffected.
    assert es.output_excluded_columns("sessions") == frozenset(
        {"id", "customer_id", "started_at"}
    )


def test_missing_primary_key_warns():
    with pytest.warns(
        MissingPrimaryKeyWarning, match="cannot be used as a relationship parent"
    ):
        tusk.Database("x").add_table("t", pl.LazyFrame({"a": [1]}))


def test_unknown_column_raises():
    with pytest.raises(SchemaError, match="nope"):
        tusk.Database("x").add_table("t", pl.LazyFrame({"a": [1]}), primary_key="nope")


def test_composite_key_raises():
    with pytest.raises(SchemaError, match="[Cc]omposite"):
        tusk.Database("x").add_table(
            "t",
            pl.LazyFrame({"a": [1]}),
            primary_key=["a"],  # ty: ignore[invalid-argument-type]
        )


def test_parent_without_primary_key_raises():
    es = tusk.Database("x").add_table("p", pl.LazyFrame({"a": [1]}), primary_key="a")
    with pytest.warns(MissingPrimaryKeyWarning):
        es.add_table("c", pl.LazyFrame({"a": [1], "p_a": [1]}))
    with pytest.raises(SchemaError, match="primary_key"):
        es.add_relationship(parent="c", child="p", foreign_key="a")


def test_unknown_foreign_key_raises(es):
    with pytest.raises(SchemaError, match="missing"):
        es.add_relationship(parent="customers", child="sessions", foreign_key="nope")


def test_eager_input_is_recorded_and_lazified():
    es = tusk.Database("x").add_table("t", pl.DataFrame({"a": [1]}), primary_key="a")
    assert es.is_eager is True
    assert isinstance(es.frame("t"), nw.LazyFrame)


def test_self_reference_is_allowed():
    es = tusk.Database("x").add_table(
        "employees",
        pl.LazyFrame({"id": [1, 2], "manager_id": [None, 1]}),
        primary_key="id",
    )
    es.add_relationship(parent="employees", child="employees", foreign_key="manager_id")
    assert es.children_of("employees")[0].foreign_key == "manager_id"


def test_duplicate_table_name_raises():
    es = tusk.Database("x").add_table("t", pl.LazyFrame({"a": [1]}), primary_key="a")
    with pytest.raises(SchemaError, match="'t'"):
        es.add_table("t", pl.LazyFrame({"a": [1]}), primary_key="a")


def test_backend_mismatch_raises():
    es = tusk.Database("x").add_table("t", pl.LazyFrame({"a": [1]}), primary_key="a")
    with pytest.raises(SchemaError, match="polars.*pyarrow|pyarrow.*polars"):
        es.add_table("u", pa.table({"a": [1]}), primary_key="a")
