import narwhals as nw
import polars as pl
import pytest

import tusk
from tusk.entityset import Relationship
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


def test_key_columns_include_pk_fk_and_time(es):
    assert es.key_columns("sessions") == frozenset({"id", "customer_id", "started_at"})


def test_missing_primary_key_warns():
    with pytest.warns(
        MissingPrimaryKeyWarning, match="cannot be used as a relationship parent"
    ):
        tusk.EntitySet("x").add_dataframe("t", pl.LazyFrame({"a": [1]}))


def test_unknown_column_raises():
    with pytest.raises(SchemaError, match="nope"):
        tusk.EntitySet("x").add_dataframe(
            "t", pl.LazyFrame({"a": [1]}), primary_key="nope"
        )


def test_composite_key_raises():
    with pytest.raises(SchemaError, match="[Cc]omposite"):
        tusk.EntitySet("x").add_dataframe(
            "t",
            pl.LazyFrame({"a": [1]}),
            primary_key=["a"],  # ty: ignore[invalid-argument-type]
        )


def test_parent_without_primary_key_raises():
    es = tusk.EntitySet("x").add_dataframe(
        "p", pl.LazyFrame({"a": [1]}), primary_key="a"
    )
    with pytest.warns(MissingPrimaryKeyWarning):
        es.add_dataframe("c", pl.LazyFrame({"a": [1], "p_a": [1]}))
    with pytest.raises(SchemaError, match="primary_key"):
        es.add_relationship(parent="c", child="p", foreign_key="a")


def test_unknown_foreign_key_raises(es):
    with pytest.raises(SchemaError, match="missing"):
        es.add_relationship(parent="customers", child="sessions", foreign_key="nope")


def test_eager_input_is_recorded_and_lazified():
    es = tusk.EntitySet("x").add_dataframe(
        "t", pl.DataFrame({"a": [1]}), primary_key="a"
    )
    assert es.is_eager is True
    assert isinstance(es.frame("t"), nw.LazyFrame)


def test_self_reference_is_allowed():
    es = tusk.EntitySet("x").add_dataframe(
        "employees",
        pl.LazyFrame({"id": [1, 2], "manager_id": [None, 1]}),
        primary_key="id",
    )
    es.add_relationship(parent="employees", child="employees", foreign_key="manager_id")
    assert es.children_of("employees")[0].foreign_key == "manager_id"


def test_duplicate_table_name_raises():
    es = tusk.EntitySet("x").add_dataframe(
        "t", pl.LazyFrame({"a": [1]}), primary_key="a"
    )
    with pytest.raises(SchemaError, match="'t'"):
        es.add_dataframe("t", pl.LazyFrame({"a": [1]}), primary_key="a")


def test_backend_mismatch_raises():
    pd = pytest.importorskip("pandas")
    es = tusk.EntitySet("x").add_dataframe(
        "t", pl.LazyFrame({"a": [1]}), primary_key="a"
    )
    with pytest.raises(SchemaError, match="backend"):
        es.add_dataframe("u", pd.DataFrame({"a": [1]}), primary_key="a")
