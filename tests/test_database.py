import datetime as dt
import warnings

import narwhals as nw
import polars as pl
import pyarrow as pa
import pytest

import tusk
from tusk.database import Relationship
from tusk.exceptions import (
    ImplicitEarlierValueWarning,
    MissingPrimaryKeyWarning,
    SchemaError,
    ValidationError,
)


def test_schema_is_read_from_the_frame(db):
    schema = db.schema("transactions")
    assert schema.primary_key == "id"
    assert schema.row_creation_time == "occurred_at"
    assert schema.dtypes["amount"] == nw.Float64


def test_relationship_accessors(db):
    assert db.children_of("customers") == [
        Relationship(parent="customers", child="sessions", foreign_key="customer_id"),
    ]
    assert db.parents_of("transactions") == [
        Relationship(parent="sessions", child="transactions", foreign_key="session_id"),
    ]
    assert db.children_of("transactions") == []


def test_input_exclusions_are_join_keys_only(db):
    # sessions has primary_key "id" and one parent relationship whose foreign
    # key is "customer_id". started_at is the row_creation_time, which is a
    # measurement, not a join key, so it stays available to primitives.
    assert db.input_excluded_columns("sessions") == frozenset({"id", "customer_id"})


def test_output_exclusions_add_the_row_creation_time(db):
    # The matrix drops the time index as a raw passthrough column, on top of
    # the join keys. Features derived from it are unaffected.
    assert db.output_excluded_columns("sessions") == frozenset(
        {"id", "customer_id", "started_at"},
    )


def test_missing_primary_key_warns():
    with pytest.warns(
        MissingPrimaryKeyWarning,
        match="cannot be used as a relationship parent",
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
    db = tusk.Database("x").add_table("p", pl.LazyFrame({"a": [1]}), primary_key="a")
    with pytest.warns(MissingPrimaryKeyWarning):
        db.add_table("c", pl.LazyFrame({"a": [1], "p_a": [1]}))
    with pytest.raises(SchemaError, match="primary_key"):
        db.add_relationship(parent="c", child="p", foreign_key="a")


def test_unknown_foreign_key_raises(db):
    with pytest.raises(SchemaError, match="missing"):
        db.add_relationship(parent="customers", child="sessions", foreign_key="nope")


@pytest.mark.parametrize(
    "table",
    [
        pl.DataFrame({"a": [1]}),
        pl.LazyFrame({"a": [1]}),
        nw.from_native(pl.DataFrame({"a": [1]})),
        nw.from_native(pl.LazyFrame({"a": [1]})),
    ],
    ids=["native-eager", "native-lazy", "narwhals-eager", "narwhals-lazy"],
)
def test_every_input_frame_form_is_lazified(table):
    db = tusk.Database("x").add_table("t", table, primary_key="a")
    assert isinstance(db.frame("t"), nw.LazyFrame)


def test_self_reference_is_allowed():
    db = tusk.Database("x").add_table(
        "employees",
        pl.LazyFrame({"id": [1, 2], "manager_id": [None, 1]}),
        primary_key="id",
    )
    db.add_relationship(parent="employees", child="employees", foreign_key="manager_id")
    assert db.children_of("employees")[0].foreign_key == "manager_id"


def test_duplicate_table_name_raises():
    db = tusk.Database("x").add_table("t", pl.LazyFrame({"a": [1]}), primary_key="a")
    with pytest.raises(SchemaError, match="'t'"):
        db.add_table("t", pl.LazyFrame({"a": [1]}), primary_key="a")


def test_backend_mismatch_raises():
    db = tusk.Database("x").add_table("t", pl.LazyFrame({"a": [1]}), primary_key="a")
    with pytest.raises(SchemaError, match="polars.*pyarrow|pyarrow.*polars"):
        db.add_table("u", pa.table({"a": [1]}), primary_key="a")


def test_row_update_times_land_on_the_schema():
    db = tusk.Database("d").add_table(
        "orders",
        pl.LazyFrame(
            {
                "id": [1],
                "status": ["delivered"],
                "updated_at": [dt.datetime(2024, 9, 1)],
            },
        ),
        primary_key="id",
        row_update_times={"updated_at": {"status": "pending", "updated_at": None}},
    )
    assert db.schema("orders").row_update_times == {
        "updated_at": {"status": "pending", "updated_at": None},
    }


def test_column_updates_flattens_the_declaration():
    db = tusk.Database("d").add_table(
        "orders",
        pl.LazyFrame(
            {
                "id": [1],
                "status": ["delivered"],
                "updated_at": [dt.datetime(2024, 9, 1)],
            },
        ),
        primary_key="id",
        row_update_times={"updated_at": {"status": "pending", "updated_at": None}},
    )
    assert db.schema("orders").column_updates == (
        ("updated_at", "status", "pending"),
        ("updated_at", "updated_at", None),
    )


def test_a_table_without_row_update_times_has_an_empty_mapping():
    db = tusk.Database("d").add_table(
        "orders", pl.LazyFrame({"id": [1]}), primary_key="id"
    )
    assert db.schema("orders").row_update_times == {}
    assert db.schema("orders").column_updates == ()


def test_an_update_time_that_does_not_list_itself_is_completed_and_warned():
    with pytest.warns(ImplicitEarlierValueWarning, match="'updated_at'"):
        db = tusk.Database("d").add_table(
            "orders",
            pl.LazyFrame(
                {
                    "id": [1],
                    "status": ["delivered"],
                    "updated_at": [dt.datetime(2024, 9, 1)],
                },
            ),
            primary_key="id",
            row_update_times={"updated_at": {"status": "pending"}},
        )
    assert db.schema("orders").row_update_times["updated_at"]["updated_at"] is None


def test_an_update_time_that_lists_itself_warns_nothing():
    with warnings.catch_warnings():
        warnings.simplefilter("error", ImplicitEarlierValueWarning)
        tusk.Database("d").add_table(
            "orders",
            pl.LazyFrame(
                {"id": [1], "updated_at": [dt.datetime(2024, 9, 1)]},
            ),
            primary_key="id",
            row_update_times={"updated_at": {"updated_at": dt.datetime(2024, 1, 1)}},
        )


def test_a_chained_update_time_warns_about_both_and_is_rejected():
    # 'first' updates 'second', and 'second' also lacks a pre-update value of
    # its own, so both get an implicit self-entry -- which leaves 'second'
    # listed under both 'first' and itself, and singly_updated_columns rejects
    # that chain rather than letting a later mask read a not-yet-masked value.
    with pytest.warns(ImplicitEarlierValueWarning) as caught:
        with pytest.raises(ValidationError, match="'second'"):
            tusk.Database("d").add_table(
                "orders",
                pl.LazyFrame(
                    {
                        "id": [1],
                        "first": [dt.datetime(2024, 9, 1)],
                        "second": [dt.datetime(2024, 9, 2)],
                    },
                ),
                primary_key="id",
                row_update_times={"first": {"second": None}, "second": {}},
            )
    assert len(caught) == 2
    messages = {str(warning.message) for warning in caught}
    assert any("'first'" in message for message in messages)
    assert any("'second'" in message for message in messages)


def test_an_unknown_update_time_column_is_rejected():
    with pytest.raises(SchemaError, match="row_update_time 'nope'"):
        tusk.Database("d").add_table(
            "orders",
            pl.LazyFrame({"id": [1]}),
            primary_key="id",
            row_update_times={"nope": {}},
        )


def test_an_unknown_updated_column_is_rejected():
    with pytest.raises(SchemaError, match="'nope'"):
        tusk.Database("d").add_table(
            "orders",
            pl.LazyFrame({"id": [1], "updated_at": [dt.datetime(2024, 9, 1)]}),
            primary_key="id",
            row_update_times={"updated_at": {"nope": 1}},
        )


def test_a_flat_row_update_times_mapping_is_rejected():
    with pytest.raises(SchemaError, match="row_update_time 'updated_at'"):
        tusk.Database("d").add_table(
            "orders",
            pl.LazyFrame({"id": [1], "updated_at": [dt.datetime(2024, 9, 1)]}),
            primary_key="id",
            row_update_times={"updated_at": ["status"]},  # ty: ignore[invalid-argument-type]
        )


def test_a_non_mapping_row_update_times_is_rejected():
    with pytest.raises(SchemaError, match="row_update_times"):
        tusk.Database("d").add_table(
            "orders",
            pl.LazyFrame({"id": [1]}),
            primary_key="id",
            row_update_times=["updated_at"],  # ty: ignore[invalid-argument-type]
        )


def test_the_declaration_is_copied_not_aliased():
    declared = {"updated_at": {"status": "pending", "updated_at": None}}
    db = tusk.Database("d").add_table(
        "orders",
        pl.LazyFrame(
            {
                "id": [1],
                "status": ["delivered"],
                "updated_at": [dt.datetime(2024, 9, 1)],
            },
        ),
        primary_key="id",
        row_update_times=declared,
    )
    declared["updated_at"]["status"] = "mutated"
    assert db.schema("orders").row_update_times["updated_at"]["status"] == "pending"


def test_add_table_rejects_a_date_row_update_time_by_default():
    with pytest.raises(ValidationError, match="row_update_time 'updated_on'"):
        tusk.Database("d").add_table(
            "orders",
            pl.LazyFrame({"id": [1], "updated_on": [dt.date(2024, 9, 1)]}),
            primary_key="id",
            row_update_times={"updated_on": {"updated_on": None}},
        )


def test_add_table_rejects_an_updated_primary_key_by_default():
    with pytest.raises(ValidationError, match="primary_key 'id'"):
        tusk.Database("d").add_table(
            "orders",
            pl.LazyFrame({"id": [1], "updated_at": [dt.datetime(2024, 9, 1)]}),
            primary_key="id",
            row_update_times={"updated_at": {"id": None, "updated_at": None}},
        )


def test_add_table_rejects_a_misfitting_pre_update_value_by_default():
    with pytest.raises(ValidationError, match="'status'"):
        tusk.Database("d").add_table(
            "orders",
            pl.LazyFrame(
                {
                    "id": [1],
                    "status": [1],
                    "updated_at": [dt.datetime(2024, 9, 1)],
                },
            ),
            primary_key="id",
            row_update_times={"updated_at": {"status": "pending", "updated_at": None}},
        )


def test_add_table_rejects_a_chained_row_update_time_by_default():
    # 'updated_at' masks 'shipped_at' and 'shipped_at' in turn masks 'status'.
    # A single with_columns reads every mask's condition off the original
    # frame, so 'status' would read shipped_at's raw, unmasked, post-cutoff
    # value and leak it -- add_table must refuse the declaration outright
    # rather than produce a frame that could leak.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ImplicitEarlierValueWarning)
        with pytest.raises(ValidationError, match="'shipped_at'"):
            tusk.Database("d").add_table(
                "orders",
                pl.LazyFrame(
                    {
                        "id": [1],
                        "placed_at": [dt.datetime(2024, 1, 1)],
                        "status": ["delivered"],
                        "shipped_at": [dt.datetime(2024, 4, 1)],
                        "updated_at": [dt.datetime(2024, 9, 1)],
                    },
                ),
                primary_key="id",
                row_creation_time="placed_at",
                row_update_times={
                    "updated_at": {"shipped_at": None, "updated_at": None},
                    "shipped_at": {"status": "pending"},
                },
            )


def test_add_table_still_does_not_scan_by_default():
    # unique_primary_key and ordered_row_times both read rows; neither may run.
    tusk.Database("d").add_table(
        "orders",
        pl.LazyFrame(
            {
                "id": [1, 1],
                "created_at": [dt.datetime(2024, 3, 1)] * 2,
                "updated_at": [dt.datetime(2024, 1, 1)] * 2,
            },
        ),
        primary_key="id",
        row_creation_time="created_at",
        row_update_times={"updated_at": {"updated_at": None}},
    )
