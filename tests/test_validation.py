import datetime as dt
from zoneinfo import ZoneInfo

import narwhals as nw
import polars as pl
import pyarrow as pa
import pytest

import tusk
from tusk.database import TableSchema
from tusk.exceptions import TuskError, ValidationError
from tusk.validation import (
    check_datetime_row_creation_time,
    check_non_null_primary_key,
    check_unique_primary_key,
    validate_table,
)


def frame(values):
    """A one-column lazy frame keyed by 'id'."""
    return nw.from_native(pl.LazyFrame({"id": values}))


def schema(primary_key="id"):
    return TableSchema("customers", primary_key, None, {"id": nw.Int64})


def test_duplicate_primary_key_is_reported():
    with pytest.raises(ValidationError) as excinfo:
        check_unique_primary_key(frame([7, 7, 12, 12, 19, 19, 1]), schema())
    message = str(excinfo.value)
    assert "primary_key 'id'" in message
    assert "'customers'" in message
    assert "7 rows" in message
    assert "4 distinct" in message


def test_unique_primary_key_passes():
    check_unique_primary_key(frame([1, 2, 3]), schema())


def test_empty_table_passes():
    check_unique_primary_key(frame([]), schema())


def test_table_without_a_primary_key_is_skipped():
    # add_table already warned about this; failing here would make
    # db.validate() unusable on any database holding a keyless table.
    check_unique_primary_key(frame([7, 7]), schema(primary_key=None))


def test_repeated_nulls_are_not_unique():
    with pytest.raises(ValidationError, match="not unique"):
        check_unique_primary_key(frame([1, None, None]), schema())


def test_a_single_null_passes_the_uniqueness_check():
    # A lone null key is a nullability defect, not a uniqueness one.
    check_unique_primary_key(frame([1, 2, None]), schema())


def test_validation_error_is_a_tusk_error():
    assert issubclass(ValidationError, TuskError)


def test_validate_table_runs_the_registry_by_default():
    with pytest.raises(ValidationError):
        validate_table(frame([7, 7]), schema())


def test_validate_table_runs_nothing_when_false():
    validate_table(frame([7, 7]), schema(), False)


@pytest.mark.parametrize(
    "selector",
    [
        True,
        "unique_primary_key",
        ["unique_primary_key"],
        ("unique_primary_key",),
        {"unique_primary_key"},
    ],
)
def test_every_selector_form_runs_the_check(selector):
    with pytest.raises(ValidationError, match="not unique"):
        validate_table(frame([7, 7]), schema(), selector)


def test_an_empty_list_runs_nothing():
    validate_table(frame([7, 7]), schema(), [])


def test_checks_run_in_the_order_given(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "tusk.validation.CHECKS",
        {
            "first": lambda f, s: calls.append("first"),
            "second": lambda f, s: calls.append("second"),
        },
    )
    validate_table(frame([1]), schema(), ["second", "first"])
    assert calls == ["second", "first"]


def test_an_unknown_check_name_raises_value_error():
    with pytest.raises(ValueError) as excinfo:
        validate_table(frame([1]), schema(), "uniqe_primary_key")
    message = str(excinfo.value)
    assert "uniqe_primary_key" in message
    assert "unique_primary_key" in message


def test_an_unknown_check_name_is_not_a_validation_error():
    # `except ValidationError` must never swallow a typo.
    with pytest.raises(ValueError) as excinfo:
        validate_table(frame([1]), schema(), ["unique_primary_key", "nope"])
    assert not isinstance(excinfo.value, TuskError)


@pytest.mark.parametrize("selector", [None, 1])
def test_an_unsupported_selector_form_raises_value_error(selector):
    # A non-iterable selector gets a clear ValueError naming the accepted
    # forms, never a raw TypeError leaking out of `list()`.
    with pytest.raises(ValueError, match="invalid checks selector") as excinfo:
        validate_table(frame([1]), schema(), selector)
    assert not isinstance(excinfo.value, TuskError)


def test_a_generator_selector_still_checks_every_table():
    # A generator is one-shot. If Database.validate passed it straight down
    # per table, table 'b' -- the one with the duplicate -- would see an
    # already-exhausted iterator and silently pass.
    db = (
        tusk.Database("x")
        .add_table("a", pl.LazyFrame({"id": [1, 2]}), primary_key="id")
        .add_table("b", pl.LazyFrame({"id": [7, 7]}), primary_key="id")
    )
    with pytest.raises(ValidationError, match="'b'"):
        db.validate(n for n in ["unique_primary_key"])


@pytest.fixture
def spy(monkeypatch):
    """Replace the registry with a recorder, so plumbing is observable."""
    calls = []
    monkeypatch.setattr(
        "tusk.validation.CHECKS",
        {"unique_primary_key": lambda f, s: calls.append(s.name)},
    )
    return calls


def dupes():
    return pl.LazyFrame({"id": [7, 7, 12], "v": [1.0, 2.0, 3.0]})


def test_add_table_does_not_validate_by_default(spy):
    db = tusk.Database("x").add_table("t", dupes(), primary_key="id")
    assert spy == []
    assert db.table_names == ("t",)


def test_add_table_validates_when_asked(spy):
    tusk.Database("x").add_table("t", dupes(), primary_key="id", validate=True)
    assert spy == ["t"]


def test_add_table_accepts_every_selector_form():
    for selector in (True, "unique_primary_key", ["unique_primary_key"]):
        with pytest.raises(ValidationError, match="not unique"):
            tusk.Database("x").add_table(
                "t", dupes(), primary_key="id", validate=selector
            )


def test_a_failed_add_table_leaves_the_database_unchanged():
    db = tusk.Database("x")
    with pytest.raises(ValidationError):
        db.add_table("t", dupes(), primary_key="id", validate=True)
    assert db.table_names == ()
    with pytest.raises(tusk.exceptions.SchemaError):
        db.frame("t")


def test_schema_errors_still_precede_validation():
    # A primary_key that is not a column must stay a SchemaError; querying a
    # column that does not exist would report it as something else.
    with pytest.raises(tusk.exceptions.SchemaError, match="nope"):
        tusk.Database("x").add_table("t", dupes(), primary_key="nope", validate=True)


def test_database_validate_runs_every_table_in_insertion_order(spy):
    db = (
        tusk.Database("x")
        .add_table("a", pl.LazyFrame({"id": [1]}), primary_key="id")
        .add_table("b", pl.LazyFrame({"id": [2]}), primary_key="id")
    )
    assert db.validate() is db
    assert spy == ["a", "b"]


def test_database_validate_reports_the_offending_table():
    db = (
        tusk.Database("x")
        .add_table("a", pl.LazyFrame({"id": [1, 2]}), primary_key="id")
        .add_table("b", dupes(), primary_key="id")
    )
    with pytest.raises(ValidationError, match="'b'"):
        db.validate()


def test_database_validate_false_is_a_no_op(spy):
    db = tusk.Database("x").add_table("t", dupes(), primary_key="id")
    assert db.validate(False) is db
    assert spy == []


def test_database_validate_skips_a_keyless_table():
    with pytest.warns(tusk.exceptions.MissingPrimaryKeyWarning):
        db = tusk.Database("x").add_table("t", dupes())
    assert db.validate() is db


def test_unknown_check_names_reach_both_entry_points():
    with pytest.raises(ValueError, match="unknown check"):
        tusk.Database("x").add_table(
            "t", pl.LazyFrame({"id": [1]}), primary_key="id", validate="nope"
        )
    db = tusk.Database("x").add_table("t", pl.LazyFrame({"id": [1]}), primary_key="id")
    with pytest.raises(ValueError, match="unknown check"):
        db.validate("nope")


def test_database_validate_checks_every_table_not_just_the_first():
    db = (
        tusk.Database("x")
        .add_table("a", pl.LazyFrame({"id": [1, 2]}), primary_key="id")
        .add_table("b", dupes(), primary_key="id")
    )
    with pytest.raises(ValidationError, match="'b'"):
        db.validate(["unique_primary_key"])


def test_a_failed_add_table_of_the_first_table_still_accepts_other_backends():
    db = tusk.Database("x")
    with pytest.raises(ValidationError):
        db.add_table("t", pa.table({"id": [1, 1, 2]}), primary_key="id", validate=True)

    # The failed add must not have pinned the database to pyarrow's backend.
    db.add_table("ok", pl.LazyFrame({"id": [1, 2]}), primary_key="id")
    assert db.table_names == ("ok",)


def test_a_failed_add_table_of_an_eager_frame_leaves_is_eager_false():
    db = tusk.Database("x")
    with pytest.raises(ValidationError):
        db.add_table("t", pl.DataFrame({"id": [1, 1]}), primary_key="id", validate=True)
    assert db.is_eager is False


def test_a_null_primary_key_is_reported():
    with pytest.raises(ValidationError) as excinfo:
        check_non_null_primary_key(frame([1, None, 3]), schema())
    message = str(excinfo.value)
    assert "primary_key 'id'" in message
    assert "'customers'" in message
    assert "1 null" in message


def test_a_fully_populated_primary_key_passes_the_null_check():
    check_non_null_primary_key(frame([1, 2, 3]), schema())


def test_an_empty_table_passes_the_null_check():
    check_non_null_primary_key(frame([]), schema())


def test_a_table_without_a_primary_key_skips_the_null_check():
    check_non_null_primary_key(frame([None, None]), schema(primary_key=None))


def test_the_null_check_runs_before_the_uniqueness_check():
    # Two nulls fail both checks. validate=True must report the null defect,
    # the more specific of the two.
    with pytest.raises(ValidationError, match="null"):
        tusk.Database("x").add_table(
            "t", pl.LazyFrame({"id": [None, None]}), primary_key="id", validate=True
        )


def test_an_explicit_list_runs_in_the_order_given_not_registry_order():
    # The caller asked for uniqueness first, so they get the uniqueness error
    # even though the null check would also have failed.
    with pytest.raises(ValidationError, match="not unique"):
        validate_table(
            frame([None, None]),
            schema(),
            ["unique_primary_key", "non_null_primary_key"],
        )


def temporal_schema(dtype, column="created_at"):
    return TableSchema("events", "id", column, {"id": nw.Int64, column: dtype})


def test_a_date_row_creation_time_is_reported():
    with pytest.raises(ValidationError) as excinfo:
        check_datetime_row_creation_time(frame([1]), temporal_schema(nw.Date))
    message = str(excinfo.value)
    assert "row_creation_time 'created_at'" in message
    assert "expected Datetime" in message


def test_a_datetime_row_creation_time_passes():
    check_datetime_row_creation_time(frame([1]), temporal_schema(nw.Datetime()))


def test_a_tz_aware_row_creation_time_passes():
    check_datetime_row_creation_time(
        frame([1]), temporal_schema(nw.Datetime(time_zone="UTC"))
    )


def test_a_non_temporal_row_creation_time_is_reported():
    with pytest.raises(ValidationError, match="expected Datetime"):
        check_datetime_row_creation_time(frame([1]), temporal_schema(nw.Int64))


def test_a_table_without_a_row_creation_time_is_skipped():
    check_datetime_row_creation_time(
        frame([1]), TableSchema("events", "id", None, {"id": nw.Int64})
    )


def temporal_db(*time_zones):
    """A database with one datetime column per given zone; None means naive."""
    db = tusk.Database("x")
    for index, zone in enumerate(time_zones):
        stamp = dt.datetime(2024, 1, 1, tzinfo=ZoneInfo(zone) if zone else None)
        db.add_table(
            f"t{index}",
            pl.LazyFrame({"id": [index], "at": [stamp]}),
            primary_key="id",
        )
    return db


def test_mixing_tz_aware_and_naive_datetimes_is_reported():
    db = temporal_db("UTC", None, None)
    with pytest.raises(ValidationError) as excinfo:
        db.validate("consistent_time_zones")
    # One message, not a tuple of them: passing several positional args to an
    # exception makes str() render a tuple repr, parens and quotes included.
    assert len(excinfo.value.args) == 1
    assert str(excinfo.value) == (
        "database mixes tz-aware and tz-naive datetimes: "
        "1 tz-aware (t0.at), 2 tz-naive (t1.at, t2.at)"
    )


def test_differing_time_zones_pass_as_long_as_all_are_aware():
    temporal_db("UTC", "Europe/Berlin").validate("consistent_time_zones")


def test_all_naive_datetimes_pass():
    temporal_db(None, None).validate("consistent_time_zones")


def test_a_database_check_runs_once_not_per_table(monkeypatch):
    calls = []
    db = temporal_db(None, None)
    monkeypatch.setattr(
        "tusk.validation.DATABASE_CHECKS", {"counted": lambda d: calls.append(d)}
    )
    db.validate("counted")
    assert calls == [db]


def test_a_database_check_name_is_rejected_by_add_table():
    # add_table sees one table, so it cannot answer a database-wide question.
    with pytest.raises(ValueError, match="unknown check 'consistent_time_zones'"):
        tusk.Database("x").add_table(
            "t",
            pl.LazyFrame({"id": [1]}),
            primary_key="id",
            validate="consistent_time_zones",
        )


def test_database_validate_accepts_names_from_both_registries():
    db = temporal_db(None, None)
    assert db.validate(["unique_primary_key", "consistent_time_zones"]) is db


def linked(parent_ids, child_fks, parent_dtype=pl.Int64, child_dtype=pl.Int64):
    """A two-table database with one customers -> sessions relationship."""
    parents = pl.LazyFrame({"id": parent_ids}, schema={"id": parent_dtype})
    children = pl.LazyFrame(
        {"id": list(range(len(child_fks))), "customer_id": child_fks},
        schema={"id": pl.Int64, "customer_id": child_dtype},
    )
    return (
        tusk.Database("retail")
        .add_table("customers", parents, primary_key="id")
        .add_table("sessions", children, primary_key="id")
        .add_relationship(
            parent="customers", child="sessions", foreign_key="customer_id"
        )
    )


def test_relationships_are_exposed_in_insertion_order():
    db = linked([1], [1])
    assert db.relationships == (
        tusk.Relationship(
            parent="customers", child="sessions", foreign_key="customer_id"
        ),
    )


def test_an_orphan_foreign_key_is_reported():
    db = linked([1, 2], [1, 99])
    with pytest.raises(ValidationError) as excinfo:
        db.validate("referential_integrity")
    message = str(excinfo.value)
    assert "1 rows of 'sessions'" in message
    assert "'customer_id'" in message
    assert "'customers'" in message


def test_every_foreign_key_matching_passes():
    linked([1, 2, 3], [1, 2, 2]).validate("referential_integrity")


def test_a_null_foreign_key_is_not_an_orphan():
    # A null foreign key means the row has no parent, which is allowed.
    linked([1, 2], [1, None]).validate("referential_integrity")


def test_a_childless_parent_is_not_a_defect():
    linked([1, 2, 3], [1]).validate("referential_integrity")


def test_mismatched_key_dtypes_are_reported():
    db = linked([1, 2], ["1", "2"], parent_dtype=pl.Int64, child_dtype=pl.String)
    with pytest.raises(ValidationError) as excinfo:
        db.validate("matching_key_dtypes")
    message = str(excinfo.value)
    assert "foreign_key 'customer_id' of 'sessions' is String" in message
    assert "primary_key 'id' of 'customers' is Int64" in message


@pytest.mark.parametrize(
    ("parent_dtype", "child_dtype"),
    [
        # pyarrow refuses to join mismatched integer widths, and polars
        # refuses int to float, so neither may pass validation.
        (pl.Int64, pl.Int32),
        (pl.Int64, pl.Float64),
        # polars refuses every crossing of the string family, including two
        # Enums whose categories differ.
        (pl.String, pl.Categorical),
        (pl.String, pl.Enum(["a", "b"])),
        (pl.Categorical, pl.Enum(["a", "b"])),
        (pl.Enum(["a", "b"]), pl.Enum(["a", "b", "c"])),
    ],
)
def test_key_dtypes_must_match_exactly(parent_dtype, child_dtype):
    values = [1, 2] if parent_dtype == pl.Int64 else ["a", "b"]
    db = linked(values, values, parent_dtype=parent_dtype, child_dtype=child_dtype)
    with pytest.raises(ValidationError, match="foreign_key 'customer_id'"):
        db.validate("matching_key_dtypes")


@pytest.mark.parametrize(
    "dtype", [pl.Int64, pl.String, pl.Categorical, pl.Enum(["a", "b"])]
)
def test_identical_key_dtypes_pass(dtype):
    values = [1, 2] if dtype == pl.Int64 else ["a", "b"]
    linked(values, values, parent_dtype=dtype, child_dtype=dtype).validate(
        "matching_key_dtypes"
    )


def test_relationship_checks_run_once_per_relationship(monkeypatch):
    seen = []
    monkeypatch.setattr(
        "tusk.validation.RELATIONSHIP_CHECKS",
        {"counted": lambda d, r: seen.append(r.child)},
    )
    db = linked([1], [1]).add_table(
        "transactions", pl.LazyFrame({"id": [1], "session_id": [0]}), primary_key="id"
    )
    db.add_relationship(
        parent="sessions", child="transactions", foreign_key="session_id"
    )
    db.validate("counted")
    assert seen == ["sessions", "transactions"]


def test_a_database_with_no_relationships_runs_no_relationship_checks():
    db = tusk.Database("x").add_table("t", pl.LazyFrame({"id": [1]}), primary_key="id")
    assert db.validate("referential_integrity") is db


def test_a_relationship_check_name_is_rejected_by_add_table():
    with pytest.raises(ValueError, match="unknown check 'referential_integrity'"):
        tusk.Database("x").add_table(
            "t",
            pl.LazyFrame({"id": [1]}),
            primary_key="id",
            validate="referential_integrity",
        )


def test_validate_true_runs_all_three_registries():
    db = linked([1, 2], [1, 99])
    with pytest.raises(ValidationError, match="no matching"):
        db.validate()


def test_names_from_all_three_registries_select_together():
    db = linked([1, 2], [1, 2])
    assert (
        db.validate(
            ["unique_primary_key", "referential_integrity", "consistent_time_zones"]
        )
        is db
    )


def test_the_dtype_check_precedes_the_join_that_would_fail_on_it():
    # polars raises NarwhalsError when join key dtypes cross the string
    # family. matching_key_dtypes must report that first, or validate=True
    # surfaces the backend's error instead of ours.
    db = linked(["a"], ["a"], parent_dtype=pl.String, child_dtype=pl.Categorical)
    with pytest.raises(ValidationError, match="foreign_key 'customer_id'"):
        db.validate()

    # And the join really would have failed, which is why the order matters.
    with pytest.raises(Exception) as excinfo:
        db.validate("referential_integrity")
    assert not isinstance(excinfo.value, ValidationError)
