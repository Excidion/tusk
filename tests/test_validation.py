import narwhals as nw
import polars as pl
import pyarrow as pa
import pytest

import tusk
from tusk.database import TableSchema
from tusk.exceptions import TuskError, ValidationError
from tusk.validation import (
    CHECKS,
    check_unique_primary_key,
    resolve_checks,
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
    assert "7" in message and "12" in message and "19" in message


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


def test_selector_forms_all_select_the_check():
    for selector in (True, "unique_primary_key", ["unique_primary_key"]):
        assert resolve_checks(selector) == (CHECKS["unique_primary_key"],)


def test_empty_selectors_select_nothing():
    assert resolve_checks(False) == ()
    assert resolve_checks([]) == ()


def test_a_repeated_name_runs_the_check_once():
    assert resolve_checks(["unique_primary_key", "unique_primary_key"]) == (
        CHECKS["unique_primary_key"],
    )


def test_checks_run_in_registry_order_not_argument_order(monkeypatch):
    calls = []
    ordered = {
        "first": lambda f, s: calls.append("first"),
        "second": lambda f, s: calls.append("second"),
    }
    monkeypatch.setattr("tusk.validation.CHECKS", ordered)
    validate_table(frame([1]), schema(), ["second", "first"])
    assert calls == ["first", "second"]


def test_an_unknown_check_name_raises_value_error():
    with pytest.raises(ValueError) as excinfo:
        resolve_checks("uniqe_primary_key")
    message = str(excinfo.value)
    assert "uniqe_primary_key" in message
    assert "unique_primary_key" in message


def test_an_unknown_check_name_is_not_a_validation_error():
    # `except ValidationError` must never swallow a typo.
    with pytest.raises(ValueError) as excinfo:
        resolve_checks(["unique_primary_key", "nope"])
    assert not isinstance(excinfo.value, TuskError)


@pytest.mark.parametrize("selector", [None, 1])
def test_an_invalid_selector_raises_value_error_not_type_error(selector):
    # A non-bool, non-str, non-iterable selector must get the same clear
    # ValueError treatment as an unknown check name, not a raw TypeError
    # leaking out of `set()`.
    with pytest.raises(ValueError, match="unique_primary_key") as excinfo:
        resolve_checks(selector)
    assert not isinstance(excinfo.value, TuskError)


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


def test_database_validate_with_a_generator_checks_every_table():
    # A generator is a one-shot iterable: if the selector were re-resolved
    # per table, table 'b' -- the one with the actual duplicate -- would see
    # an already-exhausted generator and silently pass.
    db = (
        tusk.Database("x")
        .add_table("a", pl.LazyFrame({"id": [1, 2]}), primary_key="id")
        .add_table("b", dupes(), primary_key="id")
    )
    with pytest.raises(ValidationError, match="'b'"):
        db.validate(n for n in ["unique_primary_key"])


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
