import narwhals as nw
import polars as pl
import pytest

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
