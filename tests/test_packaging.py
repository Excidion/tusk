"""Packaging and exception hierarchy smoke tests."""

import pytest

import tusk
from tusk.exceptions import (
    MissingPrimaryKeyWarning,
    PrimitiveError,
    SchemaError,
    TuskError,
)


def test_package_imports():
    assert tusk.__version__


@pytest.mark.parametrize("exc", [SchemaError, PrimitiveError])
def test_errors_share_a_base(exc):
    assert issubclass(exc, TuskError)


def test_missing_primary_key_warning_is_filterable():
    assert issubclass(MissingPrimaryKeyWarning, UserWarning)


def test_container_is_named_database():
    db = tusk.Database("retail")
    assert db.name == "retail"
    assert not hasattr(tusk, "EntitySet")
    assert not hasattr(db, "add_dataframe")
