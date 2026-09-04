"""Packaging and exception hierarchy smoke tests."""

import importlib

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


def test_eagerness_is_not_part_of_the_public_surface():
    db = tusk.Database("retail")
    assert not hasattr(db, "is_eager")


def test_entry_points_are_spelled_out():
    assert callable(tusk.deep_feature_synthesis)
    assert callable(tusk.apply_features)
    assert not hasattr(tusk, "dfs")
    assert not hasattr(tusk, "calculate_feature_matrix")


def test_entityset_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("tusk.entityset")


def test_schema_diagram_is_exported():
    assert callable(tusk.Database("d").plot)
    assert tusk.SchemaDiagram
