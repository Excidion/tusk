"""Deep feature synthesis for narwhals dataframes."""

from __future__ import annotations

from tusk import exceptions
from tusk.api import apply_features, deep_feature_synthesis
from tusk.database import Database, Relationship, TableSchema

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Database",
    "Relationship",
    "TableSchema",
    "apply_features",
    "deep_feature_synthesis",
    "exceptions",
]
