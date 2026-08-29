"""Deep feature synthesis for narwhals dataframes."""

from __future__ import annotations

from tusk import exceptions
from tusk.api import apply_features, deep_feature_synthesis
from tusk.database import Database, Relationship, TableSchema
from tusk.feature_list import FeatureList

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Database",
    "FeatureList",
    "Relationship",
    "TableSchema",
    "apply_features",
    "deep_feature_synthesis",
    "exceptions",
]
