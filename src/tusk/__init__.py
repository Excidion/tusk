"""Deep feature synthesis for narwhals lazy dataframes."""

from __future__ import annotations

from tusk import exceptions
from tusk.api import calculate_feature_matrix, dfs
from tusk.database import Database, Relationship, TableSchema

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Database",
    "Relationship",
    "TableSchema",
    "calculate_feature_matrix",
    "dfs",
    "exceptions",
]
