"""Exception and warning types raised by tusk."""

from __future__ import annotations


class TuskError(Exception):
    """Base class for all errors raised by tusk."""


class SchemaError(TuskError):
    """Raised when an entity set's schema is invalid or inconsistent."""


class PrimitiveError(TuskError):
    """Raised when a primitive is unknown or cannot be applied."""


class MissingPrimaryKeyWarning(UserWarning):
    """Warns that a table without a primary key has reduced capabilities."""
