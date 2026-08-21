"""Exception and warning types raised by tusk."""

from __future__ import annotations


class TuskError(Exception):
    """Base class for all errors raised by tusk."""


class SchemaError(TuskError):
    """Raised when a database's schema is invalid or inconsistent."""


class PrimitiveError(TuskError):
    """Raised when a primitive is unknown or cannot be applied."""


class ValidationError(TuskError):
    """Raised when a validation check finds a defect in a table's data.

    Distinct from :class:`SchemaError`, which reports a malformed database
    before any row is read. A ``ValidationError`` means the declarations are
    well-formed but the data contradicts them.

    An unknown check *name* is a caller mistake rather than a data defect and
    raises :class:`ValueError`, so ``except ValidationError`` never silently
    swallows a typo.
    """


class MissingPrimaryKeyWarning(UserWarning):
    """Warns that a table without a primary key has reduced capabilities."""


class CategoricalDtypeWarning(UserWarning):
    """Warns that a Categorical or Enum column was skipped by a string primitive."""


class UnmatchedPrimitiveWarning(UserWarning):
    """Warns that a requested primitive found no column of its input dtypes.

    Skipping such a primitive is correct -- raising would break a
    zero-configuration ``deep_feature_synthesis()`` on any schema that happens
    to lack a dtype family. Skipping it *silently* is not: the user asked for
    a primitive and got no column and no explanation. Its own class, so it
    can be filtered independently.
    """
