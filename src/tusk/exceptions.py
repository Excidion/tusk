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


class LineageError(TuskError):
    """Raised when a kept encoded column has no counterpart after the refit.

    The only way sentinel lineage can *under*-keep is by missing a source, so
    this is the tripwire for that case: if a feature was wrongly pruned, a name
    the frozen mask needs goes missing from the refitted encoder's output.
    Failing here is the alternative to silently serving different columns than
    the ones the selector chose.
    """


class EncoderError(TuskError):
    """Raised when the supplied encoder cannot be refit on a column subset.

    Pruning narrows the matrix, so the encoder is refit on fewer columns than
    it first saw. A ``ColumnTransformer`` naming its columns explicitly cannot
    survive that -- and after DFS it is already a mistake for two other
    reasons, so it is refused rather than worked around.
    """


class UnencodedFeatureWarning(UserWarning):
    """Warns that a feature fed no encoded column, so it was pruned.

    The encoder simply never looked at it -- most often a ``ColumnTransformer``
    covering only some dtypes while ``remainder`` stays at its ``"drop"``
    default. Pruning it is self-consistent, but doing so silently would let a
    user lose every numeric feature without a word. Its own class, so it can be
    filtered independently.
    """


class LineageWarning(UserWarning):
    """Warns that lineage was unrecoverable, so nothing was pruned.

    An encoder whose output names do not mention their inputs -- ``PCA`` gives
    ``pca0`` -- severs the link between kept columns and tusk features. Keeping
    every feature is the correct fallback, since pruning is an optimization and
    no result depends on it. Its own class, so it can be filtered independently.
    """
