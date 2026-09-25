"""Exception and warning types raised by tusk."""

from __future__ import annotations


class TuskError(Exception):
    """Base class for all errors raised by tusk."""


class SchemaError(TuskError):
    """Exception raised when a database's schema is invalid or inconsistent."""


class PrimitiveError(TuskError):
    """Exception raised when a primitive is unknown or cannot be applied."""


class ValidationError(TuskError):
    """Exception raised when a validation check finds a defect in a table's data.

    Distinct from :class:`SchemaError`, which reports a malformed database
    before any row is read. A ``ValidationError`` means the declarations are
    well-formed. The data contradicts them.

    An unknown check *name* is a caller mistake, not a data defect. It
    raises :class:`ValueError` instead of ``ValidationError``. This means
    ``except ValidationError`` never silently swallows a typo.
    """


class MissingPrimaryKeyWarning(UserWarning):
    """Warning raised when a table without a primary key has reduced capabilities."""


class ImplicitEarlierValueWarning(UserWarning):
    """Warning raised when a row update time was given a null pre-update value.

    A row update time not listed under any update time is read straight
    from the table. ``MAX(updated_at)`` can then return a timestamp from
    after the cutoff time. tusk fills the gap instead of allowing that
    leak.
    Its null value is only a guess at what the column held before. This
    warning has its own class. A caller can filter it independently of
    the others.
    """


class CategoricalDtypeWarning(UserWarning):
    """Warning raised when a string primitive skips a Categorical or Enum column."""


class UnmatchedPrimitiveWarning(UserWarning):
    """Warning raised when a requested primitive finds no column of its input dtypes.

    tusk skips such a primitive instead of raising an error. This keeps
    zero-configuration ``deep_feature_synthesis()`` calls working on any
    schema, even one that lacks a dtype family. The skip would otherwise
    happen silently, with no explanation. This warning reports it. This
    warning has its own class. A caller can filter it independently of
    the others.
    """


class UnmatchedConditionWarning(UserWarning):
    """Warning raised when no table declares a condition for ``conditional_primitives``.

    This differs from :class:`UnmatchedPrimitiveWarning`, which matches a
    primitive to a column by dtype alone. A conditional primitive such as
    ``count`` or ``sum`` is usually also in ``agg_primitives``. There it
    is already marked matched. This holds even when it produced zero
    conditional features. That marking suppresses
    :class:`UnmatchedPrimitiveWarning`. This warning checks the condition
    dimension instead. It fires when no table declares a ``where`` or
    ``when`` condition. Then ``conditional_primitives`` has nothing to
    mask. This warning has its own class. A caller can filter it
    independently of the others.
    """


class LineageError(TuskError):
    """Exception raised when a kept encoded column has no counterpart after the refit.

    Sentinel lineage can only under-keep by omitting a source. This
    exception is the check for that case. A wrongly pruned feature can
    leave a name the frozen mask needs absent from the refitted
    encoder's output. This exception then fires. Raising it replaces
    silently serving different columns than the ones the selector chose.
    """


class EncoderError(TuskError):
    """Exception raised when the supplied encoder cannot refit on a column subset.

    Pruning narrows the feature matrix. The encoder then refits on fewer
    columns than it first saw. A ``ColumnTransformer`` that names its
    columns explicitly cannot handle that change. tusk refuses the refit
    instead of working around it.
    """


class UnencodedFeatureWarning(UserWarning):
    """Warning raised when a feature feeds no encoded column, so tusk prunes it.

    The encoder never looked at that feature. The most common cause is a
    ``ColumnTransformer`` that covers only some dtypes while
    ``remainder`` stays at its default of ``"drop"``. Pruning the
    feature is self-consistent. Doing so silently could let a user lose
    every numeric feature without notice. This warning has its own
    class. A caller can filter it independently of the others.
    """


class LineageWarning(UserWarning):
    """Warning raised when lineage is unrecoverable, so tusk prunes nothing.

    An encoder whose output names do not mention their inputs severs the
    link between kept columns and tusk features. ``PCA`` is an example:
    it gives names such as ``pca0``. Keeping every feature is the
    correct fallback. Pruning is only an optimization. No result depends
    on it. This warning has its own class. A caller can filter it
    independently of the others.
    """
