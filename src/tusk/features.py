"""Feature definitions: the immutable output of phase 1.

Features are frozen dataclasses with structural equality, so a feature reached
by two different routes deduplicates in a set with no extra bookkeeping. Every
dtype here is derived from primitive metadata, never from data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from tusk.database import Relationship
from tusk.exceptions import PrimitiveError
from tusk.primitives.base import AggregationPrimitive, Primitive, TransformPrimitive


@dataclass(frozen=True)
class Feature(ABC):
    """Base class for every feature definition."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The column name in the feature matrix.

        A plain SQL identifier: parts are joined with ``__`` rather than the
        dots, parentheses and spaces a conventional DFS name uses, since a
        backend that generates SQL parses those as table qualifiers and
        function calls rather than as part of an identifier. See
        :attr:`display_name` for the readable form.
        """

    @property
    @abstractmethod
    def display_name(self) -> str:
        """The readable name, e.g. ``MEAN(transactions.amount)``.

        Carries the same meaning as :attr:`name` in the conventional DFS
        notation. Used for documentation, logging and error messages; never
        as a column name.
        """

    @property
    @abstractmethod
    def dtype(self) -> Any:
        """Output dtype, computed statically."""

    @property
    @abstractmethod
    def depth(self) -> int:
        """Number of stacked primitive applications."""

    @property
    @abstractmethod
    def table(self) -> str:
        """Table this feature is a column of."""

    @property
    @abstractmethod
    def base_features(self) -> tuple[Feature, ...]:
        """Features this one is computed from."""

    @property
    def output_names(self) -> tuple[str, ...]:
        """One column name per output; more than one for multi-output primitives."""
        return (self.name,)

    @property
    def display_output_names(self) -> tuple[str, ...]:
        """One readable name per output, parallel to :attr:`output_names`."""
        return (self.display_name,)

    @property
    def is_multi_output(self) -> bool:
        """Whether this feature materializes more than one column.

        Only the indexed names in :attr:`output_names` are ever materialized,
        so a multi-output feature has no single column another primitive could
        read. It is a valid output of synthesis but never a valid input.

        Deriving this from :attr:`output_names` rather than from a primitive's
        ``number_of_outputs`` keeps it well defined for
        :class:`IdentityFeature` and :class:`DirectFeature`, which have no
        primitive at all.
        """
        return len(self.output_names) > 1


@dataclass(frozen=True)
class IdentityFeature(Feature):
    """A raw column of a table.

    Attributes:
        table_name: Table the column belongs to.
        column: Column name.
        column_dtype: The column's narwhals dtype.
    """

    table_name: str
    column: str
    column_dtype: Any

    @property
    def name(self) -> str:
        """The column's own name."""
        return self.column

    @property
    def display_name(self) -> str:
        """A raw column reads the same either way."""
        return self.column

    @property
    def dtype(self) -> Any:
        """The column's dtype."""
        return self.column_dtype

    @property
    def depth(self) -> int:
        """Identity features are depth zero."""
        return 0

    @property
    def table(self) -> str:
        """Table the column belongs to."""
        return self.table_name

    @property
    def base_features(self) -> tuple[Feature, ...]:
        """Identity features have no bases."""
        return ()


@dataclass(frozen=True)
class TransformFeature(Feature):
    """A primitive applied row-wise to features of one table.

    Attributes:
        primitive: The transform primitive.
        bases: Input features, all on the same table.
    """

    primitive: Primitive
    bases: tuple[Feature, ...]

    def __post_init__(self) -> None:
        """Confirm the primitive transforms rather than aggregates.

        Raises :class:`~tusk.exceptions.PrimitiveError`, via
        :func:`_require_primitive_kind`, if it does not.
        """
        _require_primitive_kind(self.primitive, TransformPrimitive, "transform")

    @property
    def name(self) -> str:
        """Generated name, e.g. ``MONTH__started_at``."""
        return self.primitive.generate_name([b.name for b in self.bases])

    @property
    def display_name(self) -> str:
        """Readable name, e.g. ``MONTH(started_at)``."""
        return self.primitive.generate_display_name(
            [b.display_name for b in self.bases],
        )

    @property
    def dtype(self) -> Any:
        """Dtype derived from the primitive and its inputs."""
        return self.primitive.return_dtype(tuple(b.dtype for b in self.bases))

    @property
    def depth(self) -> int:
        """One deeper than the deepest input."""
        return 1 + max(b.depth for b in self.bases)

    @property
    def table(self) -> str:
        """The table its inputs live on."""
        return self.bases[0].table

    @property
    def base_features(self) -> tuple[Feature, ...]:
        """Its input features."""
        return self.bases

    @property
    def output_names(self) -> tuple[str, ...]:
        """One name per output column."""
        return self.primitive.output_names(self.name)

    @property
    def display_output_names(self) -> tuple[str, ...]:
        """One readable name per output column."""
        return self.primitive.display_output_names(self.display_name)


@dataclass(frozen=True)
class AggregationFeature(Feature):
    """A primitive applied to a child table's rows, grouped by foreign key.

    Attributes:
        primitive: The aggregation primitive.
        bases: Input features on the child table. Empty for zero-arity
            primitives such as ``count``.
        relationship: The parent-child link being aggregated across.
    """

    primitive: Primitive
    bases: tuple[Feature, ...]
    relationship: Relationship

    def __post_init__(self) -> None:
        """Confirm the primitive aggregates rather than transforms.

        Raises :class:`~tusk.exceptions.PrimitiveError`, via
        :func:`_require_primitive_kind`, if it does not.
        """
        _require_primitive_kind(self.primitive, AggregationPrimitive, "aggregation")

    @property
    def name(self) -> str:
        """Generated name, e.g. ``MEAN__transactions__amount``.

        Zero-arity primitives name the child table instead of a column, giving
        ``COUNT__transactions``.
        """
        child = self.relationship.child
        if not self.bases:
            return self.primitive.generate_name([child])
        return self.primitive.generate_name([f"{child}__{b.name}" for b in self.bases])

    @property
    def display_name(self) -> str:
        """Readable name, e.g. ``MEAN(transactions.amount)``."""
        child = self.relationship.child
        if not self.bases:
            return self.primitive.generate_display_name([child])
        return self.primitive.generate_display_name(
            [f"{child}.{b.display_name}" for b in self.bases],
        )

    @property
    def dtype(self) -> Any:
        """Dtype derived from the primitive and its inputs."""
        return self.primitive.return_dtype(tuple(b.dtype for b in self.bases))

    @property
    def depth(self) -> int:
        """One deeper than the deepest input; 1 when there are none."""
        return 1 + max((b.depth for b in self.bases), default=0)

    @property
    def table(self) -> str:
        """The parent table the aggregate lands on."""
        return self.relationship.parent

    @property
    def base_features(self) -> tuple[Feature, ...]:
        """Its input features on the child table."""
        return self.bases

    @property
    def output_names(self) -> tuple[str, ...]:
        """One name per output column."""
        return self.primitive.output_names(self.name)

    @property
    def display_output_names(self) -> tuple[str, ...]:
        """One readable name per output column."""
        return self.primitive.display_output_names(self.display_name)


@dataclass(frozen=True)
class DirectFeature(Feature):
    """A parent's feature joined down onto the child.

    Attributes:
        base_feature: The feature on the parent table.
        relationship: The parent-child link being traversed.
    """

    base_feature: Feature
    relationship: Relationship

    @property
    def name(self) -> str:
        """Generated name, e.g. ``customers__age``."""
        return f"{self.relationship.parent}__{self.base_feature.name}"

    @property
    def display_name(self) -> str:
        """Readable name, e.g. ``customers.age``."""
        return f"{self.relationship.parent}.{self.base_feature.display_name}"

    @property
    def dtype(self) -> Any:
        """The parent feature's dtype, unchanged."""
        return self.base_feature.dtype

    @property
    def depth(self) -> int:
        """One deeper than the parent feature."""
        return 1 + self.base_feature.depth

    @property
    def table(self) -> str:
        """The child table the value lands on."""
        return self.relationship.child

    @property
    def base_features(self) -> tuple[Feature, ...]:
        """The single parent feature."""
        return (self.base_feature,)


@dataclass(frozen=True)
class GroupByTransformFeature(Feature):
    """A transform applied within groups defined by a foreign key.

    Attributes:
        primitive: The transform primitive.
        bases: Input features on the child table.
        relationship: The link whose foreign key defines the groups.
    """

    primitive: Primitive
    bases: tuple[Feature, ...]
    relationship: Relationship

    def __post_init__(self) -> None:
        """Confirm the primitive transforms rather than aggregates.

        Raises :class:`~tusk.exceptions.PrimitiveError`, via
        :func:`_require_primitive_kind`, if it does not.
        """
        _require_primitive_kind(self.primitive, TransformPrimitive, "transform")

    @property
    def name(self) -> str:
        """Generated name, e.g. ``CUM_SUM__amount__by__session_id``."""
        stem = self.primitive.generate_name([b.name for b in self.bases])
        return f"{stem}__by__{self.relationship.foreign_key}"

    @property
    def display_name(self) -> str:
        """Readable name, e.g. ``CUM_SUM(amount) by session_id``."""
        stem = self.primitive.generate_display_name(
            [b.display_name for b in self.bases],
        )
        return f"{stem} by {self.relationship.foreign_key}"

    @property
    def dtype(self) -> Any:
        """Dtype derived from the primitive and its inputs."""
        return self.primitive.return_dtype(tuple(b.dtype for b in self.bases))

    @property
    def depth(self) -> int:
        """One deeper than the deepest input."""
        return 1 + max(b.depth for b in self.bases)

    @property
    def table(self) -> str:
        """The child table the values land on."""
        return self.relationship.child

    @property
    def base_features(self) -> tuple[Feature, ...]:
        """Its input features."""
        return self.bases

    @property
    def output_names(self) -> tuple[str, ...]:
        """One name per output column."""
        return self.primitive.output_names(self.name)

    @property
    def display_output_names(self) -> tuple[str, ...]:
        """One readable name per output column."""
        return self.primitive.display_output_names(self.display_name)


def _require_primitive_kind(
    primitive: Primitive,
    required: type[Primitive],
    needed: str,
) -> None:
    """Confirm a primitive is the kind its feature class requires.

    Args:
        primitive: The primitive to check.
        required: The primitive base class the feature class requires.
        needed: ``required``'s kind, ``"aggregation"`` or ``"transform"``.

    Raises:
        PrimitiveError: If ``primitive`` is not an instance of ``required``.
    """
    if isinstance(primitive, required):
        return
    is_aggregation = isinstance(primitive, AggregationPrimitive)
    actual = "aggregation" if is_aggregation else "transform"
    raise PrimitiveError(
        f"{primitive.name!r} is {_article(actual)} {actual} primitive; "
        f"{_article(needed)} {needed} feature needs {_article(needed)} "
        f"{needed} primitive",
    )


def _article(word: str) -> str:
    """Pick the indefinite article a following ``word`` needs."""
    return "an" if word[0] in "aeiou" else "a"
