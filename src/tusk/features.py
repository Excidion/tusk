"""Feature definitions: the immutable output of phase 1.

Features are frozen dataclasses with structural equality, so a feature reached
by two different routes deduplicates in a set with no extra bookkeeping. Every
dtype here is derived from primitive metadata, never from data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tusk.entityset import Relationship
from tusk.primitives.base import Primitive


@dataclass(frozen=True)
class Feature:
    """Base class for every feature definition."""

    @property
    def name(self) -> str:
        """Display name, and the column name in the feature matrix."""
        raise NotImplementedError

    @property
    def dtype(self) -> Any:
        """Output dtype, computed statically."""
        raise NotImplementedError

    @property
    def depth(self) -> int:
        """Number of stacked primitive applications."""
        raise NotImplementedError

    @property
    def table(self) -> str:
        """Table this feature is a column of."""
        raise NotImplementedError

    @property
    def base_features(self) -> tuple[Feature, ...]:
        """Features this one is computed from."""
        raise NotImplementedError

    @property
    def output_names(self) -> tuple[str, ...]:
        """One column name per output; more than one for multi-output primitives."""
        return (self.name,)

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

    @property
    def name(self) -> str:
        """Generated name, e.g. ``MONTH(started_at)``."""
        return self.primitive.generate_name([b.name for b in self.bases])

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

    @property
    def name(self) -> str:
        """Generated name, e.g. ``MEAN(transactions.amount)``.

        Zero-arity primitives name the child table instead of a column, giving
        ``COUNT(transactions)``.
        """
        child = self.relationship.child
        if not self.bases:
            return self.primitive.generate_name([child])
        return self.primitive.generate_name([f"{child}.{b.name}" for b in self.bases])

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
        """Generated name, e.g. ``customers.age``."""
        return f"{self.relationship.parent}.{self.base_feature.name}"

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

    @property
    def name(self) -> str:
        """Generated name, e.g. ``CUM_SUM(amount) by session_id``."""
        stem = self.primitive.generate_name([b.name for b in self.bases])
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
