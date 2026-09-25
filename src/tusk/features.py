"""Feature definitions: the immutable output of phase 1.

Features are frozen dataclasses with structural equality, so a feature
reached by two different routes deduplicates in a set with no extra
bookkeeping. Primitive metadata alone determines every dtype here, never
data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from tusk.database import Relationship
from tusk.exceptions import PrimitiveError
from tusk.primitives.base import (
    AggregationPrimitive,
    GroupTransformPrimitive,
    Primitive,
    TransformPrimitive,
)


@dataclass(frozen=True)
class Feature(ABC):
    """Base class for every feature definition."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The column name in the feature matrix.

        It is a plain SQL identifier. The name joins parts with ``__``, not
        with the dots, parentheses and spaces a conventional DFS name
        uses. A backend that builds SQL reads those characters as table
        qualifiers and function calls, not as part of one identifier. See
        :attr:`display_name` for the readable form.
        """

    @property
    @abstractmethod
    def display_name(self) -> str:
        """The readable name, e.g. ``MEAN(transactions.amount)``.

        Carries the same meaning as :attr:`name` in the conventional DFS
        notation. It appears in documentation, logging and error messages,
        never as a column name.
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
        """The table this feature is a column of."""

    @property
    @abstractmethod
    def base_features(self) -> tuple[Feature, ...]:
        """The features this one is computed from."""

    @property
    def output_names(self) -> tuple[str, ...]:
        """One column name per output.

        A multi-output primitive has more than one name here.
        """
        return (self.name,)

    @property
    def display_output_names(self) -> tuple[str, ...]:
        """One readable name per output, parallel to :attr:`output_names`."""
        return (self.display_name,)

    @property
    def is_multi_output(self) -> bool:
        """Whether this feature materializes more than one column.

        Only the indexed names in :attr:`output_names` are ever
        materialized. A multi-output feature has no single column another
        primitive could read. It is a valid output of synthesis. It is
        never a valid input.

        This is derived from :attr:`output_names`, not a primitive's
        ``number_of_outputs``. It is defined even for
        :class:`IdentityFeature` and :class:`DirectFeature`, which have no
        primitive.
        """
        return len(self.output_names) > 1


@dataclass(frozen=True)
class IdentityFeature(Feature):
    """A raw column of a table.

    Attributes:
        table_name: The table the column belongs to.
        column: The column's name.
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
        """Identity features have depth zero."""
        return 0

    @property
    def table(self) -> str:
        """The table the column belongs to."""
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
        """Confirm the primitive is a transform that reads only its own row.

        This raises :class:`~tusk.exceptions.PrimitiveError`, via
        :func:`_require_kind`, if the primitive does not transform, or if
        it is a :class:`~tusk.primitives.base.GroupTransformPrimitive`.
        """
        _require_kind(self.primitive, TransformPrimitive, "a transform feature")
        if isinstance(self.primitive, GroupTransformPrimitive):
            raise PrimitiveError(
                f"primitive {self.primitive.name!r} reads other rows, so it only "
                f"runs within foreign-key groups; build a GroupByTransformFeature "
                f"from it instead.",
            )

    @property
    def name(self) -> str:
        """Built name, e.g. ``MONTH__started_at``."""
        return self.primitive.build_name([b.name for b in self.bases])

    @property
    def display_name(self) -> str:
        """Readable name, e.g. ``MONTH(started_at)``."""
        return self.primitive.build_display_name(
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
        condition: The condition masking the child's rows, as a (kind, key)
            pair where kind is ``"where"`` or ``"when"``. None aggregates
            every row.
    """

    primitive: Primitive
    bases: tuple[Feature, ...]
    relationship: Relationship
    condition: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        """Confirm the primitive aggregates rather than transforms.

        This raises :class:`~tusk.exceptions.PrimitiveError`, via
        :func:`_require_kind`, if the primitive does not aggregate.
        """
        _require_kind(
            self.primitive,
            AggregationPrimitive,
            "an aggregation feature",
        )

    @property
    def name(self) -> str:
        """Built name, e.g. ``MEAN__transactions__amount``.

        Zero-arity primitives name the child table instead of a column,
        giving ``COUNT__transactions``. A condition adds two trailing
        parts, giving ``COUNT__transactions__WHEN__current``.
        """
        child = self.relationship.child
        if not self.bases:
            return self.primitive.build_name([child, *self._condition_parts])
        names = [f"{child}__{b.name}" for b in self.bases]
        return self.primitive.build_name([*names, *self._condition_parts])

    @property
    def display_name(self) -> str:
        """Readable name, e.g. ``MEAN(transactions.amount)``.

        A condition adds to the final argument, giving
        ``COUNT(transactions WHEN current)``.
        """
        child = self.relationship.child
        if not self.bases:
            arguments = [child]
        else:
            arguments = [f"{child}.{b.display_name}" for b in self.bases]
        arguments[-1] += self._condition_display
        return self.primitive.build_display_name(arguments)

    @property
    def _condition_parts(self) -> tuple[str, ...]:
        """The condition's name parts, empty when the feature has no condition."""
        if self.condition is None:
            return ()
        kind, key = self.condition
        return (kind.upper(), key)

    @property
    def _condition_display(self) -> str:
        """The condition's readable suffix, empty when the feature has no condition."""
        if self.condition is None:
            return ""
        kind, key = self.condition
        return f" {kind.upper()} {key}"

    @property
    def dtype(self) -> Any:
        """Dtype derived from the primitive and its inputs."""
        return self.primitive.return_dtype(tuple(b.dtype for b in self.bases))

    @property
    def depth(self) -> int:
        """One deeper than the deepest input. It is 1 when there are no inputs."""
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
        """Built name, e.g. ``customers__age``."""
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
        """Confirm the primitive runs within foreign-key groups.

        This raises :class:`~tusk.exceptions.PrimitiveError`, via
        :func:`_require_kind`, if the primitive is not a
        :class:`~tusk.primitives.base.GroupTransformPrimitive`.
        """
        _require_kind(
            self.primitive,
            GroupTransformPrimitive,
            "a groupby transform feature",
        )

    @property
    def name(self) -> str:
        """Built name, e.g. ``CUM_SUM__amount__by__session_id``."""
        stem = self.primitive.build_name([b.name for b in self.bases])
        return f"{stem}__by__{self.relationship.foreign_key}"

    @property
    def display_name(self) -> str:
        """Readable name, e.g. ``CUM_SUM(amount) by session_id``."""
        stem = self.primitive.build_display_name(
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


def _require_kind(
    primitive: Primitive,
    required: type[Primitive],
    where: str,
) -> None:
    """Confirm a primitive is the kind ``where`` requires.

    Args:
        primitive: The primitive to check.
        required: The primitive base class required.
        where: What requires it, named for the error message.

    Raises:
        PrimitiveError: If ``primitive`` is not an instance of ``required``.
    """
    if not isinstance(primitive, required):
        raise PrimitiveError(
            f"primitive mismatch: {primitive.name!r} in {where} is not "
            f"{required.__name__}",
        )
