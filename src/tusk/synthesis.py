"""Phase 1: build the feature graph from schemas alone.

Nothing here touches a dataframe, and this module must never import
:mod:`tusk.compiler`. That separation is what makes the algorithm testable
without any backend at all.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Iterable, Sequence

import narwhals as nw

from tusk.database import Database, Relationship
from tusk.dtypes import DtypeFamily, matches
from tusk.exceptions import (
    CategoricalDtypeWarning,
    PrimitiveError,
    UnmatchedPrimitiveWarning,
)
from tusk.features import (
    AggregationFeature,
    DirectFeature,
    Feature,
    GroupByTransformFeature,
    IdentityFeature,
    TransformFeature,
)
from tusk.primitives.aggregation import AGG_DEFAULTS
from tusk.primitives.base import Primitive
from tusk.primitives.registry import resolve_all
from tusk.primitives.transform import TRANS_DEFAULTS


def synthesize(
    database: Database,
    target_table: str,
    agg_primitives: Iterable[str | Primitive] | None = None,
    trans_primitives: Iterable[str | Primitive] | None = None,
    groupby_trans_primitives: Iterable[str | Primitive] | None = None,
    max_depth: int = 2,
) -> list[Feature]:
    """Generate feature definitions for a target table.

    ``database.schema()`` raises :class:`~tusk.exceptions.SchemaError` if the
    target table is unknown.

    Args:
        database: The schema to walk.
        target_table: Table to build features for.
        agg_primitives: Aggregation primitives, as names or instances. None
            selects ``AGG_DEFAULTS``.
        trans_primitives: Transform primitives, as names or instances. None
            selects ``TRANS_DEFAULTS``.
        groupby_trans_primitives: Transform primitives applied within
            foreign-key groups. None selects none.
        max_depth: Maximum number of stacked primitive applications.

    Returns:
        features: Feature definitions on the target table, deduplicated and
            excluding the target's own key columns.

    Warns:
        CategoricalDtypeWarning: If a Categorical or Enum column is skipped
            because a requested primitive requires a string input.
        UnmatchedPrimitiveWarning: If a requested primitive matched no column
            of its input dtypes anywhere in the walk.
    """
    database.schema(target_table)
    context = _Context(
        database=database,
        agg=resolve_all(AGG_DEFAULTS if agg_primitives is None else agg_primitives),
        trans=resolve_all(
            TRANS_DEFAULTS if trans_primitives is None else trans_primitives,
        ),
        groupby=resolve_all(groupby_trans_primitives or ()),
    )
    features = context.build(target_table, max_depth, ())
    context.warn_unmatched()
    keys = database.output_excluded_columns(target_table)
    kept = [
        f for f in features if not (isinstance(f, IdentityFeature) and f.column in keys)
    ]
    return list(dict.fromkeys(kept))


class _Context:
    """Carries the database and resolved primitives through the recursion."""

    def __init__(
        self,
        database: Database,
        agg: Sequence[Primitive],
        trans: Sequence[Primitive],
        groupby: Sequence[Primitive],
    ) -> None:
        """Store the walk's inputs.

        Args:
            database: The schema to walk.
            agg: Resolved aggregation primitives.
            trans: Resolved transform primitives.
            groupby: Resolved groupby-transform primitives.
        """
        self.database = database
        self.agg = agg
        self.trans = trans
        self.groupby = groupby
        self._categorical_warned: set[tuple[str, str, str]] = set()
        self._matched: set[str] = set()
        self._unmatched: dict[tuple[str, str], None] = {}

    def build(
        self,
        table: str,
        depth_limit: int,
        path: tuple[Relationship, ...],
    ) -> list[Feature]:
        """Build every feature on ``table`` of depth at most ``depth_limit``.

        Args:
            table: Table to build features for.
            depth_limit: Maximum depth of returned features.
            path: Relationships already traversed, never traversed again.

        Returns:
            Feature definitions on the table, deduplicated.
        """
        schema = self.database.schema(table)
        features: list[Feature] = [
            IdentityFeature(table, column, dtype)
            for column, dtype in schema.dtypes.items()
        ]

        if depth_limit > 0:
            features.extend(self._aggregations(table, depth_limit, path))
            features.extend(self._directs(table, depth_limit, path))
            features.extend(self._transforms(table, features, depth_limit))
            features.extend(
                self._groupby_transforms(table, features, depth_limit, path),
            )

        return list(dict.fromkeys(features))

    def _aggregations(
        self,
        table: str,
        depth_limit: int,
        path: tuple[Relationship, ...],
    ) -> list[Feature]:
        """Aggregate each child table's features up into this table.

        Args:
            table: The parent table.
            depth_limit: Maximum depth of returned features.
            path: Relationships already traversed.

        Returns:
            Aggregation features on the table.
        """
        out: list[Feature] = []
        for rel in self.database.children_of(table):
            if rel in path:
                continue
            child_features = self.build(rel.child, depth_limit - 1, path + (rel,))
            usable = self._usable(rel.child, child_features)
            for primitive in self.agg:
                if not primitive.input_dtypes:
                    out.append(AggregationFeature(primitive, (), rel))
                    continue
                for combo in self._combinations(primitive, usable, rel.child):
                    out.append(AggregationFeature(primitive, combo, rel))
        return out

    def _directs(
        self,
        table: str,
        depth_limit: int,
        path: tuple[Relationship, ...],
    ) -> list[Feature]:
        """Join each parent table's features down onto this table.

        Args:
            table: The child table.
            depth_limit: Maximum depth of returned features.
            path: Relationships already traversed.

        Returns:
            Direct features on the table.
        """
        out: list[Feature] = []
        for rel in self.database.parents_of(table):
            if rel in path:
                continue
            excluded_outputs = self.database.output_excluded_columns(rel.parent)
            for base in self._usable(
                rel.parent,
                self.build(rel.parent, depth_limit - 1, path + (rel,)),
            ):
                # A DirectFeature carries one column across the join, so it
                # cannot carry a multi-output parent feature: only that
                # feature's indexed columns exist to be carried.
                if base.is_multi_output:
                    continue
                # A raw row_creation_time is a legal *input* (it is not in
                # input_excluded_columns, which is why _usable let it
                # through), but it is never a legal *output*: carrying it
                # across the join as a DirectFeature would put the parent's
                # (or grandparent's, recursively) raw time index straight
                # into the feature matrix. Derived features over it, such as
                # MONTH(signed_up_at), are not IdentityFeatures and pass
                # through untouched.
                if isinstance(base, IdentityFeature) and (
                    base.column in excluded_outputs
                ):
                    continue
                out.append(DirectFeature(base, rel))
        return out

    def _transforms(
        self,
        table: str,
        existing: Sequence[Feature],
        depth_limit: int,
    ) -> list[Feature]:
        """Apply transform primitives to features already on this table.

        Args:
            table: The table being built.
            existing: Features produced so far.
            depth_limit: Maximum depth of returned features.

        Returns:
            features: Transform features on the table. ``_check_ordering``
                raises :class:`~tusk.exceptions.PrimitiveError` if an
                order-dependent primitive is requested for a table with no
                ``row_creation_time``.
        """
        usable = self._usable(table, existing)
        out: list[Feature] = []
        for primitive in self.trans:
            self._check_ordering(primitive, table)
            for combo in self._combinations(primitive, usable, table):
                feature = TransformFeature(primitive, combo)
                if feature.depth <= depth_limit:
                    out.append(feature)
        return out

    def _groupby_transforms(
        self,
        table: str,
        existing: Sequence[Feature],
        depth_limit: int,
        path: tuple[Relationship, ...],
    ) -> list[Feature]:
        """Apply transform primitives within each foreign-key group.

        Args:
            table: The table being built.
            existing: Features produced so far.
            depth_limit: Maximum depth of returned features.
            path: Relationships already traversed.

        Returns:
            features: Groupby-transform features on the table.
                ``_check_ordering`` raises
                :class:`~tusk.exceptions.PrimitiveError` if an order-dependent
                primitive is requested for a table with no
                ``row_creation_time``.
        """
        if not self.groupby:
            return []
        usable = self._usable(table, existing)
        out: list[Feature] = []
        for rel in self.database.parents_of(table):
            if rel in path:
                continue
            for primitive in self.groupby:
                self._check_ordering(primitive, table)
                for combo in self._combinations(primitive, usable, table):
                    feature = GroupByTransformFeature(primitive, combo, rel)
                    if feature.depth <= depth_limit:
                        out.append(feature)
        return out

    def warn_unmatched(self) -> None:
        """Warn about requested primitives that matched nothing anywhere.

        Skipping is the right behaviour -- raising would break a
        zero-configuration ``deep_feature_synthesis()`` on any schema missing
        a dtype family -- but skipping silently leaves the user with a
        primitive they asked for, no column, and no explanation.

        A primitive that produced features somewhere is not reported: it being
        inapplicable to one particular table is ordinary, and warning about it
        would bury the genuinely unusable case in noise. Each surviving
        (primitive, table) pair warns once.
        """
        for primitive_name, table in self._unmatched:
            if primitive_name in self._matched:
                continue
            warnings.warn(
                f"primitive {primitive_name!r} was requested but no column on "
                f"table {table!r} matches its input dtypes, so it generated no "
                f"features there. Check the column dtypes, or drop the "
                f"primitive from the request.",
                UnmatchedPrimitiveWarning,
                stacklevel=4,
            )

    def _warn_categorical(
        self,
        primitive: Primitive,
        candidates: Sequence[Feature],
    ) -> None:
        """Warn when a Categorical or Enum column is skipped by a STRING slot.

        Casting a column to ``Categorical`` asserts that its values are labels
        rather than text, so a string primitive skipping it is correct. Skipping
        it *silently* is not: the user would get a feature matrix with columns
        quietly missing and nothing to explain why.

        Each (primitive, column) pair warns at most once per synthesis run.

        Args:
            primitive: The primitive whose inputs are being matched.
            candidates: Features available as inputs.
        """
        if DtypeFamily.STRING not in primitive.input_dtypes:
            return
        for feature in candidates:
            if feature.dtype not in (nw.Categorical, nw.Enum):
                continue
            key = (primitive.name, feature.table, feature.name)
            if key in self._categorical_warned:
                continue
            self._categorical_warned.add(key)
            warnings.warn(
                f"column {feature.name!r} on {feature.table!r} has dtype "
                f"{feature.dtype}, so primitive {primitive.name!r} (which requires "
                f"a string input) will not be applied to it. Cast the column to "
                f"String if you want text primitives to use it.",
                CategoricalDtypeWarning,
                stacklevel=2,
            )

    def _check_ordering(self, primitive: Primitive, table: str) -> None:
        """Reject order-dependent primitives on tables that cannot be ordered.

        Narwhals requires ``order_by`` for these expressions on lazy backends,
        and the ordering column is the table's ``row_creation_time``. Checking
        here keeps the failure in phase 1, before any query is built.

        Args:
            primitive: The primitive being applied.
            table: The table it would be applied to.

        Raises:
            PrimitiveError: If the primitive is order-dependent and the table
                has no ``row_creation_time``.
        """
        if not getattr(primitive, "order_dependent", False):
            return
        if self.database.schema(table).row_creation_time is None:
            raise PrimitiveError(
                f"primitive {primitive.name!r} is order-dependent, so table "
                f"{table!r} needs a row_creation_time",
            )

    def _usable(self, table: str, features: Sequence[Feature]) -> list[Feature]:
        """Drop join keys, which identify rows rather than measuring anything.

        Only the primary key and foreign keys go: the ``row_creation_time`` is
        a genuine measurement and stays available as a primitive input, which
        is what makes ``MONTH(signed_up_at)``-style temporal transforms, and
        ``N_UNIQUE`` or ``CUM_COUNT`` over a temporal column, reachable. It is
        dropped later, from the matrix's raw passthrough columns only (see
        :meth:`~tusk.database.Database.output_excluded_columns`).

        Args:
            table: The table the features belong to.
            features: Candidate features.

        Returns:
            Features usable as primitive inputs.
        """
        keys = self.database.input_excluded_columns(table)
        return [
            f
            for f in features
            if not (isinstance(f, IdentityFeature) and f.column in keys)
        ]

    def _combinations(
        self,
        primitive: Primitive,
        candidates: Sequence[Feature],
        table: str,
    ) -> list[tuple[Feature, ...]]:
        """Enumerate input tuples a primitive accepts.

        Args:
            primitive: The primitive to match inputs for.
            candidates: Available features.
            table: Table the candidates live on, recorded so an unmatched
                primitive can be named alongside it once the walk finishes.

        Returns:
            One tuple per valid input combination.
        """
        # A multi-output feature materializes only its indexed columns
        # (``QUANTILES(x)[0]`` ...), never the bare stem, so nothing can read
        # it as an input. It stays a valid output; it is just not stackable.
        candidates = [f for f in candidates if not f.is_multi_output]
        per_slot = [
            [f for f in candidates if matches(f.dtype, family)]
            for family in primitive.input_dtypes
        ]
        self._warn_categorical(primitive, candidates)

        combos: list[tuple[Feature, ...]]
        if len(per_slot) == 1:
            combos = [(f,) for f in per_slot[0]]
        else:
            combos = [c for c in itertools.product(*per_slot) if len(set(c)) == len(c)]
            if primitive.commutative:
                seen: set[frozenset[Feature]] = set()
                deduped = []
                for combo in combos:
                    key = frozenset(combo)
                    if key not in seen:
                        seen.add(key)
                        deduped.append(combo)
                combos = deduped

        if not primitive.stack_on_self:
            combos = [c for c in combos if not any(_uses(f, primitive) for f in c)]

        # Only a primitive that actually produced a feature here counts as
        # matched: dtype-compatible slots are not enough on their own (e.g. a
        # commutative pair primitive with exactly one eligible column has a
        # non-empty slot but zero valid combos). Recording every table with
        # zero combos as unmatched -- not just the empty-slot case -- is what
        # lets warn_unmatched catch a primitive that never produced a single
        # feature anywhere.
        if combos:
            self._matched.add(primitive.name)
        else:
            self._unmatched.setdefault((primitive.name, table), None)
        return combos


def _uses(feature: Feature, primitive: Primitive) -> bool:
    """Report whether a feature was produced by a given primitive.

    Args:
        feature: The feature to inspect.
        primitive: The primitive to look for.

    Returns:
        True if the feature's own primitive matches.
    """
    return getattr(feature, "primitive", None) == primitive
