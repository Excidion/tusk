"""Phase 1: the feature graph built from schemas alone.

Nothing here touches a table. The walk reads schemas and primitive metadata
only. Features flow one way, from phase 1 to phase 2: this module must
never call the compiler.
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
    SchemaError,
    UnmatchedConditionWarning,
    UnmatchedPrimitiveWarning,
)
from tusk.feature_list import FeatureList
from tusk.features import (
    AggregationFeature,
    DirectFeature,
    Feature,
    GroupByTransformFeature,
    IdentityFeature,
    TransformFeature,
    _require_kind,
)
from tusk.primitives.aggregation import AGG_DEFAULTS, CONDITIONAL_DEFAULTS
from tusk.primitives.base import (
    AggregationPrimitive,
    GroupTransformPrimitive,
    OrderedAggregationPrimitive,
    OrderedTransformPrimitive,
    Primitive,
    TransformPrimitive,
)
from tusk.primitives.registry import resolve_all
from tusk.primitives.transform import TRANS_DEFAULTS


def synthesize(
    database: Database,
    target_table: str,
    agg_primitives: Iterable[str | Primitive] | None = None,
    trans_primitives: Iterable[str | Primitive] | None = None,
    conditional_primitives: Iterable[str | Primitive] | None = None,
    max_depth: int = 2,
) -> FeatureList:
    """Build feature definitions for a target table.

    ``database.get_schema()`` raises :class:`~tusk.exceptions.SchemaError` if
    the target table is unknown.

    It also raises :class:`~tusk.exceptions.PrimitiveError`, via
    :func:`~tusk.features._require_kind`, if a primitive resolved from
    ``agg_primitives`` is not an
    :class:`~tusk.primitives.base.AggregationPrimitive`, or if one resolved
    from ``trans_primitives`` is not a
    :class:`~tusk.primitives.base.TransformPrimitive`. This function checks
    that eagerly, instead of leaving it to each :class:`Feature` subclass's
    own check. A primitive that matches no column is never built into a
    feature at all, so only the argument it was passed to can name the
    mistake.

    Args:
        database: The schema to walk.
        target_table: Table to build features for.
        agg_primitives: Aggregation primitives, as names or instances. None
            selects ``AGG_DEFAULTS``.
        trans_primitives: Transform primitives, as names or instances. This
            function applies each
            :class:`~tusk.primitives.base.GroupTransformPrimitive` within
            each foreign-key group. It applies every other transform
            primitive to each row. None selects ``TRANS_DEFAULTS``.
        conditional_primitives: Aggregation primitives to compute over only
            the rows each declared condition keeps, as names or instances.
            This is a separate list from ``agg_primitives``. A primitive
            named here yields only the conditional features, so name it in
            both lists for the unconditional ones too. None selects
            ``CONDITIONAL_DEFAULTS``. ``()`` computes no conditional
            features.
        max_depth: Maximum number of stacked primitive applications.

    Returns:
        features: Feature definitions on the target table, deduplicated. It
            excludes the target's own key columns.

    Raises:
        SchemaError: If the walk built no features at all. This is a dead
            end, not an empty result, because nothing downstream can be
            computed from it.

    Warns:
        CategoricalDtypeWarning: If a Categorical or Enum column is skipped
            because a requested primitive's text inputs require a String
            column.
        UnmatchedPrimitiveWarning: If a requested primitive matched no column
            of its input dtypes anywhere in the walk.
        UnmatchedConditionWarning: If ``conditional_primitives`` was explicitly
            requested but no table in the database declares a ``where`` or
            ``when`` condition.
    """
    database.get_schema(target_table)
    agg = resolve_all(AGG_DEFAULTS if agg_primitives is None else agg_primitives)
    trans = resolve_all(
        TRANS_DEFAULTS if trans_primitives is None else trans_primitives,
    )
    conditional_agg = resolve_all(
        CONDITIONAL_DEFAULTS
        if conditional_primitives is None
        else conditional_primitives,
    )
    for primitive in agg:
        _require_kind(primitive, AggregationPrimitive, "agg_primitives")
    for primitive in trans:
        _require_kind(primitive, TransformPrimitive, "trans_primitives")
    for primitive in conditional_agg:
        _require_kind(primitive, AggregationPrimitive, "conditional_primitives")
    _warn_if_conditional_primitives_are_unusable(
        database,
        conditional_primitives,
        conditional_agg,
    )
    context = _Context(
        database=database,
        agg=agg,
        trans=[p for p in trans if not isinstance(p, GroupTransformPrimitive)],
        groupby=[p for p in trans if isinstance(p, GroupTransformPrimitive)],
        conditional_agg=conditional_agg,
    )
    features = context.build(target_table, max_depth, ())
    context.warn_unmatched()
    keys = database.output_excluded_columns(target_table)
    kept = [
        f for f in features if not (isinstance(f, IdentityFeature) and f.column in keys)
    ]
    if not kept:
        # Raised here rather than in FeatureList because only the walk
        # knows which primitives were asked for and how deep it went.
        raise SchemaError(
            f"no features generated for {target_table!r}: no primitive "
            f"matched a column within max_depth={max_depth}. Add matching "
            "primitives, raise max_depth, or add relationships.",
        )
    return FeatureList(dict.fromkeys(kept))


def _warn_if_conditional_primitives_are_unusable(
    database: Database,
    conditional_primitives: Iterable[str | Primitive] | None,
    conditional_agg: Sequence[Primitive],
) -> None:
    """Warn when ``conditional_primitives`` was requested but no condition can use it.

    This check cannot use ``_matched``/``_unmatched``. A conditional
    primitive such as ``count`` or ``sum`` is usually also in
    ``agg_primitives``. There, it counts as matched already, so a warning
    keyed on the primitive would stay suppressed. This check instead keys on
    the condition dimension, independent of primitive matching.

    ``conditional_primitives=None`` selects ``CONDITIONAL_DEFAULTS`` and
    stays silent. ``conditional_primitives=()`` explicitly disables
    conditional features and stays silent too.

    Args:
        database: The database to check for declared conditions.
        conditional_primitives: The caller's own argument, unresolved, used
            only to tell an explicit request apart from the ``None`` default.
        conditional_agg: ``conditional_primitives`` resolved to primitive
            instances.

    Warns:
        UnmatchedConditionWarning: If ``conditional_primitives`` is neither
            None nor empty, and no table in the database declares a
            ``where`` or ``when`` condition.
    """
    if conditional_primitives is None or not conditional_agg:
        return
    if any(database.get_schema(name).conditions for name in database.table_names):
        return
    warnings.warn(
        "conditional_primitives was requested but no table declares a where "
        "or when condition, so it generated no conditional features. Declare "
        "where=... or when=... on a table's add_table(), or drop "
        "conditional_primitives.",
        UnmatchedConditionWarning,
        stacklevel=4,
    )


class _Context:
    """The database and resolved primitives, carried through the recursive walk."""

    def __init__(
        self,
        database: Database,
        agg: Sequence[Primitive],
        trans: Sequence[Primitive],
        groupby: Sequence[Primitive],
        conditional_agg: Sequence[Primitive],
    ) -> None:
        """Store the walk's inputs.

        Args:
            database: The schema to walk.
            agg: Resolved aggregation primitives.
            trans: Resolved transform primitives applied to each row.
            groupby: Resolved group transform primitives, applied within
                each foreign-key group.
            conditional_agg: Resolved aggregation primitives that
                additionally get one masked variant per condition declared on
                the child table.
        """
        self.database = database
        self.agg = agg
        self.trans = trans
        self.groupby = groupby
        self.conditional_agg = conditional_agg
        self._categorical_warned: set[tuple[str, str, str]] = set()
        self._matched: set[str] = set()
        self._unmatched: dict[tuple[str, str], str] = {}

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
        schema = self.database.get_schema(table)
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
            Aggregation features on the table. ``_check_ordering`` raises
                :class:`~tusk.exceptions.PrimitiveError` if an order-dependent
                primitive is requested for a child table with no
                ``row_creation_time``.
        """
        out: list[Feature] = []
        for rel in self.database.children_of(table):
            if rel in path:
                continue
            child_features = self.build(rel.child, depth_limit - 1, path + (rel,))
            usable = self._usable(rel.child, child_features)
            for primitive in self.agg:
                self._check_ordering(primitive, rel.child)
                out.extend(self._build_aggregations(primitive, rel, usable, None))
            conditions = self.database.get_schema(rel.child).conditions
            for primitive in self.conditional_agg:
                self._check_ordering(primitive, rel.child)
                for condition in conditions:
                    out.extend(
                        self._build_aggregations(primitive, rel, usable, condition),
                    )
        return out

    def _build_aggregations(
        self,
        primitive: Primitive,
        relationship: Relationship,
        usable: Sequence[Feature],
        condition: tuple[str, str] | None,
    ) -> list[Feature]:
        """Build every aggregation of one primitive across one relationship.

        Args:
            primitive: The aggregation primitive to apply.
            relationship: The parent-child link being aggregated across.
            usable: Features on the child that may serve as inputs.
            condition: The (kind, key) pair masking the child's rows, or None.

        Returns:
            One feature per usable input combination, or a single zero-arity
            feature when the primitive declares no signatures.
        """
        if not primitive.signatures:
            return [AggregationFeature(primitive, (), relationship, condition)]
        return [
            AggregationFeature(primitive, combo, relationship, condition)
            for combo in self._combinations(primitive, usable, relationship.child)
        ]

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
            features: Transform features on the table.
        """
        usable = self._usable(table, existing)
        out: list[Feature] = []
        for primitive in self.trans:
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
        """Apply group transform primitives within each foreign-key group.

        Args:
            table: The table being built.
            existing: Features produced so far.
            depth_limit: Maximum depth of returned features.
            path: Relationships already traversed.

        Returns:
            features: Groupby-transform features on the table, one per parent
                relationship not on ``path``. ``_check_ordering`` raises
                :class:`~tusk.exceptions.PrimitiveError` if an order-dependent
                primitive is requested for a table with no
                ``row_creation_time``.
        """
        if not self.groupby:
            return []
        parents = self.database.parents_of(table)
        if not parents:
            self._record_ungroupable(table)
        relationships = [rel for rel in parents if rel not in path]
        usable = self._usable(table, existing)
        out: list[Feature] = []
        for rel in relationships:
            for primitive in self.groupby:
                self._check_ordering(primitive, table)
                for combo in self._combinations(primitive, usable, table):
                    feature = GroupByTransformFeature(primitive, combo, rel)
                    if feature.depth <= depth_limit:
                        out.append(feature)
        return out

    def _record_ungroupable(self, table: str) -> None:
        """Record every group transform primitive as unmatched on ``table``.

        This function records only a table with no parent relationship at
        all. A table whose parents the walk already consumed on its path
        still has a relationship. Naming it here would tell the user to add
        one they already have.

        Args:
            table: A table with no parent relationship.
        """
        for primitive in self.groupby:
            self._unmatched.setdefault(
                (primitive.name, table),
                f"table {table!r} has no parent relationship to group its rows "
                f"by, so it generated no features there. Add a relationship, or "
                f"drop the primitive from the request.",
            )

    def warn_unmatched(self) -> None:
        """Warn about requested primitives that matched nothing anywhere.

        Skipping is the right behaviour. Raising would break a
        zero-configuration ``deep_feature_synthesis()`` on any schema
        lacking a dtype family. However, skipping silently leaves the user
        with a primitive they asked for, no column, and no explanation.

        A primitive that produced features somewhere is not reported. Being
        inapplicable to one particular table is ordinary. Warning about it
        would bury the genuinely unusable case in noise. Each surviving
        (primitive, table) pair warns once.
        """
        for (primitive_name, _), reason in self._unmatched.items():
            if primitive_name in self._matched:
                continue
            warnings.warn(
                f"primitive {primitive_name!r} was requested but {reason}",
                UnmatchedPrimitiveWarning,
                stacklevel=4,
            )

    def _warn_categorical(
        self,
        primitive: Primitive,
        candidates: Sequence[Feature],
    ) -> None:
        """Warn when a Categorical or Enum column is skipped by a STRING slot.

        Casting a column to ``Categorical`` asserts that its values are
        labels, not text. So it is correct for a string primitive to skip
        it. Skipping it *silently* is not correct. The user would then get
        a feature matrix with columns quietly absent and no explanation for
        it.

        It does not warn about a primitive that has a CATEGORICAL signature.

        Each (primitive, column) pair warns at most once per synthesis run.

        Args:
            primitive: The primitive whose inputs are being matched.
            candidates: Features available as inputs.
        """
        if not any(
            DtypeFamily.STRING in signature for signature in primitive.signatures
        ):
            return
        if any(
            DtypeFamily.CATEGORICAL in signature for signature in primitive.signatures
        ):
            return
        # Arity is uniform across a primitive's signatures, so one answers for all.
        takes_two_inputs = len(primitive.signatures[0]) == 2
        for feature in candidates:
            if feature.dtype not in (nw.Categorical, nw.Enum):
                continue
            key = (primitive.name, feature.table, feature.name)
            if key in self._categorical_warned:
                continue
            self._categorical_warned.add(key)
            message = (
                f"column {feature.name!r} on {feature.table!r} has dtype "
                f"{feature.dtype}, so primitive {primitive.name!r} (whose text "
                f"inputs require a String column) will not be applied to it. Cast "
                f"the column to String if you want text primitives to use it."
            )
            if takes_two_inputs:
                message += (
                    " If you want to compare labels for equality instead, "
                    "'equal_categorical' / 'not_equal_categorical' take two "
                    "Categorical columns."
                )
            warnings.warn(
                message,
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
            PrimitiveError: If the primitive is an
                :class:`~tusk.primitives.base.OrderedTransformPrimitive` or an
                :class:`~tusk.primitives.base.OrderedAggregationPrimitive` and
                the table has no ``row_creation_time``.
        """
        if not isinstance(
            primitive,
            (OrderedTransformPrimitive, OrderedAggregationPrimitive),
        ):
            return
        if self.database.get_schema(table).row_creation_time is None:
            raise PrimitiveError(
                f"primitive {primitive.name!r} is order-dependent, so table "
                f"{table!r} needs a row_creation_time",
            )

    def _usable(self, table: str, features: Sequence[Feature]) -> list[Feature]:
        """Drop join keys, which identify rows rather than measuring anything.

        Only the primary key and foreign keys are dropped here. The
        ``row_creation_time`` is a genuine measurement, so it stays available
        as a primitive input. This is what makes ``MONTH(signed_up_at)``-style
        temporal transforms reachable, along with ``N_UNIQUE`` or
        ``CUM_COUNT`` over a temporal column. The ``row_creation_time`` is
        dropped later instead, from the feature matrix's raw passthrough
        columns only (see
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
        """Enumerate input tuples a primitive accepts, across all its shapes.

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
        self._warn_categorical(primitive, candidates)

        # Dtype families overlap -- a Datetime column matches both HAS_DATE
        # and TEMPORAL -- so two shapes can yield the same combination. A
        # commutative primitive can also have its two argument orders land in
        # *different* signatures (e.g. (NUMERIC, ANY) and (ANY, NUMERIC)), so
        # the commutative collapse has to run once here, over the union, not
        # per signature.
        all_combos = (
            combo
            for signature in primitive.signatures
            for combo in self._fill_slots(
                primitive,
                candidates,
                signature,
            )
        )
        dedup_key = frozenset if primitive.commutative else tuple
        seen: set[frozenset[Feature] | tuple[Feature, ...]] = set()
        combos: list[tuple[Feature, ...]] = []
        for combo in all_combos:
            key = dedup_key(combo)
            if key not in seen:
                seen.add(key)
                combos.append(combo)

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
            self._unmatched.setdefault(
                (primitive.name, table),
                f"no column on table {table!r} matches its input dtypes, so it "
                f"generated no features there. Check the column dtypes, or drop "
                f"the primitive from the request.",
            )
        return combos

    def _fill_slots(
        self,
        primitive: Primitive,
        candidates: Sequence[Feature],
        signature: tuple[DtypeFamily, ...],
    ) -> list[tuple[Feature, ...]]:
        """Enumerate input tuples matching one of a primitive's input shapes.

        Args:
            primitive: The primitive to match inputs for.
            candidates: Available features, already filtered of multi-output
                ones.
            signature: One dtype family per input slot.

        Returns:
            One tuple per valid input combination for this shape.
        """
        per_slot = [
            [f for f in candidates if matches(f.dtype, family)] for family in signature
        ]

        combos: list[tuple[Feature, ...]]
        if len(per_slot) == 1:
            combos = [(f,) for f in per_slot[0]]
        else:
            combos = [c for c in itertools.product(*per_slot) if len(set(c)) == len(c)]

        if not primitive.stack_on_self:
            combos = [c for c in combos if not any(_uses(f, primitive) for f in c)]

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
