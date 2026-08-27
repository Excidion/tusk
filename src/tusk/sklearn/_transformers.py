"""Deep feature synthesis as scikit-learn estimators.

``X`` is the target table's primary-key column, not the database. That
inversion is what makes ``X`` array-like, so scikit-learn can split it: the
keys *are* the rows, which makes misalignment against ``y`` impossible rather
than merely discouraged, and lets ``cross_val_score`` and ``GridSearchCV``
work at all. The database travels as routed metadata, because ``clone``
deep-copies constructor parameters and a duckdb relation cannot be pickled.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from datetime import datetime
from typing import Any

import narwhals as nw
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

from tusk.api import apply_features
from tusk.database import Database
from tusk.exceptions import (
    LineageError,
    LineageWarning,
    SchemaError,
    UnencodedFeatureWarning,
)
from tusk.features import Feature
from tusk.primitives.aggregation import AGG_DEFAULTS
from tusk.primitives.base import Primitive
from tusk.primitives.transform import TRANS_DEFAULTS
from tusk.sklearn._encoders import encoder_prefix, selector_of, validate_inner
from tusk.sklearn._frames import as_keys, backend_hint, collect_matrix
from tusk.sklearn._lineage import Sentinels, make_sentinels
from tusk.synthesis import synthesize


class DFSTransformer(TransformerMixin, BaseEstimator):
    """Deep feature synthesis as a pipeline step.

    Fitting sets ``features_``, the synthesized feature definitions, and
    ``database_``, the database seen at fit -- used as :meth:`transform`'s
    fallback when none is routed there directly. Neither is declared as a
    class-level attribute: both depend on ``target_table`` and are only known
    once :meth:`fit` has run.
    """

    __metadata_request__fit = {"database": True}
    __metadata_request__transform = {"database": True}

    def __init__(
        self,
        target_table: str,
        agg_primitives: Iterable[str | Primitive] | None = None,
        trans_primitives: Iterable[str | Primitive] | None = None,
        groupby_trans_primitives: Iterable[str | Primitive] | None = None,
        max_depth: int = 2,
        cutoff_time: datetime | None = None,
        output_backend: str | None = None,
    ) -> None:
        """Configure synthesis.

        Every argument is stored unmodified under its own name: scikit-learn
        requires ``__init__`` to only assign, or ``get_params`` and ``clone``
        cannot round-trip the estimator.

        Args:
            target_table: Table to build features for.
            agg_primitives: Aggregation primitives; None selects the defaults.
            trans_primitives: Transform primitives; None selects the defaults.
            groupby_trans_primitives: Transforms within foreign-key groups.
            max_depth: Maximum stacked primitive applications.
            cutoff_time: Only rows at or before this are visible. Clone-safe,
                unlike the database, so it lives here and can be searched.
            output_backend: Backend to collect to. None collects natively,
                which keeps narwhals-native transformers on their own frame
                type.
        """
        self.target_table = target_table
        self.agg_primitives = agg_primitives
        self.trans_primitives = trans_primitives
        self.groupby_trans_primitives = groupby_trans_primitives
        self.max_depth = max_depth
        self.cutoff_time = cutoff_time
        self.output_backend = output_backend

    def fit(
        self,
        X: Any,
        y: Any = None,
        database: Database | None = None,
    ) -> DFSTransformer:
        """Synthesize feature definitions.

        Reads the schema and no rows, so this is cheap; the expensive pass is
        in :meth:`transform`.

        Args:
            X: The target's primary-key column. Unused here beyond validation,
                since synthesis depends only on the schema.
            y: Ignored; present for the scikit-learn signature.
            database: The database, routed as metadata.

        Returns:
            This estimator.

        Raises:
            SchemaError: If no database was supplied.
        """
        if database is None:
            raise SchemaError(
                "no database supplied; pass database=… to fit, and enable "
                "sklearn.set_config(enable_metadata_routing=True) when fitting "
                "inside a Pipeline",
            )
        as_keys(X)
        self.database_ = database
        self.features_ = list(
            synthesize(
                database=database,
                target_table=self.target_table,
                agg_primitives=AGG_DEFAULTS
                if self.agg_primitives is None
                else self.agg_primitives,
                trans_primitives=TRANS_DEFAULTS
                if self.trans_primitives is None
                else self.trans_primitives,
                groupby_trans_primitives=self.groupby_trans_primitives or (),
                max_depth=self.max_depth,
            ),
        )
        return self

    def transform(self, X: Any, database: Database | None = None) -> Any:
        """Compute the feature matrix for the keys in ``X``.

        Args:
            X: The target's primary-key column. Its order becomes the matrix's
                row order.
            database: The database, routed as metadata. When absent, the one
                seen at fit is used -- scikit-learn's scorers call ``predict``
                with no metadata, so without this fallback every
                cross-validated score would be ``nan``.

        Returns:
            An eager native frame, one row per key, in key order.
        """
        check_is_fitted(self, "features_")
        db = self.database_ if database is None else database
        return collect_matrix(
            apply_features(self.features_, db, self.cutoff_time),
            self._primary_key(db),
            as_keys(X),
            self.output_backend,
        )

    def fit_transform(
        self,
        X: Any,
        y: Any = None,
        database: Database | None = None,
        **kwargs: Any,
    ) -> Any:
        """Fit, then transform, forwarding the database to both.

        ``TransformerMixin.fit_transform`` drops metadata on the way to
        ``transform``; scikit-learn warns about exactly this. Overriding is the
        prescribed fix.

        Args:
            X: The target's primary-key column.
            y: Ignored.
            database: The database, routed as metadata.
            **kwargs: Ignored; absorbs scikit-learn's fit parameters.

        Returns:
            The feature matrix.
        """
        return self.fit(X, y, database=database).transform(X, database=database)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Column names of the matrix, in column order.

        A multi-output primitive contributes several names, so this is wider
        than ``len(features_)``.

        Args:
            input_features: Ignored; present for the scikit-learn signature.

        Returns:
            The names, as an object array.
        """
        check_is_fitted(self, "features_")
        names = [n for f in self.features_ for n in f.output_names]
        return np.asarray(names, dtype=object)

    def _primary_key(self, database: Database) -> str:
        """The target table's key column.

        Args:
            database: The database to read the schema from.

        Returns:
            The primary key column name.

        Raises:
            SchemaError: If the target table declares none.
        """
        primary_key = database.schema(self.target_table).primary_key
        if primary_key is None:
            raise SchemaError(
                f"target table {self.target_table!r} needs a primary_key: "
                "it is what X names",
            )
        return primary_key


class DFSSelectorTransformer(DFSTransformer):
    """DFS whose feature list is pruned to what a selector actually kept.

    The point is inference cost. A run that generates eight hundred features
    and keeps forty should compute forty when it next sees data, not eight
    hundred.

    Two column spaces meet here and must never be conflated: *tusk space*, the
    matrix, indexed by feature ``output_names``, and *encoded space*, the
    encoder's output, indexed by ``get_feature_names_out()``. The selector's
    mask indexes encoded space, pruning happens in tusk space, and sentinel
    lineage is the only bridge between them.

    Fitting adds ``encoder_``, the encoder prefix refitted on the surviving
    columns; ``kept_names_``, the encoded-space names the selector chose, in
    encoder order; and ``sentinels_``, the renaming that recovers lineage.
    Like the parent's fitted attributes, none is declared at class level:
    all three are only known once :meth:`fit` has run.
    """

    def __init__(
        self,
        target_table: str,
        inner: Any = None,
        agg_primitives: Iterable[str | Primitive] | None = None,
        trans_primitives: Iterable[str | Primitive] | None = None,
        groupby_trans_primitives: Iterable[str | Primitive] | None = None,
        max_depth: int = 2,
        cutoff_time: datetime | None = None,
        output_backend: str | None = None,
    ) -> None:
        """Configure synthesis and selection.

        The parent's parameters are repeated rather than absorbed into
        ``**kwargs``, because scikit-learn discovers them by introspecting this
        signature.

        Args:
            target_table: Table to build features for.
            inner: An estimator ending in a ``SelectorMixin``; everything
                before it encodes.
            agg_primitives: Aggregation primitives; None selects the defaults.
            trans_primitives: Transform primitives; None selects the defaults.
            groupby_trans_primitives: Transforms within foreign-key groups.
            max_depth: Maximum stacked primitive applications.
            cutoff_time: Only rows at or before this are visible.
            output_backend: Backend to collect to; None collects natively.
        """
        super().__init__(
            target_table=target_table,
            agg_primitives=agg_primitives,
            trans_primitives=trans_primitives,
            groupby_trans_primitives=groupby_trans_primitives,
            max_depth=max_depth,
            cutoff_time=cutoff_time,
            output_backend=output_backend,
        )
        self.inner = inner

    def fit(
        self,
        X: Any,
        y: Any = None,
        database: Database | None = None,
    ) -> DFSSelectorTransformer:
        """Synthesize, select, then prune.

        The selector is fitted once here and then frozen; only the encoder is
        refitted on the survivors. Refitting the selector would let it reach a
        different verdict -- ``SelectKBest(k=50)`` choosing 50 out of 60 need
        not choose the 50 it chose out of 500.

        Args:
            X: The target's primary-key column.
            y: Training targets, passed to the selector.
            database: The database, routed as metadata.

        Returns:
            This estimator.

        Raises:
            SchemaError: If selection eliminated every feature.
            LineageError: If a kept column vanished from the refitted encoder,
                meaning lineage missed a source and a feature was wrongly
                pruned.

        Warns:
            LineageWarning: If any kept column's provenance was unrecoverable,
                in which case nothing is pruned.
            UnencodedFeatureWarning: If a feature fed no encoded column at
                all, so the encoder never gave the selector a chance to keep
                it.
        """
        validate_inner(self.inner)
        super().fit(X, y, database=database)
        db = self.database_ if database is None else database

        matrix = nw.from_native(super().transform(X, database=db), eager_only=True)
        sentinels = make_sentinels(list(matrix.columns))
        renamed = matrix.rename(sentinels.mapping)
        probe = renamed.to_native()

        inner = clone(self.inner)
        with backend_hint(probe):
            inner.fit(probe, y)
        # These are the *pre*-selection names: slicing the pipeline is what
        # gets them, since ``inner.get_feature_names_out()`` on a pipeline
        # ending in a selector reports what survived, which cannot be zipped
        # against the mask. The inputs are named explicitly because an inner
        # that is a bare selector has no encoder to have been fitted, and the
        # identity standing in for one can only report names if it is told
        # them.
        encoded = list(
            encoder_prefix(inner).get_feature_names_out(list(renamed.columns)),
        )
        # strict=True: a mask that does not line up with the names is a broken
        # encoder contract, and zipping it short would silently mis-attribute
        # every column after the mismatch.
        mask = selector_of(inner).get_support()
        kept = [name for name, keep in zip(encoded, mask, strict=True) if keep]

        self.features_ = self._prune(kept, encoded, sentinels)
        if not self.features_:
            raise SchemaError("feature selection eliminated every feature")

        surviving_columns = [n for f in self.features_ for n in f.output_names]
        narrowed = matrix.select(surviving_columns).rename(
            {c: sentinels.mapping[c] for c in surviving_columns},
        )
        self.encoder_ = encoder_prefix(clone(self.inner))
        with backend_hint(narrowed.to_native()):
            self.encoder_.fit(narrowed.to_native())

        refit = list(self.encoder_.get_feature_names_out())
        missing = [n for n in kept if n not in set(refit)]
        if missing:
            raise LineageError(
                f"{len(missing)} selected columns vanished when the encoder was "
                f"refit on the surviving features, e.g. "
                f"{[sentinels.restore(n) for n in missing[:3]]}. Lineage missed "
                "a source and pruned a feature that was still needed.",
            )
        self.sentinels_ = sentinels
        self.kept_names_ = [n for n in refit if n in set(kept)]
        return self

    def transform(self, X: Any, database: Database | None = None) -> Any:
        """Compute the surviving features, encode, and apply the frozen mask.

        Nothing is fitted here. Both fits happen in :meth:`fit`; a Pipeline
        calls ``fit_transform`` on intermediate steps during ``fit`` and
        ``transform`` during ``predict``.

        Args:
            X: The target's primary-key column.
            database: The database, routed as metadata.

        Returns:
            The encoded, selected matrix.
        """
        check_is_fitted(self, "kept_names_")
        db = self.database_ if database is None else database
        matrix = nw.from_native(super().transform(X, database=db), eager_only=True)
        surviving_columns = [n for f in self.features_ for n in f.output_names]
        probe = matrix.select(surviving_columns).rename(
            {c: self.sentinels_.mapping[c] for c in surviving_columns},
        )
        with backend_hint(probe.to_native()):
            encoded = self.encoder_.transform(probe.to_native())
        return self._select(encoded)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Selected column names, with sentinels substituted back.

        Args:
            input_features: Ignored; present for the scikit-learn signature.

        Returns:
            Readable names such as ``oh__MODE__transactions__category_a``.
        """
        check_is_fitted(self, "kept_names_")
        names = [self.sentinels_.restore(n) for n in self.kept_names_]
        return np.asarray(names, dtype=object)

    def _prune(
        self,
        kept: list[str],
        encoded: list[str],
        sentinels: Sentinels,
    ) -> list[Feature]:
        """Features with at least one column feeding a kept encoded column.

        A multi-output feature is all-or-nothing: one output cannot be computed
        without the others, so any live column keeps the whole feature.

        Args:
            kept: Encoded-space names the selector chose.
            encoded: Every encoded-space name, kept or not. Needed to tell a
                feature the selector *rejected* from one the encoder never
                looked at, which are worth reporting differently.
            sentinels: The renaming used to read provenance.

        Returns:
            The surviving features, in their original order.

        Warns:
            LineageWarning: If any kept name mentions no sentinel, in which
                case every feature is kept.
            UnencodedFeatureWarning: If a matrix column fed no encoded column
                at all, so its feature is pruned for a reason the user may not
                have intended.
        """
        sources = {name: sentinels.sources(name) for name in kept}
        opaque = [name for name, s in sources.items() if not s]
        if opaque:
            warnings.warn(
                f"{len(opaque)} selected columns do not name their inputs "
                f"(e.g. {opaque[:3]}), so tusk cannot tell which features they "
                "came from and has pruned nothing. Selection still applies; "
                "only the inference-time saving is lost.",
                LineageWarning,
                stacklevel=3,
            )
            return list(self.features_)

        live = {column for s in sources.values() for column in s}
        touched = {c for name in encoded for c in sentinels.sources(name)}
        unencoded = [c for c in sentinels.columns if c not in touched]
        if unencoded:
            warnings.warn(
                f"{len(unencoded)} features fed no encoded column (e.g. "
                f"{unencoded[:3]}) and have been pruned. The encoder never "
                'looked at them -- a ColumnTransformer with remainder="drop" '
                "covering only some dtypes is the usual cause.",
                UnencodedFeatureWarning,
                stacklevel=3,
            )
        return [f for f in self.features_ if any(n in live for n in f.output_names)]

    def _select(self, encoded: Any) -> Any:
        """Apply the frozen mask to the encoder's output.

        Resolved by name against the fitted encoder's own ordering rather than
        by stored position: with ``output_backend=None`` the output may be a
        frame or a bare array, and a stored index would silently mis-slice if
        that changed between fit and transform.

        Args:
            encoded: Whatever the encoder returned.

        Returns:
            Only the selected columns, in encoder order.
        """
        order = list(self.encoder_.get_feature_names_out())
        indices = [order.index(n) for n in self.kept_names_]
        frame = nw.from_native(encoded, eager_only=True, pass_through=True)
        if isinstance(frame, nw.DataFrame):
            return frame.select([frame.columns[i] for i in indices]).to_native()
        return encoded[:, indices]
