"""Deep feature synthesis as scikit-learn estimators.

Both estimators take ``X`` as the target table's primary key and
receive the database as routed metadata, so scikit-learn must be configured
with ``sklearn.set_config(enable_metadata_routing=True)`` to use them inside
a ``Pipeline``.

:class:`DFSTransformer` synthesizes features and computes them for the keys
in ``X``. :class:`DFSSelectorTransformer` additionally fits a supplied
encode-and-select pipeline, drops the features whose columns the selector did
not keep, and computes only the rest thereafter.
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

from tusk.database import Database
from tusk.exceptions import (
    LineageError,
    LineageWarning,
    SchemaError,
    UnencodedFeatureWarning,
)
from tusk.feature_list import FeatureList
from tusk.features import Feature
from tusk.primitives.base import Primitive
from tusk.sklearn._encoders import (
    get_encoder_prefix,
    get_last_step,
    validate_selection_pipeline,
)
from tusk.sklearn._frames import backend_hint, collect_matrix, read_keys
from tusk.sklearn._lineage import Sentinels, make_sentinels
from tusk.synthesis import synthesize


class DFSTransformer(TransformerMixin, BaseEstimator):
    """Deep feature synthesis as a pipeline step.

    :meth:`fit` sets ``features_``, the synthesized definitions as a
    :class:`~tusk.FeatureList`, and ``database_``, the database it was given.
    :meth:`transform` computes those features for the keys in ``X``, returning
    one row per key in key order.
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

        Args:
            target_table: Table to build features for.
            agg_primitives: Aggregation primitives; None selects the defaults.
            trans_primitives: Transform primitives; None selects the defaults.
            groupby_trans_primitives: Transforms within foreign-key groups.
            max_depth: Maximum stacked primitive applications.
            cutoff_time: Only rows at or before this are visible.
            output_backend: Backend to collect the matrix to. None collects to
                the database's own backend.
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
        """Synthesize feature definitions from the database's schema.

        Reads no rows; :meth:`transform` does the computation.

        Args:
            X: Ignored. Synthesis depends only on the schema.
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
        self.database_ = database
        self.features_ = synthesize(
            database=database,
            target_table=self.target_table,
            agg_primitives=self.agg_primitives,
            trans_primitives=self.trans_primitives,
            groupby_trans_primitives=self.groupby_trans_primitives,
            max_depth=self.max_depth,
        )
        return self

    def transform(self, X: Any, database: Database | None = None) -> Any:
        """Compute the feature matrix for the keys in ``X``.

        Args:
            X: The target's primary key. Its order becomes the matrix's
                row order.
            database: The database, routed as metadata. When absent, the one
                seen at fit is used.

        Returns:
            An eager native frame, one row per key, in key order.
        """
        check_is_fitted(self, "features_")
        # scikit-learn's scorers call predict() with no metadata, so without
        # this fallback every cross-validated score would come back nan.
        db = self.database_ if database is None else database
        return collect_matrix(
            matrix=self.features_.apply(db, self.cutoff_time),
            primary_key=self._require_primary_key(db),
            keys=read_keys(X),
            output_backend=self.output_backend,
        )

    def fit_transform(
        self,
        X: Any,
        y: Any = None,
        database: Database | None = None,
        **kwargs: Any,
    ) -> Any:
        """Fit, then transform, passing ``database`` to both.

        Args:
            X: The target's primary key.
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
        return np.asarray(self.features_.output_names, dtype=object)

    def _require_primary_key(self, database: Database) -> str:
        """Return the target table's primary key.

        Args:
            database: The database to read the schema from.

        Returns:
            The primary primary key name.

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
    """DFS whose feature list is dropped to what a selector actually kept.

    The point is inference cost. A run that generates eight hundred features
    and keeps forty should compute forty when it next sees data, not eight
    hundred.

    Two column spaces meet here and must never be conflated: *tusk space*, the
    matrix, indexed by feature ``output_names``, and *encoded space*, the
    encoder's output, indexed by ``get_feature_names_out()``. The selector's
    mask indexes encoded space, pruning happens in tusk space, and sentinel
    lineage is the only bridge between them.

    Fitting adds ``encoder_``, the encoder prefix refitted on the kept
    columns; ``kept_names_``, the encoded-space names the selector chose, in
    encoder order; and ``sentinels_``, the renaming that recovers lineage.
    Like the parent's fitted attributes, none is declared at class level:
    all three are only known once :meth:`fit` has run.
    """

    def __init__(
        self,
        target_table: str,
        selection_pipeline: Any = None,
        agg_primitives: Iterable[str | Primitive] | None = None,
        trans_primitives: Iterable[str | Primitive] | None = None,
        groupby_trans_primitives: Iterable[str | Primitive] | None = None,
        max_depth: int = 2,
        cutoff_time: datetime | None = None,
        output_backend: str | None = None,
    ) -> None:
        """Configure synthesis and selection.

        Args:
            target_table: Table to build features for.
            selection_pipeline: An estimator ending in a ``SelectorMixin``;
                everything before it encodes.
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
        self.selection_pipeline = selection_pipeline

    def fit(
        self,
        X: Any,
        y: Any = None,
        database: Database | None = None,
    ) -> DFSSelectorTransformer:
        """Synthesize features, fit the selection pipeline, drop the rest.

        Args:
            X: The target's primary key.
            y: Training targets, passed to the selector.
            database: The database, routed as metadata.

        Returns:
            This estimator.

        Raises:
            SchemaError: If selection eliminated every feature.
            LineageError: If a kept column vanished from the refitted encoder,
                meaning lineage missed a source and a feature was wrongly
                dropped.

        Warns:
            LineageWarning: If any kept column's provenance was unrecoverable,
                in which case nothing is dropped.
            UnencodedFeatureWarning: If a feature fed no encoded column at
                all, so the encoder never gave the selector a chance to keep
                it.
        """
        validate_selection_pipeline(self.selection_pipeline)
        super().fit(X, y, database=database)
        db = self.database_ if database is None else database

        matrix = nw.from_native(super().transform(X, database=db), eager_only=True)
        sentinels = make_sentinels(list(matrix.columns))
        renamed = matrix.rename(sentinels.mapping)
        probe = renamed.to_native()

        # The selector is fitted once and then frozen; only the encoder is
        # refitted on the kept columns. Refitting the selector would let it
        # reach a different verdict -- SelectKBest(k=50) choosing 50 out of 60
        # need not choose the 50 it chose out of 500.
        selection_pipeline = clone(self.selection_pipeline)
        with backend_hint(probe):
            selection_pipeline.fit(probe, y)
        # Slicing gets the *pre*-selection names: calling
        # get_feature_names_out() on the whole pipeline reports what survived
        # selection, which cannot be zipped against the mask. The input names
        # are passed explicitly because a bare selector has no encoder to have
        # been fitted, and the identity standing in for one can only report
        # names if it is given them.
        encoded = list(
            get_encoder_prefix(selection_pipeline).get_feature_names_out(
                list(renamed.columns)
            ),
        )
        # strict=True: a mask that does not line up with the names is a broken
        # encoder contract, and zipping it short would silently mis-attribute
        # every column after the mismatch.
        mask = get_last_step(selection_pipeline).get_support()
        kept = [name for name, keep in zip(encoded, mask, strict=True) if keep]

        survivors = self._find_kept_features(kept, encoded, sentinels)
        if not survivors:
            raise SchemaError("feature selection eliminated every feature")
        self.features_ = FeatureList(survivors)

        kept_columns = list(self.features_.output_names)
        narrowed = matrix.select(kept_columns).rename(
            {c: sentinels.mapping[c] for c in kept_columns},
        )
        self.encoder_ = get_encoder_prefix(clone(self.selection_pipeline))
        with backend_hint(narrowed.to_native()):
            self.encoder_.fit(narrowed.to_native(), y)

        refit = list(self.encoder_.get_feature_names_out())
        missing = [n for n in kept if n not in set(refit)]
        if missing:
            raise LineageError(
                f"{len(missing)} selected columns vanished when the encoder was "
                f"refit on the kept features, e.g. "
                f"{[sentinels.restore(n) for n in missing[:3]]}. Lineage missed "
                "a source and dropped a feature that was still needed.",
            )
        self.sentinels_ = sentinels
        self.kept_names_ = [n for n in refit if n in set(kept)]
        return self

    def transform(self, X: Any, database: Database | None = None) -> Any:
        """Compute the kept features, encode them, apply the frozen mask.

        Args:
            X: The target's primary key.
            database: The database, routed as metadata.

        Returns:
            The encoded, selected matrix.
        """
        check_is_fitted(self, "kept_names_")
        db = self.database_ if database is None else database
        matrix = nw.from_native(super().transform(X, database=db), eager_only=True)
        kept_columns = list(self.features_.output_names)
        probe = matrix.select(kept_columns).rename(
            {c: self.sentinels_.mapping[c] for c in kept_columns},
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

    def _find_kept_features(
        self,
        kept: list[str],
        encoded: list[str],
        sentinels: Sentinels,
    ) -> list[Feature]:
        """Return the features with a column feeding a kept encoded column.

        A feature survives if any of its output columns is a source of a kept
        encoded column, and then contributes all of them.

        Args:
            kept: Encoded-space names the selector chose.
            encoded: Every encoded-space name, kept or not.
            sentinels: The renaming used to read provenance.

        Returns:
            The kept features, in their original order.

        Warns:
            LineageWarning: If any kept name mentions no sentinel, in which
                case every feature is kept.
            UnencodedFeatureWarning: If a matrix column fed no encoded column
                at all, so its feature is dropped for a reason the user may not
                have intended.
        """
        sources = {name: sentinels.sources(name) for name in kept}
        opaque = [name for name, s in sources.items() if not s]
        if opaque:
            warnings.warn(
                f"{len(opaque)} selected columns do not name their inputs "
                f"(e.g. {opaque[:3]}), so tusk cannot tell which features they "
                "came from and has dropped nothing. Selection still applies; "
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
                f"{unencoded[:3]}) and have been dropped. The encoder never "
                'looked at them -- a ColumnTransformer with remainder="drop" '
                "covering only some dtypes is the usual cause.",
                UnencodedFeatureWarning,
                stacklevel=3,
            )
        return [f for f in self.features_ if any(n in live for n in f.output_names)]

    def _select(self, encoded: Any) -> Any:
        """Return only the selected columns of the encoder's output.

        Args:
            encoded: Whatever the encoder returned -- a frame or an array.

        Returns:
            Only the selected columns, in encoder order.
        """
        # Resolved by name rather than by stored position: with
        # output_backend=None the output may be a frame or a bare array, and a
        # stored index would mis-slice if that changed between fit and
        # transform.
        order = list(self.encoder_.get_feature_names_out())
        indices = [order.index(n) for n in self.kept_names_]
        frame = nw.from_native(encoded, eager_only=True, pass_through=True)
        if isinstance(frame, nw.DataFrame):
            return frame.select([frame.columns[i] for i in indices]).to_native()
        return encoded[:, indices]
