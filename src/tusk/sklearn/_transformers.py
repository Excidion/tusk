"""Deep feature synthesis as scikit-learn estimators.

``X`` is the target table's primary-key column, not the database. That
inversion is what makes ``X`` array-like, so scikit-learn can split it: the
keys *are* the rows, which makes misalignment against ``y`` impossible rather
than merely discouraged, and lets ``cross_val_score`` and ``GridSearchCV``
work at all. The database travels as routed metadata, because ``clone``
deep-copies constructor parameters and a duckdb relation cannot be pickled.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from tusk.api import apply_features
from tusk.database import Database
from tusk.exceptions import SchemaError
from tusk.primitives.aggregation import AGG_DEFAULTS
from tusk.primitives.base import Primitive
from tusk.primitives.transform import TRANS_DEFAULTS
from tusk.sklearn._frames import as_keys, collect_matrix
from tusk.synthesis import synthesize


class DFSTransformer(BaseEstimator, TransformerMixin):
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
