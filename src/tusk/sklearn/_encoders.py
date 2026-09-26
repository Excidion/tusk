"""Inspection and validation of a selection pipeline.

:func:`get_last_step` and :func:`get_encoder_prefix` split a pipeline into the
part that encodes and the selector that ends it.
:func:`validate_selection_pipeline` rejects pipelines this module cannot
support. :class:`dtype_selector` picks columns by dtype for a
``ColumnTransformer``.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from tusk.dtypes import DtypeFamily, matches
from tusk.exceptions import EncoderError


class dtype_selector:  # noqa: N801
    """A selector of columns by :class:`~tusk.dtypes.DtypeFamily`, on any backend.

    It serves the same role as scikit-learn's ``make_column_selector``. It
    reads the schema through narwhals. This lets it work on every backend a
    tusk database can use, not pandas alone.

    Families are :class:`~tusk.dtypes.DtypeFamily` values. ``"string"`` here
    means what it means to a primitive: ``String``, not ``Categorical`` or
    ``Enum``.

    It is callable. Each call re-evaluates the family match against the
    table it is given. A narrowed feature matrix then narrows the selection.

    Attributes:
        family: The :class:`~tusk.dtypes.DtypeFamily` to select.
    """

    family: DtypeFamily

    def __init__(self, family: DtypeFamily | str) -> None:
        """Build a selector for one dtype family.

        Args:
            family: A ``DtypeFamily`` or its string value, such as
                ``"numeric"`` or ``"string"``. An unrecognized string raises
                ``ValueError`` listing the valid values.
        """
        self.family = DtypeFamily(family)

    def __call__(self, X: Any) -> list[str]:
        """Return the matching column names.

        Args:
            X: The table the encoder is being fitted on.

        Returns:
            Matching column names, in table order.
        """
        schema = nw.from_native(X, eager_only=True).schema
        return [c for c, d in schema.items() if matches(d, self.family)]

    def __repr__(self) -> str:
        """Show the family value. A cloned estimator then prints readably."""
        return f"dtype_selector({self.family.value!r})"


def get_last_step(selection_pipeline: Any) -> Any:
    """Return the final step of ``selection_pipeline``.

    Args:
        selection_pipeline: The user's encode-and-select estimator.

    Returns:
        step: Its last step if it is a pipeline, else ``selection_pipeline``
            itself. After :func:`validate_selection_pipeline` runs, this is
            the selector. This function does not check that.
    """
    return (
        selection_pipeline[-1]
        if isinstance(selection_pipeline, Pipeline)
        else selection_pipeline
    )


def get_encoder_prefix(selection_pipeline: Any) -> Any:
    """Return everything in ``selection_pipeline`` before its final step.

    A bare selector, or a one-step pipeline wrapping one, has no encoder. An
    identity transformer stands in for it.

    Args:
        selection_pipeline: The user's encode-and-select estimator.

    Returns:
        An unfitted estimator that produces encoded space from tusk space.
    """
    # Slicing a one-step pipeline yields an empty one, which cannot be fitted,
    # hence the length check rather than an unconditional [:-1].
    if isinstance(selection_pipeline, Pipeline) and len(selection_pipeline.steps) > 1:
        return selection_pipeline[:-1]
    return FunctionTransformer(feature_names_out="one-to-one")


def validate_selection_pipeline(selection_pipeline: Any) -> None:
    """Raise unless ``selection_pipeline`` can be used for selection.

    Args:
        selection_pipeline: The user's encode-and-select estimator.

    Raises:
        EncoderError: If it does not end in a ``SelectorMixin``. If any
            ``ColumnTransformer`` within it names its columns explicitly.
    """
    if not isinstance(get_last_step(selection_pipeline), SelectorMixin):
        raise EncoderError(
            "selection_pipeline must end in a scikit-learn SelectorMixin; its "
            "get_support() mask is what tells tusk which features to keep. Got "
            f"{type(get_last_step(selection_pipeline)).__name__}",
        )
    _reject_explicit_columns(selection_pipeline)
    _require_feature_names(get_encoder_prefix(selection_pipeline))


def _require_feature_names(estimator: Any) -> None:
    """Raise unless every step can report its output names.

    Args:
        estimator: An estimator to inspect, recursively.

    Raises:
        EncoderError: If any step cannot report output names.
    """
    # Each step is checked rather than the pipeline as a whole: Pipeline always
    # has get_feature_names_out and only fails on reaching the step that does
    # not, deep inside the fit and far from the cause.
    if isinstance(estimator, Pipeline):
        for _, step in estimator.steps:
            _require_feature_names(step)
        return
    if isinstance(estimator, ColumnTransformer):
        for _, transformer, _columns in estimator.transformers:
            if transformer not in ("drop", "passthrough"):
                _require_feature_names(transformer)
        return
    if not hasattr(estimator, "get_feature_names_out"):
        raise EncoderError(
            f"{type(estimator).__name__} has no get_feature_names_out(), so "
            "tusk cannot tell which features its output columns came from. "
            "Every step before the selector must implement it.",
        )


def _reject_explicit_columns(estimator: Any) -> None:
    """Raise if a ``ColumnTransformer`` names its columns explicitly.

    Args:
        estimator: An estimator to inspect, recursively.

    Raises:
        EncoderError: If an explicit column list is found.
    """
    # An explicit list cannot be refit once selection narrows the matrix, and
    # DFS generates its column names, so they cannot be written down in advance
    # without going stale.
    if isinstance(estimator, Pipeline):
        for _, step in estimator.steps:
            _reject_explicit_columns(step)
    elif isinstance(estimator, ColumnTransformer):
        for name, transformer, columns in estimator.transformers:
            if not callable(columns):
                raise EncoderError(
                    f"ColumnTransformer step {name!r} names its columns "
                    f"explicitly ({list(columns)[:3]}...), which cannot be "
                    "refit once selection narrows the matrix, and DFS "
                    "generates its column names so they cannot be known in "
                    "advance anyway. Use a callable instead, such as "
                    "tusk.sklearn.dtype_selector('numeric') or "
                    "dtype_selector('string').",
                )
            _reject_explicit_columns(transformer)
