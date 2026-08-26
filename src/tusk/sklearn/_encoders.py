"""What the user's ``inner`` pipeline must look like, and why.

Pruning narrows the matrix, so the encoder is refit on fewer columns than it
first saw. That makes indifference to which columns are present a hard
requirement rather than a nicety, and it is the reason a whole shape of
``ColumnTransformer`` is refused here.
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
    """Select columns by :class:`~tusk.dtypes.DtypeFamily`, on any backend.

    scikit-learn's ``make_column_selector`` is the right shape but rejects
    anything that is not pandas, which collides with collecting natively. skrub
    and scikit-lego were surveyed too: skrub's selectors are not callable and
    ``ColumnTransformer`` rejects them, and scikit-lego's ``TypeSelector`` is a
    transformer with no ``get_feature_names_out``, which lineage depends on.
    So this is the same idea through narwhals.

    The families are tusk's own rather than a new set, so ``"string"`` means
    here exactly what it means to a primitive: ``String`` and not
    ``Categorical`` or ``Enum``.

    Being a callable rather than a fixed list is what makes the refit work: it
    re-evaluates against whatever frame it is handed, so narrowing the matrix
    narrows the selection instead of breaking it.

    Attributes:
        family: The :class:`~tusk.dtypes.DtypeFamily` to select.
    """

    family: DtypeFamily

    def __init__(self, family: DtypeFamily | str) -> None:
        """Create a selector.

        Coercing through ``DtypeFamily`` here, rather than storing the raw
        argument, is what rejects an unknown family name eagerly -- with the
        enum's own ``ValueError`` and its list of valid values -- instead of
        failing later and further from the mistake.

        Args:
            family: A ``DtypeFamily`` or its string value, such as
                ``"numeric"`` or ``"string"``.
        """
        self.family = DtypeFamily(family)

    def __call__(self, X: Any) -> list[str]:
        """Return the matching column names.

        Args:
            X: The frame the encoder is being fitted on.

        Returns:
            Matching column names, in frame order.
        """
        schema = nw.from_native(X, eager_only=True).schema
        return [c for c, d in schema.items() if matches(d, self.family)]

    def __repr__(self) -> str:
        """Show the family, so cloned estimators print readably."""
        return f"dtype_selector({self.family.value!r})"


def selector_of(inner: Any) -> Any:
    """The selector at the end of ``inner``.

    Args:
        inner: The user's encode-and-select estimator.

    Returns:
        Its last step if it is a pipeline, else ``inner`` itself.
    """
    return inner[-1] if isinstance(inner, Pipeline) else inner


def encoder_prefix(inner: Any) -> Any:
    """Everything in ``inner`` before the selector.

    A bare selector, or a one-step pipeline wrapping one, has no encoder; an
    identity transformer stands in so callers need no special case. Slicing a
    one-step pipeline would give an empty one, which cannot be fitted.

    Args:
        inner: The user's encode-and-select estimator.

    Returns:
        An unfitted estimator producing encoded space from tusk space.
    """
    if isinstance(inner, Pipeline) and len(inner.steps) > 1:
        return inner[:-1]
    return FunctionTransformer(feature_names_out="one-to-one")


def validate_inner(inner: Any) -> None:
    """Check ``inner`` against the contract, before anything is fitted.

    Args:
        inner: The user's encode-and-select estimator.

    Raises:
        EncoderError: If it does not end in a ``SelectorMixin``, or if any
            ``ColumnTransformer`` within it names its columns explicitly.
    """
    if not isinstance(selector_of(inner), SelectorMixin):
        raise EncoderError(
            "inner must end in a scikit-learn SelectorMixin -- its get_support() "
            "mask is what tells tusk which features to keep; got "
            f"{type(selector_of(inner)).__name__}",
        )
    _reject_explicit_columns(inner)
    _require_feature_names(encoder_prefix(inner))


def _require_feature_names(estimator: Any) -> None:
    """Refuse an encoder step that cannot report its output names.

    Lineage reads ``get_feature_names_out`` and nothing else, so a transformer
    without it is wholly opaque. Checking each step rather than the pipeline is
    the point: ``Pipeline`` always *has* the attribute and only fails when it
    reaches the step that does not, deep inside the fit and far from the cause.
    scikit-lego's ``TypeSelector`` is a real example.

    Args:
        estimator: An estimator to inspect, recursively.

    Raises:
        EncoderError: If any step cannot report output names.
    """
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
    """Refuse a ``ColumnTransformer`` that names DFS-generated columns.

    Such an encoder is wrong here for three independent reasons: the names do
    not exist until DFS has run, so they must be hardcoded and go stale
    silently; ``remainder="drop"`` discards every generated feature the user
    did not name; and it is the one shape that cannot be refit on the pruned
    matrix.

    Args:
        estimator: An estimator to inspect, recursively.

    Raises:
        EncoderError: If an explicit column list is found.
    """
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
