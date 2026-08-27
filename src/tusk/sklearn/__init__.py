"""scikit-learn estimators for deep feature synthesis.

:class:`DFSTransformer` runs synthesis as a pipeline step.
:class:`DFSSelectorTransformer` additionally drops the features a selector
did not keep, so later calls compute only the rest.
:class:`dtype_selector` picks columns by dtype for a ``ColumnTransformer``.

Requires the ``sklearn`` extra: ``pip install "tusk[sklearn]"``. :mod:`tusk`
does not import this package, so it must be imported by name.
"""

from __future__ import annotations

from tusk.sklearn._encoders import dtype_selector
from tusk.sklearn.transformers import DFSSelectorTransformer, DFSTransformer

__all__ = ["DFSSelectorTransformer", "DFSTransformer", "dtype_selector"]
