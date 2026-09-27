"""scikit-learn estimators for deep feature synthesis.

:class:`DFSTransformer` runs synthesis as a pipeline step.
:class:`DFSSelectorTransformer` also drops the features a selector did not
keep. Later calls then compute only the rest.
:class:`dtype_selector` picks columns by dtype for a ``ColumnTransformer``.

This package needs the ``sklearn`` extra: ``pip install "tusk[sklearn]"``.
:mod:`tusk` does not import this package. Import it by name.
"""

from __future__ import annotations

from tusk.sklearn._encoders import dtype_selector
from tusk.sklearn.transformers import DFSSelectorTransformer, DFSTransformer

__all__ = ["DFSSelectorTransformer", "DFSTransformer", "dtype_selector"]
