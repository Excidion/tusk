"""scikit-learn estimators wrapping deep feature synthesis.

Deliberately not imported by :mod:`tusk`: importing scikit-learn eagerly would
break every install without the ``sklearn`` extra. Reach these through
``from tusk.sklearn import DFSTransformer``, as with ``xgboost.sklearn``.
"""

from __future__ import annotations

from tusk.sklearn._encoders import dtype_selector
from tusk.sklearn._transformers import DFSSelectorTransformer, DFSTransformer

__all__ = ["DFSSelectorTransformer", "DFSTransformer", "dtype_selector"]
