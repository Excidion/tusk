"""scikit-learn estimators wrapping deep feature synthesis.

Deliberately not imported by :mod:`tusk`: importing scikit-learn eagerly would
break every install without the ``sklearn`` extra. Reach these through
``from tusk.sklearn import DFSTransformer``, as with ``xgboost.sklearn``.
"""

from __future__ import annotations

__all__: list[str] = []
