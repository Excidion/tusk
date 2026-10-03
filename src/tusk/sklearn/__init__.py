"""scikit-learn estimators for deep feature synthesis.

:class:`DFSTransformer` runs synthesis as a pipeline step.
:class:`DFSSelectorTransformer` also drops the features a selector did not
keep. Later calls then compute only the rest.
:class:`TableEncoder` encodes each column by its narwhals dtype.
:class:`NarwhalsMixin` makes any estimator accept a table narwhals can read.
:class:`NarwhalsEncoder` converts a table to what ``set_output`` asks for.

This package needs the ``sklearn`` extra: ``pip install "tusk[sklearn]"``.
:mod:`tusk` does not import this package. Import it by name.
"""

from __future__ import annotations

from tusk.sklearn._column_encoders import (
    DateEncoder,
    DatetimeEncoder,
    DurationEncoder,
    EnumOrdinalEncoder,
    TemporalEncoder,
    TfIdfSvdEncoder,
    TimeEncoder,
)
from tusk.sklearn._narwhals import NarwhalsEncoder, NarwhalsMixin
from tusk.sklearn._table_encoder import TableEncoder
from tusk.sklearn.transformers import DFSSelectorTransformer, DFSTransformer

__all__ = [
    "DFSSelectorTransformer",
    "DFSTransformer",
    "DateEncoder",
    "DatetimeEncoder",
    "DurationEncoder",
    "EnumOrdinalEncoder",
    "NarwhalsEncoder",
    "NarwhalsMixin",
    "TableEncoder",
    "TemporalEncoder",
    "TfIdfSvdEncoder",
    "TimeEncoder",
]
