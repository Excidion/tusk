"""scikit-learn estimators for deep feature synthesis.

:class:`DFSTransformer` runs synthesis as a pipeline step.
:class:`DFSSelectorTransformer` also drops the features a selector did not
keep. Later calls then compute only the rest.
:class:`TableEncoder` encodes each column by its narwhals dtype.
:class:`ConvertInputMixin` makes any estimator accept a table narwhals can read.
:class:`TableConverter` converts a table to numpy, pandas or polars.

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
from tusk.sklearn._narwhals import ConvertInputMixin, TableConverter
from tusk.sklearn._table_encoder import TableEncoder
from tusk.sklearn.transformers import DFSSelectorTransformer, DFSTransformer

__all__ = [
    "ConvertInputMixin",
    "DFSSelectorTransformer",
    "DFSTransformer",
    "DateEncoder",
    "DatetimeEncoder",
    "DurationEncoder",
    "EnumOrdinalEncoder",
    "TableConverter",
    "TableEncoder",
    "TemporalEncoder",
    "TfIdfSvdEncoder",
    "TimeEncoder",
]
