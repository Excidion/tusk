"""scikit-learn estimators for deep feature synthesis.

[`DFSTransformer`][tusk.sklearn.DFSTransformer] runs synthesis as a
pipeline step.
[`DFSSelectorTransformer`][tusk.sklearn.DFSSelectorTransformer] also drops
the features a selector did not keep. Later calls then compute only the rest.
[`TableEncoder`][tusk.sklearn.TableEncoder] encodes each column by its
narwhals dtype. [`ConvertInputMixin`][tusk.sklearn.ConvertInputMixin] makes
any estimator accept a table narwhals can read.
[`TableConverter`][tusk.sklearn.TableConverter] converts a table to numpy,
pandas or polars.

This package needs the `sklearn` extra: `pip install "tusk[sklearn]"`.
[`tusk`][tusk] does not import this package. Import it by name.
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
from tusk.sklearn._transformers import DFSSelectorTransformer, DFSTransformer

__all__ = [
    "DFSTransformer",
    "DFSSelectorTransformer",
    "TableEncoder",
    "EnumOrdinalEncoder",
    "TemporalEncoder",
    "DateEncoder",
    "TimeEncoder",
    "DatetimeEncoder",
    "DurationEncoder",
    "TfIdfSvdEncoder",
    "ConvertInputMixin",
    "TableConverter",
]
