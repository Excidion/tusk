"""Estimators that accept any table narwhals can read.

:class:`NarwhalsMixin` makes a scikit-learn estimator accept such a table.
:class:`NarwhalsEncoder` is the base of the tusk encoders. Used alone, it
converts a table to what ``set_output`` asks for.
"""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any, ClassVar, Literal

import narwhals as nw
import numpy as np
from sklearn import get_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from tusk.exceptions import EncoderError, TuskError

ConvertTo = Literal["narwhals", "numpy", "pandas", "polars"]
OUTPUTS = ("default", "pandas", "polars")


class NarwhalsMixin:
    """A mixin that lets an estimator accept any table narwhals can read.

    Mix it in before the estimator's class. The class attribute
    ``convert_to`` sets what the estimator receives: ``"numpy"``,
    ``"pandas"``, ``"polars"`` or ``"narwhals"``. Input that is not a table,
    such as a numpy array, reaches the estimator unchanged.

    Fitting sets ``narwhals_columns_``, the column names of the table.
    :meth:`get_feature_names_out` uses them when it is given no names.

    Attributes:
        convert_to: What the estimator receives.
    """

    convert_to: ClassVar[ConvertTo] = "numpy"

    def fit(self, X: Any, y: Any = None, **params: Any) -> Any:
        """Fit the estimator on ``X`` converted to ``convert_to``.

        Args:
            X: A table narwhals can read, eager or lazy.
            y: Targets, passed on unchanged.
            **params: Fit parameters, passed on unchanged.

        Returns:
            This estimator.
        """
        self._record_columns(X)
        return super().fit(self._convert(X), y, **params)  # ty: ignore[unresolved-attribute]

    def transform(self, X: Any, **params: Any) -> Any:
        """Transform ``X`` converted to ``convert_to``.

        Args:
            X: A table narwhals can read, eager or lazy.
            **params: Transform parameters, passed on unchanged.

        Returns:
            The estimator's output.
        """
        return super().transform(self._convert(X), **params)  # ty: ignore[unresolved-attribute]

    def fit_transform(self, X: Any, y: Any = None, **params: Any) -> Any:
        """Fit and transform ``X`` converted to ``convert_to``.

        Args:
            X: A table narwhals can read, eager or lazy.
            y: Targets, passed on unchanged.
            **params: Fit parameters, passed on unchanged.

        Returns:
            The estimator's output.
        """
        self._record_columns(X)
        return super().fit_transform(self._convert(X), y, **params)  # ty: ignore[unresolved-attribute]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return the output names. Input names default to those seen at fit.

        Args:
            input_features: Input column names. None uses
                ``narwhals_columns_``.

        Returns:
            The output names.
        """
        if input_features is None:
            input_features = getattr(self, "narwhals_columns_", None)
        return super().get_feature_names_out(input_features)  # ty: ignore[unresolved-attribute]

    def _record_columns(self, X: Any) -> None:
        """Store the column names of ``X`` if it is a table.

        Args:
            X: The input to fit.
        """
        # TransformerMixin.fit_transform calls fit again on the converted
        # input. A numpy array carries no names, so it must not overwrite them.
        table = nw.from_native(X, pass_through=True)
        if isinstance(table, nw.DataFrame | nw.LazyFrame):
            self.narwhals_columns_ = list(table.collect_schema().names())

    def _convert(self, X: Any) -> Any:
        """Return ``X`` as ``convert_to``.

        Args:
            X: The input.

        Returns:
            The converted table, or ``X`` unchanged if it is not a table.
        """
        return convert_table(X, self.convert_to, type(self).__name__)


class NarwhalsEncoder(
    NarwhalsMixin,
    TransformerMixin,
    BaseEstimator,
    auto_wrap_output_keys=None,
):
    """An encoder that works on narwhals tables, on any backend.

    Used alone, it passes every column through unchanged. ``set_output``
    sets what it returns: numpy, pandas or polars. It is then the step that
    hands a feature matrix to scikit-learn.

    :meth:`fit` reads the schema and does not collect. It sets
    ``schema_in_``, column name to narwhals dtype, ``feature_names_in_`` and
    ``n_features_in_``. Subclasses override :meth:`_fit`,
    :meth:`_expressions`, :meth:`_output_names` and, if they need values,
    :meth:`_transform`.

    Attributes:
        convert_to: ``"narwhals"``. The encoder receives a narwhals table.
    """

    convert_to: ClassVar[ConvertTo] = "narwhals"
    _fits_on_schema: ClassVar[bool] = True

    def fit(self, X: Any, y: Any = None, **params: Any) -> NarwhalsEncoder:
        """Read the schema of ``X`` and fit the encoder.

        Args:
            X: A table narwhals can read, eager or lazy.
            y: Targets, passed to :meth:`_fit`.
            **params: Ignored. Present for the scikit-learn signature.

        Returns:
            This encoder.
        """
        table = read_table(X, type(self).__name__)
        self.schema_in_ = dict(table.collect_schema())
        self.feature_names_in_ = np.asarray(list(self.schema_in_), dtype=object)
        self.n_features_in_ = len(self.schema_in_)
        self._fit(table, y)
        return self

    def transform(self, X: Any, **params: Any) -> Any:
        """Encode ``X`` and return it as ``set_output`` asks.

        Args:
            X: A table narwhals can read, eager or lazy, with the columns and
                dtypes seen at fit.
            **params: Ignored. Present for the scikit-learn signature.

        Returns:
            The encoded table as numpy, pandas or polars.
        """
        check_is_fitted(self, "schema_in_")
        table = read_table(X, type(self).__name__)
        reject_changed_schema(self.schema_in_, dict(table.collect_schema()))
        return self._convert_output(collect(self._transform(table)))

    def fit_transform(self, X: Any, y: Any = None, **params: Any) -> Any:
        """Fit on ``X``, then encode it.

        Args:
            X: A table narwhals can read, eager or lazy.
            y: Targets, passed to :meth:`fit`.
            **params: Ignored. Present for the scikit-learn signature.

        Returns:
            The encoded table as numpy, pandas or polars.
        """
        return self.fit(X, y).transform(X)

    def set_output(self, *, transform: str | None = None) -> NarwhalsEncoder:
        """Set what :meth:`transform` returns.

        Args:
            transform: ``"default"`` for numpy, ``"pandas"`` or ``"polars"``.
                None leaves the setting unchanged.

        Returns:
            This encoder.

        Raises:
            ValueError: If ``transform`` is not one of the accepted values.
        """
        if transform is None:
            return self
        if transform not in OUTPUTS:
            raise ValueError(
                f"set_output(transform={transform!r}) is not supported; choose "
                f"from {list(OUTPUTS)}",
            )
        if transform != "default":
            require_package(transform, type(self).__name__)
        # This attribute name is the one sklearn.base.clone copies, so a
        # cloned encoder keeps its output setting.
        self._sklearn_output_config = {"transform": transform}
        return self

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return the output column names.

        Args:
            input_features: Ignored. The names seen at fit are used.

        Returns:
            The output names, as an object array.
        """
        check_is_fitted(self, "schema_in_")
        return np.asarray(self._output_names(), dtype=object)

    def _fit(self, table: nw.DataFrame | nw.LazyFrame, y: Any) -> None:
        """Learn what the encoder needs. The default learns nothing.

        Args:
            table: The input table, eager or lazy.
            y: Targets.
        """

    def _expressions(self) -> list[nw.Expr]:
        """Return one expression per output column, in output order.

        Returns:
            The expressions. The default selects every input column.
        """
        return [nw.col(name) for name in self.schema_in_]

    def _output_names(self) -> list[str]:
        """Return the output column names, in output order.

        Returns:
            The names. The default is the input column names.
        """
        return list(self.schema_in_)

    def _transform(
        self, table: nw.DataFrame | nw.LazyFrame
    ) -> nw.DataFrame | nw.LazyFrame:
        """Encode the table.

        Args:
            table: The input table, eager or lazy.

        Returns:
            The encoded table. The default selects :meth:`_expressions`.
        """
        return table.select(self._expressions())

    def _convert_output(self, table: nw.DataFrame) -> Any:
        """Return ``table`` as ``set_output`` asks.

        Args:
            table: The encoded table.

        Returns:
            A numpy array, a pandas DataFrame or a polars DataFrame.
        """
        output = getattr(self, "_sklearn_output_config", {}).get(
            "transform", get_config()["transform_output"]
        )
        if output == "pandas":
            return table.to_pandas()
        if output == "polars":
            return table.to_polars()
        return table.to_numpy()


def convert_table(X: Any, convert_to: ConvertTo, owner: str) -> Any:
    """Return ``X`` as ``convert_to``. Input that is not a table is unchanged.

    Args:
        X: The input.
        convert_to: What to convert to.
        owner: The estimator's class name, for the error message.

    Returns:
        The converted table, or ``X`` itself.
    """
    table = nw.from_native(X, pass_through=True)
    if not isinstance(table, nw.DataFrame | nw.LazyFrame):
        return X
    if convert_to == "narwhals":
        return table
    eager = collect(table)
    if convert_to == "numpy":
        return eager.to_numpy()
    require_package(convert_to, owner)
    return eager.to_pandas() if convert_to == "pandas" else eager.to_polars()


def read_table(X: Any, owner: str) -> nw.DataFrame | nw.LazyFrame:
    """Return ``X`` as a narwhals table.

    Args:
        X: The input.
        owner: The encoder's class name, for the error message.

    Returns:
        The table, eager or lazy.

    Raises:
        TypeError: If narwhals cannot read ``X``.
    """
    table = nw.from_native(X, pass_through=True)
    if not isinstance(table, nw.DataFrame | nw.LazyFrame):
        raise TypeError(
            f"{owner} takes a table narwhals can read, such as a polars, "
            f"pandas or pyarrow table; got {type(X).__name__}",
        )
    return table


def collect(table: nw.DataFrame | nw.LazyFrame) -> nw.DataFrame:
    """Return ``table`` eager, collecting it if it is lazy.

    Args:
        table: The table.

    Returns:
        The eager table.
    """
    return table.collect() if isinstance(table, nw.LazyFrame) else table


def require_package(package: str, owner: str) -> None:
    """Raise unless ``package`` is installed.

    Args:
        package: ``"pandas"`` or ``"polars"``.
        owner: The class name that needs it, for the error message.

    Raises:
        TuskError: If the package is not installed.
    """
    if find_spec(package) is None:
        raise TuskError(
            f"{owner} converts to {package}, which is not installed; "
            f"`uv add {package}`",
        )


def reject_changed_schema(
    fitted: dict[str, nw.dtypes.DType], given: dict[str, nw.dtypes.DType]
) -> None:
    """Raise if ``given`` has other columns or dtypes than ``fitted``.

    Args:
        fitted: The schema seen at fit.
        given: The schema seen now.

    Raises:
        EncoderError: If a column was added or removed, or changed dtype.
    """
    extra = [name for name in given if name not in fitted]
    absent = [name for name in fitted if name not in given]
    if extra or absent:
        raise EncoderError(
            f"the table has other columns than at fit: extra {extra[:5]}, "
            f"absent {absent[:5]}",
        )
    changed = [name for name in fitted if given[name] != fitted[name]]
    if changed:
        name = changed[0]
        raise EncoderError(
            f"column {name!r} was {fitted[name]} at fit and is {given[name]} "
            f"now; {len(changed)} columns changed dtype",
        )
