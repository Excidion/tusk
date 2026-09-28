# TableEncoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `dtype_selector` with `TableEncoder`, which encodes every column of a feature matrix by its narwhals dtype, on any backend.

**Architecture:** `NarwhalsMixin` makes any scikit-learn estimator accept a table narwhals can read and converts it to the estimator's `convert_to`. `NarwhalsEncoder` is the base of tusk's encoders: it fits on the schema alone and converts its output through narwhals for `set_output`. The column encoders (`EnumEncoder`, the `TemporalEncoder` family, `StringEncoder`) and `TableEncoder` subclass it. `TableEncoder` splits the columns into ten disjoint dtype groups, builds one `select` on the native table and collects it once.

**Tech Stack:** Python ≥ 3.10, narwhals ≥ 2.24, scikit-learn ≥ 1.4, pytest; polars, pandas, pyarrow and duckdb in tests only.

**Spec:** `agents/superpowers/specs/2026-09-28-table-encoder-design.md`

## Global Constraints

- Work on branch `table-encoder`. Do not push. Do not merge.
- Every module keeps `from __future__ import annotations` and Google-style docstrings; `just lint` (ruff, ruff format, ty, interrogate at 100 %, pydoclint) passes after every task.
- Use the terms in `agents/GLOSSARY.md`: "table" (not frame), "null", "build", "compute", "column", "row", "backend".
- Comments say why, never what. Docstrings say what, not why, except on exception and warning classes.
- Do not make a public class private, or a planned public class private, without the maintainer's consent. The public classes are `NarwhalsMixin`, `NarwhalsEncoder`, `EnumEncoder`, `TemporalEncoder`, `DateEncoder`, `TimeEncoder`, `DatetimeEncoder`, `DurationEncoder`, `StringEncoder`, `TableEncoder`.
- No new runtime dependency. pandas and polars stay optional; a missing one raises `TuskError` ending in `` `uv add <package>` ``.
- Default components: `DateEncoder` `("month", "day")`, `TimeEncoder` `("hour", "minute")`, `DatetimeEncoder` `("month", "day", "hour", "minute")`, `DurationEncoder` `("total_seconds",)`. `StringEncoder` `n_components=30`.
- Group order and output names `{group}__{name}` are part of the contract: `numeric`, `boolean`, `string`, `categorical`, `enum`, `date`, `time`, `datetime`, `duration`, `other`.

## File Structure

| File | Responsibility |
| --- | --- |
| `src/tusk/sklearn/_narwhals.py` (new) | `NarwhalsMixin`, `NarwhalsEncoder`, conversion helpers |
| `src/tusk/sklearn/_column_encoders.py` (new) | `EnumEncoder`, `TemporalEncoder` and its four subclasses, `StringEncoder` |
| `src/tusk/sklearn/_table_encoder.py` (new) | `TableEncoder`, the dtype groups, dispatch helpers |
| `src/tusk/sklearn/__init__.py` | exports |
| `src/tusk/sklearn/_encoders.py` | loses `dtype_selector`; its error message names `TableEncoder` |
| `src/tusk/dtypes.py` | docstring loses its `dtype_selector` sentence |
| `tests/test_sklearn_narwhals.py` (new) | mixin and base encoder |
| `tests/test_sklearn_column_encoders.py` (new) | column encoders |
| `tests/test_sklearn_table_encoder.py` (new) | `TableEncoder`, including inside `DFSSelectorTransformer` |
| `tests/test_sklearn_encoders.py`, `tests/test_sklearn_selector_transformer.py`, `tests/test_sklearn_dfs_transformer.py` | migrate from `dtype_selector` to `TableEncoder` |
| `docs/guide/sklearn.md` | the `TableEncoder` section replaces the `dtype_selector` one |

The code below was run before this plan was written: all hooks of `just lint` pass, and `just test` passes (863 passed, 12 skipped) on scikit-learn 1.6.1. The sklearn test files also pass on scikit-learn 1.4.2.

---

### Task 1: `NarwhalsMixin` and `NarwhalsEncoder`

**Files:**
- Create: `src/tusk/sklearn/_narwhals.py`
- Modify: `src/tusk/sklearn/__init__.py`
- Test: `tests/test_sklearn_narwhals.py`

**Interfaces:**
- Consumes: `tusk.exceptions.EncoderError`, `tusk.exceptions.TuskError`.
- Produces:
  - `ConvertTo = Literal["narwhals", "numpy", "pandas", "polars"]`
  - `class NarwhalsMixin` with `convert_to: ClassVar[ConvertTo] = "numpy"`; overrides `fit`, `transform`, `fit_transform`, `get_feature_names_out`; sets `narwhals_columns_`.
  - `class NarwhalsEncoder(NarwhalsMixin, TransformerMixin, BaseEstimator, auto_wrap_output_keys=None)` with `convert_to = "narwhals"`, `_fits_on_schema: ClassVar[bool] = True`; `fit` sets `schema_in_: dict[str, DType]`, `feature_names_in_`, `n_features_in_`; hooks `_fit(table, y) -> None`, `_expressions() -> list[nw.Expr]`, `_output_names() -> list[str]`, `_transform(table) -> nw.DataFrame | nw.LazyFrame`; `set_output(*, transform)`.
  - Module functions `convert_table(X, convert_to, owner)`, `read_table(X, owner)`, `collect(table) -> nw.DataFrame`, `require_package(package, owner)`, `reject_changed_schema(fitted, given)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sklearn_narwhals.py`:

```python
import warnings

import duckdb
import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

import tusk
from tusk.exceptions import EncoderError, TuskError
from tusk.sklearn import DFSTransformer, NarwhalsEncoder, NarwhalsMixin, _narwhals

NUMBERS = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 9.0]})


class NarwhalsScaler(NarwhalsMixin, StandardScaler):
    pass


class PandasTargetEncoder(NarwhalsMixin, TargetEncoder):
    convert_to = "pandas"


class TypeRecorder(TransformerMixin, BaseEstimator):
    """Records the type of what it is fitted on."""

    def fit(self, X, y=None):
        self.received_ = type(X)
        return self

    def transform(self, X):
        return X


def as_duckdb(frame):
    connection = duckdb.connect()
    connection.register("t", frame.to_arrow())
    return connection.sql("select * from t")


@pytest.mark.parametrize("backend", ["polars", "pandas", "pyarrow", "duckdb"])
def test_the_mixin_accepts_any_backend(backend):
    native = {
        "polars": lambda: NUMBERS,
        "pandas": NUMBERS.to_pandas,
        "pyarrow": NUMBERS.to_arrow,
        "duckdb": lambda: as_duckdb(NUMBERS),
    }[backend]()
    scaler = NarwhalsScaler().fit(native)
    assert list(scaler.get_feature_names_out()) == ["a", "b"]
    np.testing.assert_allclose(
        scaler.transform(native).mean(axis=0), [0.0, 0.0], atol=1e-12
    )


@pytest.mark.parametrize(
    ("convert_to", "expected"),
    [
        ("numpy", np.ndarray),
        ("pandas", pd.DataFrame),
        ("polars", pl.DataFrame),
        ("narwhals", nw.LazyFrame),
    ],
)
def test_convert_to_sets_what_the_estimator_receives(convert_to, expected):
    recorder = type("Recorder", (NarwhalsMixin, TypeRecorder), {})
    recorder.convert_to = convert_to
    assert recorder().fit(as_duckdb(NUMBERS)).received_ is expected


def test_a_numpy_fit_then_transform_issues_no_feature_name_warning():
    scaler = NarwhalsScaler().fit(NUMBERS)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scaler.transform(NUMBERS)


def test_an_estimator_with_its_own_fit_transform_receives_the_conversion():
    frame = pl.DataFrame({"c": ["x", "y"] * 10})
    y = [0, 1] * 10
    assert PandasTargetEncoder().fit_transform(frame, y).shape == (20, 1)
    fitted = PandasTargetEncoder().fit(frame, y)
    assert list(fitted.get_feature_names_out()) == ["c"]


def test_set_output_pandas_names_the_columns():
    out = NarwhalsScaler().set_output(transform="pandas").fit_transform(NUMBERS)
    assert list(out.columns) == ["a", "b"]


def test_clone_keeps_convert_to():
    assert clone(PandasTargetEncoder()).convert_to == "pandas"


def test_an_absent_package_names_the_install_command(monkeypatch):
    monkeypatch.setattr(_narwhals, "find_spec", lambda name: None)
    with pytest.raises(TuskError, match="uv add pandas"):
        PandasTargetEncoder().fit(pl.DataFrame({"c": ["x", "y"]}), [0, 1])


def test_the_encoder_fits_a_lazy_table_without_collecting(monkeypatch):
    relation = as_duckdb(NUMBERS)

    def refuse(*args, **kwargs):
        raise AssertionError("fit collected the table")

    monkeypatch.setattr(nw.LazyFrame, "collect", refuse)
    encoder = NarwhalsEncoder().fit(relation)
    assert list(encoder.feature_names_in_) == ["a", "b"]


def test_the_encoder_returns_what_set_output_asks_for():
    categories = pl.DataFrame({"c": pl.Series(["x", "y"]).cast(pl.Categorical)})
    assert isinstance(NarwhalsEncoder().fit_transform(categories), np.ndarray)
    as_pandas = NarwhalsEncoder().set_output(transform="pandas")
    assert str(as_pandas.fit_transform(categories)["c"].dtype) == "category"
    as_polars = NarwhalsEncoder().set_output(transform="polars")
    assert isinstance(as_polars.fit_transform(as_duckdb(NUMBERS)), pl.DataFrame)


def test_clone_keeps_the_output_setting():
    encoder = NarwhalsEncoder().set_output(transform="polars").fit(NUMBERS)
    assert isinstance(clone(encoder).fit_transform(NUMBERS), pl.DataFrame)


def test_set_output_rejects_an_unsupported_value():
    with pytest.raises(ValueError, match="choose from"):
        NarwhalsEncoder().set_output(transform="pyarrow")


def test_the_encoder_rejects_other_columns_or_dtypes():
    encoder = NarwhalsEncoder().fit(NUMBERS)
    with pytest.raises(EncoderError, match="absent"):
        encoder.transform(NUMBERS.select("a"))
    with pytest.raises(EncoderError, match="'a'"):
        encoder.transform(NUMBERS.with_columns(pl.col("a").cast(pl.Int64)))


def test_the_encoder_is_a_convert_step_between_dfs_and_a_model():
    customers = duckdb.sql(
        "select * from (values "
        "(1, 30.0, timestamp '2024-01-01'), "
        "(2, 40.0, timestamp '2024-01-01'), "
        "(3, 50.0, timestamp '2024-01-01'), "
        "(4, 60.0, timestamp '2024-01-01')) t(id, age, signed_up_at)",
    )
    database = tusk.Database("shop").add_table(
        "customers", customers, primary_key="id", row_creation_time="signed_up_at"
    )
    received = []

    class Model(LogisticRegression):
        def fit(self, X, y, **params):
            received.append(X)
            return super().fit(X, y, **params)

    with sklearn.config_context(enable_metadata_routing=True):
        pipeline = Pipeline(
            [
                ("dfs", DFSTransformer(target_table="customers")),
                ("convert", NarwhalsEncoder().set_output(transform="pandas")),
                ("model", Model()),
            ]
        )
        pipeline.fit([1, 2, 3, 4], [0, 0, 1, 1], database=database)
    assert isinstance(received[0], pd.DataFrame)
    assert received[0]["age"].tolist() == [30.0, 40.0, 50.0, 60.0]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_sklearn_narwhals.py -q`
Expected: collection error, `ImportError: cannot import name 'NarwhalsEncoder' from 'tusk.sklearn'`.

- [ ] **Step 3: Write the implementation**

Create `src/tusk/sklearn/_narwhals.py`:

```python
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
```

Why the non-obvious parts are there:
- `auto_wrap_output_keys=None` turns off scikit-learn's output wrapper, so `NarwhalsEncoder` converts through narwhals and a polars `Categorical` becomes a pandas `category`, not an object array. Its own `set_output` stores the setting under `_sklearn_output_config`, the attribute `sklearn.base.clone` copies.
- The `# ty: ignore[unresolved-attribute]` comments are needed: ty cannot see that a mixin's `super()` has `fit`.

Replace `src/tusk/sklearn/__init__.py` with:

```python
"""scikit-learn estimators for deep feature synthesis.

:class:`DFSTransformer` runs synthesis as a pipeline step.
:class:`DFSSelectorTransformer` also drops the features a selector did not
keep. Later calls then compute only the rest.
:class:`dtype_selector` picks columns by dtype for a ``ColumnTransformer``.
:class:`NarwhalsMixin` makes any estimator accept a table narwhals can read.
:class:`NarwhalsEncoder` converts a table to what ``set_output`` asks for.

This package needs the ``sklearn`` extra: ``pip install "tusk[sklearn]"``.
:mod:`tusk` does not import this package. Import it by name.
"""

from __future__ import annotations

from tusk.sklearn._encoders import dtype_selector
from tusk.sklearn._narwhals import NarwhalsEncoder, NarwhalsMixin
from tusk.sklearn.transformers import DFSSelectorTransformer, DFSTransformer

__all__ = [
    "DFSSelectorTransformer",
    "DFSTransformer",
    "NarwhalsEncoder",
    "NarwhalsMixin",
    "dtype_selector",
]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_sklearn_narwhals.py -q`
Expected: `19 passed`.

- [ ] **Step 5: Lint**

```bash
git add -A
just lint
```

Expected: every hook `Passed` or `Skipped`.

- [ ] **Step 6: Commit**

```bash
git add src/tusk/sklearn/_narwhals.py src/tusk/sklearn/__init__.py tests/test_sklearn_narwhals.py
git commit -m "feat: add NarwhalsMixin and NarwhalsEncoder

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LzdNzyhemKvk3JKEHFajLx"
```

---

### Task 2: Column encoders

**Files:**
- Create: `src/tusk/sklearn/_column_encoders.py`
- Modify: `src/tusk/sklearn/__init__.py`
- Test: `tests/test_sklearn_column_encoders.py`

**Interfaces:**
- Consumes: `NarwhalsEncoder`, `collect` from `tusk.sklearn._narwhals` (Task 1).
- Produces:
  - `EnumEncoder()`: output `{column}_code`, `Float64`, null → NaN.
  - `TemporalEncoder(NarwhalsEncoder)` with class attributes `accepted_dtype`, `allowed_components` and instance attribute `components`.
  - `DateEncoder(components=("month", "day"))`, `TimeEncoder(components=("hour", "minute"))`, `DatetimeEncoder(components=("month", "day", "hour", "minute"))`, `DurationEncoder(components=("total_seconds",))`: output `{column}_{component}`.
  - `StringEncoder(n_components=30)` with `_fits_on_schema = False`: output `{column}_svd_{i}`; fitting sets `vectorizers_`.
  - Constants `DATE_COMPONENTS`, `TIME_COMPONENTS`, `DURATION_COMPONENTS`.
  - Every schema-only encoder implements `_expressions()` in the same order as `_output_names()`; Task 3 relies on that.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sklearn_column_encoders.py`:

```python
import datetime as dt

import duckdb
import numpy as np
import pandas as pd
import polars as pl
import pytest

from tusk.exceptions import EncoderError
from tusk.sklearn import (
    DateEncoder,
    DatetimeEncoder,
    DurationEncoder,
    EnumEncoder,
    StringEncoder,
    TimeEncoder,
)

ENUM = pl.DataFrame({"e": pl.Series(["b", "a", None], dtype=pl.Enum(["b", "a"]))})
TEMPORAL = pl.DataFrame(
    {
        "d": [dt.date(2024, 3, 5), None],
        "t": [dt.time(9, 30), None],
        "w": [dt.datetime(2024, 3, 5, 9, 30), None],
        "u": [dt.timedelta(minutes=90), None],
    }
)


def test_enum_codes_follow_the_category_order():
    encoder = EnumEncoder().fit(ENUM)
    np.testing.assert_array_equal(encoder.transform(ENUM).ravel(), [0.0, 1.0, np.nan])
    assert list(encoder.get_feature_names_out()) == ["e_code"]


def test_a_pandas_ordered_categorical_gets_the_same_codes():
    ordered = pd.Categorical(["b", "a", None], categories=["b", "a"], ordered=True)
    codes = EnumEncoder().fit_transform(pd.DataFrame({"e": ordered}))
    np.testing.assert_array_equal(codes.ravel(), [0.0, 1.0, np.nan])


def test_enum_codes_are_computed_on_duckdb():
    connection = duckdb.connect()
    connection.execute("create type mood as enum ('b', 'a')")
    relation = connection.sql(
        "select * from (values ('b'::mood), ('a'::mood), (null::mood)) t(e)"
    )
    codes = EnumEncoder().fit_transform(relation)
    np.testing.assert_array_equal(codes.ravel(), [0.0, 1.0, np.nan])


def test_the_enum_encoder_rejects_another_dtype():
    with pytest.raises(EncoderError, match="EnumEncoder encodes Enum columns"):
        EnumEncoder().fit(pl.DataFrame({"s": ["a"]}))


@pytest.mark.parametrize(
    ("encoder", "column", "expected", "names"),
    [
        (DateEncoder(), "d", [3, 5], ["d_month", "d_day"]),
        (TimeEncoder(), "t", [9, 30], ["t_hour", "t_minute"]),
        (
            DatetimeEncoder(),
            "w",
            [3, 5, 9, 30],
            ["w_month", "w_day", "w_hour", "w_minute"],
        ),
        (DurationEncoder(), "u", [5400], ["u_total_seconds"]),
    ],
)
def test_the_default_components(encoder, column, expected, names):
    out = encoder.fit_transform(TEMPORAL.select(column))
    assert out[0].tolist() == expected
    assert np.isnan(out[1]).all()
    assert list(encoder.get_feature_names_out()) == names


def test_components_can_be_chosen():
    encoder = DateEncoder(components=["year", "weekday", "ordinal_day"])
    assert encoder.fit_transform(TEMPORAL.select("d"))[0].tolist() == [2024, 2, 65]


def test_an_unrecognized_component_lists_the_allowed_ones():
    with pytest.raises(ValueError, match="choose from"):
        DateEncoder(components=["hour"]).fit(TEMPORAL.select("d"))


def test_a_temporal_encoder_rejects_another_dtype():
    with pytest.raises(EncoderError, match="DateEncoder encodes Date columns"):
        DateEncoder().fit(TEMPORAL.select("w"))


CITIES = pl.DataFrame(
    {"c": ["Zurich", "Zürich", "Zurich HB", "London", "Londres", None] * 3}
)


def test_the_string_encoder_output_has_n_components_columns():
    encoder = StringEncoder(n_components=4).fit(CITIES)
    assert encoder.transform(CITIES).shape == (18, 4)
    assert list(encoder.get_feature_names_out()) == [f"c_svd_{i}" for i in range(4)]


def test_shared_n_grams_are_closer_than_none():
    out = StringEncoder(n_components=4).fit_transform(CITIES)
    assert np.linalg.norm(out[0] - out[2]) < np.linalg.norm(out[0] - out[3])


def test_a_value_first_seen_at_transform_is_encoded():
    encoder = StringEncoder(n_components=4).fit(CITIES)
    unseen = encoder.transform(pl.DataFrame({"c": ["Zurichberg"]}))
    assert unseen.shape == (1, 4)
    assert np.abs(unseen).sum() > 0


@pytest.mark.parametrize(
    "values",
    [["ab", "cd"], ["a", "a"], [None, ""]],
    ids=["few rows", "one n-gram", "no n-grams"],
)
def test_a_small_vocabulary_is_padded_to_n_components(values):
    frame = pl.DataFrame({"c": values}, schema={"c": pl.String})
    assert StringEncoder().fit_transform(frame).shape == (2, 30)


def test_no_n_grams_encode_as_zeros():
    frame = pl.DataFrame({"c": [None, ""]}, schema={"c": pl.String})
    assert not StringEncoder().fit_transform(frame).any()


def test_the_string_encoder_rejects_another_dtype():
    with pytest.raises(EncoderError, match="StringEncoder encodes String columns"):
        StringEncoder().fit(pl.DataFrame({"n": [1]}))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_sklearn_column_encoders.py -q`
Expected: collection error, `ImportError: cannot import name 'DateEncoder' from 'tusk.sklearn'`.

- [ ] **Step 3: Write the implementation**

Create `src/tusk/sklearn/_column_encoders.py`:

```python
"""Encoders for the columns of one narwhals dtype.

:class:`EnumEncoder` encodes ``Enum`` columns as codes. :class:`DateEncoder`,
:class:`TimeEncoder`, :class:`DatetimeEncoder` and :class:`DurationEncoder`
encode temporal columns as numeric components. :class:`StringEncoder`
encodes ``String`` columns as coordinates of their character n-grams.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import narwhals as nw
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from tusk.exceptions import EncoderError
from tusk.sklearn._narwhals import NarwhalsEncoder, collect

DATE_COMPONENTS = ("year", "month", "day", "weekday", "ordinal_day", "timestamp")
TIME_COMPONENTS = (
    "hour",
    "minute",
    "second",
    "millisecond",
    "microsecond",
    "nanosecond",
)
DURATION_COMPONENTS = (
    "total_minutes",
    "total_seconds",
    "total_milliseconds",
    "total_microseconds",
    "total_nanoseconds",
)


class EnumEncoder(NarwhalsEncoder):
    """An encoder of ``Enum`` columns as their position in the category order.

    The categories come from the dtype, so fitting reads no rows. A null
    becomes NaN. Output name: ``{column}_code``.
    """

    def _fit(self, table: nw.DataFrame | nw.LazyFrame, y: Any) -> None:
        """Check that every column is an ``Enum``.

        Args:
            table: The input table.
            y: Ignored.
        """
        reject_other_dtypes(self.schema_in_, nw.Enum, type(self).__name__)

    def _expressions(self) -> list[nw.Expr]:
        """Return one code expression per column.

        Returns:
            The expressions.
        """
        return [
            code_expression(name, list(dtype.categories))  # ty: ignore[unresolved-attribute]
            for name, dtype in self.schema_in_.items()
        ]

    def _output_names(self) -> list[str]:
        """Return ``{column}_code`` per column.

        Returns:
            The names.
        """
        return [f"{name}_code" for name in self.schema_in_]


class TemporalEncoder(NarwhalsEncoder):
    """The base of the encoders of temporal columns as numeric components.

    A subclass sets ``accepted_dtype`` and ``allowed_components``, and takes
    ``components`` in its ``__init__``. Each component is a method of
    narwhals' ``dt`` namespace. Output name: ``{column}_{component}``.

    Attributes:
        accepted_dtype: The dtype class the encoder encodes.
        allowed_components: The components ``components`` may name.
        components: The components to compute, in output order.
    """

    accepted_dtype: ClassVar[type[nw.dtypes.DType]]
    allowed_components: ClassVar[tuple[str, ...]]
    components: Sequence[str]

    def _fit(self, table: nw.DataFrame | nw.LazyFrame, y: Any) -> None:
        """Check the components and the column dtypes.

        Args:
            table: The input table.
            y: Ignored.

        Raises:
            ValueError: If a component is not in ``allowed_components``.
        """
        unrecognized = [c for c in self.components if c not in self.allowed_components]
        if unrecognized:
            raise ValueError(
                f"{type(self).__name__} has no component {unrecognized[0]!r}; "
                f"choose from {list(self.allowed_components)}",
            )
        reject_other_dtypes(self.schema_in_, self.accepted_dtype, type(self).__name__)

    def _expressions(self) -> list[nw.Expr]:
        """Return one expression per column and component.

        Returns:
            The expressions, column by column.
        """
        return [
            getattr(nw.col(name).dt, component)().alias(f"{name}_{component}")
            for name in self.schema_in_
            for component in self.components
        ]

    def _output_names(self) -> list[str]:
        """Return ``{column}_{component}``, column by column.

        Returns:
            The names.
        """
        return [
            f"{name}_{component}"
            for name in self.schema_in_
            for component in self.components
        ]


class DateEncoder(TemporalEncoder):
    """An encoder of ``Date`` columns as calendar components.

    Attributes:
        accepted_dtype: ``Date``.
        allowed_components: ``year``, ``month``, ``day``, ``weekday``,
            ``ordinal_day`` and ``timestamp``.
    """

    accepted_dtype = nw.Date
    allowed_components = DATE_COMPONENTS

    def __init__(self, components: Sequence[str] = ("month", "day")) -> None:
        """Configure the components.

        Args:
            components: Any of ``year``, ``month``, ``day``, ``weekday``,
                ``ordinal_day`` and ``timestamp``.
        """
        self.components = components


class TimeEncoder(TemporalEncoder):
    """An encoder of ``Time`` columns as clock components.

    Attributes:
        accepted_dtype: ``Time``.
        allowed_components: ``hour``, ``minute``, ``second``,
            ``millisecond``, ``microsecond`` and ``nanosecond``.
    """

    accepted_dtype = nw.Time
    allowed_components = TIME_COMPONENTS

    def __init__(self, components: Sequence[str] = ("hour", "minute")) -> None:
        """Configure the components.

        Args:
            components: Any of ``hour``, ``minute``, ``second``,
                ``millisecond``, ``microsecond`` and ``nanosecond``.
        """
        self.components = components


class DatetimeEncoder(TemporalEncoder):
    """An encoder of ``Datetime`` columns as calendar and clock components.

    Attributes:
        accepted_dtype: ``Datetime``.
        allowed_components: The components of :class:`DateEncoder` and
            :class:`TimeEncoder`.
    """

    accepted_dtype = nw.Datetime
    allowed_components = DATE_COMPONENTS + TIME_COMPONENTS

    def __init__(
        self, components: Sequence[str] = ("month", "day", "hour", "minute")
    ) -> None:
        """Configure the components.

        Args:
            components: Any component of :class:`DateEncoder` or
                :class:`TimeEncoder`.
        """
        self.components = components


class DurationEncoder(TemporalEncoder):
    """An encoder of ``Duration`` columns as their total length.

    Attributes:
        accepted_dtype: ``Duration``.
        allowed_components: ``total_minutes``, ``total_seconds``,
            ``total_milliseconds``, ``total_microseconds`` and
            ``total_nanoseconds``.
    """

    accepted_dtype = nw.Duration
    allowed_components = DURATION_COMPONENTS

    def __init__(self, components: Sequence[str] = ("total_seconds",)) -> None:
        """Configure the components.

        Args:
            components: Any of ``total_minutes``, ``total_seconds``,
                ``total_milliseconds``, ``total_microseconds`` and
                ``total_nanoseconds``.
        """
        self.components = components


class StringEncoder(NarwhalsEncoder):
    """An encoder of ``String`` columns as coordinates of their character n-grams.

    Each column is encoded separately. A value becomes a TF-IDF vector over
    the character 3- and 4-grams seen at fit. ``TruncatedSVD`` reduces that
    vector to ``n_components`` coordinates. A null is encoded as ``""``. Zero
    columns fill the output when the column has too few n-grams or rows for
    ``n_components``. Output name: ``{column}_svd_{i}``.

    Fitting sets ``vectorizers_``, column name to its fitted TF-IDF and SVD
    steps.
    """

    _fits_on_schema = False

    def __init__(self, n_components: int = 30) -> None:
        """Configure the output width.

        Args:
            n_components: The number of output columns per input column.
        """
        self.n_components = n_components

    def _fit(self, table: nw.DataFrame | nw.LazyFrame, y: Any) -> None:
        """Fit TF-IDF and SVD per column.

        Args:
            table: The input table.
            y: Ignored.
        """
        reject_other_dtypes(self.schema_in_, nw.String, type(self).__name__)
        eager = collect(table)
        self.vectorizers_ = {
            name: fit_string_column(read_strings(eager, name), self.n_components)
            for name in self.schema_in_
        }

    def _output_names(self) -> list[str]:
        """Return ``{column}_svd_{i}``, column by column.

        Returns:
            The names.
        """
        return [
            f"{name}_svd_{i}"
            for name in self.schema_in_
            for i in range(self.n_components)
        ]

    def _transform(self, table: nw.DataFrame | nw.LazyFrame) -> nw.DataFrame:
        """Encode each column.

        Args:
            table: The input table.

        Returns:
            The coordinates, column by column.
        """
        eager = collect(table)
        blocks = [
            encode_string_column(
                self.vectorizers_[name], read_strings(eager, name), self.n_components
            )
            for name in self.schema_in_
        ]
        coordinates = np.hstack([np.zeros((len(eager), 0)), *blocks])
        return nw.from_numpy(
            coordinates.reshape(len(eager), -1),
            schema=self._output_names(),
            backend=eager.implementation,
        )


def reject_other_dtypes(
    schema: dict[str, nw.dtypes.DType],
    accepted: type[nw.dtypes.DType],
    owner: str,
) -> None:
    """Raise if a column is not of the ``accepted`` dtype.

    Args:
        schema: Column name to dtype.
        accepted: The dtype class the encoder encodes.
        owner: The encoder's class name, for the error message.

    Raises:
        EncoderError: If a column has another dtype.
    """
    for name, dtype in schema.items():
        if dtype != accepted:
            raise EncoderError(
                f"{owner} encodes {accepted.__name__} columns; column {name!r} "
                f"is {dtype}",
            )


def code_expression(name: str, categories: list[str]) -> nw.Expr:
    """Return the position of each value of ``name`` in ``categories``.

    Args:
        name: The column name.
        categories: The categories, in order.

    Returns:
        A ``Float64`` expression, null where the value is null.
    """
    # default=None: pandas casts a null category to the string "nan", which
    # replace_strict would otherwise reject as a value with no code.
    return (
        nw.col(name)
        .cast(nw.String)
        .replace_strict(
            categories,
            [float(position) for position in range(len(categories))],
            default=None,
            return_dtype=nw.Float64,
        )
        .alias(f"{name}_code")
    )


def read_strings(table: nw.DataFrame, name: str) -> list[str]:
    """Return the values of column ``name``, with null as ``""``.

    Args:
        table: The table.
        name: The column name.

    Returns:
        The values.
    """
    return table[name].fill_null("").to_list()


def fit_string_column(
    values: list[str], n_components: int
) -> tuple[TfidfVectorizer, TruncatedSVD | None] | None:
    """Fit TF-IDF and SVD on one column's values.

    Args:
        values: The column's values.
        n_components: The number of coordinates wanted.

    Returns:
        The fitted TF-IDF and SVD steps. The SVD is None when there are fewer
        than two n-grams. None when there are no n-grams at all.
    """
    if not any(value.strip() for value in values):
        return None
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4))
    tfidf = vectorizer.fit_transform(values)
    # TruncatedSVD needs at least two features.
    if tfidf.shape[1] < 2:
        return vectorizer, None
    svd = TruncatedSVD(n_components=min(n_components, tfidf.shape[1]))
    svd.fit(tfidf)
    return vectorizer, svd


def encode_string_column(
    fitted: tuple[TfidfVectorizer, TruncatedSVD | None] | None,
    values: list[str],
    n_components: int,
) -> np.ndarray:
    """Encode one column's values, padded to ``n_components`` columns.

    Args:
        fitted: The steps from :func:`fit_string_column`.
        values: The column's values.
        n_components: The output width.

    Returns:
        An array of shape ``(len(values), n_components)``.
    """
    if fitted is None:
        return np.zeros((len(values), n_components))
    vectorizer, svd = fitted
    tfidf = vectorizer.transform(values)
    coordinates = tfidf.toarray() if svd is None else svd.transform(tfidf)
    # The SVD yields fewer components than asked when the column has fewer
    # rows or n-grams; zero columns keep the output width fixed.
    return np.pad(coordinates, ((0, 0), (0, n_components - coordinates.shape[1])))
```

Why the non-obvious parts are there:
- `replace_strict(..., default=None)`: pandas casts a null category to the string `"nan"`, which `replace_strict` would otherwise reject. The probe confirmed identical codes on polars, pandas, pyarrow and duckdb.
- `TruncatedSVD` needs at least two features, and returns fewer components than asked when there are fewer rows, so `encode_string_column` pads with zeros.

Replace `src/tusk/sklearn/__init__.py` with:

```python
"""scikit-learn estimators for deep feature synthesis.

:class:`DFSTransformer` runs synthesis as a pipeline step.
:class:`DFSSelectorTransformer` also drops the features a selector did not
keep. Later calls then compute only the rest.
:class:`dtype_selector` picks columns by dtype for a ``ColumnTransformer``.
:class:`NarwhalsMixin` makes any estimator accept a table narwhals can read.
:class:`NarwhalsEncoder` converts a table to what ``set_output`` asks for.
The column encoders encode the columns of one narwhals dtype.

This package needs the ``sklearn`` extra: ``pip install "tusk[sklearn]"``.
:mod:`tusk` does not import this package. Import it by name.
"""

from __future__ import annotations

from tusk.sklearn._column_encoders import (
    DateEncoder,
    DatetimeEncoder,
    DurationEncoder,
    EnumEncoder,
    StringEncoder,
    TemporalEncoder,
    TimeEncoder,
)
from tusk.sklearn._encoders import dtype_selector
from tusk.sklearn._narwhals import NarwhalsEncoder, NarwhalsMixin
from tusk.sklearn.transformers import DFSSelectorTransformer, DFSTransformer

__all__ = [
    "DFSSelectorTransformer",
    "DFSTransformer",
    "DateEncoder",
    "DatetimeEncoder",
    "DurationEncoder",
    "EnumEncoder",
    "NarwhalsEncoder",
    "NarwhalsMixin",
    "StringEncoder",
    "TemporalEncoder",
    "TimeEncoder",
    "dtype_selector",
]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_sklearn_column_encoders.py -q`
Expected: `19 passed`.

- [ ] **Step 5: Lint**

```bash
git add -A
just lint
```

- [ ] **Step 6: Commit**

```bash
git add src/tusk/sklearn/_column_encoders.py src/tusk/sklearn/__init__.py tests/test_sklearn_column_encoders.py
git commit -m "feat: add column encoders

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LzdNzyhemKvk3JKEHFajLx"
```

---

### Task 3: `TableEncoder`

**Files:**
- Create: `src/tusk/sklearn/_table_encoder.py`
- Modify: `src/tusk/sklearn/__init__.py`
- Test: `tests/test_sklearn_table_encoder.py`

**Interfaces:**
- Consumes: `NarwhalsEncoder`, `NarwhalsMixin`, `collect` (Task 1); every column encoder (Task 2); `DFSSelectorTransformer` (existing).
- Produces:
  - `TableEncoder(numeric="passthrough", boolean="passthrough", string=StringEncoder(), categorical=OneHotEncoder(handle_unknown="ignore", sparse_output=False), enum=EnumEncoder(), date=DateEncoder(), time=TimeEncoder(), datetime=DatetimeEncoder(), duration=DurationEncoder(), other="drop")`.
  - Fitting sets `groups_: dict[str, tuple[estimator | "passthrough", list[str]]]` and `output_names_: list[str]`.
  - Module constant `GROUPS: tuple[str, ...]` in group order; Task 4's tests import it.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sklearn_table_encoder.py`:

```python
import datetime as dt
from decimal import Decimal

import duckdb
import narwhals as nw
import numpy as np
import polars as pl
import pytest
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import tusk
from tusk.sklearn import (
    DFSSelectorTransformer,
    NarwhalsMixin,
    StringEncoder,
    TableEncoder,
)

EVERY_DTYPE = pl.DataFrame(
    {
        "n": [1, 2],
        "x": [Decimal("1.5"), Decimal("2.5")],
        "b": [True, False],
        "s": ["foo", "bar"],
        "c": pl.Series(["x", "y"]).cast(pl.Categorical),
        "e": pl.Series(["a", "b"], dtype=pl.Enum(["b", "a"])),
        "d": [dt.date(2024, 3, 5), dt.date(2024, 4, 6)],
        "t": [dt.time(9, 30), dt.time(10, 0)],
        "w": [dt.datetime(2024, 3, 5, 9, 30), dt.datetime(2024, 3, 5, 9, 30)],
        "u": [dt.timedelta(minutes=90), dt.timedelta(minutes=1)],
        "l": [[1], [2]],
    }
)


class TypeRecorder(TransformerMixin, BaseEstimator):
    """Records the type of what it is fitted on, and passes it through."""

    def fit(self, X, y=None):
        self.received_ = type(X)
        return self

    def transform(self, X):
        return nw.from_native(X).to_numpy() if not isinstance(X, np.ndarray) else X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class PolarsRecorder(NarwhalsMixin, TypeRecorder):
    convert_to = "polars"


def as_duckdb(frame):
    connection = duckdb.connect()
    connection.register("t", frame.to_arrow())
    return connection.sql("select * from t")


def small_strings():
    return TableEncoder(string=StringEncoder(n_components=2))


def test_every_dtype_reaches_its_group():
    encoder = small_strings().set_output(transform="polars")
    out = encoder.fit_transform(EVERY_DTYPE)
    assert out.columns == list(encoder.get_feature_names_out())
    assert out.columns == [
        "numeric__n",
        "numeric__x",
        "boolean__b",
        "string__s_svd_0",
        "string__s_svd_1",
        "categorical__c_x",
        "categorical__c_y",
        "enum__e_code",
        "date__d_month",
        "date__d_day",
        "time__t_hour",
        "time__t_minute",
        "datetime__w_month",
        "datetime__w_day",
        "datetime__w_hour",
        "datetime__w_minute",
        "duration__u_total_seconds",
    ]


def test_decimal_is_cast_to_float64():
    out = small_strings().set_output(transform="polars").fit_transform(EVERY_DTYPE)
    assert out["numeric__x"].dtype == pl.Float64
    assert out["numeric__x"].to_list() == [1.5, 2.5]


def test_drop_and_passthrough():
    encoder = TableEncoder(numeric="drop", string="passthrough", other="drop")
    out = encoder.set_output(transform="polars").fit_transform(EVERY_DTYPE)
    assert "numeric__n" not in out.columns
    assert out["string__s"].to_list() == ["foo", "bar"]


@pytest.mark.parametrize("backend", ["pandas", "pyarrow"])
def test_other_backends(backend):
    frame = EVERY_DTYPE.drop("d", "t", "l")
    native = frame.to_pandas() if backend == "pandas" else frame.to_arrow()
    encoder = small_strings().fit(native)
    assert encoder.transform(native).shape == (2, len(encoder.get_feature_names_out()))


def test_an_empty_group_is_skipped():
    encoder = TableEncoder()
    encoder.fit(pl.DataFrame({"n": [1.0, 2.0]}))
    assert list(encoder.groups_) == ["numeric"]


def test_a_schema_only_fit_does_not_collect(monkeypatch):
    frame = pl.DataFrame({"n": [1.0, 2.0], "w": [dt.datetime(2024, 3, 5, 9, 30)] * 2})
    relation = as_duckdb(frame)

    def refuse(*args, **kwargs):
        raise AssertionError("fit collected the table")

    monkeypatch.setattr(nw.LazyFrame, "collect", refuse)
    TableEncoder().fit(relation)


def test_transform_collects_a_lazy_table_once(monkeypatch):
    frame = pl.DataFrame(
        {
            "n": [1.0, 2.0],
            "s": ["foo", "bar"],
            "w": [dt.datetime(2024, 3, 5, 9, 30)] * 2,
        }
    )
    relation = as_duckdb(frame)
    encoder = small_strings().fit(relation)
    calls = []
    collect = nw.LazyFrame.collect

    def counting(self, *args, **kwargs):
        calls.append(self)
        return collect(self, *args, **kwargs)

    monkeypatch.setattr(nw.LazyFrame, "collect", counting)
    assert encoder.transform(relation).shape == (2, 1 + 2 + 4)
    assert len(calls) == 1


def test_a_bare_estimator_receives_numpy_and_a_mixin_its_convert_to():
    encoder = TableEncoder(numeric=TypeRecorder(), string=PolarsRecorder())
    encoder.fit(EVERY_DTYPE)
    assert encoder.groups_["numeric"][0].received_ is np.ndarray
    assert encoder.groups_["string"][0].received_ is pl.DataFrame


def test_a_value_that_is_not_an_estimator_is_rejected():
    with pytest.raises(ValueError, match="numeric='scale'"):
        TableEncoder(numeric="scale").fit(EVERY_DTYPE)


def test_nested_parameters_do_not_change_the_defaults():
    changed = TableEncoder().set_params(date__components=["year"])
    assert changed.date.components == ["year"]
    assert TableEncoder().date.components == ("month", "day")
    assert clone(changed).date.components == ["year"]


def test_it_encodes_inside_dfs_selection_on_duckdb():
    customers = duckdb.sql(
        "select * from (values "
        "(1, 30, 'gold', timestamp '2024-01-01'), "
        "(2, 40, 'silver', timestamp '2024-01-01'), "
        "(3, 50, 'gold', timestamp '2024-01-01'), "
        "(4, 60, 'silver', timestamp '2024-01-01')) t(id, age, tier, signed_up_at)",
    )
    database = tusk.Database("shop").add_table(
        "customers", customers, primary_key="id", row_creation_time="signed_up_at"
    )
    encoder = TableEncoder(numeric=StandardScaler(), string="drop", datetime="drop")
    selector = DFSSelectorTransformer(
        target_table="customers",
        selection_pipeline=Pipeline(
            [("encode", encoder), ("select", SelectKBest(f_classif, k=1))]
        ),
        trans_primitives=[],
    )
    with pytest.warns(UserWarning, match="fed no encoded column"):
        selector.fit([1, 2, 3, 4], [0, 0, 1, 1], database=database)
    assert [feature.name for feature in selector.features_] == ["age"]
    assert list(selector.get_feature_names_out()) == ["numeric__age"]
    assert selector.transform([1, 2], database=database).shape == (2, 1)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_sklearn_table_encoder.py -q`
Expected: collection error, `ImportError: cannot import name 'TableEncoder' from 'tusk.sklearn'`.

- [ ] **Step 3: Write the implementation**

Create `src/tusk/sklearn/_table_encoder.py`:

```python
"""Encoding of every column of a table by its narwhals dtype.

:class:`TableEncoder` splits the columns into disjoint dtype groups and
encodes each group with its own estimator.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder

from tusk.exceptions import EncoderError
from tusk.sklearn._column_encoders import (
    DateEncoder,
    DatetimeEncoder,
    DurationEncoder,
    EnumEncoder,
    StringEncoder,
    TimeEncoder,
)
from tusk.sklearn._narwhals import NarwhalsEncoder, NarwhalsMixin, collect

GROUPS = (
    "numeric",
    "boolean",
    "string",
    "categorical",
    "enum",
    "date",
    "time",
    "datetime",
    "duration",
    "other",
)
GROUP_DTYPES = {
    "boolean": nw.Boolean,
    "string": nw.String,
    "categorical": nw.Categorical,
    "enum": nw.Enum,
    "date": nw.Date,
    "time": nw.Time,
    "datetime": nw.Datetime,
    "duration": nw.Duration,
}
KEYWORDS = ("passthrough", "drop")


class TableEncoder(NarwhalsEncoder):
    """An encoder of every column of a table by its narwhals dtype.

    Each parameter names a dtype group and takes an estimator,
    ``"passthrough"`` or ``"drop"``. Each column belongs to exactly one
    group. An estimator with :class:`NarwhalsMixin` receives a narwhals table
    and converts it itself. Any other estimator receives numpy.

    Output names are ``{group}__{name}``. Groups appear in parameter order.

    Fitting sets ``groups_``, group name to its fitted estimator or
    ``"passthrough"`` and its input columns. Empty and dropped groups are
    absent. It also sets ``output_names_``, the output column names.
    """

    _fits_on_schema = False

    def __init__(
        self,
        numeric: Any = "passthrough",
        boolean: Any = "passthrough",
        string: Any = StringEncoder(),  # noqa: B008
        categorical: Any = OneHotEncoder(handle_unknown="ignore", sparse_output=False),  # noqa: B008
        enum: Any = EnumEncoder(),  # noqa: B008
        date: Any = DateEncoder(),  # noqa: B008
        time: Any = TimeEncoder(),  # noqa: B008
        datetime: Any = DatetimeEncoder(),  # noqa: B008
        duration: Any = DurationEncoder(),  # noqa: B008
        other: Any = "drop",
    ) -> None:
        """Configure the estimator of each dtype group.

        Args:
            numeric: For ``Int*``, ``UInt*``, ``Float*`` and ``Decimal``.
                ``Decimal`` is cast to ``Float64`` first.
            boolean: For ``Boolean``.
            string: For ``String``.
            categorical: For ``Categorical``.
            enum: For ``Enum``.
            date: For ``Date``.
            time: For ``Time``.
            datetime: For ``Datetime``.
            duration: For ``Duration``.
            other: For every other dtype, such as ``List`` or ``Struct``.
        """
        self.numeric = numeric
        self.boolean = boolean
        self.string = string
        self.categorical = categorical
        self.enum = enum
        self.date = date
        self.time = time
        self.datetime = datetime
        self.duration = duration
        self.other = other

    def set_params(self, **params: Any) -> TableEncoder:
        """Set parameters, including nested ones such as ``date__components``.

        Args:
            **params: Parameter names and values.

        Returns:
            This encoder.
        """
        # The default estimators are shared by every TableEncoder. A nested
        # parameter would change the shared instance, so it is cloned first.
        defaults = type(self)().get_params(deep=False)
        for group in {key.split("__")[0] for key in params if "__" in key}:
            if getattr(self, group) is defaults[group]:
                setattr(self, group, clone(defaults[group]))
        return super().set_params(**params)

    def _fit(self, table: nw.DataFrame | nw.LazyFrame, y: Any) -> None:
        """Fit each group's estimator on the group's columns.

        Args:
            table: The input table, eager or lazy.
            y: Targets, passed to each estimator.
        """
        groups = split_into_groups(self.schema_in_)
        estimators = {group: self._get_estimator(group) for group in groups}
        kept = {g: c for g, c in groups.items() if estimators[g] != "drop"}
        cast = table.select(read_expressions(self.schema_in_))
        by_values = [g for g in kept if not fits_on_schema(estimators[g])]
        values = collect_columns(cast, [c for g in by_values for c in kept[g]])
        self.groups_ = {}
        for group, columns in kept.items():
            source = values if group in by_values else cast
            self.groups_[group] = (
                fit_group(estimators[group], source.select(columns), y),
                columns,
            )
        self.output_names_ = [
            f"{group}__{name}"
            for group, (estimator, columns) in self.groups_.items()
            for name in group_output_names(estimator, columns)
        ]

    def _output_names(self) -> list[str]:
        """Return the output names found at fit.

        Returns:
            The names.
        """
        return self.output_names_

    def _transform(self, table: nw.DataFrame | nw.LazyFrame) -> nw.DataFrame:
        """Encode each group and join the results by position.

        Args:
            table: The input table, eager or lazy.

        Returns:
            The encoded table.

        Raises:
            EncoderError: If a group's output has another row count than the
                input.
        """
        selected = nw.maybe_reset_index(collect(table.select(self._selection())))
        blocks = [self._encode_group(group, selected) for group in self.groups_]
        for group, block in zip(self.groups_, blocks, strict=True):
            if len(block) != len(selected):
                raise EncoderError(
                    f"the {group!r} group returned {len(block)} rows for "
                    f"{len(selected)} input rows",
                )
        if not blocks:
            return selected.select([])
        return nw.concat(blocks, how="horizontal")

    def _get_estimator(self, group: str) -> Any:
        """Return a clone of the group's estimator, or its keyword.

        Args:
            group: The group name.

        Returns:
            The estimator clone, ``"passthrough"`` or ``"drop"``.

        Raises:
            ValueError: If the parameter is neither an estimator nor a
                keyword.
        """
        value = getattr(self, group)
        if isinstance(value, str) and value in KEYWORDS:
            return value
        if not (hasattr(value, "fit") and hasattr(value, "transform")):
            raise ValueError(
                f"{group}={value!r} is not an estimator, 'passthrough' or 'drop'",
            )
        return clone(value)

    def _selection(self) -> list[nw.Expr]:
        """Return the one ``select`` that feeds every group.

        Schema-only estimators contribute their expressions. Every other group
        contributes its columns. Every name is prefixed with its group.

        Returns:
            The expressions.
        """
        read = dict(
            zip(self.schema_in_, read_expressions(self.schema_in_), strict=True)
        )
        expressions = []
        for group, (estimator, columns) in self.groups_.items():
            if estimator != "passthrough" and fits_on_schema(estimator):
                names = estimator.get_feature_names_out()
                sources = estimator._expressions()
                expressions += [
                    expression.alias(f"{group}__{name}")
                    for expression, name in zip(sources, names, strict=True)
                ]
                continue
            expressions += [read[c].alias(f"{group}__{c}") for c in columns]
        return expressions

    def _encode_group(self, group: str, selected: nw.DataFrame) -> nw.DataFrame:
        """Return the encoded columns of one group.

        Args:
            group: The group name.
            selected: The collected result of :meth:`_selection`.

        Returns:
            The group's output, with ``{group}__`` names.
        """
        estimator, columns = self.groups_[group]
        if estimator == "passthrough" or fits_on_schema(estimator):
            names = group_output_names(estimator, columns)
            return selected.select([f"{group}__{name}" for name in names])
        values = selected.select([f"{group}__{c}" for c in columns]).rename(
            {f"{group}__{c}": c for c in columns}
        )
        output = estimator.transform(to_estimator_input(estimator, values))
        return nw.from_numpy(
            to_dense(output),
            schema=[f"{group}__{n}" for n in group_output_names(estimator, columns)],
            backend=selected.implementation,
        )


def split_into_groups(schema: dict[str, nw.dtypes.DType]) -> dict[str, list[str]]:
    """Return group name to its columns, in group order, without empty groups.

    Args:
        schema: Column name to dtype.

    Returns:
        The groups and their columns, in table order.
    """
    columns: dict[str, list[str]] = {group: [] for group in GROUPS}
    for name, dtype in schema.items():
        columns[find_group(dtype)].append(name)
    return {group: names for group, names in columns.items() if names}


def find_group(dtype: nw.dtypes.DType) -> str:
    """Return the group a dtype belongs to.

    Args:
        dtype: A narwhals dtype.

    Returns:
        The group name.
    """
    if dtype.is_numeric():
        return "numeric"
    for group, group_dtype in GROUP_DTYPES.items():
        if dtype == group_dtype:
            return group
    return "other"


def read_expressions(schema: dict[str, nw.dtypes.DType]) -> list[nw.Expr]:
    """Return one expression per column, with ``Decimal`` cast to ``Float64``.

    Args:
        schema: Column name to dtype.

    Returns:
        The expressions, in column order.
    """
    return [
        nw.col(name).cast(nw.Float64) if dtype == nw.Decimal else nw.col(name)
        for name, dtype in schema.items()
    ]


def collect_columns(
    table: nw.DataFrame | nw.LazyFrame, columns: list[str]
) -> nw.DataFrame | nw.LazyFrame:
    """Return ``columns`` of ``table``, collected unless there are none.

    Args:
        table: The table, eager or lazy.
        columns: The columns to collect.

    Returns:
        The collected columns, or ``table`` itself when ``columns`` is empty.
    """
    if not columns:
        return table
    return collect(table.select(columns))


def fits_on_schema(estimator: Any) -> bool:
    """Report whether an estimator fits on the schema alone.

    Args:
        estimator: An estimator or ``"passthrough"``.

    Returns:
        True for ``"passthrough"`` and for tusk's schema-only encoders.
    """
    return estimator == "passthrough" or (
        isinstance(estimator, NarwhalsEncoder) and estimator._fits_on_schema
    )


def fit_group(estimator: Any, table: nw.DataFrame | nw.LazyFrame, y: Any) -> Any:
    """Fit a group's estimator on the group's columns.

    Args:
        estimator: The cloned estimator or ``"passthrough"``.
        table: The group's columns.
        y: Targets.

    Returns:
        The fitted estimator, or ``"passthrough"``.
    """
    if estimator == "passthrough":
        return estimator
    return estimator.fit(to_estimator_input(estimator, table), y)


def to_estimator_input(estimator: Any, table: nw.DataFrame | nw.LazyFrame) -> Any:
    """Return what the estimator receives: the table, or numpy.

    Args:
        estimator: The estimator.
        table: The group's columns.

    Returns:
        The narwhals table for an estimator with :class:`NarwhalsMixin`, else
        a numpy array.
    """
    if isinstance(estimator, NarwhalsMixin):
        return table
    return collect(table).to_numpy()


def group_output_names(estimator: Any, columns: list[str]) -> list[str]:
    """Return a group's output names, without the group prefix.

    Args:
        estimator: The fitted estimator or ``"passthrough"``.
        columns: The group's input columns.

    Returns:
        The names.
    """
    if estimator == "passthrough":
        return columns
    return list(estimator.get_feature_names_out(columns))


def to_dense(output: Any) -> np.ndarray:
    """Return an estimator's output as a dense 2-D array.

    Args:
        output: A numpy array, a sparse matrix or a table.

    Returns:
        The dense array.
    """
    if hasattr(output, "toarray"):
        return output.toarray()
    table = nw.from_native(output, pass_through=True)
    if isinstance(table, nw.DataFrame):
        return table.to_numpy()
    return np.asarray(output)
```

Why the non-obvious parts are there:
- `set_params` clones a shared default before a nested parameter reaches it. Without that, `TableEncoder().set_params(date__components=["year"])` changes the default `DateEncoder` of every later `TableEncoder`.
- `nw.maybe_reset_index`: a pandas table sorted by `DFSTransformer` keeps its old index, and the horizontal concatenation would align on it instead of by position.
- Every expression in the one `select` is aliased `{group}__{name}`, so a component name such as `signup_month` cannot collide with an input column of the same name.

Replace `src/tusk/sklearn/__init__.py` with:

```python
"""scikit-learn estimators for deep feature synthesis.

:class:`DFSTransformer` runs synthesis as a pipeline step.
:class:`DFSSelectorTransformer` also drops the features a selector did not
keep. Later calls then compute only the rest.
:class:`TableEncoder` encodes each column by its narwhals dtype.
:class:`dtype_selector` picks columns by dtype for a ``ColumnTransformer``.
:class:`NarwhalsMixin` makes any estimator accept a table narwhals can read.
:class:`NarwhalsEncoder` converts a table to what ``set_output`` asks for.
The column encoders encode the columns of one narwhals dtype.

This package needs the ``sklearn`` extra: ``pip install "tusk[sklearn]"``.
:mod:`tusk` does not import this package. Import it by name.
"""

from __future__ import annotations

from tusk.sklearn._column_encoders import (
    DateEncoder,
    DatetimeEncoder,
    DurationEncoder,
    EnumEncoder,
    StringEncoder,
    TemporalEncoder,
    TimeEncoder,
)
from tusk.sklearn._encoders import dtype_selector
from tusk.sklearn._narwhals import NarwhalsEncoder, NarwhalsMixin
from tusk.sklearn._table_encoder import TableEncoder
from tusk.sklearn.transformers import DFSSelectorTransformer, DFSTransformer

__all__ = [
    "DFSSelectorTransformer",
    "DFSTransformer",
    "DateEncoder",
    "DatetimeEncoder",
    "DurationEncoder",
    "EnumEncoder",
    "NarwhalsEncoder",
    "NarwhalsMixin",
    "StringEncoder",
    "TableEncoder",
    "TemporalEncoder",
    "TimeEncoder",
    "dtype_selector",
]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_sklearn_table_encoder.py -q`
Expected: `12 passed`.

- [ ] **Step 5: Lint**

```bash
git add -A
just lint
```

- [ ] **Step 6: Commit**

```bash
git add src/tusk/sklearn/_table_encoder.py src/tusk/sklearn/__init__.py tests/test_sklearn_table_encoder.py
git commit -m "feat: add TableEncoder

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LzdNzyhemKvk3JKEHFajLx"
```

---

### Task 4: Remove `dtype_selector`

**Files:**
- Modify: `src/tusk/sklearn/_encoders.py`, `src/tusk/sklearn/__init__.py`, `src/tusk/dtypes.py`
- Test: `tests/test_sklearn_encoders.py`, `tests/test_sklearn_selector_transformer.py`, `tests/test_sklearn_dfs_transformer.py`

**Interfaces:**
- Consumes: `TableEncoder`, `GROUPS` (Task 3).
- Produces: `tusk.sklearn` no longer exports `dtype_selector`. `_reject_explicit_columns` raises `EncoderError` whose message names `tusk.sklearn.TableEncoder`.

- [ ] **Step 1: Migrate the tests**

Save this patch as `migrate-tests.diff` and apply it with `git apply migrate-tests.diff`, then delete the file. It replaces every `ColumnTransformer` built with `dtype_selector` by a `TableEncoder`, keeps each test's intent (groups the old `ColumnTransformer` did not cover are set to `"drop"`), renames the asserted encoded names (`oh__` → `string__`, `num__` → `numeric__`), and makes the explicit-columns test expect `TableEncoder` in the message.

```diff
diff --git a/tests/test_sklearn_dfs_transformer.py b/tests/test_sklearn_dfs_transformer.py
index 486f14b..61cc8f2 100644
--- a/tests/test_sklearn_dfs_transformer.py
+++ b/tests/test_sklearn_dfs_transformer.py
@@ -6,7 +6,6 @@ import numpy as np
 import polars as pl
 import pytest
 import sklearn
-from sklearn.compose import ColumnTransformer
 from sklearn.dummy import DummyClassifier
 from sklearn.impute import SimpleImputer
 from sklearn.linear_model import LogisticRegression
@@ -15,7 +14,8 @@ from sklearn.pipeline import Pipeline

 import tusk
 from tusk.exceptions import SchemaError, UnmatchedPrimitiveWarning
-from tusk.sklearn import DFSTransformer, dtype_selector
+from tusk.sklearn import DFSTransformer, TableEncoder
+from tusk.sklearn._table_encoder import GROUPS

 KEYS = [1, 2, 3]
 Y = [0, 1, 0]
@@ -89,14 +89,9 @@ def test_it_routes_the_database_through_a_pipeline(db):
                 ("dfs", _transformer()),
                 (
                     "impute",
-                    ColumnTransformer(
-                        [
-                            (
-                                "numbers",
-                                SimpleImputer(keep_empty_features=True),
-                                dtype_selector("numeric"),
-                            ),
-                        ],
+                    TableEncoder(
+                        **{group: "drop" for group in GROUPS}
+                        | {"numeric": SimpleImputer(keep_empty_features=True)}
                     ),
                 ),
                 ("clf", LogisticRegression()),
diff --git a/tests/test_sklearn_encoders.py b/tests/test_sklearn_encoders.py
index be5c611..3e2bd0e 100644
--- a/tests/test_sklearn_encoders.py
+++ b/tests/test_sklearn_encoders.py
@@ -1,16 +1,12 @@
-import datetime as dt
-
 import polars as pl
 import pytest
 from sklearn.base import BaseEstimator, TransformerMixin
-from sklearn.compose import ColumnTransformer
+from sklearn.compose import ColumnTransformer, make_column_selector
 from sklearn.feature_selection import SelectKBest, f_classif
 from sklearn.pipeline import Pipeline
-from sklearn.preprocessing import OneHotEncoder, StandardScaler
+from sklearn.preprocessing import StandardScaler

-from tusk.dtypes import DtypeFamily
 from tusk.exceptions import EncoderError
-from tusk.sklearn import dtype_selector
 from tusk.sklearn._encoders import (
     get_encoder_prefix,
     get_last_step,
@@ -22,73 +18,6 @@ FRAME = pl.DataFrame(
 )


-def test_dtype_selector_splits_numeric_from_string():
-    assert dtype_selector("numeric")(FRAME) == ["age", "cnt"]
-    assert dtype_selector("string")(FRAME) == ["cat"]
-
-
-def test_dtype_selector_keeps_booleans_and_dates_out_of_string():
-    frame = pl.DataFrame(
-        {
-            "n": [1.0],
-            "s": ["a"],
-            "b": [True],
-            "d": [dt.datetime(2024, 1, 1)],
-        },
-    )
-    assert dtype_selector("numeric")(frame) == ["n"]
-    assert dtype_selector("string")(frame) == ["s"]
-    assert dtype_selector("boolean")(frame) == ["b"]
-    assert dtype_selector("temporal")(frame) == ["d"]
-
-
-def test_dtype_selector_separates_categorical_from_string():
-    frame = pl.DataFrame(
-        {
-            "s": ["t"],
-            "c": pl.Series(["x"]).cast(pl.Categorical),
-            "e": pl.Series(["a"]).cast(pl.Enum(["a", "b"])),
-        },
-    )
-    assert dtype_selector("string")(frame) == ["s"]
-    assert dtype_selector(DtypeFamily.CATEGORICAL)(frame) == ["c", "e"]
-
-
-def test_dtype_selector_rejects_an_unknown_family():
-    with pytest.raises(ValueError):
-        dtype_selector("texty")
-
-
-def test_dtype_selector_separates_has_date_from_duration():
-    """The narrow families are the supported way to route a duration column."""
-    frame = pl.DataFrame(
-        {
-            "when": [dt.datetime(2024, 1, 1)],
-            "elapsed": [dt.timedelta(hours=3)],
-            "n": [1],
-        },
-    )
-    assert dtype_selector("has_date")(frame) == ["when"]
-    assert dtype_selector("duration")(frame) == ["elapsed"]
-    assert dtype_selector("temporal")(frame) == ["when", "elapsed"]
-
-
-def test_dtype_selector_has_time_matches_datetime_and_time_only():
-    """HAS_TIME routes a bare time-of-day column alongside a full datetime."""
-    frame = pl.DataFrame(
-        {
-            "when": [dt.datetime(2024, 1, 1)],
-            "at": [dt.time(9, 30)],
-            "on": [dt.date(2024, 1, 1)],
-        },
-    )
-    assert dtype_selector("has_time")(frame) == ["when", "at"]
-
-
-def test_dtype_selector_reevaluates_on_a_subset():
-    assert dtype_selector("numeric")(FRAME.select(["age"])) == ["age"]
-
-
 # sklearn's feature-name detection falls back to the dataframe interchange
 # protocol for any non-pandas frame; polars has deprecated that protocol, so
 # fitting a raw polars frame directly (bypassing narwhals, as ColumnTransformer
@@ -100,22 +29,6 @@ _INTERCHANGE_DEPRECATION = (
 )


-@pytest.mark.filterwarnings(_INTERCHANGE_DEPRECATION)
-def test_dtype_selector_works_inside_a_column_transformer_on_polars():
-    encoder = ColumnTransformer(
-        [
-            ("oh", OneHotEncoder(handle_unknown="ignore"), dtype_selector("string")),
-            ("num", StandardScaler(), dtype_selector("numeric")),
-        ],
-    ).fit(FRAME)
-    assert list(encoder.get_feature_names_out()) == [
-        "oh__cat_a",
-        "oh__cat_b",
-        "num__age",
-        "num__cnt",
-    ]
-
-
 def test_get_last_step_returns_the_final_step():
     selector = SelectKBest(f_classif, k=1)
     assert (
@@ -150,7 +63,7 @@ def test_explicit_column_lists_are_refused():
             ("sel", SelectKBest(f_classif, k=1)),
         ],
     )
-    with pytest.raises(EncoderError, match="dtype_selector"):
+    with pytest.raises(EncoderError, match="TableEncoder"):
         validate_selection_pipeline(selection_pipeline)


@@ -183,7 +96,13 @@ def test_a_callable_column_list_is_accepted():
             (
                 "enc",
                 ColumnTransformer(
-                    [("num", StandardScaler(), dtype_selector("numeric"))]
+                    [
+                        (
+                            "num",
+                            StandardScaler(),
+                            make_column_selector(dtype_include="number"),
+                        )
+                    ]
                 ),
             ),
             ("sel", SelectKBest(f_classif, k=1)),
diff --git a/tests/test_sklearn_selector_transformer.py b/tests/test_sklearn_selector_transformer.py
index f3d73ef..3ebac7b 100644
--- a/tests/test_sklearn_selector_transformer.py
+++ b/tests/test_sklearn_selector_transformer.py
@@ -6,7 +6,6 @@ import polars as pl
 import pytest
 import sklearn
 from sklearn.base import BaseEstimator, TransformerMixin
-from sklearn.compose import ColumnTransformer
 from sklearn.decomposition import PCA
 from sklearn.feature_selection import SelectKBest, SelectorMixin, f_classif
 from sklearn.impute import SimpleImputer
@@ -22,7 +21,8 @@ from tusk.exceptions import (
     SchemaError,
     UnencodedFeatureWarning,
 )
-from tusk.sklearn import DFSSelectorTransformer, DFSTransformer, dtype_selector
+from tusk.sklearn import DFSSelectorTransformer, DFSTransformer, TableEncoder
+from tusk.sklearn._table_encoder import GROUPS

 # Fitting a raw polars frame directly (bypassing narwhals, as scikit-learn's own
 # validation does internally) raises a DeprecationWarning that is sklearn/polars
@@ -114,12 +114,15 @@ def shop():
     )


+def _only(**groups):
+    """A TableEncoder that drops every group not named."""
+    return TableEncoder(**{group: "drop" for group in GROUPS} | groups)
+
+
 def _encoder():
-    return ColumnTransformer(
-        [
-            ("oh", OneHotEncoder(handle_unknown="ignore"), dtype_selector("string")),
-            ("num", StandardScaler(), dtype_selector("numeric")),
-        ],
+    return _only(
+        string=OneHotEncoder(handle_unknown="ignore"),
+        numeric=StandardScaler(),
     )


@@ -250,8 +253,8 @@ def test_pruned_features_are_never_computed(shop):
 def test_the_survivors_are_exactly_the_features_feeding_kept_columns(shop):
     matrix = _matrix(shop)
     positions = [
-        _encoded_position(matrix, "oh__region_south"),
-        _encoded_position(matrix, "num__MAX__transactions__amount"),
+        _encoded_position(matrix, "string__region_south"),
+        _encoded_position(matrix, "numeric__MAX__transactions__amount"),
     ]
     transformer = DFSSelectorTransformer(
         target_table="customers",
@@ -271,7 +274,7 @@ def test_the_survivors_are_exactly_the_features_feeding_kept_columns(shop):
 def test_a_multi_output_feature_survives_whole(shop):
     quantiles = {"agg_primitives": ["quantiles"], "trans_primitives": []}
     matrix = _matrix(shop, **quantiles)
-    position = _encoded_position(matrix, "num__QUANTILES__transactions__amount__1")
+    position = _encoded_position(matrix, "numeric__QUANTILES__transactions__amount__1")
     transformer = DFSSelectorTransformer(
         target_table="customers",
         selection_pipeline=Pipeline(
@@ -318,12 +321,7 @@ def test_an_opaque_encoder_keeps_every_feature_and_warns(shop):
     # handling.
     selection_pipeline = Pipeline(
         [
-            (
-                "enc",
-                ColumnTransformer(
-                    [("pca", PCA(n_components=2), dtype_selector("numeric"))],
-                ),
-            ),
+            ("enc", _only(numeric=PCA(n_components=2))),
             ("sel", SelectKBest(f_classif, k=1)),
         ],
     )
@@ -350,22 +348,11 @@ def test_an_opaque_encoder_keeps_every_feature_and_warns(shop):

 @pytest.mark.filterwarnings(_INTERCHANGE_DEPRECATION)
 def test_a_partial_encoder_warns_about_features_it_never_saw(shop):
-    # Only string columns are encoded, and remainder defaults to "drop", so
+    # Only string columns are encoded and every other group is dropped, so
     # every numeric feature silently feeds nothing and gets dropped.
     selection_pipeline = Pipeline(
         [
-            (
-                "enc",
-                ColumnTransformer(
-                    [
-                        (
-                            "oh",
-                            OneHotEncoder(handle_unknown="ignore"),
-                            dtype_selector("string"),
-                        ),
-                    ],
-                ),
-            ),
+            ("enc", _only(string=OneHotEncoder(handle_unknown="ignore"))),
             ("sel", SelectKBest(f_classif, k=1)),
         ],
     )
@@ -440,12 +427,7 @@ def test_a_supervised_encoder_is_refitted_with_y(shop):
             [
                 (
                     "encode",
-                    ColumnTransformer(
-                        [
-                            ("target", TargetEncoder(), dtype_selector("string")),
-                            ("numbers", StandardScaler(), dtype_selector("numeric")),
-                        ],
-                    ),
+                    _only(string=TargetEncoder(), numeric=StandardScaler()),
                 ),
                 ("select", KeepPositions(positions=(0,))),
             ],
```

- [ ] **Step 2: Run the migrated tests to verify the message test fails**

Run: `uv run pytest tests/test_sklearn_encoders.py tests/test_sklearn_selector_transformer.py tests/test_sklearn_dfs_transformer.py -q`
Expected: one failure, `test_explicit_column_lists_are_refused`: `Regex pattern did not match` (the message still names `dtype_selector`). Every other test passes.

- [ ] **Step 3: Remove `dtype_selector` and update the message**

Save this patch as `remove-dtype-selector.diff`, apply it with `git apply remove-dtype-selector.diff`, then delete the file:

```diff
diff --git a/src/tusk/dtypes.py b/src/tusk/dtypes.py
index 754fa9d..e62442d 100644
--- a/src/tusk/dtypes.py
+++ b/src/tusk/dtypes.py
@@ -22,13 +22,12 @@ class DtypeFamily(Enum):
     reports this distinction.

     ``TEMPORAL`` matches every dtype the narrower temporal families match,
-    plus every dtype that is temporal. It stays broad for
-    ``dtype_selector``. ``HAS_DATE`` matches ``Datetime`` and ``Date``,
-    the dtypes a calendar position can be read from. ``HAS_TIME`` matches
-    ``Datetime`` and ``Time``, the dtypes an hour or minute can be read
-    from. Both families match ``Datetime``. ``DURATION`` is elapsed time.
-    Neither ``HAS_DATE`` nor ``HAS_TIME`` matches it, even though it is
-    temporal.
+    plus every dtype that is temporal. ``HAS_DATE`` matches ``Datetime``
+    and ``Date``, the dtypes a calendar position can be read from.
+    ``HAS_TIME`` matches ``Datetime`` and ``Time``, the dtypes an hour or
+    minute can be read from. Both families match ``Datetime``.
+    ``DURATION`` is elapsed time. Neither ``HAS_DATE`` nor ``HAS_TIME``
+    matches it, even though it is temporal.
     """

     NUMERIC = "numeric"
diff --git a/src/tusk/sklearn/_encoders.py b/src/tusk/sklearn/_encoders.py
index 3042a61..38653ba 100644
--- a/src/tusk/sklearn/_encoders.py
+++ b/src/tusk/sklearn/_encoders.py
@@ -3,71 +3,21 @@
 :func:`get_last_step` and :func:`get_encoder_prefix` split a pipeline into the
 part that encodes and the selector that ends it.
 :func:`validate_selection_pipeline` rejects pipelines this module cannot
-support. :class:`dtype_selector` picks columns by dtype for a
-``ColumnTransformer``.
+support.
 """

 from __future__ import annotations

 from typing import Any

-import narwhals as nw
 from sklearn.compose import ColumnTransformer
 from sklearn.feature_selection import SelectorMixin
 from sklearn.pipeline import Pipeline
 from sklearn.preprocessing import FunctionTransformer

-from tusk.dtypes import DtypeFamily, matches
 from tusk.exceptions import EncoderError


-class dtype_selector:  # noqa: N801
-    """A selector of columns by :class:`~tusk.dtypes.DtypeFamily`, on any backend.
-
-    It serves the same role as scikit-learn's ``make_column_selector``. It
-    reads the schema through narwhals. This lets it work on every backend a
-    tusk database can use, not pandas alone.
-
-    Families are :class:`~tusk.dtypes.DtypeFamily` values. ``"string"`` here
-    means what it means to a primitive: ``String``, not ``Categorical`` or
-    ``Enum``.
-
-    It is callable. Each call re-evaluates the family match against the
-    table it is given. A narrowed feature matrix then narrows the selection.
-
-    Attributes:
-        family: The :class:`~tusk.dtypes.DtypeFamily` to select.
-    """
-
-    family: DtypeFamily
-
-    def __init__(self, family: DtypeFamily | str) -> None:
-        """Build a selector for one dtype family.
-
-        Args:
-            family: A ``DtypeFamily`` or its string value, such as
-                ``"numeric"`` or ``"string"``. An unrecognized string raises
-                ``ValueError`` listing the valid values.
-        """
-        self.family = DtypeFamily(family)
-
-    def __call__(self, X: Any) -> list[str]:
-        """Return the matching column names.
-
-        Args:
-            X: The table the encoder is being fitted on.
-
-        Returns:
-            Matching column names, in table order.
-        """
-        schema = nw.from_native(X, eager_only=True).schema
-        return [c for c, d in schema.items() if matches(d, self.family)]
-
-    def __repr__(self) -> str:
-        """Show the family value. A cloned estimator then prints readably."""
-        return f"dtype_selector({self.family.value!r})"
-
-
 def get_last_step(selection_pipeline: Any) -> Any:
     """Return the final step of ``selection_pipeline``.

@@ -177,8 +127,7 @@ def _reject_explicit_columns(estimator: Any) -> None:
                     f"explicitly ({list(columns)[:3]}...), which cannot be "
                     "refit once selection narrows the feature matrix, and DFS "
                     "builds its column names so they cannot be known in "
-                    "advance anyway. Use a callable instead, such as "
-                    "tusk.sklearn.dtype_selector('numeric') or "
-                    "dtype_selector('string').",
+                    "advance anyway. Use tusk.sklearn.TableEncoder, which "
+                    "encodes columns by dtype.",
                 )
             _reject_explicit_columns(transformer)
```

Replace `src/tusk/sklearn/__init__.py` with:

```python
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
    EnumEncoder,
    StringEncoder,
    TemporalEncoder,
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
    "EnumEncoder",
    "NarwhalsEncoder",
    "NarwhalsMixin",
    "StringEncoder",
    "TableEncoder",
    "TemporalEncoder",
    "TimeEncoder",
]
```

- [ ] **Step 4: Run the whole suite**

Run: `just test`
Expected: `863 passed, 12 skipped` (counts may differ if main moved; no failures). Also check nothing still names it:

Run: `grep -rn "dtype_selector" src tests docs/api`
Expected: no output.

- [ ] **Step 5: Lint**

```bash
git add -A
just lint
```

- [ ] **Step 6: Commit**

```bash
git add src/tusk/sklearn/_encoders.py src/tusk/sklearn/__init__.py src/tusk/dtypes.py tests/test_sklearn_encoders.py tests/test_sklearn_selector_transformer.py tests/test_sklearn_dfs_transformer.py
git commit -m "refactor!: replace dtype_selector with TableEncoder

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LzdNzyhemKvk3JKEHFajLx"
```

---

### Task 5: Docs

**Files:**
- Modify: `docs/guide/sklearn.md`

**Interfaces:**
- Consumes: the public names from Tasks 1–3.
- Produces: the guide section `## Encoding by dtype with \`TableEncoder\``, linked from the introduction as `#encoding-by-dtype-with-tableencoder`. The heading "Frame backends" becomes "Table backends" (the glossary bans "frame"; nothing links to the old anchor).

- [ ] **Step 1: Apply the docs change**

Save this patch as `docs.diff`, apply it with `git apply docs.diff`, then delete the file:

```diff
diff --git a/docs/guide/sklearn.md b/docs/guide/sklearn.md
index c1602a2..41ba6f5 100644
--- a/docs/guide/sklearn.md
+++ b/docs/guide/sklearn.md
@@ -13,25 +13,18 @@ metadata:

 ```python
 import sklearn
-from sklearn.compose import ColumnTransformer
-from sklearn.ensemble import ExtraTreesClassifier
-from sklearn.impute import SimpleImputer
+from sklearn.ensemble import HistGradientBoostingClassifier
 from sklearn.pipeline import Pipeline

-from tusk.sklearn import DFSTransformer, dtype_selector
+from tusk.sklearn import DFSTransformer, TableEncoder

 sklearn.set_config(enable_metadata_routing=True)

 pipeline = Pipeline(
     steps=[
         ("dfs", DFSTransformer(target_table="customers", max_depth=2)),
-        (
-            "encode",
-            ColumnTransformer(
-                [("numbers", SimpleImputer(), dtype_selector("numeric"))],
-            ),
-        ),
-        ("model", ExtraTreesClassifier()),
+        ("encode", TableEncoder()),
+        ("model", HistGradientBoostingClassifier()),
     ],
 )

@@ -46,8 +39,10 @@ rows. `transform` computes them and returns one row per key, in key order.
 reach the transformer through the pipeline. Set it once per process.

 The feature matrix holds whatever dtypes synthesis produced, so it can carry
-strings and nulls, which most estimators do not take. How to encode them, and
-what value to substitute for a null, is yours to choose.
+strings and nulls, which most estimators do not take.
+[`TableEncoder`](#encoding-by-dtype-with-tableencoder) encodes each column by
+its dtype. A null stays a null (NaN). Choose a model that accepts NaN, such as
+`HistGradientBoostingClassifier`, or give the `numeric` group an imputer.

 Pass `database=` to `predict` to score a different set of keys, from either
 the same database or another one built to the same schema.
@@ -132,7 +127,7 @@ remains:

 ```python
 from sklearn.feature_selection import SelectKBest
-from sklearn.preprocessing import OneHotEncoder, StandardScaler
+from sklearn.impute import SimpleImputer

 from tusk.sklearn import DFSSelectorTransformer

@@ -140,19 +135,7 @@ selector = DFSSelectorTransformer(
     target_table="customers",
     selection_pipeline=Pipeline(
         steps=[
-            (
-                "encode",
-                ColumnTransformer(
-                    [
-                        (
-                            "categories",
-                            OneHotEncoder(handle_unknown="ignore"),
-                            dtype_selector("string"),
-                        ),
-                        ("numbers", StandardScaler(), dtype_selector("numeric")),
-                    ],
-                ),
-            ),
+            ("encode", TableEncoder(numeric=SimpleImputer())),
             ("select", SelectKBest(k=50)),
         ],
     ),
@@ -167,7 +150,7 @@ selector.fit(keys, y_train, database=db)
 After fitting, `features_` holds the kept feature definitions as a
 [`FeatureList`][tusk.FeatureList]. `get_feature_names_out()` gives the encoded
 column names with tusk's names substituted back, such as
-`categories__MODE__orders__products__category_a`.
+`string__MODE__orders__products__category_svd_0`.

 Two cases change what you get:

@@ -180,39 +163,89 @@ Two cases change what you get:
   Consider placing the `PCA` further downstream in the main `Pipeline` and
   not in the `selection_pipeline`.
 - If a feature reaches no encoder, tusk drops it and warns with
-  `UnencodedFeatureWarning` naming how many. Cover every dtype in your
-  feature matrix, or set `remainder="passthrough"`.
+  `UnencodedFeatureWarning` naming how many. A `TableEncoder` group set to
+  `"drop"` is the usual cause. The `other` group, for dtypes such as `List`
+  and `Struct`, drops by default.
+
+## Encoding by dtype with `TableEncoder`
+
+`TableEncoder` puts each column into one group by its narwhals dtype and
+encodes each group with its own estimator:
+
+| Parameter | Dtypes | Default |
+| --- | --- | --- |
+| `numeric` | integers, floats, `Decimal` | `"passthrough"` |
+| `boolean` | `Boolean` | `"passthrough"` |
+| `string` | `String` | `StringEncoder()` |
+| `categorical` | `Categorical` | `OneHotEncoder(handle_unknown="ignore")` |
+| `enum` | `Enum` | `EnumEncoder()` |
+| `date` | `Date` | `DateEncoder()`: month, day |
+| `time` | `Time` | `TimeEncoder()`: hour, minute |
+| `datetime` | `Datetime` | `DatetimeEncoder()`: month, day, hour, minute |
+| `duration` | `Duration` | `DurationEncoder()`: total seconds |
+| `other` | every other dtype | `"drop"` |
+
+Each parameter takes an estimator, `"passthrough"` or `"drop"`. Output names
+are `{group}__{name}`, such as `date__signed_up_at_month`.
+
+The dtype of a text column chooses its encoding:
+
+- `Categorical`: one column per value (one-hot). Use it for a few unordered
+  values.
+- `Enum`: the value's position in the category order. Use it for ordered
+  values.
+- `String`: `StringEncoder` computes a TF-IDF vector over the value's
+  character 3- and 4-grams and reduces it to 30 coordinates. Values that
+  share spelling get close coordinates. Use it for free text and for values
+  with many distinct entries.
+
+Cast a column in the database to choose its encoding.
+
+The temporal encoders take `components`:
+
+```python
+from tusk.sklearn import DateEncoder

-## Choosing columns with `dtype_selector`
+TableEncoder(date=DateEncoder(components=["year", "month", "weekday"]))
+```

 Synthesis builds the feature matrix's column names, so you cannot know them
-all in advance. Which features exist depends on your schema, your primitives
-and `max_depth`. tusk therefore rejects a `ColumnTransformer` given an
-explicit list of names, raising `EncoderError`.
-
-`dtype_selector` picks columns by dtype instead. It takes a
-[`DtypeFamily`](../api/dtypes.md), the same families that decide which
-primitives apply to which columns:
-
-| Family | Matches |
-| --- | --- |
-| `"numeric"` | integers and floats |
-| `"temporal"` | `Date`, `Datetime`, `Duration`, `Time` (every temporal dtype) |
-| `"has_date"` | `Date`, `Datetime` (columns you can read a calendar position from) |
-| `"has_time"` | `Datetime`, `Time` (columns you can read an hour or minute from) |
-| `"duration"` | `Duration` |
-| `"string"` | `String` |
-| `"categorical"` | `Categorical`, `Enum` |
-| `"boolean"` | `Boolean` |
-
-`"string"` and `"categorical"` are separate: a column you declared
-`Categorical` is a label, a `String` column carries no such declaration.
-
-scikit-learn's own `make_column_selector` does the same job but accepts only
-pandas. `dtype_selector` reads the schema through narwhals, so it works on
-every backend a tusk database can use.
-
-## Frame backends
+all in advance. tusk therefore rejects a `ColumnTransformer` given an
+explicit list of names, raising `EncoderError`. `TableEncoder` needs no
+names.
+
+### Any estimator, any backend
+
+A group gets numpy unless its estimator has `NarwhalsMixin`. Mix it in to
+give an estimator a table narwhals can read, from any backend, converted to
+`convert_to`:
+
+```python
+from sklearn.preprocessing import TargetEncoder
+
+from tusk.sklearn import NarwhalsMixin
+
+
+class PandasTargetEncoder(NarwhalsMixin, TargetEncoder):
+    convert_to = "pandas"
+
+
+TableEncoder(string=PandasTargetEncoder())
+```
+
+`convert_to` is `"numpy"`, `"pandas"`, `"polars"` or `"narwhals"`.
+
+`NarwhalsEncoder` on its own passes every column through unchanged.
+`set_output` sets what it returns. Use it to hand a feature matrix from any
+backend to scikit-learn:
+
+```python
+from tusk.sklearn import NarwhalsEncoder
+
+NarwhalsEncoder().set_output(transform="pandas")
+```
+
+## Table backends

 tusk collects the feature matrix to whatever backend the database already
 uses, so narwhals-native transformers get the native table type they want.
@@ -227,6 +260,7 @@ not include it. Two cases need it:

 - `ColumnTransformer` cannot read pyarrow tables, which is what a duckdb
   database collects to. Use `"pandas"` or `"polars"` in that case.
+  `TableEncoder` and `NarwhalsEncoder` read every backend.
 - scikit-learn reads polars DataFrames through a dataframe interchange
   protocol that polars deprecated, so fitting one emits harmless
   `DeprecationWarning`s. `"pandas"` avoids them.
```

- [ ] **Step 2: Build the site**

Run: `just docs`
Expected: `No issues found`.

- [ ] **Step 3: Check the terms**

Run: `grep -n -i "frame\b\|dataframe\|missing\|generate\|calculate" docs/guide/sklearn.md`
Expected: only lines that name a library type, such as "pandas DataFrame" or "polars DataFrames", or none.

- [ ] **Step 4: Commit**

```bash
git add docs/guide/sklearn.md
git commit -m "docs: document TableEncoder

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LzdNzyhemKvk3JKEHFajLx"
```

---

### Task 6: Final verification

- [ ] **Step 1: Lint and test everything**

Run: `just lint && just test`
Expected: every hook passes; no test fails.

- [ ] **Step 2: Test the scikit-learn floor**

Run: `uv run --with "scikit-learn==1.4.2" --with "numpy<2" pytest tests/test_sklearn_narwhals.py tests/test_sklearn_column_encoders.py tests/test_sklearn_table_encoder.py tests/test_sklearn_selector_transformer.py tests/test_sklearn_dfs_transformer.py tests/test_sklearn_encoders.py -q`
Expected: `91 passed`.

- [ ] **Step 3: Review the diff against the spec**

Run: `git diff main --stat` and read `git diff main -- src`.
Check: no `dtype_selector` left; every public class in Global Constraints is exported; no new dependency in `pyproject.toml`; no class the spec names public has a leading underscore.
