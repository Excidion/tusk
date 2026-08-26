# scikit-learn Transformers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `tusk.sklearn`, an optional-extra module giving `DFSTransformer` (deep feature synthesis as a pipeline step) and `DFSSelectorTransformer` (the same, plus pruning tusk features down to what a user-supplied encode-and-select pipeline actually kept, so inference computes only the survivors).

**Architecture:** A new `src/tusk/sklearn/` package with four private modules and one public `__init__`. `_frames.py` owns the collect boundary — tusk never collects, and this is the one place that does, because sklearn cannot consume a query plan. `_lineage.py` owns the sentinel-renaming trick that reconnects the selector's support mask (which indexes *encoded* columns) to tusk features (which index *matrix* columns). `_encoders.py` owns the contract the user's encoder must satisfy. `_transformers.py` holds the two estimators. Six tasks: helpers bottom-up, then the two estimators, then docs.

**Tech Stack:** Python 3.10+, narwhals, scikit-learn (optional extra), polars/pyarrow/duckdb backends, pytest, uv, ruff + ty + interrogate + pydoclint via pre-commit, zensical for docs.

**Spec:** `docs/superpowers/specs/2026-08-24-sklearn-transformer-design.md`

**One deviation from the spec:** it names a single module `src/tusk/sklearn.py`; this plan uses a `src/tusk/sklearn/` package instead. tusk's docstring density puts the whole thing well past 500 lines in one file, and the four concerns split cleanly. The public import path is identical — `from tusk.sklearn import DFSTransformer` — so no documented contract changes.

## Global Constraints

- **Standard test command:** `uv run --group validation pytest -q`. The `validation` group must be present or `tests/differential/` fails to *collect* (it imports numpy/pandas/featuretools) even though the marker deselects it. That group is named for the featuretools differential tests and has nothing to do with this feature.
- **The baseline is NOT green.** Before any work: `2 failed, 266 passed, 1 skipped, 41 deselected`. The two failures are `test_a_date_cutoff_is_rejected` and `test_a_string_cutoff_is_rejected` in `tests/test_deep_feature_synthesis.py`; they expect `"must be a datetime.datetime"` while `src/tusk/api.py:139` raises `"'cutoff_time' must be a datetime."`. **These are pre-existing and out of scope. Do not fix them, and do not count them as your failures.** Your task passes when the counts are `2 failed, N passed` with N grown by your new tests.
- **`tusk/__init__.py` must NOT import `tusk.sklearn`.** Doing so makes `import tusk` fail for everyone without the extra. Nothing here is added to `tusk.__all__`. There is a test for this in Task 4.
- **Metadata routing is required at runtime**, declared per class:
  ```python
  __metadata_request__fit = {"database": True}
  __metadata_request__transform = {"database": True}
  ```
  Never call `set_fit_request`/`set_transform_request` in `__init__` — those raise `RuntimeError` when `enable_metadata_routing` is off, which it is by default. Tests that exercise a `Pipeline` must wrap in `sklearn.config_context(enable_metadata_routing=True)`.
- **No `**kwargs` in any `__init__`.** Every parameter is declared explicitly and stored unmodified under its own name, or `BaseEstimator.get_params`/`clone`/`GridSearchCV` break. Do not validate or transform parameters in `__init__`; sklearn requires `__init__` to only assign. Validation happens in `fit`.
- **Two column spaces, never conflated.** *tusk space* = the matrix, `M` columns, indexed by feature `output_names`. *encoded space* = the encoder prefix's output, `E` columns, indexed by `get_feature_names_out()`. The selector's support mask indexes **encoded** space. Pruning happens in **tusk** space. If you find yourself indexing the matrix with `support`, stop — that is the central bug this design guards against.
- **Multi-output features are all-or-nothing.** A `Feature` may own several matrix columns via `output_names`. You cannot compute one output without the others, so a feature survives if *any* of its columns is live, and then contributes *all* of them.
- **Pre-commit gates every commit** with ruff (`E,F,I,UP,B,D`), ty, interrogate (`fail-under = 100`) and pydoclint. Every function, including private ones, needs a complete Google-style docstring whose `Args:` names match the signature, plus `Returns:`/`Raises:`/`Warns:` where they apply. A missing section fails at commit time, not test time.
- **Docstrings explain why, not what.** A docstring restating the signature is noise. Match the voice of `src/tusk/database.py` and `src/tusk/compiler.py`, which explain rationale and trade-offs.
- **Out of scope, do not build:** upstream's `DFSSelectionTransformer` and any null-ratio or single-value heuristics; a `restrict_columns` rewriter for explicit-column-list encoders (they are refused, see Task 3); padding dropped columns to preserve encoder width (measured at 20x memory and rejected); per-row cutoff times.

---

### Task 1: Packaging, exceptions, and the collect boundary

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/tusk/exceptions.py`
- Create: `src/tusk/sklearn/__init__.py`
- Create: `src/tusk/sklearn/_frames.py`
- Test: `tests/test_sklearn_frames.py` (create)

**Interfaces:**
- Consumes: `tusk.exceptions.TuskError`, narwhals, numpy.
- Produces:
  - `tusk.exceptions.LineageWarning(UserWarning)`
  - `tusk.exceptions.LineageError(TuskError)`
  - `tusk.exceptions.EncoderError(TuskError)`
  - `tusk.sklearn._frames.as_keys(X: Any) -> list[Any]`
  - `tusk.sklearn._frames.collect_matrix(lazy_native: Any, primary_key: str, keys: Sequence[Any], output_backend: str | None) -> Any`
  - `tusk.sklearn._frames.backend_hint(frame: Any) -> AbstractContextManager[None]`

- [ ] **Step 1: Add the extra and the dev dependency**

In `pyproject.toml`, after the `dependencies` line add:

```toml
[project.optional-dependencies]
sklearn = ["scikit-learn>=1.4"]
```

and add `"scikit-learn>=1.4",` to the `dev` list in `[dependency-groups]`, immediately after `"ruff>=0.16.3",`.

Then run `uv sync --group dev` and confirm `uv run python -c "import sklearn; print(sklearn.__version__)"` prints a version.

- [ ] **Step 2: Write the failing test**

Create `tests/test_sklearn_frames.py`:

```python
import narwhals as nw
import numpy as np
import polars as pl
import pytest

from tusk.exceptions import SchemaError
from tusk.sklearn._frames import as_keys, collect_matrix


@pytest.mark.parametrize(
    ("label", "value"),
    [
        ("list", [1, 2, 3]),
        ("range", range(1, 4)),
        ("1d array", np.array([1, 2, 3])),
        ("column array", np.array([[1], [2], [3]])),
        ("one column frame", pl.DataFrame({"id": [1, 2, 3]})),
    ],
)
def test_as_keys_accepts_every_documented_shape(label, value):
    assert as_keys(value) == [1, 2, 3]


@pytest.mark.parametrize(
    "value",
    [np.array([[1, 2], [3, 4]]), pl.DataFrame({"a": [1], "b": [2]})],
)
def test_as_keys_rejects_more_than_one_column(value):
    with pytest.raises(SchemaError, match="exactly one column"):
        as_keys(value)


def test_collect_matrix_returns_rows_in_key_order():
    frame = pl.LazyFrame({"id": [1, 2, 3, 4], "a": [10.0, 20.0, 30.0, 40.0]})
    out = nw.from_native(collect_matrix(frame, "id", [3, 1, 4], None))
    assert out["a"].to_list() == [30.0, 10.0, 40.0]


def test_collect_matrix_drops_the_primary_key():
    frame = pl.LazyFrame({"id": [1, 2], "a": [10.0, 20.0]})
    out = nw.from_native(collect_matrix(frame, "id", [1, 2], None))
    assert out.columns == ["a"]


def test_collect_matrix_rejects_a_duplicate_key():
    frame = pl.LazyFrame({"id": [1, 2], "a": [10.0, 20.0]})
    with pytest.raises(SchemaError, match="duplicate"):
        collect_matrix(frame, "id", [1, 1], None)


def test_collect_matrix_rejects_a_key_with_no_row():
    frame = pl.LazyFrame({"id": [1, 2], "a": [10.0, 20.0]})
    with pytest.raises(SchemaError, match="no row"):
        collect_matrix(frame, "id", [1, 99], None)


def test_backend_hint_reraises_the_original_exception_type():
    from tusk.sklearn._frames import backend_hint

    with pytest.raises(ValueError, match="boom"):
        with backend_hint(pl.DataFrame({"a": [1]})):
            raise ValueError("boom")


def test_backend_hint_names_the_backend_and_the_fix():
    import sys

    from tusk.sklearn._frames import backend_hint

    try:
        with backend_hint(pl.DataFrame({"a": [1]})):
            raise ValueError("boom")
    except ValueError as exc:
        hint = " ".join(getattr(exc, "__notes__", []))
        if sys.version_info < (3, 11):
            return
    assert "polars" in hint.lower()
    assert "output_backend" in hint
```

Wrapping the error would change its type and break a user's `except ValueError`
around their own pipeline, so the first test is the one that matters.

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run --group validation pytest tests/test_sklearn_frames.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tusk.sklearn'`

- [ ] **Step 4: Add the three exception types**

Append to `src/tusk/exceptions.py`:

```python
class LineageError(TuskError):
    """Raised when a kept encoded column has no counterpart after the refit.

    The only way sentinel lineage can *under*-keep is by missing a source, so
    this is the tripwire for that case: if a feature was wrongly pruned, a name
    the frozen mask needs goes missing from the refitted encoder's output.
    Failing here is the alternative to silently serving different columns than
    the ones the selector chose.
    """


class EncoderError(TuskError):
    """Raised when the supplied encoder cannot be refit on a column subset.

    Pruning narrows the matrix, so the encoder is refit on fewer columns than
    it first saw. A ``ColumnTransformer`` naming its columns explicitly cannot
    survive that -- and after DFS it is already a mistake for two other
    reasons, so it is refused rather than worked around.
    """


class UnencodedFeatureWarning(UserWarning):
    """Warns that a feature fed no encoded column, so it was pruned.

    The encoder simply never looked at it -- most often a ``ColumnTransformer``
    covering only some dtypes while ``remainder`` stays at its ``"drop"``
    default. Pruning it is self-consistent, but doing so silently would let a
    user lose every numeric feature without a word. Its own class, so it can be
    filtered independently.
    """


class LineageWarning(UserWarning):
    """Warns that lineage was unrecoverable, so nothing was pruned.

    An encoder whose output names do not mention their inputs -- ``PCA`` gives
    ``pca0`` -- severs the link between kept columns and tusk features. Keeping
    every feature is the correct fallback, since pruning is an optimization and
    no result depends on it. Its own class, so it can be filtered independently.
    """
```

- [ ] **Step 5: Create the package init**

Create `src/tusk/sklearn/__init__.py`:

```python
"""scikit-learn estimators wrapping deep feature synthesis.

Deliberately not imported by :mod:`tusk`: importing scikit-learn eagerly would
break every install without the ``sklearn`` extra. Reach these through
``from tusk.sklearn import DFSTransformer``, as with ``xgboost.sklearn``.
"""

from __future__ import annotations

__all__: list[str] = []
```

- [ ] **Step 6: Write the frame helpers**

Create `src/tusk/sklearn/_frames.py`:

```python
"""The collect boundary.

tusk never collects; this is the one module that does, because scikit-learn
cannot consume a query plan. Everything here is about crossing that line
exactly once, in a way that cannot silently misalign rows against ``y``.
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Iterator, Sequence
from typing import Any

import narwhals as nw
import numpy as np

from tusk.exceptions import SchemaError

_POSITION = "__tusk_position"


def as_keys(X: Any) -> list[Any]:
    """Normalize ``X`` to a flat list of primary-key values.

    Args:
        X: A 1-D sequence, an ``(n, 1)`` array, or a one-column frame.

    Returns:
        The key values, in the order given. That order is the contract: it
        becomes the row order of the feature matrix, which is what keeps ``X``
        aligned with ``y``.

    Raises:
        SchemaError: If ``X`` has more than one column, or more than two
            dimensions.
    """
    frame = nw.from_native(X, eager_only=True, pass_through=True)
    if isinstance(frame, nw.DataFrame):
        if len(frame.columns) != 1:
            raise SchemaError(
                f"X must have exactly one column, the target's primary key; "
                f"got {len(frame.columns)}",
            )
        return frame[frame.columns[0]].to_list()
    array = np.asarray(X)
    if array.ndim == 2:
        if array.shape[1] != 1:
            raise SchemaError(
                f"X must have exactly one column, the target's primary key; "
                f"got {array.shape[1]}",
            )
        array = array[:, 0]
    elif array.ndim != 1:
        raise SchemaError(f"X must be 1- or 2-dimensional; got {array.ndim}")
    return array.tolist()


def collect_matrix(
    lazy_native: Any,
    primary_key: str,
    keys: Sequence[Any],
    output_backend: str | None,
) -> Any:
    """Materialize the feature matrix for ``keys``, in ``keys`` order.

    Args:
        lazy_native: The uncomputed matrix from ``apply_features``.
        primary_key: The target table's key column.
        keys: Key values selecting and ordering the rows.
        output_backend: Backend to collect to, or None to collect natively.

    Returns:
        An eager native frame with one row per key, in key order, without the
        primary key column -- it is a join key, not a feature.

    Raises:
        SchemaError: If ``keys`` repeats a value, or names a key that produced
            no row.
        TuskError: If ``output_backend`` names an uninstalled package.
    """
    keys = list(keys)
    if len(set(keys)) != len(keys):
        raise SchemaError(
            "X contains duplicate keys; the feature matrix is keyed by the "
            "target's primary key, so each row must be requested at most once",
        )

    frame = nw.from_native(lazy_native).lazy()
    filtered = frame.filter(nw.col(primary_key).is_in(keys))
    collected = _collect(filtered, output_backend)

    found = set(collected[primary_key].to_list())
    missing = [k for k in keys if k not in found]
    if missing:
        raise SchemaError(
            f"no row for {len(missing)} of {len(keys)} keys, e.g. "
            f"{missing[:5]}; they are absent from {primary_key!r} or were "
            f"excluded by cutoff_time",
        )

    # The key dtype must be taken from the collected frame rather than
    # inferred: inference gives int64, and a duckdb database keyed by int32
    # then fails the join with ArrowInvalid.
    order = nw.from_dict(
        {primary_key: keys, _POSITION: list(range(len(keys)))},
        schema={primary_key: collected.schema[primary_key], _POSITION: nw.Int64()},
        backend=collected.implementation,
    )
    joined = order.join(collected, on=primary_key, how="left").sort(_POSITION)
    return joined.drop([primary_key, _POSITION]).to_native()


def _collect(frame: nw.LazyFrame, output_backend: str | None) -> nw.DataFrame:
    """Collect, translating a missing backend package into a tusk error.

    Args:
        frame: The filtered lazy matrix.
        output_backend: Backend name, or None for the database's own.

    Returns:
        The collected frame.

    Raises:
        TuskError: If ``output_backend`` names a package that is not installed.
    """
    from tusk.exceptions import TuskError

    if output_backend is None:
        return frame.collect()
    try:
        return frame.collect(backend=output_backend)
    except ModuleNotFoundError as exc:
        raise TuskError(
            f"output_backend={output_backend!r} needs the {output_backend} "
            f"package, which is not installed; `uv add {output_backend}`",
        ) from exc


@contextlib.contextmanager
def backend_hint(frame: Any) -> Iterator[None]:
    """Attach a backend hint to whatever the user's pipeline raises.

    Many sklearn transformers reject non-pandas frames in ways that read as
    unrelated type errors -- ``ColumnTransformer`` on pyarrow gives
    ``TypeError: Index must either be string or integer``. The hint names the
    backend and the fix.

    The exception is re-raised **unchanged**. Wrapping it would change its
    type and break a user's ``except ValueError`` around their own pipeline.

    Args:
        frame: The matrix handed to the pipeline, used to name the backend.

    Yields:
        None.
    """
    try:
        yield
    except Exception as exc:
        backend = nw.from_native(frame, eager_only=True, pass_through=True)
        name = getattr(getattr(backend, "implementation", None), "name", "unknown")
        hint = (
            f"the feature matrix was collected as {name}; if this pipeline "
            f'needs pandas, set output_backend="pandas"'
        )
        if hasattr(exc, "add_note"):
            exc.add_note(hint)
        else:
            warnings.warn(hint, UserWarning, stacklevel=2)
        raise
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run --group validation pytest tests/test_sklearn_frames.py -q`
Expected: PASS, 11 tests.

- [ ] **Step 8: Run the full suite**

Run: `uv run --group validation pytest -q`
Expected: `2 failed, 277 passed, 1 skipped, 41 deselected` — the same two pre-existing failures, nothing new.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock src/tusk/exceptions.py src/tusk/sklearn tests/test_sklearn_frames.py
git commit -m "feat: add tusk[sklearn] extra and the collect boundary"
```

---

### Task 2: Sentinel lineage

**Files:**
- Create: `src/tusk/sklearn/_lineage.py`
- Test: `tests/test_sklearn_lineage.py` (create)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `tusk.sklearn._lineage.Sentinels` — frozen dataclass with fields `prefix: str`, `width: int`, `columns: tuple[str, ...]`, `mapping: dict[str, str]`
  - `tusk.sklearn._lineage.make_sentinels(columns: Sequence[str]) -> Sentinels`
  - `Sentinels.sources(name: str) -> list[str]`
  - `Sentinels.restore(name: str) -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sklearn_lineage.py`:

```python
import pytest

from tusk.sklearn._lineage import make_sentinels

COLUMNS = ["age", "MODE__transactions__category", "COUNT__transactions"]


def test_every_column_gets_a_distinct_sentinel():
    sentinels = make_sentinels(COLUMNS)
    assert len(set(sentinels.mapping.values())) == len(COLUMNS)


def test_sentinels_are_fixed_width_so_prefixes_cannot_collide():
    sentinels = make_sentinels([f"c{i}" for i in range(11)])
    lengths = {len(name) for name in sentinels.mapping.values()}
    assert len(lengths) == 1


def test_a_one_hot_name_resolves_to_its_source_column():
    sentinels = make_sentinels(COLUMNS)
    encoded = f"oh__{sentinels.mapping['MODE__transactions__category']}_a"
    assert sentinels.sources(encoded) == ["MODE__transactions__category"]


def test_a_multi_input_name_resolves_to_every_source():
    sentinels = make_sentinels(COLUMNS)
    encoded = f"{sentinels.mapping['age']} {sentinels.mapping['COUNT__transactions']}"
    assert sentinels.sources(encoded) == ["age", "COUNT__transactions"]


def test_an_opaque_name_resolves_to_nothing():
    assert make_sentinels(COLUMNS).sources("pca0") == []


def test_restore_puts_real_names_back():
    sentinels = make_sentinels(COLUMNS)
    encoded = f"oh__{sentinels.mapping['MODE__transactions__category']}_a"
    assert sentinels.restore(encoded) == "oh__MODE__transactions__category_a"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --group validation pytest tests/test_sklearn_lineage.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tusk.sklearn._lineage'`

- [ ] **Step 3: Write the implementation**

Create `src/tusk/sklearn/_lineage.py`:

```python
"""Reconnecting encoded columns to the tusk features they came from.

featuretools can prune feature definitions because ``encode_features`` returns
definitions for the one-hot columns, keeping the selector's mask and the
feature list 1:1. Handing encoding to scikit-learn severs that link: a support
mask indexes *encoded* columns and says nothing about which feature produced
them.

This module rebuilds the link by exploiting scikit-learn's own convention that
output names derive from input names. Rename the matrix columns to opaque
tokens, and every downstream name carries its provenance in plain sight.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Sentinels:
    """An opaque renaming of matrix columns, and the map back.

    Attributes:
        prefix: Random per-fit token prefix.
        width: Zero-padded index width, fixed so no sentinel is a prefix of
            another.
        columns: The tusk-space column names, positionally indexed.
        mapping: Column name to sentinel.
    """

    prefix: str
    width: int
    columns: tuple[str, ...]
    mapping: dict[str, str] = field(compare=False)

    @property
    def _pattern(self) -> re.Pattern[str]:
        """Matches one sentinel and captures its index."""
        return re.compile(rf"{re.escape(self.prefix)}(\d{{{self.width}}})")

    def sources(self, name: str) -> list[str]:
        """Columns an encoded name derives from.

        Every match is collected rather than the first, so a multi-input
        transformer such as ``PolynomialFeatures`` correctly reports both of
        its sources and keeps both features alive.

        Args:
            name: One name from the encoder's ``get_feature_names_out()``.

        Returns:
            The tusk-space columns it mentions, in column order. Empty means
            lineage is unrecoverable for this name -- the caller must then keep
            every feature rather than guess.
        """
        found = {int(m) for m in self._pattern.findall(name)}
        return [self.columns[i] for i in sorted(found) if i < len(self.columns)]

    def restore(self, name: str) -> str:
        """Substitute sentinels back to real column names.

        Args:
            name: One name from the encoder's ``get_feature_names_out()``.

        Returns:
            The same name with every sentinel replaced, so users read
            ``oh__MODE__transactions__category_a`` rather than
            ``oh___t9f3a_0001_a``.
        """

        def replace(match: re.Match[str]) -> str:
            index = int(match.group(1))
            return self.columns[index] if index < len(self.columns) else match.group(0)

        return self._pattern.sub(replace, name)


def make_sentinels(columns: Sequence[str]) -> Sentinels:
    """Build an opaque renaming for one fit.

    The prefix is random per fit. A data *value* that happens to look like a
    sentinel would otherwise be misread as a column reference; randomness makes
    that vanishingly unlikely, and its only consequence is an extra source,
    which over-keeps rather than over-prunes.

    Args:
        columns: The matrix's column names, in order.

    Returns:
        The sentinel mapping and the map back.
    """
    prefix = f"_t{secrets.token_hex(4)}_"
    width = max(4, len(str(max(len(columns) - 1, 0))))
    mapping = {c: f"{prefix}{i:0{width}d}" for i, c in enumerate(columns)}
    return Sentinels(prefix, width, tuple(columns), mapping)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --group validation pytest tests/test_sklearn_lineage.py -q`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add src/tusk/sklearn/_lineage.py tests/test_sklearn_lineage.py
git commit -m "feat: recover encoded-column lineage via sentinel names"
```

---

### Task 3: The encoder contract and `dtype_selector`

**Files:**
- Create: `src/tusk/sklearn/_encoders.py`
- Modify: `src/tusk/sklearn/__init__.py`
- Test: `tests/test_sklearn_encoders.py` (create)

**Interfaces:**
- Consumes: `tusk.exceptions.EncoderError` (Task 1).
- Produces:
  - `tusk.sklearn.dtype_selector(kind: str)` — callable column selector, `kind` in `{"numeric", "categorical"}`
  - `tusk.sklearn._encoders.encoder_prefix(inner: Any) -> Any`
  - `tusk.sklearn._encoders.selector_of(inner: Any) -> Any`
  - `tusk.sklearn._encoders.validate_inner(inner: Any) -> None`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sklearn_encoders.py`:

```python
import datetime as dt

import polars as pl
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tusk.exceptions import EncoderError
from tusk.sklearn import dtype_selector
from tusk.sklearn._encoders import encoder_prefix, selector_of, validate_inner

FRAME = pl.DataFrame(
    {"age": [20.0, 30.0], "cat": ["a", "b"], "cnt": [1.0, 2.0]},
)


def test_dtype_selector_splits_numeric_from_categorical():
    assert dtype_selector("numeric")(FRAME) == ["age", "cnt"]
    assert dtype_selector("categorical")(FRAME) == ["cat"]


def test_dtype_selector_keeps_booleans_and_dates_out_of_categorical():
    frame = pl.DataFrame(
        {
            "n": [1.0],
            "s": ["a"],
            "b": [True],
            "d": [dt.datetime(2024, 1, 1)],
        },
    )
    assert dtype_selector("numeric")(frame) == ["n"]
    assert dtype_selector("categorical")(frame) == ["s"]
    assert dtype_selector("boolean")(frame) == ["b"]
    assert dtype_selector("temporal")(frame) == ["d"]


def test_dtype_selector_rejects_an_unknown_family():
    with pytest.raises(ValueError, match="kind must be one of"):
        dtype_selector("stringy")


def test_dtype_selector_reevaluates_on_a_subset():
    assert dtype_selector("numeric")(FRAME.select(["age"])) == ["age"]


def test_dtype_selector_works_inside_a_column_transformer_on_polars():
    encoder = ColumnTransformer(
        [
            ("oh", OneHotEncoder(handle_unknown="ignore"), dtype_selector("categorical")),
            ("num", StandardScaler(), dtype_selector("numeric")),
        ],
    ).fit(FRAME)
    assert list(encoder.get_feature_names_out()) == [
        "oh__cat_a",
        "oh__cat_b",
        "num__age",
        "num__cnt",
    ]


def test_selector_of_finds_the_last_step():
    selector = SelectKBest(f_classif, k=1)
    assert selector_of(Pipeline([("s", StandardScaler()), ("sel", selector)])) is selector
    assert selector_of(selector) is selector


def test_encoder_prefix_is_identity_when_there_is_no_encoder():
    prefix = encoder_prefix(SelectKBest(f_classif, k=1))
    fitted = prefix.fit(FRAME)
    assert list(fitted.get_feature_names_out()) == ["age", "cat", "cnt"]


def test_encoder_prefix_of_a_single_step_pipeline_is_identity():
    prefix = encoder_prefix(Pipeline([("sel", SelectKBest(f_classif, k=1))]))
    assert list(prefix.fit(FRAME).get_feature_names_out()) == ["age", "cat", "cnt"]


def test_inner_must_end_in_a_selector():
    with pytest.raises(EncoderError, match="SelectorMixin"):
        validate_inner(Pipeline([("s", StandardScaler())]))


def test_explicit_column_lists_are_refused():
    inner = Pipeline(
        [
            ("enc", ColumnTransformer([("num", StandardScaler(), ["age", "cnt"])])),
            ("sel", SelectKBest(f_classif, k=1)),
        ],
    )
    with pytest.raises(EncoderError, match="dtype_selector"):
        validate_inner(inner)


class _Opaque(BaseEstimator, TransformerMixin):
    """A transformer with no get_feature_names_out.

    TransformerMixin does not supply one -- only OneToOneFeatureMixin and
    ClassNamePrefixFeaturesOutMixin do -- so this needs no trickery. It stands
    in for scikit-lego's TypeSelector without taking the dependency.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


def test_a_step_without_feature_names_is_refused():
    inner = Pipeline(
        [("opaque", _Opaque()), ("sel", SelectKBest(f_classif, k=1))],
    )
    with pytest.raises(EncoderError, match="get_feature_names_out"):
        validate_inner(inner)


def test_a_callable_column_list_is_accepted():
    inner = Pipeline(
        [
            ("enc", ColumnTransformer([("num", StandardScaler(), dtype_selector("numeric"))])),
            ("sel", SelectKBest(f_classif, k=1)),
        ],
    )
    validate_inner(inner)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --group validation pytest tests/test_sklearn_encoders.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tusk.sklearn._encoders'`

- [ ] **Step 3: Write the implementation**

Create `src/tusk/sklearn/_encoders.py`:

```python
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

from tusk.exceptions import EncoderError


FAMILIES = ("numeric", "temporal", "boolean", "categorical")


def _matches(kind: str, dtype: Any) -> bool:
    """Whether a dtype belongs to a family.

    Four explicit families rather than "numeric or not": narwhals reports
    ``is_numeric()`` for numbers and ``is_temporal()`` for dates and reports
    **neither** for ``Boolean``, ``String`` and ``Categorical``, so a binary
    split would feed Booleans and Datetimes to a one-hot encoder.

    Args:
        kind: One of :data:`FAMILIES`.
        dtype: A narwhals dtype.

    Returns:
        Whether the dtype is in that family.
    """
    if kind == "numeric":
        return bool(dtype.is_numeric())
    if kind == "temporal":
        return bool(dtype.is_temporal())
    if kind == "boolean":
        return bool(dtype == nw.Boolean)
    return dtype in (nw.String, nw.Categorical, nw.Enum)


class dtype_selector:  # noqa: N801
    """Select columns by dtype family, on any backend narwhals supports.

    scikit-learn's ``make_column_selector`` is the right shape but rejects
    anything that is not pandas, which collides with collecting natively. skrub
    and scikit-lego were surveyed too: skrub's selectors are not callable and
    ``ColumnTransformer`` rejects them, and scikit-lego's ``TypeSelector`` is a
    transformer with no ``get_feature_names_out``, which lineage depends on.
    So this is the same idea through narwhals.

    Being a callable rather than a fixed list is what makes the refit work: it
    re-evaluates against whatever frame it is handed, so narrowing the matrix
    narrows the selection instead of breaking it.

    Attributes:
        kind: One of :data:`FAMILIES`.
    """

    def __init__(self, kind: str) -> None:
        """Create a selector.

        Args:
            kind: One of :data:`FAMILIES`.

        Raises:
            ValueError: If ``kind`` is not a known family.
        """
        if kind not in FAMILIES:
            raise ValueError(f"kind must be one of {FAMILIES}; got {kind!r}")
        self.kind = kind

    def __call__(self, X: Any) -> list[str]:
        """Return the matching column names.

        Args:
            X: The frame the encoder is being fitted on.

        Returns:
            Matching column names, in frame order.
        """
        schema = nw.from_native(X, eager_only=True).schema
        return [c for c, d in schema.items() if _matches(self.kind, d)]

    def __repr__(self) -> str:
        """Show the kind, so cloned estimators print readably."""
        return f"dtype_selector({self.kind!r})"


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
                    f"explicitly ({list(columns)[:3]}...). DFS generates column "
                    "names, so they cannot be known in advance, and "
                    'remainder="drop" would silently discard every feature not '
                    "named. Use tusk.sklearn.dtype_selector('numeric') or "
                    "dtype_selector('categorical') instead.",
                )
            _reject_explicit_columns(transformer)
```

- [ ] **Step 4: Export `dtype_selector`**

Replace the `__all__` line in `src/tusk/sklearn/__init__.py`:

```python
from tusk.sklearn._encoders import dtype_selector

__all__ = ["dtype_selector"]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --group validation pytest tests/test_sklearn_encoders.py -q`
Expected: PASS, 12 tests.

- [ ] **Step 6: Commit**

```bash
git add src/tusk/sklearn tests/test_sklearn_encoders.py
git commit -m "feat: add dtype_selector and the inner-pipeline contract"
```

---

### Task 4: `DFSTransformer`

**Files:**
- Create: `src/tusk/sklearn/_transformers.py`
- Modify: `src/tusk/sklearn/__init__.py`
- Test: `tests/test_sklearn_dfs_transformer.py` (create)

**Interfaces:**
- Consumes: `_frames.as_keys`, `_frames.collect_matrix` (Task 1); `tusk.synthesis.synthesize`, `tusk.api.apply_features`, `tusk.primitives.aggregation.AGG_DEFAULTS`, `tusk.primitives.transform.TRANS_DEFAULTS`.
- Produces:
  - `tusk.sklearn.DFSTransformer` with `fit(X, y=None, database=None)`, `transform(X, database=None)`, `fit_transform(X, y=None, database=None, **kwargs)`, `get_feature_names_out(input_features=None)`, and fitted attributes `features_: list[Feature]`, `database_: Database`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sklearn_dfs_transformer.py`:

```python
import warnings

import narwhals as nw
import numpy as np
import pytest
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

from tusk.exceptions import SchemaError
from tusk.sklearn import DFSTransformer

KEYS = [1, 2, 3]
Y = [0, 1, 0]


def _transformer():
    return DFSTransformer(target_table="customers", max_depth=2)


def test_fit_reads_no_rows_and_produces_feature_definitions(db):
    fitted = _transformer().fit(KEYS, database=db)
    assert fitted.features_
    assert all(f.table == "customers" for f in fitted.features_)


def test_transform_returns_one_row_per_key_in_key_order(db):
    fitted = _transformer().fit(KEYS, database=db)
    out = nw.from_native(fitted.transform([3, 1], database=db))
    assert out.shape[0] == 2


def test_the_primary_key_is_not_a_feature(db):
    fitted = _transformer().fit(KEYS, database=db)
    out = nw.from_native(fitted.transform(KEYS, database=db))
    assert "id" not in out.columns


def test_feature_names_match_the_matrix_columns(db):
    fitted = _transformer().fit(KEYS, database=db)
    out = nw.from_native(fitted.transform(KEYS, database=db))
    assert list(fitted.get_feature_names_out()) == list(out.columns)


def test_an_unknown_key_raises_rather_than_dropping_a_row(db):
    fitted = _transformer().fit(KEYS, database=db)
    with pytest.raises(SchemaError, match="no row"):
        fitted.transform([1, 999], database=db)


def test_transform_falls_back_to_the_database_seen_at_fit(db):
    fitted = _transformer().fit(KEYS, database=db)
    out = nw.from_native(fitted.transform(KEYS))
    assert out.shape[0] == 3


def test_it_routes_the_database_through_a_pipeline(db):
    with sklearn.config_context(enable_metadata_routing=True):
        pipe = Pipeline([("dfs", _transformer()), ("clf", LogisticRegression())])
        pipe.fit(np.array(KEYS).reshape(-1, 1), Y, database=db)
        assert len(pipe.predict(np.array([[1], [2]]), database=db)) == 2


def test_cross_val_score_produces_finite_scores(db):
    with sklearn.config_context(enable_metadata_routing=True):
        pipe = Pipeline([("dfs", _transformer()), ("clf", LogisticRegression())])
        scores = cross_val_score(
            pipe,
            np.array(KEYS).reshape(-1, 1),
            Y,
            cv=3,
            params={"database": db},
        )
    assert np.isfinite(scores).all()


def test_grid_search_can_tune_a_dfs_parameter(db):
    with sklearn.config_context(enable_metadata_routing=True):
        search = GridSearchCV(
            Pipeline([("dfs", _transformer()), ("clf", LogisticRegression())]),
            {"dfs__max_depth": [1, 2]},
            cv=3,
        )
        search.fit(np.array(KEYS).reshape(-1, 1), Y, database=db)
    assert search.best_params_["dfs__max_depth"] in (1, 2)


def test_fit_transform_forwards_the_database_without_warning(db):
    # TransformerMixin's fit_transform drops metadata on the way to transform,
    # and sklearn warns about exactly that. The database_ fallback would hide a
    # broken override, so the absence of the warning is what we assert.
    with sklearn.config_context(enable_metadata_routing=True):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            _transformer().fit_transform(KEYS, Y, database=db)


def test_importing_tusk_does_not_import_sklearn():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import sys, tusk; print('sklearn' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --group validation pytest tests/test_sklearn_dfs_transformer.py -q`
Expected: FAIL — `ImportError: cannot import name 'DFSTransformer'`

- [ ] **Step 3: Write the implementation**

Create `src/tusk/sklearn/_transformers.py`:

```python
"""Deep feature synthesis as scikit-learn estimators.

``X`` is the target table's primary-key column, not the database. That
inversion is what makes ``X`` array-like, so scikit-learn can split it: the
keys *are* the rows, which makes misalignment against ``y`` impossible rather
than merely discouraged, and lets ``cross_val_score`` and ``GridSearchCV``
work at all. The database travels as routed metadata, because ``clone``
deep-copies constructor parameters and a duckdb relation cannot be pickled.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from tusk.api import apply_features
from tusk.database import Database
from tusk.exceptions import SchemaError
from tusk.primitives.aggregation import AGG_DEFAULTS
from tusk.primitives.base import Primitive
from tusk.primitives.transform import TRANS_DEFAULTS
from tusk.sklearn._frames import as_keys, collect_matrix
from tusk.synthesis import synthesize


class DFSTransformer(BaseEstimator, TransformerMixin):
    """Deep feature synthesis as a pipeline step.

    Attributes:
        features_: Feature definitions, set by :meth:`fit`.
        database_: The database seen at fit, used when none is routed later.
    """

    __metadata_request__fit = {"database": True}
    __metadata_request__transform = {"database": True}

    def __init__(
        self,
        target_table: str,
        agg_primitives: Iterable[str | Primitive] | None = None,
        trans_primitives: Iterable[str | Primitive] | None = None,
        groupby_trans_primitives: Iterable[str | Primitive] | None = None,
        max_depth: int = 2,
        cutoff_time: datetime | None = None,
        output_backend: str | None = None,
    ) -> None:
        """Configure synthesis.

        Every argument is stored unmodified under its own name: scikit-learn
        requires ``__init__`` to only assign, or ``get_params`` and ``clone``
        cannot round-trip the estimator.

        Args:
            target_table: Table to build features for.
            agg_primitives: Aggregation primitives; None selects the defaults.
            trans_primitives: Transform primitives; None selects the defaults.
            groupby_trans_primitives: Transforms within foreign-key groups.
            max_depth: Maximum stacked primitive applications.
            cutoff_time: Only rows at or before this are visible. Clone-safe,
                unlike the database, so it lives here and can be searched.
            output_backend: Backend to collect to. None collects natively,
                which keeps narwhals-native transformers on their own frame
                type.
        """
        self.target_table = target_table
        self.agg_primitives = agg_primitives
        self.trans_primitives = trans_primitives
        self.groupby_trans_primitives = groupby_trans_primitives
        self.max_depth = max_depth
        self.cutoff_time = cutoff_time
        self.output_backend = output_backend

    def fit(
        self,
        X: Any,
        y: Any = None,
        database: Database | None = None,
    ) -> DFSTransformer:
        """Synthesize feature definitions.

        Reads the schema and no rows, so this is cheap; the expensive pass is
        in :meth:`transform`.

        Args:
            X: The target's primary-key column. Unused here beyond validation,
                since synthesis depends only on the schema.
            y: Ignored; present for the scikit-learn signature.
            database: The database, routed as metadata.

        Returns:
            This estimator.

        Raises:
            SchemaError: If no database was supplied.
        """
        if database is None:
            raise SchemaError(
                "no database supplied; pass database=… to fit, and enable "
                "sklearn.set_config(enable_metadata_routing=True) when fitting "
                "inside a Pipeline",
            )
        as_keys(X)
        self.database_ = database
        self.features_ = list(
            synthesize(
                database=database,
                target_table=self.target_table,
                agg_primitives=AGG_DEFAULTS
                if self.agg_primitives is None
                else self.agg_primitives,
                trans_primitives=TRANS_DEFAULTS
                if self.trans_primitives is None
                else self.trans_primitives,
                groupby_trans_primitives=self.groupby_trans_primitives or (),
                max_depth=self.max_depth,
            ),
        )
        return self

    def transform(self, X: Any, database: Database | None = None) -> Any:
        """Compute the feature matrix for the keys in ``X``.

        Args:
            X: The target's primary-key column. Its order becomes the matrix's
                row order.
            database: The database, routed as metadata. When absent, the one
                seen at fit is used -- scikit-learn's scorers call ``predict``
                with no metadata, so without this fallback every
                cross-validated score would be ``nan``.

        Returns:
            An eager native frame, one row per key, in key order.
        """
        check_is_fitted(self, "features_")
        db = self.database_ if database is None else database
        return collect_matrix(
            apply_features(self.features_, db, self.cutoff_time),
            self._primary_key(db),
            as_keys(X),
            self.output_backend,
        )

    def fit_transform(
        self,
        X: Any,
        y: Any = None,
        database: Database | None = None,
        **kwargs: Any,
    ) -> Any:
        """Fit, then transform, forwarding the database to both.

        ``TransformerMixin.fit_transform`` drops metadata on the way to
        ``transform``; scikit-learn warns about exactly this. Overriding is the
        prescribed fix.

        Args:
            X: The target's primary-key column.
            y: Ignored.
            database: The database, routed as metadata.
            **kwargs: Ignored; absorbs scikit-learn's fit parameters.

        Returns:
            The feature matrix.
        """
        return self.fit(X, y, database=database).transform(X, database=database)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Column names of the matrix, in column order.

        A multi-output primitive contributes several names, so this is wider
        than ``len(features_)``.

        Args:
            input_features: Ignored; present for the scikit-learn signature.

        Returns:
            The names, as an object array.
        """
        check_is_fitted(self, "features_")
        names = [n for f in self.features_ for n in f.output_names]
        return np.asarray(names, dtype=object)

    def _primary_key(self, database: Database) -> str:
        """The target table's key column.

        Args:
            database: The database to read the schema from.

        Returns:
            The primary key column name.

        Raises:
            SchemaError: If the target table declares none.
        """
        primary_key = database.schema(self.target_table).primary_key
        if primary_key is None:
            raise SchemaError(
                f"target table {self.target_table!r} needs a primary_key: "
                "it is what X names",
            )
        return primary_key
```

- [ ] **Step 4: Export it**

In `src/tusk/sklearn/__init__.py`:

```python
from tusk.sklearn._encoders import dtype_selector
from tusk.sklearn._transformers import DFSTransformer

__all__ = ["DFSTransformer", "dtype_selector"]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --group validation pytest tests/test_sklearn_dfs_transformer.py -q`
Expected: PASS, 11 tests.

- [ ] **Step 6: Run the full suite**

Run: `uv run --group validation pytest -q`
Expected: `2 failed, ...` — the same two pre-existing failures only.

- [ ] **Step 7: Commit**

```bash
git add src/tusk/sklearn tests/test_sklearn_dfs_transformer.py
git commit -m "feat: add DFSTransformer"
```

---

### Task 5: `DFSSelectorTransformer`

**Files:**
- Modify: `src/tusk/sklearn/_transformers.py`
- Modify: `src/tusk/sklearn/__init__.py`
- Test: `tests/test_sklearn_selector_transformer.py` (create)

**Interfaces:**
- Consumes: `DFSTransformer` (Task 4); `_lineage.make_sentinels` (Task 2); `_encoders.encoder_prefix`, `selector_of`, `validate_inner` (Task 3); `_frames.backend_hint` (Task 1).
- Produces:
  - `tusk.sklearn.DFSSelectorTransformer`, adding constructor parameter `inner` and fitted attributes `encoder_`, `kept_names_`, `sentinels_`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sklearn_selector_transformer.py`:

```python
import narwhals as nw
import numpy as np
import pytest
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tusk.exceptions import LineageWarning, UnencodedFeatureWarning
from tusk.sklearn import DFSSelectorTransformer, dtype_selector

KEYS = [1, 2, 3]
Y = [0, 1, 0]


def _inner(k=2):
    return Pipeline(
        [
            (
                "enc",
                ColumnTransformer(
                    [
                        (
                            "oh",
                            OneHotEncoder(handle_unknown="ignore"),
                            dtype_selector("categorical"),
                        ),
                        ("num", StandardScaler(), dtype_selector("numeric")),
                    ],
                ),
            ),
            ("sel", SelectKBest(f_classif, k=k)),
        ],
    )


def _transformer(k=2):
    return DFSSelectorTransformer(target_table="customers", inner=_inner(k), max_depth=2)


def test_fitting_prunes_the_feature_list(db):
    unpruned = len(_transformer().fit(KEYS, database=db).features_)
    from tusk.sklearn import DFSTransformer

    everything = len(
        DFSTransformer(target_table="customers", max_depth=2)
        .fit(KEYS, database=db)
        .features_,
    )
    assert unpruned < everything


def test_transform_width_equals_the_selectors_choice(db):
    fitted = _transformer(k=2).fit(KEYS, database=db)
    out = np.asarray(fitted.transform(KEYS, database=db))
    assert out.shape == (3, 2)


def test_pruned_features_are_never_computed(db):
    from tusk.sklearn import DFSTransformer

    everything = DFSTransformer(target_table="customers", max_depth=2).fit(
        KEYS,
        database=db,
    )
    all_columns = set(everything.get_feature_names_out())

    fitted = _transformer().fit(KEYS, database=db)
    kept = {n for f in fitted.features_ for n in f.output_names}
    dropped = all_columns - kept
    assert dropped, "selection kept every feature, so this proves nothing"

    # The parent's transform computes the tusk-space matrix for whatever
    # features_ currently holds, which after fit is the pruned list.
    computed = set(
        nw.from_native(DFSTransformer.transform(fitted, KEYS, database=db)).columns,
    )
    assert computed == kept
    assert not computed & dropped


def test_feature_names_out_are_readable_not_sentinels(db):
    fitted = _transformer().fit(KEYS, database=db)
    names = list(fitted.get_feature_names_out())
    assert names
    assert all(fitted.sentinels_.prefix not in name for name in names)


def test_an_opaque_encoder_keeps_every_feature_and_warns(db):
    inner = Pipeline([("pca", PCA(n_components=2)), ("sel", SelectKBest(f_classif, k=1))])
    transformer = DFSSelectorTransformer(
        target_table="customers",
        inner=inner,
        agg_primitives=["count"],
        trans_primitives=[],
    )
    from tusk.sklearn import DFSTransformer

    everything = len(
        DFSTransformer(
            target_table="customers",
            agg_primitives=["count"],
            trans_primitives=[],
        )
        .fit(KEYS, database=db)
        .features_,
    )
    with pytest.warns(LineageWarning):
        fitted = transformer.fit(KEYS, database=db)
    assert len(fitted.features_) == everything


def test_a_partial_encoder_warns_about_features_it_never_saw(db):
    # Only categorical columns are encoded, and remainder defaults to "drop",
    # so every numeric feature silently feeds nothing and gets pruned.
    inner = Pipeline(
        [
            (
                "enc",
                ColumnTransformer(
                    [
                        (
                            "oh",
                            OneHotEncoder(handle_unknown="ignore"),
                            dtype_selector("categorical"),
                        ),
                    ],
                ),
            ),
            ("sel", SelectKBest(f_classif, k=1)),
        ],
    )
    transformer = DFSSelectorTransformer(target_table="customers", inner=inner)
    with pytest.warns(UnencodedFeatureWarning, match="fed no encoded column"):
        transformer.fit(KEYS, Y, database=db)


def test_it_works_as_a_pipeline_step(db):
    with sklearn.config_context(enable_metadata_routing=True):
        pipe = Pipeline([("dfs", _transformer()), ("clf", LogisticRegression())])
        pipe.fit(np.array(KEYS).reshape(-1, 1), Y, database=db)
        assert len(pipe.predict(np.array([[1], [2]]), database=db)) == 2


def test_predicting_against_another_database_uses_the_pruned_features(db):
    fitted = _transformer().fit(KEYS, database=db)
    before = list(fitted.features_)
    fitted.transform([1, 2], database=db)
    assert fitted.features_ == before
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --group validation pytest tests/test_sklearn_selector_transformer.py -q`
Expected: FAIL — `ImportError: cannot import name 'DFSSelectorTransformer'`

- [ ] **Step 3: Write the implementation**

Append to `src/tusk/sklearn/_transformers.py`:

```python
class DFSSelectorTransformer(DFSTransformer):
    """DFS whose feature list is pruned to what a selector actually kept.

    The point is inference cost. A run that generates eight hundred features
    and keeps forty should compute forty when it next sees data, not eight
    hundred.

    Attributes:
        encoder_: The encoder prefix, refitted on the surviving columns.
        kept_names_: Encoded-space names the selector chose, in encoder order.
        sentinels_: The opaque renaming used to recover lineage.
    """

    def __init__(
        self,
        target_table: str,
        inner: Any = None,
        agg_primitives: Iterable[str | Primitive] | None = None,
        trans_primitives: Iterable[str | Primitive] | None = None,
        groupby_trans_primitives: Iterable[str | Primitive] | None = None,
        max_depth: int = 2,
        cutoff_time: datetime | None = None,
        output_backend: str | None = None,
    ) -> None:
        """Configure synthesis and selection.

        The parent's parameters are repeated rather than absorbed into
        ``**kwargs``, because scikit-learn discovers them by introspecting this
        signature.

        Args:
            target_table: Table to build features for.
            inner: An estimator ending in a ``SelectorMixin``; everything
                before it encodes.
            agg_primitives: Aggregation primitives; None selects the defaults.
            trans_primitives: Transform primitives; None selects the defaults.
            groupby_trans_primitives: Transforms within foreign-key groups.
            max_depth: Maximum stacked primitive applications.
            cutoff_time: Only rows at or before this are visible.
            output_backend: Backend to collect to; None collects natively.
        """
        super().__init__(
            target_table=target_table,
            agg_primitives=agg_primitives,
            trans_primitives=trans_primitives,
            groupby_trans_primitives=groupby_trans_primitives,
            max_depth=max_depth,
            cutoff_time=cutoff_time,
            output_backend=output_backend,
        )
        self.inner = inner

    def fit(
        self,
        X: Any,
        y: Any = None,
        database: Database | None = None,
    ) -> DFSSelectorTransformer:
        """Synthesize, select, then prune.

        Args:
            X: The target's primary-key column.
            y: Training targets, passed to the selector.
            database: The database, routed as metadata.

        Returns:
            This estimator.

        Raises:
            SchemaError: If selection eliminated every feature.
            LineageError: If a kept column vanished from the refitted encoder,
                meaning lineage missed a source and a feature was wrongly
                pruned.

        Warns:
            LineageWarning: If any kept column's provenance was unrecoverable,
                in which case nothing is pruned.
        """
        validate_inner(self.inner)
        super().fit(X, y, database=database)
        db = self.database_ if database is None else database

        matrix = nw.from_native(super().transform(X, database=db), eager_only=True)
        sentinels = make_sentinels(list(matrix.columns))
        probe = matrix.rename(sentinels.mapping).to_native()

        inner = clone(self.inner)
        with backend_hint(probe):
            inner.fit(probe, y)
        encoded = list(encoder_prefix(inner).get_feature_names_out())
        kept = [
            name
            for name, keep in zip(encoded, selector_of(inner).get_support())
            if keep
        ]

        self.features_ = self._prune(kept, encoded, sentinels)
        if not self.features_:
            raise SchemaError("feature selection eliminated every feature")

        surviving = [n for f in self.features_ for n in f.output_names]
        narrowed = matrix.select(surviving).rename(
            {c: sentinels.mapping[c] for c in surviving},
        )
        self.encoder_ = encoder_prefix(clone(self.inner))
        with backend_hint(narrowed.to_native()):
            self.encoder_.fit(narrowed.to_native())

        refit = list(self.encoder_.get_feature_names_out())
        missing = [n for n in kept if n not in set(refit)]
        if missing:
            raise LineageError(
                f"{len(missing)} selected columns vanished when the encoder was "
                f"refit on the surviving features, e.g. "
                f"{[sentinels.restore(n) for n in missing[:3]]}. Lineage missed "
                "a source and pruned a feature that was still needed.",
            )
        self.sentinels_ = sentinels
        self.kept_names_ = [n for n in refit if n in set(kept)]
        return self

    def transform(self, X: Any, database: Database | None = None) -> Any:
        """Compute the surviving features, encode, and apply the frozen mask.

        Nothing is fitted here. Both fits happen in :meth:`fit`; a Pipeline
        calls ``fit_transform`` on intermediate steps during ``fit`` and
        ``transform`` during ``predict``.

        Args:
            X: The target's primary-key column.
            database: The database, routed as metadata.

        Returns:
            The encoded, selected matrix.
        """
        check_is_fitted(self, "kept_names_")
        db = self.database_ if database is None else database
        matrix = nw.from_native(super().transform(X, database=db), eager_only=True)
        surviving = [n for f in self.features_ for n in f.output_names]
        probe = matrix.select(surviving).rename(
            {c: self.sentinels_.mapping[c] for c in surviving},
        )
        with backend_hint(probe.to_native()):
            encoded = self.encoder_.transform(probe.to_native())
        return self._select(encoded)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Selected column names, with sentinels substituted back.

        Args:
            input_features: Ignored; present for the scikit-learn signature.

        Returns:
            Readable names such as ``oh__MODE__transactions__category_a``.
        """
        check_is_fitted(self, "kept_names_")
        names = [self.sentinels_.restore(n) for n in self.kept_names_]
        return np.asarray(names, dtype=object)

    def _prune(
        self,
        kept: list[str],
        encoded: list[str],
        sentinels: Sentinels,
    ) -> list[Feature]:
        """Features with at least one column feeding a kept encoded column.

        A multi-output feature is all-or-nothing: one output cannot be computed
        without the others, so any live column keeps the whole feature.

        Args:
            kept: Encoded-space names the selector chose.
            encoded: Every encoded-space name, kept or not. Needed to tell a
                feature the selector *rejected* from one the encoder never
                looked at, which are worth reporting differently.
            sentinels: The renaming used to read provenance.

        Returns:
            The surviving features, in their original order.

        Warns:
            LineageWarning: If any kept name mentions no sentinel, in which
                case every feature is kept.
            UnencodedFeatureWarning: If a matrix column fed no encoded column
                at all, so its feature is pruned for a reason the user may not
                have intended.
        """
        sources = {name: sentinels.sources(name) for name in kept}
        opaque = [name for name, s in sources.items() if not s]
        if opaque:
            warnings.warn(
                f"{len(opaque)} selected columns do not name their inputs "
                f"(e.g. {opaque[:3]}), so tusk cannot tell which features they "
                "came from and has pruned nothing. Selection still applies; "
                "only the inference-time saving is lost.",
                LineageWarning,
                stacklevel=3,
            )
            return list(self.features_)

        live = {column for s in sources.values() for column in s}
        touched = {c for name in encoded for c in sentinels.sources(name)}
        unencoded = [c for c in sentinels.columns if c not in touched]
        if unencoded:
            warnings.warn(
                f"{len(unencoded)} features fed no encoded column (e.g. "
                f"{unencoded[:3]}) and have been pruned. The encoder never "
                'looked at them -- a ColumnTransformer with remainder="drop" '
                "covering only some dtypes is the usual cause.",
                UnencodedFeatureWarning,
                stacklevel=3,
            )
        return [
            f for f in self.features_ if any(n in live for n in f.output_names)
        ]

    def _select(self, encoded: Any) -> Any:
        """Apply the frozen mask to the encoder's output.

        Resolved by name against the fitted encoder's own ordering rather than
        by stored position: with ``output_backend=None`` the output may be a
        frame or a bare array, and a stored index would silently mis-slice if
        that changed between fit and transform.

        Args:
            encoded: Whatever the encoder returned.

        Returns:
            Only the selected columns, in encoder order.
        """
        order = list(self.encoder_.get_feature_names_out())
        indices = [order.index(n) for n in self.kept_names_]
        frame = nw.from_native(encoded, eager_only=True, pass_through=True)
        if isinstance(frame, nw.DataFrame):
            return frame.select([frame.columns[i] for i in indices]).to_native()
        return encoded[:, indices]
```

Add to the imports at the top of `_transformers.py`:

```python
import warnings

import narwhals as nw
from sklearn.base import clone

from tusk.exceptions import (
    LineageError,
    LineageWarning,
    UnencodedFeatureWarning,
)
from tusk.features import Feature
from tusk.sklearn._encoders import encoder_prefix, selector_of, validate_inner
from tusk.sklearn._frames import backend_hint
from tusk.sklearn._lineage import Sentinels, make_sentinels
```

`Feature` belongs here rather than in Task 4: it is only ever used as an
annotation on `_prune`, and an unused import fails ruff's `F401` at commit
time.

- [ ] **Step 4: Export it**

In `src/tusk/sklearn/__init__.py`:

```python
from tusk.sklearn._encoders import dtype_selector
from tusk.sklearn._transformers import DFSSelectorTransformer, DFSTransformer

__all__ = ["DFSSelectorTransformer", "DFSTransformer", "dtype_selector"]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --group validation pytest tests/test_sklearn_selector_transformer.py -q`
Expected: PASS, 8 tests.

- [ ] **Step 6: Run the full suite**

Run: `uv run --group validation pytest -q`
Expected: `2 failed, ...` — the same two pre-existing failures only.

- [ ] **Step 7: Commit**

```bash
git add src/tusk/sklearn tests/test_sklearn_selector_transformer.py
git commit -m "feat: add DFSSelectorTransformer with lineage-aware pruning"
```

---

### Task 6: Backend coverage and documentation

**Files:**
- Create: `tests/test_sklearn_backends.py`
- Create: `docs/api/sklearn.md`
- Create: `docs/guide/sklearn.md`
- Modify: `zensical.toml`
- Modify: `README.md`
- Modify: `docs/guide/index.md`

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces: no new code surface.

- [ ] **Step 1: Write the backend test**

Create `tests/test_sklearn_backends.py`:

```python
import datetime as dt

import duckdb
import narwhals as nw
import pytest

import tusk
from tusk.sklearn import DFSTransformer


@pytest.fixture
def duckdb_db():
    customers = duckdb.sql(
        "select * from (values "
        "(1, 30, timestamp '2024-01-01'), "
        "(2, 40, timestamp '2024-01-01'), "
        "(3, 50, timestamp '2024-01-01')) t(id, age, signed_up_at)",
    )
    sessions = duckdb.sql(
        "select * from (values "
        "(10, 1, timestamp '2024-03-04'), "
        "(20, 1, timestamp '2024-03-05'), "
        "(30, 2, timestamp '2024-03-06')) t(id, customer_id, started_at)",
    )
    return (
        tusk.Database("retail")
        .add_table("customers", customers, primary_key="id", row_creation_time="signed_up_at")
        .add_table("sessions", sessions, primary_key="id", row_creation_time="started_at")
        .add_relationship(parent="customers", child="sessions", foreign_key="customer_id")
    )


def test_a_duckdb_database_reorders_despite_a_narrower_key_dtype(duckdb_db):
    fitted = DFSTransformer(target_table="customers").fit([1, 2, 3], database=duckdb_db)
    out = nw.from_native(fitted.transform([3, 1], database=duckdb_db))
    assert out.shape[0] == 2


def test_output_backend_pandas_returns_a_pandas_frame(duckdb_db):
    pytest.importorskip("pandas")
    fitted = DFSTransformer(
        target_table="customers",
        output_backend="pandas",
    ).fit([1, 2, 3], database=duckdb_db)
    out = fitted.transform([1, 2], database=duckdb_db)
    assert type(out).__module__.startswith("pandas")
```

- [ ] **Step 2: Run it**

Run: `uv run --group validation pytest tests/test_sklearn_backends.py -q`
Expected: PASS, 2 tests.

- [ ] **Step 3: Write the API reference page**

Create `docs/api/sklearn.md`:

```markdown
# scikit-learn

::: tusk.sklearn
```

- [ ] **Step 4: Write the guide**

Create `docs/guide/sklearn.md`:

````markdown
# scikit-learn pipelines

Install the extra:

```bash
uv add "tusk[sklearn]"
```

## `X` is the key column

Unlike featuretools' wrapper, `X` is the target table's primary-key column and
the database travels as routed metadata. That makes `X` array-like, so
scikit-learn can split it — which is what lets cross-validation and grid
search work at all, and makes it impossible for the feature matrix to fall out
of alignment with `y`.

Routing is a global opt-in:

```python
import sklearn
sklearn.set_config(enable_metadata_routing=True)
```

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline

from tusk.sklearn import DFSTransformer

pipeline = Pipeline([
    ("dfs", DFSTransformer(target_table="customers", max_depth=2)),
    ("clf", ExtraTreesClassifier()),
])

pipeline.fit(train_ids, y, database=db)
pipeline.predict(test_ids, database=db_test)
```

Because `X` is ordinary array-like data, so is everything built on it:

```python
from sklearn.model_selection import GridSearchCV

search = GridSearchCV(pipeline, {"dfs__max_depth": [1, 2, 3]})
search.fit(train_ids, y, database=db)
```

## Computing only the features you kept

`DFSSelectorTransformer` fits an encode-and-select pipeline you supply, works
out which tusk features the selector actually used, and drops the rest. Later
calls compute only the survivors.

```python
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tusk.sklearn import DFSSelectorTransformer, dtype_selector

selector = DFSSelectorTransformer(
    target_table="customers",
    inner=Pipeline([
        ("enc", ColumnTransformer([
            ("oh", OneHotEncoder(handle_unknown="ignore"), dtype_selector("categorical")),
            ("num", StandardScaler(), dtype_selector("numeric")),
        ])),
        ("sel", SelectKBest(k=50)),
    ]),
)
```

tusk never encodes. Encoding is your choice of scikit-learn transformer, so
the strategy stays yours.

## Why `dtype_selector`

Give a `ColumnTransformer` an explicit list of column names and tusk raises.
DFS *generates* its column names, so they do not exist until it has run:
writing them out means hardcoding names like `MEAN__transactions__amount`
that go stale the moment a primitive changes, and `remainder="drop"` would
silently discard every feature you did not name.

`dtype_selector` picks columns by dtype instead. scikit-learn's own
`make_column_selector` does the same job but only accepts pandas;
`dtype_selector` goes through narwhals and works on every backend a tusk
database can use.

## Frame backends

By default the matrix is collected to whatever the database already uses, so
narwhals-native transformers keep their own frame type. If a transformer in
your pipeline insists on pandas, say so:

```python
DFSTransformer(target_table="customers", output_backend="pandas")
```

`tusk[sklearn]` does not depend on pandas, so install it yourself if you use
that option. Note that `ColumnTransformer` cannot consume pyarrow frames, so a
duckdb-backed database usually wants `output_backend="pandas"` or
`"polars"`.
````

- [ ] **Step 5: Add both pages to the nav**

In `zensical.toml`, add `"guide/sklearn.md",` after `"guide/custom-primitives.md",` and `"api/sklearn.md",` after `"api/validation.md",`.

- [ ] **Step 6: Link the guide from the docs index**

In `docs/guide/index.md`, add this line to the bullet list at the bottom, between `Custom primitives` and `Differences from featuretools`:

```markdown
- [scikit-learn pipelines](sklearn.md)
```

In `README.md`, add to the documentation list, after the `Custom primitives` bullet:

```markdown
- [scikit-learn](docs/guide/sklearn.md) — DFS as a pipeline step, and computing only the features you kept.
```

- [ ] **Step 7: Verify the docs build**

Run: `uv run --group docs zensical build`
Expected: completes without warnings about missing nav pages.

- [ ] **Step 8: Run the full suite**

Run: `uv run --group validation pytest -q`
Expected: `2 failed, ...` — the same two pre-existing failures only.

- [ ] **Step 9: Commit**

```bash
git add tests/test_sklearn_backends.py docs zensical.toml README.md
git commit -m "docs: add the scikit-learn guide and API reference"
```

---

## Open items for the implementer

- **The `scikit-learn` floor is a guess at `>=1.4`.** Metadata routing through
  `Pipeline.predict` to `transform` is verified on 1.6.1 only. In Task 4, after
  the routing tests pass, try `uv pip install "scikit-learn==1.4.*"` and rerun
  `tests/test_sklearn_dfs_transformer.py`. If routing fails there, raise the
  floor in `pyproject.toml` to the oldest version that passes and say so in the
  commit message.
