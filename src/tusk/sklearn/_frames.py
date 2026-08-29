"""Materializes a lazy feature matrix for scikit-learn.

:func:`read_keys` normalizes the primary key ``X`` to a list.
:func:`collect_matrix` filters the matrix to those keys, collects it, and
returns the rows in key order. :func:`backend_hint` annotates exceptions from
a user's pipeline with the frame backend in play.

This is the only module in tusk that collects: scikit-learn cannot consume a
query plan.
"""

from __future__ import annotations

import contextlib
import sys
import warnings
from collections.abc import Iterable, Iterator
from typing import Any

import narwhals as nw
from narwhals.typing import IntoLazyFrame

from tusk.exceptions import SchemaError

_POSITION = "__tusk_position"


def read_keys(X: Iterable[Any]) -> list[Any]:
    """Return the primary-key values in ``X`` as a list, in the order given.

    The order given becomes the feature matrix's row order.

    Args:
        X: An iterable of key values -- a list, a 1-D array, a Series.

    Returns:
        The key values.

    Raises:
        TypeError: If ``X`` is not iterable, or its elements are not single
            values.
    """
    try:
        keys = list(X)
    except TypeError:
        raise TypeError(
            f"X must be an iterable of key values; got {type(X).__name__}",
        ) from None
    if keys and hasattr(keys[0], "__len__") and not isinstance(keys[0], str):
        raise TypeError(
            f"X must be one-dimensional -- each element is one key -- but its "
            f"elements are {type(keys[0]).__name__}. A column vector of shape "
            f"(n, 1) is the usual cause; pass the keys themselves.",
        )
    return keys


def collect_matrix(
    matrix: IntoLazyFrame,
    primary_key: str,
    keys: list[Any],
    output_backend: str | None,
) -> Any:
    """Materialize the feature matrix for ``keys``, in ``keys`` order.

    Args:
        matrix: The uncomputed feature matrix from ``apply_features``.
        primary_key: The target table's primary key.
        keys: Key values selecting and ordering the rows.
        output_backend: Backend to collect to, or None to collect natively.

    Returns:
        matrix: An eager native frame with one row per key, in key order,
            without the primary key -- it is a join key, not a feature.

    Raises:
        SchemaError: If ``keys`` repeats a value, or names a key that produced
            no row.
    """
    if len(set(keys)) != len(keys):
        raise SchemaError(
            "X contains duplicate keys; the feature matrix is keyed by the "
            "target's primary key, so each row must be requested at most once",
        )

    frame = nw.from_native(matrix)
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
        # narwhals types `backend` as a 3-way Literal but accepts a plain
        # string. It rejects an unknown name itself, with a ValueError listing
        # what it accepts; this clause is for a name it accepts whose package
        # is not installed.
        return frame.collect(backend=output_backend)  # ty: ignore[invalid-argument-type]
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
        None: Control returns to the caller's ``with`` block.

    Raises:
        Exception: Whatever the wrapped block raised, unchanged, with a
            backend hint attached as a note (or, on Python 3.10, warned).
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
        if sys.version_info >= (3, 11):
            exc.add_note(hint)
        else:
            warnings.warn(hint, UserWarning, stacklevel=2)
        raise
