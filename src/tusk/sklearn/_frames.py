"""The collect boundary.

tusk never collects; this is the one module that does, because scikit-learn
cannot consume a query plan. Everything here is about crossing that line
exactly once, in a way that cannot silently misalign rows against ``y``.
"""

from __future__ import annotations

import contextlib
import sys
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
        # narwhals types `backend` as a 3-way Literal, but it resolves any
        # installed backend module by name; a bad name surfaces below as
        # ModuleNotFoundError, which is what this except clause is for.
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
