"""Key handling for scikit-learn.

`read_keys` normalizes the primary key `X` to a list.
`check_keys_are_visible` rejects keys the target table has no row for.
`backend_hint` annotates exceptions from a user's pipeline with the
table's backend.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Iterator
from datetime import datetime
from typing import Any

import narwhals as nw

from tusk.database import Database
from tusk.exceptions import SchemaError


def read_keys(X: Iterable[Any]) -> list[Any]:
    """Return the primary-key values in `X` as a list, in the order given.

    The order given becomes the feature matrix's row order.

    Args:
        X: An iterable of key values. For example, a list, a 1-D array or a
            Series.

    Returns:
        The key values.

    Raises:
        TypeError: If `X` is not iterable. If its elements are not single
            values.
        SchemaError: If `X` repeats a key.
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
    if len(set(keys)) != len(keys):
        raise SchemaError(
            "X contains duplicate keys; the feature matrix is keyed by the "
            "target's primary key, so each row must be requested at most once",
        )
    return keys


def check_keys_are_visible(
    database: Database,
    target_table: str,
    primary_key: str,
    keys: list[Any],
    cutoff_time: datetime | None,
) -> None:
    """Fail if the target table has no row for a key at `cutoff_time`.

    It reads the target's primary key alone. The function reports a stale
    key before it computes the feature matrix. Raises
    [`SchemaError`][tusk.exceptions.SchemaError], from `_reject_missing_keys`,
    if a key names no visible row.

    Args:
        database: The database holding the tables.
        target_table: Table the features are built for.
        primary_key: The target table's primary key.
        keys: Key values the caller asked for.
        cutoff_time: The cutoff time, or None.
    """
    visible = (
        database._get_table(target_table, cutoff_time)
        .select(primary_key)
        .filter(nw.col(primary_key).is_in(keys))
        .collect()
    )
    _reject_missing_keys(keys, set(visible[primary_key].to_list()), primary_key)


def _reject_missing_keys(keys: list[Any], found: set[Any], primary_key: str) -> None:
    """Fail if a requested key is not among the ones found.

    Args:
        keys: Key values the caller asked for.
        found: Key values that produced a row.
        primary_key: The target table's primary key, named in the message.

    Raises:
        SchemaError: If a key produced no row.
    """
    missing = [k for k in keys if k not in found]
    if not missing:
        return
    raise SchemaError(
        f"no row for {len(missing)} of {len(keys)} keys, e.g. "
        f"{missing[:5]}; they are absent from {primary_key!r} or were "
        f"excluded by cutoff_time",
    )


@contextlib.contextmanager
def backend_hint(table: Any) -> Iterator[None]:
    """Attach a backend hint to whatever the user's pipeline raises.

    Many sklearn transformers reject non-pandas tables in ways that read as
    unrelated type errors -- `ColumnTransformer` on pyarrow gives
    `TypeError: Index must either be string or integer`. The hint names the
    backend and the fix.

    This context manager re-raises the exception **unchanged**, with its
    original type.

    Args:
        table: The feature matrix handed to the pipeline, used to name the
            backend.

    Yields:
        None: Control returns to the caller's `with` block.

    Raises:
        Exception: Whatever the wrapped block raised, unchanged, with a
            backend hint attached as a note.
    """
    try:
        yield
    except Exception as exc:
        backend = nw.from_native(table, eager_only=True, pass_through=True)
        name = getattr(getattr(backend, "implementation", None), "name", "unknown")
        hint = (
            f"the feature matrix was collected as {name}; if this pipeline "
            f'needs pandas, set output_backend="pandas"'
        )
        exc.add_note(hint)
        raise
