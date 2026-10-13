import datetime as dt

import numpy as np
import polars as pl
import pytest

from tusk.exceptions import SchemaError
from tusk.sklearn._tables import (
    check_keys_are_visible,
    read_keys,
)


@pytest.mark.parametrize(
    ("label", "value"),
    [
        ("list", [1, 2, 3]),
        ("range", range(1, 4)),
        ("1d array", np.array([1, 2, 3])),
        ("series", pl.Series("id", [1, 2, 3])),
    ],
)
def test_read_keys_accepts_every_documented_shape(label, value):
    assert read_keys(value) == [1, 2, 3]


def test_read_keys_rejects_a_frame():
    # A frame's row order is not guaranteed on every backend, and key order is
    # what aligns the matrix against y.
    with pytest.raises(TypeError, match="one-dimensional"):
        read_keys(pl.DataFrame({"id": [1, 2, 3]}))


def test_read_keys_rejects_a_column_vector_by_naming_the_cause():
    # (n, 1) is the conventional sklearn X shape, so this is the mistake users
    # will actually make; without this check it dies later as "unhashable type".
    with pytest.raises(TypeError, match=r"one-dimensional.*\(n, 1\)"):
        read_keys(np.array([[1], [2], [3]]))


def test_read_keys_rejects_something_that_is_not_iterable():
    with pytest.raises(TypeError, match="iterable of key values"):
        # Deliberately outside the declared type: the point is that callers
        # who ignore the annotation still get a clear error at runtime.
        read_keys(object())  # ty: ignore[invalid-argument-type]


def test_read_keys_rejects_a_duplicate_key():
    with pytest.raises(SchemaError, match="duplicate"):
        read_keys([1, 1])


def test_backend_hint_reraises_the_original_exception_type():
    from tusk.sklearn._tables import backend_hint

    with pytest.raises(ValueError, match="boom"):
        with backend_hint(pl.DataFrame({"a": [1]})):
            raise ValueError("boom")


def test_backend_hint_names_the_backend_and_the_fix():
    from tusk.sklearn._tables import backend_hint

    with pytest.raises(ValueError) as caught:
        with backend_hint(pl.DataFrame({"a": [1]})):
            raise ValueError("boom")
    hint = " ".join(caught.value.__notes__)
    assert "polars" in hint.lower()
    assert "output_backend" in hint


def test_check_keys_are_visible_names_cutoff_time_for_an_excluded_key(db):
    with pytest.raises(SchemaError, match="cutoff_time"):
        check_keys_are_visible(db, "customers", "id", [1], dt.datetime(2023, 12, 31))


def test_check_keys_are_visible_passes_when_every_key_survives_the_cutoff(db):
    check_keys_are_visible(db, "customers", "id", [1, 2], dt.datetime(2024, 6, 1))
