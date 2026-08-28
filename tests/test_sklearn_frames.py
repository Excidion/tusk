import narwhals as nw
import numpy as np
import polars as pl
import pytest

from tusk.exceptions import SchemaError
from tusk.sklearn._frames import collect_matrix, read_keys


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
    import warnings

    from tusk.sklearn._frames import backend_hint

    # Below 3.11, backend_hint warns instead of using add_note (no such
    # method exists yet); catch_warnings keeps that expected UserWarning
    # from leaking into the test run's output on those interpreters.
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="boom"):
            with backend_hint(pl.DataFrame({"a": [1]})):
                raise ValueError("boom")


def test_backend_hint_names_the_backend_and_the_fix():
    import warnings

    from tusk.sklearn._frames import backend_hint

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with backend_hint(pl.DataFrame({"a": [1]})):
                raise ValueError("boom")
        except ValueError as exc:
            notes = " ".join(getattr(exc, "__notes__", []))
            warned = " ".join(str(w.message) for w in caught)
    # 3.11+ carries the hint as a note on the exception; below that, as the
    # warning caught above -- exactly one of the two is populated.
    hint = notes or warned
    assert "polars" in hint.lower()
    assert "output_backend" in hint
