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
