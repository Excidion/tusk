"""Cross-check tusk's boolean transform primitives against featuretools.

``not`` agrees everywhere. ``and`` and ``or`` agree on every row where both
inputs are known and disagree by design where one is null: tusk follows the
three-valued logic of the SQL engines it pushes down to, while featuretools'
``np.logical_and`` propagates the null. Both halves are asserted here, so the
divergence stays a tested claim rather than a note in the docs.

Run with: uv run --group validation pytest -m differential

Verified against featuretools 1.31.0.
"""

import pytest

import tusk
from differential import _as_tusk

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pl = pytest.importorskip("polars")
featuretools = pytest.importorskip("featuretools")

pytestmark = pytest.mark.differential


@pytest.fixture
def rows():
    """Every pairing of True, False and null across two boolean columns."""
    is_active = [True, True, True, False, False, False, None, None, None]
    is_verified = [True, False, None, True, False, None, True, False, None]
    frame = pd.DataFrame(
        {
            "id": np.arange(1, len(is_active) + 1),
            "is_active": pd.array(is_active, dtype="boolean"),
            "is_verified": pd.array(is_verified, dtype="boolean"),
        },
    )
    _assert_rows_invariants(frame)
    return frame


def _assert_rows_invariants(frame):
    """Guard the cases ``rows`` is built to cover.

    A pairing of two known values is what the agreement tests read, and a
    pairing with a null is what the divergence test reads; losing either
    would leave a test passing over nothing.

    Args:
        frame: The table built by ``rows``.
    """
    both_known = frame["is_active"].notna() & frame["is_verified"].notna()
    assert both_known.any()
    assert (~both_known).any()


@pytest.mark.parametrize("primitive_name", ["and", "or"])
def test_binary_boolean_transforms_match_featuretools_where_nothing_is_null(
    rows,
    primitive_name,
):
    """With both inputs known, three-valued logic and featuretools coincide."""
    ours, theirs = _both_matrices(rows, primitive_name)
    known = rows.set_index("id")[["is_active", "is_verified"]].notna().all(axis=1)
    column = f"{primitive_name.upper()}(is_active, is_verified)"
    assert _nullable(ours[_as_tusk(column)][known]) == _nullable(theirs[column][known])


def test_not_matches_featuretools_on_every_row(rows):
    """NOT has no divergence to make: a null negates to a null on both sides."""
    ours, theirs = _both_matrices(rows, "not")
    assert _nullable(ours[_as_tusk("NOT(is_active)")]) == _nullable(
        theirs["NOT(is_active)"],
    )


@pytest.mark.parametrize(
    ("primitive_name", "expected"),
    [("and", False), ("or", True)],
)
def test_binary_boolean_transforms_diverge_where_one_input_is_null(
    rows,
    primitive_name,
    expected,
):
    """A null is an unknown value, and an unknown cannot change these answers.

    FALSE AND unknown is FALSE and TRUE OR unknown is TRUE whatever the
    unknown turns out to be, which is what every SQL engine answers.
    featuretools returns a null for both.
    """
    ours, theirs = _both_matrices(rows, primitive_name)
    row = rows.set_index("id")
    settled_by_one_input = (row["is_active"] == expected) & row["is_verified"].isna()
    assert settled_by_one_input.any()
    column = f"{primitive_name.upper()}(is_active, is_verified)"
    assert _nullable(ours[_as_tusk(column)][settled_by_one_input]) == [expected]
    assert theirs[column][settled_by_one_input].isna().all()


def _nullable(series):
    """Render a boolean column as a comparable list, with nulls as None.

    Args:
        series: A boolean column from either matrix.

    Returns:
        One True, False or None per row.
    """
    return [None if pd.isna(value) else bool(value) for value in series]


def _both_matrices(frame, primitive_name):
    """Build the same one-primitive matrix on both sides.

    Args:
        frame: The table to build features over.
        primitive_name: The primitive's name, spelled the same way by tusk
            and featuretools.

    Returns:
        A tuple of the tusk matrix and the featuretools matrix, both indexed
        by the table's primary key.
    """
    return (
        _tusk_matrix(frame, primitive_name),
        _featuretools_matrix(frame, primitive_name),
    )


def _featuretools_matrix(frame, primitive_name):
    """Run one transform primitive through featuretools and return its matrix.

    Args:
        frame: The table to build features over.
        primitive_name: The primitive's featuretools name, the only entry in
            ``trans_primitives`` so exactly one feature is built.

    Returns:
        The feature matrix, sorted by the table's index.
    """
    es = featuretools.EntitySet("rows")
    es = es.add_dataframe(dataframe_name="rows", dataframe=frame, index="id")
    matrix, _ = featuretools.dfs(
        entityset=es,
        target_dataframe_name="rows",
        agg_primitives=[],
        trans_primitives=[primitive_name],
        max_depth=1,
    )
    return matrix.sort_index()


def _tusk_matrix(frame, primitive_name):
    """Run one transform primitive through tusk and return its matrix.

    Args:
        frame: The table to build features over.
        primitive_name: The primitive's name, the only entry in
            ``trans_primitives`` so exactly one feature is built.

    Returns:
        The feature matrix, collected and sorted by the table's primary key.
    """
    db = tusk.Database("rows_db").add_table(
        "rows",
        pl.from_pandas(frame).lazy(),
        primary_key="id",
    )
    matrix, _ = tusk.deep_feature_synthesis(
        database=db,
        target_table="rows",
        agg_primitives=[],
        trans_primitives=[primitive_name],
        max_depth=1,
    )
    return matrix.collect().sort("id").to_pandas().set_index("id")
