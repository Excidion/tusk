"""Cross-check tusk's general transform primitives against featuretools.

Each test builds one primitive on both sides over the shared table in
``tests/transform_cases.py`` and compares the one resulting column.

Run with: uv run --group validation pytest -m differential

Verified against featuretools 1.31.0.
"""

import math

import pytest

from differential.transform_matrices import (
    assert_agree,
    featuretools_values,
    tusk_values,
)

pytestmark = pytest.mark.differential


@pytest.mark.parametrize(
    ("ours", "column", "theirs", "feature_name"),
    [
        ("is_null", "IS_NULL__value", "is_null", "IS_NULL(value)"),
        ("negate", "NEGATE__value", "negate", "-(value)"),
        ("sine", "SINE__value", "sine", "SINE(value)"),
        ("cosine", "COSINE__value", "cosine", "COSINE(value)"),
        ("absolute", "ABSOLUTE__value", "absolute", "ABSOLUTE(value)"),
    ],
)
def test_general_transforms_match_featuretools(ours, column, theirs, feature_name):
    """Covers a null, zero, a negative number and ties in ``value``."""
    assert_agree(tusk_values(ours, column), featuretools_values(theirs, feature_name))


@pytest.mark.parametrize(
    ("ours", "column", "theirs", "feature_name"),
    [
        ("square_root", "SQUARE_ROOT__value", "square_root", "SQUARE_ROOT(value)"),
        (
            "natural_log",
            "NATURAL_LOG__value",
            "natural_logarithm",
            "NATURAL_LOGARITHM(value)",
        ),
    ],
)
def test_square_root_and_natural_log_give_null_where_featuretools_gives_nan(
    ours,
    column,
    theirs,
    feature_name,
):
    """Row id 5 holds -1: tusk answers null, featuretools NaN.

    Every other row agrees.
    """
    ours_values = tusk_values(ours, column)
    theirs_values = featuretools_values(theirs, feature_name)
    negative_row = 4
    assert ours_values[negative_row] is None
    assert math.isnan(theirs_values[negative_row])
    del ours_values[negative_row]
    del theirs_values[negative_row]
    assert_agree(ours_values, theirs_values)
