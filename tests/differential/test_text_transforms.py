"""Cross-check tusk's text transform primitives against featuretools.

Each test builds one primitive on both sides over the shared table in
``tests/transform_cases.py`` and compares the one resulting column.

Run with: uv run --group validation pytest -m differential

Verified against featuretools 1.31.0.
"""

import pytest

from differential.transform_matrices import (
    assert_agree,
    featuretools_values,
    tusk_values,
)

featuretools = pytest.importorskip("featuretools")

pytestmark = pytest.mark.differential


def test_n_characters_matches_featuretools():
    """Covers a null, punctuation, whitespace only and a blank string."""
    ours = tusk_values("n_characters", "N_CHARACTERS__text")
    theirs = featuretools_values("num_characters", "NUM_CHARACTERS(text)")
    assert_agree(ours, theirs)


def test_n_words_matches_featuretools():
    """Covers a null, punctuation, repeated words and a blank string."""
    ours = tusk_values("n_words", "N_WORDS__text")
    theirs = featuretools_values("num_words", "NUM_WORDS(text)")
    assert_agree(ours, theirs)


def test_n_unique_words_matches_case_insensitive_featuretools():
    """tusk folds case; featuretools only does so when asked to.

    ``NumberOfUniqueWords`` defaults to ``case_insensitive=False``, so row
    id 3, ``"Dog dog cat"``, is 3 unique words there and 2 in tusk. Against
    the case-insensitive instance every row agrees.
    """
    ours = tusk_values("n_unique_words", "N_UNIQUE_WORDS__text")
    theirs = featuretools_values(
        featuretools.primitives.NumberOfUniqueWords(case_insensitive=True),
        "NUMBER_OF_UNIQUE_WORDS(text, case_insensitive=True)",
    )
    assert_agree(ours, theirs)


def test_n_unique_words_folds_case_where_featuretools_defaults_not_to():
    """Row id 3 holds ``"Dog dog cat"``: tusk answers 2, featuretools 3."""
    ours = tusk_values("n_unique_words", "N_UNIQUE_WORDS__text")
    theirs = featuretools_values(
        "number_of_unique_words",
        "NUMBER_OF_UNIQUE_WORDS(text)",
    )
    mixed_case_row = 2
    assert ours[mixed_case_row] == 2
    assert theirs[mixed_case_row] == 3
