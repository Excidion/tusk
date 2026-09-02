"""Shared helpers for the differential suite.

Run with: uv run --group validation pytest -m differential
"""


def _as_tusk(name: str) -> str:
    """Translate a featuretools feature name into tusk's column name.

    Every construct featuretools spells with punctuation -- application,
    argument separator, path step, multi-output index, groupby suffix --
    tusk spells with ``__``.

    Args:
        name: A featuretools feature name, e.g. ``MEAN(sessions.value)``.

    Returns:
        The equivalent tusk column name, e.g. ``MEAN__sessions__value``.
    """
    for old, new in (
        (" by ", "__by__"),
        (", ", "__"),
        ("(", "__"),
        (")", ""),
        ("[", "__"),
        ("]", ""),
        (".", "__"),
    ):
        name = name.replace(old, new)
    return name
