"""Reconnecting encoded columns to the tusk features they came from.

featuretools can prune feature definitions because ``encode_features`` returns
definitions for the one-hot columns, keeping the selector's mask and the
feature list 1:1. Handing encoding to scikit-learn severs that link: a support
mask indexes *encoded* columns and says nothing about which feature produced
them.

This module rebuilds the link by exploiting scikit-learn's own convention that
output names derive from input names. Rename the matrix columns to opaque
tokens, and every downstream name carries its provenance in plain sight.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Sentinels:
    """An opaque renaming of matrix columns, and the map back.

    Attributes:
        prefix: Random per-fit token prefix.
        width: Zero-padded index width, fixed so no sentinel is a prefix of
            another.
        columns: The tusk-space column names, positionally indexed.
        mapping: Column name to sentinel.
    """

    prefix: str
    width: int
    columns: tuple[str, ...]
    mapping: dict[str, str] = field(compare=False)

    @property
    def _pattern(self) -> re.Pattern[str]:
        """Regex matching one sentinel and capturing its index."""
        return re.compile(rf"{re.escape(self.prefix)}(\d{{{self.width}}})")

    def sources(self, name: str) -> list[str]:
        """Columns an encoded name derives from.

        Every match is collected rather than the first, so a multi-input
        transformer such as ``PolynomialFeatures`` correctly reports both of
        its sources and keeps both features alive.

        Args:
            name: One name from the encoder's ``get_feature_names_out()``.

        Returns:
            The tusk-space columns it mentions, in column order. Empty means
            lineage is unrecoverable for this name -- the caller must then keep
            every feature rather than guess.
        """
        found = {int(m) for m in self._pattern.findall(name)}
        return [self.columns[i] for i in sorted(found) if i < len(self.columns)]

    def restore(self, name: str) -> str:
        """Substitute sentinels back to real column names.

        Args:
            name: One name from the encoder's ``get_feature_names_out()``.

        Returns:
            The same name with every sentinel replaced, so users read
            ``oh__MODE__transactions__category_a`` rather than
            ``oh___t9f3a_0001_a``.
        """

        def replace(match: re.Match[str]) -> str:
            """Map one matched sentinel back to its source column name.

            Args:
                match: A single regex match of ``_pattern`` against ``name``.

            Returns:
                The source column for the matched index, or the original
                matched text if the index is out of range.
            """
            index = int(match.group(1))
            return self.columns[index] if index < len(self.columns) else match.group(0)

        return self._pattern.sub(replace, name)


def make_sentinels(columns: Sequence[str]) -> Sentinels:
    """Build an opaque renaming for one fit.

    The prefix is random per fit. A data *value* that happens to look like a
    sentinel would otherwise be misread as a column reference; randomness makes
    that vanishingly unlikely, and its only consequence is an extra source,
    which over-keeps rather than over-prunes.

    Args:
        columns: The matrix's column names, in order.

    Returns:
        The sentinel mapping and the map back.
    """
    prefix = f"_t{secrets.token_hex(4)}_"
    width = max(4, len(str(max(len(columns) - 1, 0))))
    mapping = {c: f"{prefix}{i:0{width}d}" for i, c in enumerate(columns)}
    return Sentinels(prefix, width, tuple(columns), mapping)
