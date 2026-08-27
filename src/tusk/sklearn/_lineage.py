"""Maps encoded column names back to the matrix columns they came from.

:func:`make_sentinels` renames a matrix's columns to opaque tokens.
:class:`Sentinels` then reads those tokens back out of an encoder's
``get_feature_names_out()``: :meth:`Sentinels.sources` reports which matrix
columns an encoded name derives from, and :meth:`Sentinels.restore`
substitutes the real names back for display.
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
    """Build an opaque renaming of ``columns`` for one fit.

    Args:
        columns: The matrix's column names, in order.

    Returns:
        The sentinel mapping and the map back.
    """
    # The prefix is random per fit: a data value that happens to look like a
    # sentinel would be misread as a column reference, and randomness makes
    # that vanishingly unlikely. Its only consequence is an extra source, which
    # over-keeps rather than over-drops.
    prefix = f"_t{secrets.token_hex(4)}_"
    width = max(4, len(str(max(len(columns) - 1, 0))))
    mapping = {c: f"{prefix}{i:0{width}d}" for i, c in enumerate(columns)}
    return Sentinels(prefix, width, tuple(columns), mapping)
