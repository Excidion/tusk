"""Draws a database's schema as a Mermaid entity-relationship diagram.

:meth:`tusk.Database.plot` builds the diagram source here and returns it as a
:class:`SchemaDiagram`, which displays itself in a notebook and writes itself
to a file. The rest of the module is the escaping that keeps generated source
parseable: Mermaid's grammar is narrower than narwhals' dtypes and a
dataframe's column names.
"""

from __future__ import annotations

import re
from typing import Any

import narwhals as nw

_UNSAFE_IN_TOKEN = re.compile(r"[^A-Za-z0-9_-]")


def render_dtype(dtype: Any) -> str:
    """Render a narwhals dtype as a token Mermaid's type slot accepts.

    Args:
        dtype: A narwhals dtype.

    Returns:
        The dtype's class name, with its parameters in square brackets when it
        has any.
    """
    parameters = _render_dtype_parameters(dtype)
    name = type(dtype).__name__
    return f"{name}[{parameters}]" if parameters else name


def render_table_name(name: str) -> str:
    """Render a table name as a Mermaid entity name.

    Args:
        name: The table's name.

    Returns:
        The name in double quotes, which is what allows it to contain spaces.
    """
    return f'"{name}"'


def render_column_name(name: str) -> str:
    """Render a column name as a Mermaid attribute name.

    Attribute names cannot be quoted, so a name Mermaid would reject is
    rewritten rather than escaped. The rewrite is lossy: two columns differing
    only by a space collapse to the same token. The diagram is for visual
    inspection, so that is preferred to refusing to draw it.

    Args:
        name: The column's name.

    Returns:
        A name Mermaid's attribute slot parses, as close to the original as
        the grammar allows.
    """
    # U+2007 FIGURE SPACE renders as a space but is not one to the parser,
    # which rejects a real space in an attribute name.
    parseable = name.replace(" ", " ")
    if parseable[:1].isdigit():
        return f"_{parseable}"
    return parseable


def _render_dtype_parameters(dtype: Any) -> str:
    """Render the parameters that distinguish one instance of a dtype from another.

    Args:
        dtype: A narwhals dtype.

    Returns:
        The parameter token, or an empty string for a dtype without parameters.
    """
    if isinstance(dtype, nw.Datetime):
        return _render_time_parameters(dtype.time_unit, dtype.time_zone)
    if isinstance(dtype, nw.Duration):
        return _render_time_parameters(dtype.time_unit, None)
    if isinstance(dtype, nw.List):
        return render_dtype(dtype.inner)
    if isinstance(dtype, nw.Enum):
        return str(len(dtype.categories))
    if isinstance(dtype, nw.Struct):
        return str(len(dtype.fields))
    return ""


def _render_time_parameters(time_unit: str, time_zone: str | None) -> str:
    """Render a temporal dtype's unit and zone as one token.

    Args:
        time_unit: The dtype's time unit.
        time_zone: The dtype's time zone, or None when it is naive.

    Returns:
        The unit alone when naive, otherwise unit and zone joined by a hyphen.
    """
    if time_zone is None:
        return time_unit
    return f"{time_unit}-{_UNSAFE_IN_TOKEN.sub('_', time_zone)}"
