"""Draws a database's schema as a Mermaid entity-relationship diagram.

:class:`SchemaDiagram` is the whole public surface: it builds itself from a
:class:`tusk.Database`, displays itself in a notebook, and writes itself to a
file. :meth:`tusk.Database.plot` is a shortcut for its constructor. Everything
below it is the escaping that keeps generated source parseable, because
Mermaid's grammar is narrower than narwhals' dtypes and a dataframe's column
names.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import narwhals as nw

if TYPE_CHECKING:
    from tusk.database import Database, Relationship, TableSchema


class SchemaDiagram:
    """A Mermaid diagram of a database's schema.

    Build one with :meth:`from_database`, or with
    :meth:`tusk.Database.plot`, which is a shortcut for it.

    Printing it, or reading :attr:`source`, gives the Mermaid source, which is
    the escape hatch for any renderer: this class only knows how to build
    itself, display itself, and write a file.

    Attributes:
        source: The Mermaid ``erDiagram`` source.
    """

    source: str

    @classmethod
    def from_database(
        cls,
        database: Database,
        columns: bool | str = True,
    ) -> SchemaDiagram:
        """Draw a database's schema.

        Args:
            database: The database to draw.
            columns: True lists every column, False lists none, and
                ``"structural"`` lists only the primary key, the foreign keys,
                and the ``row_creation_time``.

        Returns:
            The diagram.

        Raises:
            ValueError: If ``columns`` is not True, False, or ``"structural"``.
        """
        # Identity, not equality: `0 == False` and `1 == True`.
        if not any(columns is mode for mode in (True, False, "structural")):
            raise ValueError(
                f"columns must be True, False, or 'structural'; got {columns!r}",
            )
        lines = ["erDiagram"]
        lines.extend(render_relationship(database, r) for r in database.relationships)
        for name in database.table_names:
            lines.extend(render_table(database, name, columns))
        return cls("\n".join(lines) + "\n")

    def __init__(self, source: str) -> None:
        """Wrap already-built Mermaid source.

        Args:
            source: The Mermaid ``erDiagram`` source.
        """
        self.source = source

    def __str__(self) -> str:
        """Return the Mermaid source.

        Returns:
            The diagram source.
        """
        return self.source

    def _repr_markdown_(self) -> str:
        """Return the source as a fenced block, for notebooks and docs.

        Returns:
            The source in a ``mermaid`` code fence, which Jupyter, GitHub and
            the documentation site all render as a picture.
        """
        return f"```mermaid\n{self.source}```"

    def save(self, path: str | Path) -> None:
        """Write the diagram to a file.

        The suffix selects the format. ``.mmd`` and ``.md`` write the source
        and need nothing installed; ``.svg``, ``.png`` and ``.pdf`` render the
        diagram and need ``tusk-ml[plot]``, raising ``ImportError`` if it is
        not installed.

        Args:
            path: Where to write, including the suffix.

        Raises:
            ValueError: If the suffix names no supported format.
        """
        source_suffixes = (".mmd", ".md")
        image_suffixes = (".svg", ".png", ".pdf")
        path = Path(path)
        if path.suffix in source_suffixes:
            path.write_text(self.source, encoding="utf-8")
        elif path.suffix in image_suffixes:
            self._render_image(path)
        else:
            supported = ", ".join(source_suffixes + image_suffixes)
            raise ValueError(
                f"cannot save {path.suffix!r}; supported suffixes are {supported}",
            )

    def _render_image(self, path: Path) -> None:
        """Render the diagram to an image file.

        Args:
            path: Where to write, including the suffix.

        Raises:
            ImportError: If the renderer is not installed.
        """
        try:
            import mermaidx
        except ImportError as error:
            raise ImportError(
                f"saving {path.suffix} needs a renderer: pip install tusk-ml[plot]",
            ) from error
        mermaidx.render(self.source).save(str(path))


def render_relationship(database: Database, relationship: Relationship) -> str:
    """Render one relationship as a Mermaid edge.

    Args:
        database: The database the relationship belongs to.
        relationship: The relationship to draw.

    Returns:
        The edge line, indented.
    """
    parent = render_table_name(relationship.parent)
    child = render_table_name(relationship.child)
    return (
        f"  {parent} 1 to {_count_of_children(database, relationship)} {child}"
        f" : {render_edge_label(relationship.foreign_key)}"
    )


def _count_of_children(database: Database, relationship: Relationship) -> str:
    """Describe how many child rows one parent row can have.

    Args:
        database: The database the relationship belongs to.
        relationship: The relationship to describe.

    Returns:
        The Mermaid cardinality for the child end.
    """
    # A foreign key that is also the child's primary key is unique there, so
    # one parent can match at most one child. Both keys are declared, so this
    # costs no query.
    if relationship.foreign_key == database.schema(relationship.child).primary_key:
        return "zero or one"
    return "0+"


def render_edge_label(foreign_key: str) -> str:
    """Render a foreign key as the label on a relationship's edge.

    Args:
        foreign_key: The child's foreign key column.

    Returns:
        The name in double quotes, which lets it keep every character a
        column name can hold except the quote itself.
    """
    # Quoting means only a literal quote, which would close the label early,
    # has to go -- so the edge shows the column name as it is really spelled.
    return f'"{foreign_key.replace(chr(34), "")}"'


def render_table(database: Database, name: str, columns: bool | str) -> list[str]:
    """Render one table as a Mermaid entity.

    Args:
        database: The database the table belongs to.
        name: The table's name.
        columns: The column mode, as documented on
            :meth:`SchemaDiagram.from_database`.

    Returns:
        The entity's lines. A bare name when no columns are shown, otherwise a
        braced block.
    """
    entity = render_table_name(name)
    if columns is False:
        return [f"  {entity}"]
    attributes = [
        f"    {attribute}" for attribute in render_attributes(database, name, columns)
    ]
    return [f"  {entity} {{", *attributes, "  }"]


def render_attributes(database: Database, name: str, columns: bool | str) -> list[str]:
    """Render the attribute lines for one table.

    Args:
        database: The database the table belongs to.
        name: The table's name.
        columns: The column mode, as documented on
            :meth:`SchemaDiagram.from_database`.

    Returns:
        One line per shown column, in schema order.
    """
    schema = database.schema(name)
    foreign_keys = {r.foreign_key for r in database.parents_of(name)}
    lines = []
    for column, dtype in schema.dtypes.items():
        role = _describe_role(column, schema, foreign_keys)
        if columns == "structural" and not role:
            continue
        lines.append(
            " ".join(
                part
                for part in (render_dtype(dtype), render_column_name(column), role)
                if part
            ),
        )
    return lines


def _describe_role(column: str, schema: TableSchema, foreign_keys: set[str]) -> str:
    """Describe what a column does structurally, as Mermaid markers and a comment.

    ``row_creation_time`` gets a comment rather than a marker because Mermaid
    has only PK, FK and UK, and its ``classDef`` styling cannot target an
    individual attribute. The comment slot is where a secondary time index
    will go too.

    Args:
        column: The column's name.
        schema: The schema of the table it belongs to.
        foreign_keys: Every foreign key column on that table.

    Returns:
        The marker and comment text, or an empty string for a column with no
        structural role.
    """
    markers = []
    if column == schema.primary_key:
        markers.append("PK")
    if column in foreign_keys:
        markers.append("FK")
    parts = [", ".join(markers)] if markers else []
    if column == schema.row_creation_time:
        parts.append('"row creation time"')
    return " ".join(parts)


def render_dtype(dtype: Any) -> str:
    """Render a narwhals dtype as a token Mermaid's type slot accepts.

    Args:
        dtype: A narwhals dtype.

    Returns:
        The dtype's class name, with its parameters in square brackets when it
        has any.
    """
    parameters = render_dtype_parameters(dtype)
    name = type(dtype).__name__
    return f"{name}[{parameters}]" if parameters else name


def render_table_name(name: str) -> str:
    """Render a table name as a Mermaid entity name.

    Any double quote in the name is dropped before quoting: an embedded quote
    would close the entity name early and break the whole diagram, not just
    this one label. A name left empty by that, or empty to start with, would
    quote to `` "" ``, which Mermaid also rejects, so it falls back to `` "_" ``.

    Args:
        name: The table's name.

    Returns:
        The name, quote-stripped, unsafe-character-substituted and wrapped in
        double quotes, which is what allows it to contain spaces.
    """
    quote_stripped = name.replace(chr(34), "")
    # The only characters a quoted entity name still rejects.
    safe = re.sub(r"[\\%\n]", "_", quote_stripped)
    return f'"{safe or "_"}"'


def render_column_name(name: str) -> str:
    """Render a column name as a Mermaid attribute name.

    Attribute names cannot be quoted, so a name Mermaid would reject is
    rewritten rather than escaped. The rewrite is lossy: two columns differing
    only by a space, or only by an unsafe character, collapse to the same
    token. The diagram is for visual inspection, so that is preferred to
    refusing to draw it.

    Args:
        name: The column's name.

    Returns:
        A name Mermaid's attribute slot parses, as close to the original as
        the grammar allows.
    """
    # Found empirically. Brackets, parens and commas are safe here even though
    # they are not everywhere, so a name like count(*) survives intact.
    safe = re.sub(r"""[:;#'|/\\<=>+&!?@$~^`{}%"\t\n]""", "_", name)
    # U+2007 renders as a space but is not one to the parser.
    parseable = safe.replace(" ", "\u2007")
    if not parseable:
        # Mermaid rejects an empty attribute name.
        return "_"
    if parseable[:1].isdigit():
        return f"_{parseable}"
    return parseable


def render_dtype_parameters(dtype: Any) -> str:
    """Render the parameters that distinguish one instance of a dtype from another.

    Args:
        dtype: A narwhals dtype.

    Returns:
        The parameter token, or an empty string for a dtype without parameters.
    """
    if isinstance(dtype, nw.Datetime):
        return render_time_parameters(dtype.time_unit, dtype.time_zone)
    if isinstance(dtype, nw.Duration):
        return render_time_parameters(dtype.time_unit, None)
    if isinstance(dtype, nw.List):
        return render_dtype(dtype.inner)
    if isinstance(dtype, nw.Enum):
        return str(len(dtype.categories))
    if isinstance(dtype, nw.Struct):
        return str(len(dtype.fields))
    return ""


def render_time_parameters(time_unit: str, time_zone: str | None) -> str:
    """Render a temporal dtype's unit and zone as one token.

    Args:
        time_unit: The dtype's time unit.
        time_zone: The dtype's time zone, or None when it is naive.

    Returns:
        The unit alone when naive, otherwise unit and zone joined by a hyphen.
    """
    if time_zone is None:
        return time_unit
    # A zone like America/New_York would otherwise break the type slot.
    zone = re.sub(r"[^A-Za-z0-9_-]", "_", time_zone)
    return f"{time_unit}-{zone}"
