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
        lines: The diagram's Mermaid lines, starting with the ``erDiagram``
            header. :attr:`source` joins them.
    """

    lines: list[str]

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

        diagram = cls()
        for name in database.table_names:
            diagram._add_lines(*render_table(database, name, columns))
        for relationship in database.relationships:
            diagram._add_lines(render_relationship(database, relationship))
        return diagram

    def __init__(self) -> None:
        """Start an empty diagram, holding only the header."""
        self.lines = ["erDiagram"]

    def _add_lines(self, *new_lines: str, indent: int = 2) -> None:
        """Append lines to the diagram.

        Args:
            *new_lines: The lines to append
            indent: Spaces to prepend to each line.
        """
        self.lines.extend(" " * indent + line for line in new_lines)

    @property
    def source(self) -> str:
        """The Mermaid ``erDiagram`` source, as one newline-terminated string."""
        return "\n".join(self.lines) + "\n"

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
        The edge line, unindented.
    """
    parent = render_table_name(relationship.parent)
    child = render_table_name(relationship.child)
    return f'{parent} 1 to {get_count_of_children(database, relationship)} {child} : ""'


def get_count_of_children(database: Database, relationship: Relationship) -> str:
    """Describe how many child rows one parent row can have.

    Args:
        database: The database the relationship belongs to.
        relationship: The relationship to describe.

    Returns:
        The Mermaid cardinality for the child end.
    """
    if relationship.foreign_key == database.schema(relationship.child).primary_key:
        return "zero or one"
    return "0+"


def render_table(database: Database, name: str, columns: bool | str) -> list[str]:
    """Render one table as a Mermaid entity.

    Args:
        database: The database the table belongs to.
        name: The table's name.
        columns: The column mode, as documented on
            :meth:`SchemaDiagram.from_database`.

    Returns:
        The entity's lines, unindented apart from the attributes, which are
        nested one level. A bare name when no columns are shown, otherwise a
        braced block.
    """
    entity = render_table_name(name)
    if columns is False:
        return [entity]
    attributes = [
        f"  {attribute}" for attribute in render_attributes(database, name, columns)
    ]
    return [f"{entity} {{", *attributes, "}"]


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
    parents = collect_parents_by_foreign_key(database, name)
    lines = []
    for column, dtype in schema.dtypes.items():
        role = describe_role(column, schema, parents)
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


def collect_parents_by_foreign_key(
    database: Database, name: str
) -> dict[str, list[str]]:
    """Group a table's parents by the foreign key column pointing at them.

    Args:
        database: The database the table belongs to.
        name: The table's name.

    Returns:
        One entry per foreign key column, holding the parent tables it points
        at, in insertion order.
    """
    parents: dict[str, list[str]] = {}
    for relationship in database.parents_of(name):
        parents.setdefault(relationship.foreign_key, []).append(relationship.parent)
    return parents


def describe_role(
    column: str, schema: TableSchema, parents: dict[str, list[str]]
) -> str:
    """Describe what a column does structurally, as Mermaid markers and a comment.

    Args:
        column: The column's name.
        schema: The schema of the table it belongs to.
        parents: The table's parents, keyed by foreign key column.

    Returns:
        The marker and comment text, or an empty string for a column with no
        structural role.
    """
    markers = describe_markers(column, schema, parents)
    comment = render_comment(describe_comments(column, schema, parents))
    return " ".join(part for part in (markers, comment) if part)


def describe_markers(
    column: str, schema: TableSchema, parents: dict[str, list[str]]
) -> str:
    """Render the key markers a column carries.

    Args:
        column: The column's name.
        schema: The schema of the table it belongs to.
        parents: The table's parents, keyed by foreign key column.

    Returns:
        The comma-separated markers, or an empty string for a column that is
        neither a primary nor a foreign key.
    """
    markers = []
    if column == schema.primary_key:
        markers.append("PK")
    if column in parents:
        markers.append("FK")
    return ", ".join(markers)


def describe_comments(
    column: str,
    schema: TableSchema,
    parents: dict[str, list[str]],
) -> list[str]:
    """Describe everything about a column that Mermaid has no marker for.

    Mermaid knows only PK, FK and UK, and its ``classDef`` styling cannot
    target an individual attribute, so the tables a foreign key points at and
    the ``row_creation_time`` both have to travel in the comment slot.

    Args:
        column: The column's name.
        schema: The schema of the table it belongs to.
        parents: The table's parents, keyed by foreign key column.

    Returns:
        One phrase per role, in the order they should be shown.
    """
    comments = []
    if column in parents:
        comments.append("-> " + ", ".join(parents[column]))
    if column == schema.row_creation_time:
        comments.append("row creation time")
    return comments


def render_comment(comments: list[str]) -> str:
    """Render a column's comment phrases as one Mermaid comment.

    Args:
        comments: The phrases to show, as built by :func:`describe_comments`.

    Returns:
        The phrases joined and double quoted, or an empty string when there
        are none.
    """
    if not comments:
        return ""
    # A literal quote would close the comment early and break the diagram.
    joined = "; ".join(comments).replace(chr(34), "")
    return f'"{joined}"'


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
    # Found empirically
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
