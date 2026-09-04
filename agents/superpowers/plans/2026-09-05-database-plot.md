# Database.plot() Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `Database` a `plot()` method that returns a Mermaid ER diagram of its schema, renderable to SVG/PNG/PDF through an optional extra.

**Architecture:** A new `src/tusk/plotting.py` holds two pure token-sanitising helpers, a source builder, and the `SchemaDiagram` value object. `Database.plot()` is a thin delegate so `database.py` gains no second responsibility. Rendering is confined to `SchemaDiagram.save()`, which imports `mermaidx` at point of use.

**Tech Stack:** Python ≥3.10, narwhals, Mermaid `erDiagram` syntax, `mermaidx` (optional).

**Spec:** `agents/superpowers/specs/2026-09-04-database-plot-design.md`

## Global Constraints

- `plot()` reads no rows. It uses only `TableSchema.dtypes`, `TableSchema.primary_key`, `TableSchema.row_creation_time`, and `Database.parents_of()`. Never call `.collect()` or `.collect_schema()`.
- Attributes are emitted in literal Mermaid order: `type name [PK|FK] "comment"`.
- Cardinality is always `||--o{`.
- `columns` accepts exactly `True`, `False`, `"structural"`. Anything else raises `ValueError`.
- Extra is named `plot`, pinned `mermaidx>=0.9.5`.
- Style rules from `agents/style/CODESTYLE.md` apply: descriptive names, functions do one thing, callers above callees, Google docstrings on everything (`interrogate` requires 100%), comments explain *why* only.
- `ruff` lint selects `E,F,I,UP,B,D`; `pydoclint` runs in Google style. Run `just lint` before every commit.
- The diagram is for visual inspection only. Name sanitisation is deliberately lossy and must not raise.
- Tasks 2–5 say "append to `tests/test_plotting.py`". The *tests* append; the *imports* they show go into the module's existing top import block, because ruff's `E402` rejects a mid-file import.

---

### Task 1: Mermaid-safe dtype rendering

Turns a narwhals dtype into a token Mermaid's `type` slot accepts. Mermaid types may contain only letters, digits, hyphens, underscores, parens and brackets, and must start with a letter — so `Datetime(time_unit='ns', time_zone='America/New_York')` is a parse error and must become `Datetime[ns-America_New_York]`.

**Files:**
- Create: `src/tusk/plotting.py`
- Test: `tests/test_plotting.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `render_dtype(dtype: Any) -> str`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_plotting.py`:

```python
import narwhals as nw
import pytest

from tusk.plotting import render_dtype


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (nw.Int64(), "Int64"),
        (nw.String(), "String"),
        (nw.Float64(), "Float64"),
        (nw.Date(), "Date"),
        (nw.Datetime(time_unit="us"), "Datetime[us]"),
        (nw.Datetime(time_unit="ns", time_zone="UTC"), "Datetime[ns-UTC]"),
        (
            nw.Datetime(time_unit="ns", time_zone="America/New_York"),
            "Datetime[ns-America_New_York]",
        ),
        (nw.Duration(time_unit="ms"), "Duration[ms]"),
        (nw.List(nw.Int64()), "List[Int64]"),
        (nw.List(nw.List(nw.Int64())), "List[List[Int64]]"),
        (nw.Enum(["a", "b", "c"]), "Enum[3]"),
        (nw.Struct({"x": nw.Int64(), "y": nw.Int64()}), "Struct[2]"),
    ],
)
def test_dtype_renders_as_a_mermaid_safe_token(dtype, expected):
    assert render_dtype(dtype) == expected


def test_timezone_punctuation_is_replaced():
    # Slashes, plus signs and colons are all parse errors in Mermaid's type
    # slot, so every character outside the safe set collapses to underscore.
    rendered = render_dtype(nw.Datetime(time_unit="ns", time_zone="UTC+02:00"))
    assert rendered == "Datetime[ns-UTC_02_00]"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_plotting.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tusk.plotting'`

- [ ] **Step 3: Write the implementation**

Create `src/tusk/plotting.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_plotting.py -v`
Expected: PASS (13 tests)

- [ ] **Step 5: Lint**

Run: `just lint`
Expected: every hook passes. Fix any `pydoclint`/`interrogate` complaint before committing.

- [ ] **Step 6: Commit**

```bash
git add src/tusk/plotting.py tests/test_plotting.py
git commit -m "feat: render narwhals dtypes as mermaid-safe tokens"
```

---

### Task 2: Mermaid-safe column and table names

Mermaid attribute names cannot be quoted, cannot contain a space, and cannot start with a digit. Table names have neither restriction once quoted. Both rules are lossy on purpose — the diagram is for visual inspection, so sanitising must never raise.

**Files:**
- Modify: `src/tusk/plotting.py`
- Test: `tests/test_plotting.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `render_column_name(name: str) -> str`, `render_table_name(name: str) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_plotting.py`:

```python
from tusk.plotting import render_column_name, render_table_name

FIGURE_SPACE = " "


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("amount", "amount"),
        ("total_2024", "total_2024"),
        ("n-items", "n-items"),
        ("a.b", "a.b"),
        ("straße", "straße"),
        ("unit price", f"unit{FIGURE_SPACE}price"),
        ("2024_total", "_2024_total"),
        ("2024 total", f"_2024{FIGURE_SPACE}total"),
    ],
)
def test_column_name_is_made_parseable(name, expected):
    assert render_column_name(name) == expected


def test_table_name_is_quoted():
    # Quoting is what lets a table name contain a space, which an attribute
    # name cannot.
    assert render_table_name("order items") == '"order items"'
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_plotting.py -v`
Expected: FAIL — `ImportError: cannot import name 'render_column_name'`

- [ ] **Step 3: Write the implementation**

Add to `src/tusk/plotting.py`, below `render_dtype` and above `_render_dtype_parameters`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_plotting.py -v`
Expected: PASS (22 tests)

- [ ] **Step 5: Lint**

Run: `just lint`

- [ ] **Step 6: Commit**

```bash
git add src/tusk/plotting.py tests/test_plotting.py
git commit -m "feat: render table and column names as parseable mermaid names"
```

---

### Task 3: The diagram source builder

Walks a `Database` and emits `erDiagram` source. This is the task that adds `mermaidx` to the test group, because its last test is the one that proves the generated source actually parses — the golden-string tests only compare text.

**Files:**
- Modify: `src/tusk/plotting.py`, `pyproject.toml`
- Test: `tests/test_plotting.py`

**Interfaces:**
- Consumes: `render_dtype`, `render_table_name`, `render_column_name` from Tasks 1 and 2.
- Produces: `build_schema_source(database: Database, columns: bool | str) -> str`.

- [ ] **Step 1: Add mermaidx to the test dependency group**

In `pyproject.toml`, inside `[dependency-groups]`, add `"mermaidx>=0.9.5",` to the `test` list, keeping it alphabetically ordered:

```toml
test = [
    "duckdb>=1.0",
    "mermaidx>=0.9.5",
    "pandas>=2.0",
    "polars>=1.43",
    "pyarrow>=17",
    "pytest>=8.0",
    "scikit-learn>=1.4",
]
```

Run: `uv sync --all-groups --all-extras`

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_plotting.py`:

```python
import polars as pl

import tusk
from tusk.plotting import build_schema_source


@pytest.fixture
def two_table_db():
    """A parent and a child, plus one table related to neither."""
    return (
        tusk.Database("shop")
        .add_table(
            "customers",
            pl.LazyFrame({"id": [1], "signed_up_at": [None], "region": ["eu"]}).cast(
                {"signed_up_at": pl.Datetime("us")},
            ),
            primary_key="id",
            row_creation_time="signed_up_at",
        )
        .add_table(
            "orders",
            pl.LazyFrame({"id": [1], "customer_id": [1], "amount": [1.0]}),
            primary_key="id",
        )
        .add_table(
            "regions",
            pl.LazyFrame({"code": ["eu"]}),
            primary_key="code",
        )
        .add_relationship(parent="customers", child="orders", foreign_key="customer_id")
    )


def test_every_table_and_relationship_appears(two_table_db):
    source = build_schema_source(two_table_db, columns=True)
    assert source.startswith("erDiagram\n")
    assert '"customers" ||--o{ "orders" : customer_id' in source
    # A table with no relationships still has to be drawn.
    assert '"regions"' in source


def test_columns_true_lists_every_column_with_markers(two_table_db):
    source = build_schema_source(two_table_db, columns=True)
    assert "Int64 id PK" in source
    assert "Int64 customer_id FK" in source
    assert 'Datetime[us] signed_up_at "row creation time"' in source
    assert "Float64 amount" in source


def test_columns_false_omits_every_attribute(two_table_db):
    source = build_schema_source(two_table_db, columns=False)
    assert "{" not in source
    assert '"customers" ||--o{ "orders" : customer_id' in source


def test_columns_structural_keeps_only_keys_and_the_time_index(two_table_db):
    source = build_schema_source(two_table_db, columns="structural")
    assert "Int64 customer_id FK" in source
    assert 'Datetime[us] signed_up_at "row creation time"' in source
    assert "region" not in source
    assert "amount" not in source


def test_a_column_that_is_both_keys_gets_both_markers():
    # orders.id is the primary key and also the foreign key to customers.
    db = (
        tusk.Database("d")
        .add_table("customers", pl.LazyFrame({"id": [1]}), primary_key="id")
        .add_table("orders", pl.LazyFrame({"id": [1]}), primary_key="id")
        .add_relationship(parent="customers", child="orders", foreign_key="id")
    )
    assert "Int64 id PK, FK" in build_schema_source(db, columns=True)


def test_a_table_without_a_primary_key_has_no_marker():
    with pytest.warns(tusk.exceptions.MissingPrimaryKeyWarning):
        db = tusk.Database("d").add_table("t", pl.LazyFrame({"a": [1]}))
    source = build_schema_source(db, columns=True)
    assert "Int64 a" in source
    assert "PK" not in source


def test_an_unknown_columns_value_is_rejected(two_table_db):
    with pytest.raises(ValueError, match="structural"):
        build_schema_source(two_table_db, columns="all")


def test_generated_source_parses(two_table_db):
    # The golden-string tests above compare text and would happily accept
    # source Mermaid cannot parse. This is the test that catches an escaping
    # regression, so it renders for real.
    mermaidx = pytest.importorskip("mermaidx")
    mermaidx.render(build_schema_source(two_table_db, columns=True)).svg()


def test_hostile_names_still_parse():
    db = (
        tusk.Database("d")
        .add_table(
            "order items",
            pl.LazyFrame({"id": [1], "unit price": [1.0], "2024 total": [2.0]}),
            primary_key="id",
        )
    )
    mermaidx = pytest.importorskip("mermaidx")
    mermaidx.render(build_schema_source(db, columns=True)).svg()
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_plotting.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_schema_source'`

- [ ] **Step 4: Write the implementation**

Add to `src/tusk/plotting.py`. `build_schema_source` goes directly below the module docstring's imports and above `render_table_name`, so callers sit above callees:

```python
COLUMN_MODES = (True, False, "structural")


def build_schema_source(database: Database, columns: bool | str = True) -> str:
    """Build Mermaid ``erDiagram`` source for a database's schema.

    Args:
        database: The database to draw.
        columns: True lists every column, False lists none, and ``"structural"``
            lists only the primary key, the foreign keys, and the
            ``row_creation_time``.

    Returns:
        The diagram source.

    Raises:
        ValueError: If ``columns`` is not True, False, or ``"structural"``.
    """
    if columns not in COLUMN_MODES:
        raise ValueError(
            f"columns must be True, False, or 'structural'; got {columns!r}",
        )
    lines = ["erDiagram"]
    lines.extend(_render_relationship(r) for r in database.relationships)
    for name in database.table_names:
        lines.extend(_render_table(database, name, columns))
    return "\n".join(lines) + "\n"


def _render_relationship(relationship: Relationship) -> str:
    """Render one relationship as a Mermaid edge.

    The cardinality is always one-to-many: a tusk relationship is one by
    definition, and a nullable foreign key -- the only thing that would make
    the parent side optional -- could not be detected without reading rows.

    Args:
        relationship: The relationship to draw.

    Returns:
        The edge line, indented.
    """
    parent = render_table_name(relationship.parent)
    child = render_table_name(relationship.child)
    return f"  {parent} ||--o{{ {child} : {relationship.foreign_key}"


def _render_table(database: Database, name: str, columns: bool | str) -> list[str]:
    """Render one table as a Mermaid entity.

    Args:
        database: The database the table belongs to.
        name: The table's name.
        columns: The column mode, as documented on :func:`build_schema_source`.

    Returns:
        The entity's lines. A bare name when no columns are shown, otherwise a
        braced block.
    """
    entity = render_table_name(name)
    if columns is False:
        return [f"  {entity}"]
    attributes = [
        f"    {attribute}"
        for attribute in _render_attributes(database, name, columns)
    ]
    return [f"  {entity} {{", *attributes, "  }"]


def _render_attributes(database: Database, name: str, columns: bool | str) -> list[str]:
    """Render the attribute lines for one table.

    Args:
        database: The database the table belongs to.
        name: The table's name.
        columns: The column mode, as documented on :func:`build_schema_source`.

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
```

Add these imports at the top of the module, alongside the existing ones:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tusk.database import Database, Relationship, TableSchema
```

This guard is required, not optional. Task 5 has `database.py` import from
`plotting.py`, and a plain module-level import here would close the cycle:

```
ImportError: cannot import name 'Database' from partially initialized
module 'tusk.database' (most likely due to a circular import)
```

All three names are used in annotations only, and the module already has
`from __future__ import annotations`, so nothing is needed at runtime.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_plotting.py -v`
Expected: PASS (31 tests)

- [ ] **Step 6: Lint**

Run: `just lint`

- [ ] **Step 7: Commit**

```bash
git add src/tusk/plotting.py tests/test_plotting.py pyproject.toml uv.lock
git commit -m "feat: build mermaid erDiagram source from a database schema"
```

---

### Task 4: SchemaDiagram and the plot extra

The value object `plot()` hands back. It displays itself in a notebook, prints as source, and writes itself to a file. Only the image formats need `mermaidx`, so the import is deferred to the branch that uses it.

**Files:**
- Modify: `src/tusk/plotting.py`, `pyproject.toml`
- Test: `tests/test_plotting.py`

**Interfaces:**
- Consumes: nothing from Task 3 directly; `SchemaDiagram` is constructed from a source string.
- Produces: `SchemaDiagram(source: str)` with `__str__`, `_repr_markdown_()`, and `save(path: str | Path) -> None`.

- [ ] **Step 1: Add the plot extra**

In `pyproject.toml`, under `[project.optional-dependencies]`, add the extra beside the existing `sklearn` one:

```toml
[project.optional-dependencies]
sklearn = ["scikit-learn>=1.4"]
plot = ["mermaidx>=0.9.5"]
```

Run: `uv sync --all-groups --all-extras`

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_plotting.py`:

```python
from tusk.plotting import SchemaDiagram

SOURCE = 'erDiagram\n  "t" {\n    Int64 id PK\n  }\n'


def test_str_is_the_source():
    assert str(SchemaDiagram(SOURCE)) == SOURCE


def test_markdown_repr_is_a_mermaid_block():
    # Jupyter, GitHub and the docs site all render a fenced mermaid block.
    rendered = SchemaDiagram(SOURCE)._repr_markdown_()
    assert rendered.startswith("```mermaid\n")
    assert rendered.endswith("```")
    assert SOURCE in rendered


@pytest.mark.parametrize("suffix", [".mmd", ".md"])
def test_saving_source_needs_no_renderer(tmp_path, suffix, monkeypatch):
    # Hiding mermaidx proves the text formats never reach for it.
    monkeypatch.setitem(sys.modules, "mermaidx", None)
    path = tmp_path / f"schema{suffix}"
    SchemaDiagram(SOURCE).save(path)
    assert path.read_text(encoding="utf-8") == SOURCE


@pytest.mark.parametrize("suffix", [".svg", ".png"])
def test_saving_an_image_writes_a_file(tmp_path, suffix):
    pytest.importorskip("mermaidx")
    path = tmp_path / f"schema{suffix}"
    SchemaDiagram(SOURCE).save(path)
    assert path.stat().st_size > 0


def test_an_unsupported_suffix_is_rejected(tmp_path):
    with pytest.raises(ValueError, match=".mmd"):
        SchemaDiagram(SOURCE).save(tmp_path / "schema.gif")


def test_a_missing_renderer_names_the_extra(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "mermaidx", None)
    with pytest.raises(ImportError, match=r"tusk-ml\[plot\]"):
        SchemaDiagram(SOURCE).save(tmp_path / "schema.svg")
```

Add `import sys` to the test module's top import block; `tmp_path` is already a `Path`, so nothing else is needed.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_plotting.py -v`
Expected: FAIL — `ImportError: cannot import name 'SchemaDiagram'`

- [ ] **Step 4: Write the implementation**

Add to `src/tusk/plotting.py`, directly below `build_schema_source` so the caller stays above its callees:

```python
SOURCE_SUFFIXES = (".mmd", ".md")
IMAGE_SUFFIXES = (".svg", ".png", ".pdf")


@dataclass(frozen=True)
class SchemaDiagram:
    """A Mermaid diagram of a database's schema.

    Printing it, or reading :attr:`source`, gives the Mermaid source, which is
    the escape hatch for any renderer: this class only knows how to display
    itself and write a file.

    Attributes:
        source: The Mermaid ``erDiagram`` source.
    """

    source: str

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
        diagram and need ``tusk-ml[plot]``.

        Args:
            path: Where to write, including the suffix.

        Raises:
            ValueError: If the suffix names no supported format.
            ImportError: If an image format is asked for and the renderer is
                not installed.
        """
        path = Path(path)
        if path.suffix in SOURCE_SUFFIXES:
            path.write_text(self.source, encoding="utf-8")
        elif path.suffix in IMAGE_SUFFIXES:
            _render_to_file(self.source, path)
        else:
            supported = ", ".join(SOURCE_SUFFIXES + IMAGE_SUFFIXES)
            raise ValueError(
                f"cannot save {path.suffix!r}; supported suffixes are {supported}",
            )


def _render_to_file(source: str, path: Path) -> None:
    """Render Mermaid source to an image file.

    Args:
        source: The diagram source.
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
    mermaidx.render(source).save(str(path))
```

Add to the module's imports:

```python
from dataclasses import dataclass
from pathlib import Path
```

Note on the `ImportError` test: `monkeypatch.setitem(sys.modules, "mermaidx", None)` makes `import mermaidx` raise `ImportError`, which is what the `try` block catches.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_plotting.py -v`
Expected: PASS (39 tests)

- [ ] **Step 6: Lint**

Run: `just lint`

- [ ] **Step 7: Commit**

```bash
git add src/tusk/plotting.py tests/test_plotting.py pyproject.toml uv.lock
git commit -m "feat: add SchemaDiagram and the plot extra"
```

---

### Task 5: Database.plot(), exports, and docs

Wires the builder to the public surface and documents it. `Database.plot()` stays a delegate so `database.py` keeps its single responsibility.

**Files:**
- Modify: `src/tusk/database.py`, `src/tusk/__init__.py`, `docs/api/index.md`, `docs/guide/databases.md`, `zensical.toml`
- Create: `docs/api/plotting.md`
- Test: `tests/test_plotting.py`, `tests/test_packaging.py`

**Interfaces:**
- Consumes: `build_schema_source` and `SchemaDiagram` from Tasks 3 and 4.
- Produces: `Database.plot(columns: bool | str = True) -> SchemaDiagram`, and `tusk.SchemaDiagram`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_plotting.py`:

```python
def test_plot_returns_a_diagram_of_the_database(db):
    diagram = db.plot()
    assert isinstance(diagram, SchemaDiagram)
    assert '"customers" ||--o{ "sessions" : customer_id' in diagram.source
    assert '"sessions" ||--o{ "transactions" : session_id' in diagram.source


def test_plot_passes_the_column_mode_through(db):
    assert "age" in db.plot(columns=True).source
    assert "age" not in db.plot(columns="structural").source
    assert "age" not in db.plot(columns=False).source


def test_plot_reads_no_rows(db, monkeypatch):
    # The whole diagram comes from declared schema. Collecting would make
    # plotting cost as much as computing, on a method that looks free.
    def fail(*args, **kwargs):
        raise AssertionError("plot() must not collect")

    monkeypatch.setattr(nw.LazyFrame, "collect", fail)
    db.plot()
```

Append to `tests/test_packaging.py`:

```python
def test_schema_diagram_is_exported():
    assert callable(tusk.Database("d").plot)
    assert tusk.SchemaDiagram
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_plotting.py tests/test_packaging.py -v`
Expected: FAIL — `AttributeError: 'Database' object has no attribute 'plot'`

- [ ] **Step 3: Add the method**

In `src/tusk/database.py`, add `plot` to the `Database` class immediately after `validate()`, so it sits with the other whole-database operations and above the per-table accessors:

```python
    def plot(self, columns: bool | str = True) -> SchemaDiagram:
        """Draw the database's schema as an entity-relationship diagram.

        Reads no rows: the diagram is built entirely from declared schema.

        Args:
            columns: True lists every column, False lists none, and
                ``"structural"`` lists only the primary key, the foreign keys,
                and the ``row_creation_time``.

        Returns:
            The diagram, which renders itself in a notebook and writes itself
            to a file with :meth:`~tusk.SchemaDiagram.save`.

        Raises:
            ValueError: If ``columns`` is not True, False, or ``"structural"``.
        """
        return SchemaDiagram(build_schema_source(self, columns))
```

Import it at the bottom of `database.py`'s import block:

```python
from tusk.plotting import SchemaDiagram, build_schema_source
```

This is the import that closes the cycle with `plotting.py`. Task 3 already put `plotting.py`'s own import of `Database`, `Relationship` and `TableSchema` behind `if TYPE_CHECKING:` for exactly this reason, so this direction is the only runtime edge and the package imports cleanly. If you see `ImportError: cannot import name 'Database' from partially initialized module`, that guard is missing.

- [ ] **Step 4: Export SchemaDiagram**

In `src/tusk/__init__.py`, add the import and the `__all__` entry, keeping both alphabetical:

```python
from tusk.plotting import SchemaDiagram
```

```python
__all__ = [
    "__version__",
    "Database",
    "FeatureList",
    "Relationship",
    "SchemaDiagram",
    "TableSchema",
    "apply_features",
    "deep_feature_synthesis",
    "exceptions",
]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_plotting.py tests/test_packaging.py -v`
Expected: PASS

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest`
Expected: PASS. A circular import would surface here as a collection error across unrelated modules.

- [ ] **Step 7: Write the API page**

Create `docs/api/plotting.md`:

```markdown
# Plotting

::: tusk.plotting
    options:
      members: false

## SchemaDiagram

::: tusk.SchemaDiagram
```

- [ ] **Step 8: Add it to the docs navigation**

In `zensical.toml`, add `"api/plotting.md",` to the `API reference` list, after `"api/database.md",`.

In `docs/api/index.md`, add a bullet after the Databases line:

```markdown
- [Plotting](plotting.md) — `SchemaDiagram`, what `Database.plot()` returns.
```

- [ ] **Step 9: Document it in the guide**

Append to `docs/guide/databases.md`:

````markdown
## Looking at the schema

`plot()` draws the database as a Mermaid entity-relationship diagram. It reads
no rows, so it costs nothing:

```python
db.plot()
```

In a notebook the diagram renders inline. Elsewhere, `print(db.plot())` gives
the Mermaid source, and `save()` writes a file:

```python
db.plot().save("schema.svg")
```

`.mmd` and `.md` write the source and need nothing installed. `.svg`, `.png`
and `.pdf` render the diagram and need the extra:

```bash
pip install tusk-ml[plot]
```

Wide tables make an unreadable picture. `columns="structural"` keeps only the
primary key, the foreign keys and the `row_creation_time`; `columns=False`
keeps only the table names and the lines between them:

```python
db.plot(columns="structural")
```
````

- [ ] **Step 10: Build the docs**

Run: `just docs`
Expected: builds with no warning about `api/plotting.md`.

- [ ] **Step 11: Lint**

Run: `just lint`

- [ ] **Step 12: Commit**

```bash
git add src/tusk/database.py src/tusk/__init__.py src/tusk/plotting.py \
        tests/test_plotting.py tests/test_packaging.py \
        docs/api/plotting.md docs/api/index.md docs/guide/databases.md \
        zensical.toml
git commit -m "feat: add Database.plot()"
```

---

## Self-review notes

Checked against `agents/superpowers/specs/2026-09-04-database-plot-design.md`:

| Spec requirement | Task |
| --- | --- |
| `columns=True/False/"structural"`, `ValueError` otherwise | 3 |
| Reads no rows | 3, asserted in 5 |
| `SchemaDiagram` with `__str__`, `_repr_markdown_`, `save()` | 4 |
| Suffix dispatch, `.mmd` needing no dependency | 4 |
| Literal Mermaid attribute order | 3 |
| Cardinality always `||--o{` | 3 |
| PK / FK / `PK, FK` markers, no marker without a primary key | 3 |
| `row_creation_time` as a comment | 3 |
| Table name quoting | 2 |
| Space → U+2007, leading digit → `_` prefix | 2 |
| Dtype token rendering incl. timezone punctuation, `Duration`, nested `List` | 1 |
| `plot` extra pinned `mermaidx>=0.9.5`, deferred import, `ImportError` naming the extra | 4 |
| `mermaidx` in the test group | 3 |
| Render-parses test | 3 |
| New module, `Database.plot()` a delegate | 5 |
| Exported and documented | 5 |

Out of scope per the spec and absent from this plan by design: feature lineage graphs, secondary time index marking, theming and image dimensions.
