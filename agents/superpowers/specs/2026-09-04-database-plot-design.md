# Database.plot: a Mermaid ER diagram of the schema

Design for `Database.plot()` and the `tusk-ml[plot]` extra.

## Problem

A `Database` carries everything needed to draw its own schema — table names,
column dtypes, primary keys, foreign keys, `row_creation_time` — and offers no
way to look at it. Users check their schema by reading back the `add_table` and
`add_relationship` calls they just wrote, which verifies nothing.

featuretools answers this with `EntitySet.plot()`, which returns a
`graphviz.Digraph` and therefore requires both the graphviz Python package and
the system binary.

## Decisions

These were settled with the maintainer.

1. **Mermaid is the output.** `plot()` produces Mermaid `erDiagram` source.
   Rendering it to an image is a separate, optional step.
2. **One renderer, no plugin seam.** `save()` calls `mermaidx` directly. A
   `renderer=` argument was considered and rejected: anyone able to write a
   renderer can take `str(diagram)` and render it themselves, so the argument
   adds surface without adding capability.
3. **Scope is the schema only.** Feature lineage graphs (issue #7) are a
   different picture of a different object and get their own design.
4. **Literal Mermaid attribute order.** Attributes are emitted as
   `type name [PK|FK] "comment"`, so the source is honest ER that pastes into
   mermaid.live unchanged. This renders as dtype | name | key | comment.
5. **`plot()` reads no rows.** It uses only the declared schema, matching the
   `add_table(validate=False)` default: the common path costs nothing and never
   touches data.

## Architecture

### Public API

```python
SchemaDiagram.from_database(database, columns=True) -> SchemaDiagram
db.plot(columns=True) -> SchemaDiagram          # shortcut for the above
```

| `columns` | Shows |
| --- | --- |
| `True` (default) | Every column |
| `False` | Table names and relationship lines only |
| `"structural"` | Primary key, foreign keys, and `row_creation_time` only |

Any other value raises `ValueError` naming the three accepted values.

`"structural"` exists for wide tables. A relbench-scale table with 100+ columns
produces a box no one can read, and the columns that make a schema diagram
worth looking at are the ones that define its structure.

### SchemaDiagram

A plain class holding one attribute, `source: str`.

| Member | Returns | Notes |
| --- | --- | --- |
| `from_database(database, columns)` | `SchemaDiagram` | The named constructor; builds the source |
| `__str__` | The Mermaid source | |
| `_repr_markdown_` | A fenced ` ```mermaid ` block | Renders inline in Jupyter, GitHub, and the zensical docs |
| `save(path)` | `None` | Suffix selects the format |

`save()` dispatches on the suffix. `.mmd` and `.md` write the source and need
no dependency. `.svg`, `.png`, and `.pdf` go through `mermaidx`. Any other
suffix raises `ValueError` listing the supported ones.

The class is not a dataclass. `frozen=True` would supply `__eq__`, `__hash__`
and `__repr__` that nothing uses, and its generated `__repr__` would dump an
entire diagram. What earns the class its place is `_repr_markdown_`: a plain
`str` cannot render itself in a notebook.

The suffix tuples are locals inside `save()`, its only reader.

### Module layout

New module `src/tusk/plotting.py`. `SchemaDiagram` is its only public name;
every renderer and escaper below it is private, so the module surface matches
the documented API rather than exceeding it. `Database.plot()` is a one-line
delegate, mirroring how `Database.validate()` delegates to `validation.py` —
including the same `TYPE_CHECKING` import-cycle guard.

`SchemaDiagram` is exported from `tusk` for type annotations and documented in
a new `docs/api/plotting.md`.

## Diagram content

```
erDiagram
  "customers" 1 to 0+ "transactions" : "customer_id"
  "customers" {
    Int64            customer_id  PK
    Datetime[ns-UTC] signed_up_at    "row creation time"
    String           region
  }
  "transactions" {
    Int64            txn_id       PK
    Int64            customer_id  FK
    Datetime[ns-UTC] placed_at       "row creation time"
    Float64          amount
  }
```

One entity per table, in insertion order. One relationship line per
`Relationship`, labelled with the foreign key column.

Mermaid's word and digit aliases are used rather than the `||--o{` symbols,
because `print(db.plot())` and the `.mmd` format are first-class outputs that
people read. `1` and `0+` are aliases for `||` and `o{`; `zero or one` has no
short alias, so it is spelled out.

### Cardinality

The parent end is always `1`. That is what the schema declares; a null foreign
key or an orphan row, which would make it `zero or one`, is only discoverable
by reading rows.

The child end is `0+`, except when the foreign key **is** the child's
`primary_key`, where it is `zero or one`. A primary key is unique, so one
parent matches at most one such child. Both keys are declared, so the narrower
cardinality costs no query.

The line is always solid (`to`, never `optionally to`). Mermaid's
`optionally to` selects a dashed line and says nothing about nullability, but
it reads as though it does — a misleading word in generated source that people
read.

### Edge label

The foreign key's name, in double quotes. A quoted label accepts every
character except a literal `"`, which would close it early and is dropped. The
label is therefore verbatim, and does **not** share the attribute slot's
substitutions — which is why a column named `count(*)` keeps its punctuation.

### Key markers

`PK` for the table's `primary_key`, `FK` for every foreign key from
`parents_of()`. A column that is both is emitted as `PK, FK`, which Mermaid
accepts.

`row_creation_time` has no Mermaid marker and cannot be given one: `classDef`
applies to whole entities, never to individual attributes. It is marked with
the comment `"row creation time"` instead. This is also the slot that secondary
time indices will use when they land (issue #15).

A table with no `primary_key` simply has no `PK` marker.

## Escaping

Mermaid's grammar rejects several things narwhals produces. Each rule below was
verified by rendering through mermaidx.

| Case | Rule | Reason |
| --- | --- | --- |
| Table name | Always quoted | Lets names contain spaces |
| Column name containing a space | Space → U+2007 FIGURE SPACE | A real space is a parse error and attribute names cannot be quoted; U+2007 renders identically |
| Column name starting with a digit | Prefix `_` | `2024_total` is a parse error; `_2024_total` parses |
| Column name containing `:;#'\|/\<=>+&!?@$~^`{}%"`, tab or newline | → `_` | Each confirmed by rendering to make an attribute name unparseable |
| Column name containing `()[],` | Left alone | These parse in the attribute slot; only the edge label rejected them, and it is quoted separately |
| Empty column name | → `_` | An empty attribute name is a parse error |
| Foreign key in the edge label | Quoted, `"` dropped | A quoted label accepts everything but a literal quote, which would close it early |
| `Datetime(time_unit='ns', time_zone='UTC')` | → `Datetime[ns-UTC]` | Quotes, commas and `=` are illegal in the type slot |
| `Datetime(time_unit='us', time_zone=None)` | → `Datetime[us]` | A naive datetime shows no zone |
| `Datetime(..., time_zone='America/New_York')` | → `Datetime[ns-America_New_York]` | `/`, `+` and `:` are all parse errors; every character outside `[A-Za-z0-9_-]` becomes `_` |
| `Duration(time_unit='ms')` | → `Duration[ms]` | Same charset restriction |
| `List(Int64)` | → `List[Int64]` | Rendered recursively, so `List(List(Int64))` → `List[List[Int64]]` |
| `Enum(...)` / `Struct(...)` | → `Enum[3]` / `Struct[2]` | Variant and field counts; the full contents would not fit |
| Any other dtype | The dtype class name | Already a bare identifier |

Hyphens, dots, underscores and non-ASCII letters need no escaping in either
slot.

The two name rules are lossy in principle — two columns differing only by a
space versus U+2007 would collide. In practice the diagram is for reading, not
round-tripping, so this is accepted rather than guarded.

## Packaging

```toml
[project.optional-dependencies]
plot = ["mermaidx>=0.9.5"]
```

`mermaidx` renders through an embedded QuickJS engine and resvg. It needs no
browser, Node, or npm, and ships a pure-Python wheel; its dependencies
(`quickjs-ng`, `resvg-py`, `termaid`) publish abi3 wheels across platforms. It
is MIT licensed.

The alternatives were rejected: `mermaid-cli` pulls playwright and still needs
`playwright install chromium`; `pymmdc` requires a Node toolchain; `mermaid-py`
and `mermaidian` POST the diagram to the hosted mermaid.ink service, which
would send user table and column names to a third party.

`mermaidx` is at 0.9.5 and was first released weeks before this design, so
tusk is an early adopter. Its immaturity shows: on some invalid sources its own
error path raises `TypeError: Unknown quickjs tag: -6` instead of reporting the
parse error. This is why escaping is tusk's responsibility rather than
something a renderer error message can be relied on to surface. The risk is
otherwise contained: it is reached from exactly one function, and replacing it
changes no public API.

Importing it is deferred to the point of use. A missing install raises
`ImportError` naming `pip install tusk-ml[plot]`. Every other member of
`SchemaDiagram` works without it.

`mermaidx` is added to the `test` dependency group so the render test can run.

## Testing

- Golden Mermaid source for each `columns` value, over a fixture database with
  a parent, a child, and an unrelated table.
- `columns` rejecting an invalid value.
- Escaping: a space in a column name, a leading digit, a non-ASCII name, a
  tz-aware and a naive `Datetime`, `List`, `Enum`, `Struct`.
- A table with no `primary_key`, and a column that is both `PK` and `FK`.
- **A render through mermaidx asserting the source parses.** This is the test
  that catches escaping regressions; the golden-source tests only compare text.
- `save()` suffix dispatch, `.mmd` succeeding with no renderer involved, an
  unsupported suffix raising, and the `ImportError` message when `mermaidx` is
  absent.

## Out of scope

- Feature lineage graphs (issue #7).
- Secondary time index marking (issue #15), which reuses the comment slot.
- Theming, layout direction, and image dimensions. Users wanting control take
  `str(diagram)` and drive their renderer directly.
