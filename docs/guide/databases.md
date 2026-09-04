# Databases

A [`Database`][tusk.Database] holds the tables you want features over and the
relationships between them. By default it is pure schema: adding a table reads
its column names and dtypes, nothing else. No row is read unless you ask for
[validation](#validation).

```python
import tusk

db = tusk.Database("retail")
db.add_table("customers", customers_lf, primary_key="id", row_creation_time="signed_up_at")
db.add_table("sessions", sessions_lf, primary_key="id", row_creation_time="started_at")
db.add_relationship(parent="customers", child="sessions", foreign_key="customer_id")
```

Both `add_table` and `add_relationship` return the database, so they chain.

## Keys

tusk uses `primary_key` and `row_creation_time` rather than featuretools'
`index` and `time_index`. Narwhals has no index concept, and
`row_creation_time` names what the column actually means: when the row became
knowable.

`primary_key` is **optional**, but a table without one cannot be a relationship
parent or a DFS target, and order-dependent primitives on it have
non-deterministic tiebreaks. Omitting it raises a
[`MissingPrimaryKeyWarning`][tusk.exceptions.MissingPrimaryKeyWarning].

`row_creation_time` is required for order-dependent primitives on that table,
and is what a [cutoff time](deep-feature-synthesis.md#cutoff-times) filters on. A table without
one is *timeless*: it passes through every cutoff unfiltered.

Keys are single columns. Passing a tuple or list raises
[`SchemaError`][tusk.exceptions.SchemaError] — there are no composite keys.

## Relationships

`add_relationship(parent=, child=, foreign_key=)` takes three arguments, not
two pairs. The parent side is always the parent's `primary_key`, so only the
child's column needs naming.

```python
db.add_relationship(parent="sessions", child="transactions", foreign_key="session_id")
```

A relationship is one parent to many children. Chaining them is what gives DFS
depth: with `customers → sessions → transactions`, a depth-2 walk from
`customers` reaches transaction columns through sessions.

## Backends

One database uses one backend. The first table you add fixes it; a later
table on a different backend raises `SchemaError`, because narwhals cannot join
across backends.

Eagerness, by contrast, is not sticky and not remembered. `add_table` accepts
a native or narwhals frame in either form and immediately lazifies it, so an
eager frame and a lazy one are interchangeable — you can mix both forms of the
same backend in one database. Either way the feature matrix comes back
[uncomputed](deep-feature-synthesis.md#lazy-out-always).

## Validation

A database takes your declarations on trust. Naming a column as `primary_key`
asserts that it identifies a row; nothing confirms it. When the assertion is
false, tusk does not fail — a duplicated key fans out every join that lands on
the table, and `COUNT`, `SUM` and `MEAN` come back inflated by a factor you
cannot see.

[`validate()`][tusk.Database.validate] spends real queries to confirm the
declarations hold:

```python
db.validate()
```

It runs every check — each table, then each relationship, then the database
as a whole — and raises
[`ValidationError`][tusk.exceptions.ValidationError] on the first defect. It
returns the database, so it chains.

The error names the table, the column and the counts that disagree — for
example `4 rows, 3 distinct values` — but not which keys are duplicated.
Finding them costs a second full scan, which is a real bill on a remote
backend, so tusk leaves that query to you if you want it.

`validate()` takes one selector per scope — `tables`, `relationships` and
`database` — each of which accepts `True` for every check in that scope,
`False` for none, a name, or an iterable of names:

```python
db.validate(tables="unique_primary_key")     # one table check, nothing else
db.validate(relationships=False)             # skip the joins
db.validate(tables=["non_null_primary_key", "unique_primary_key"])
```

Each name is looked up only in its own scope, so `validate(tables=
"consistent_time_zones")` raises `ValueError` — that check is database-wide.

`add_table` and `add_relationship` take a single `validate=` selector, since
each already knows its scope:

```python
db.add_table("customers", customers_lf, primary_key="id", validate=True)
```

`add_table` defaults to `False`. That default is deliberate: without it every
`add_table` would materialize a full scan, which is a silent performance cliff
on `pl.scan_parquet` or a remote table.

`add_relationship` also takes it, but defaults to `"matching_key_dtypes"`
rather than `False`, because that check reads the declared dtypes and no
rows — it costs nothing, and a key dtype mismatch is worth hearing about at
the point you declare the link rather than at the join:

```python
db.add_relationship(parent="customers", child="sessions", foreign_key="customer_id")
# ValidationError: foreign_key 'customer_id' of 'sessions' is String,
# but primary_key 'id' of 'customers' is Int64

db.add_relationship(..., validate=True)   # also checks the keys overlap
db.add_relationship(..., validate=False)  # check nothing
```

A relationship that fails validation is not registered, exactly as a table
that fails is not added. A misspelled check name raises `ValueError`, not
`ValidationError`, so catching validation failures never swallows a typo.

### Available checks

Table checks run against one table. `add_table` and `validate()` both run them:

| Name | Confirms |
| --- | --- |
| `non_null_primary_key` | The declared `primary_key` holds no null. |
| `unique_primary_key` | The declared `primary_key` holds no repeated value. |
| `datetime_row_creation_time` | The declared `row_creation_time` is a `Datetime`, not a `Date`. |

Relationship checks run against each relationship, and database checks against
the whole database. Only `validate()` runs those — `add_table` sees one table
and raises `ValueError` if you name one:

| Name | Scope | Confirms |
| --- | --- | --- |
| `matching_key_dtypes` | relationship | The `foreign_key` dtype matches the parent's `primary_key` exactly. |
| `overlapping_keys` | relationship | The `foreign_key` matches at least one of the parent's keys. Nulls are ignored. |
| `consistent_time_zones` | database | Every `Datetime` column is tz-aware, or every one is naive. Zones may differ; mixing awareness does not. |

With `validate=True` checks run in the order above, so `non_null_primary_key`
reports a null key before `unique_primary_key` reports it as a duplicate. An
explicit list runs in the order you write it.

A table with no `primary_key` is skipped by the key checks rather than failed —
you were already warned about that at `add_table` time. A table with no
`row_creation_time` is skipped by `datetime_row_creation_time`.

`matching_key_dtypes` demands an exact match rather than anything looser,
because the backends disagree about what they will join and the strictest one
decides: pyarrow refuses `Int64` against `Int32`, polars refuses `Int64`
against `Float64`, and polars refuses every crossing of `String`,
`Categorical` and `Enum` — including two `Enum`s whose categories differ. A
looser rule would pass validation and then fail the join. Cast the column
instead.

`datetime_row_creation_time`, `matching_key_dtypes` and
`consistent_time_zones` read the declared dtypes, not the rows, so they cost
no query. `overlapping_keys` is the only one that reads rows on a
relationship, and it stops at the first match rather than scanning the child.

`overlapping_keys` asks only whether the two key columns meet at all, not
whether every foreign key resolves. Orphan rows are ordinary in real data —
a warehouse that deletes parents, an extract that cuts a table short — and
failing on them would make the check unusable. Zero overlap is the defect it
looks for: the wrong column, or two id spaces that never met.

In `unique_primary_key`, nulls count as one distinct value, so **repeated
nulls fail** while **a single null passes**. A lone null key is a real defect,
but `non_null_primary_key` is the check that reports it.

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
