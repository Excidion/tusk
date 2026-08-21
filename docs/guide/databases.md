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

The first table also fixes eagerness. If you hand tusk eager frames,
[`is_eager`][tusk.Database.is_eager] is true and the feature matrix is
collected once at the end. If you hand it lazy frames, you get a lazy frame
back and nothing is computed until you collect it yourself.

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

It runs every check against every table in insertion order and raises
[`ValidationError`][tusk.exceptions.ValidationError] on the first defect. It
returns the database, so it chains.

The error names the table, the column and the counts that disagree — for
example `4 rows, 3 distinct values` — but not which keys are duplicated.
Finding them costs a second full scan, which is a real bill on a remote
backend, so tusk leaves that query to you if you want it.

Checks are selected by name. `True` runs all of them, `False` none, and a
string or list runs those:

```python
db.validate("unique_primary_key")
db.add_table("customers", customers_lf, primary_key="id", validate=True)
```

`add_table` takes the same selector, and defaults to `False`. That default is
deliberate: without it every `add_table` would materialize a full scan, which
is a silent performance cliff on `pl.scan_parquet` or a remote table. A
misspelled check name raises `ValueError`, not `ValidationError`, so catching
validation failures never swallows a typo.

### Available checks

| Name | Confirms |
| --- | --- |
| `unique_primary_key` | The declared `primary_key` holds no repeated value. |

A table with no `primary_key` is skipped by `unique_primary_key` rather than
failed — you were already warned about that at `add_table` time.

Nulls count as one distinct value, so **repeated nulls fail** this check while
**a single null passes** it. A lone null key is a real defect, but a
nullability one rather than a uniqueness one.
