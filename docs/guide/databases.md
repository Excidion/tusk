# Databases

A [`Database`][tusk.Database] holds the tables you want features over and
the relationships between them. It is pure schema: adding a table reads its
column names and dtypes, nothing else.

```python
import tusk

db = tusk.Database("retail")
db.add_table("customers", customers_lf, primary_key="id", row_creation_time="signed_up_at")
db.add_table("sessions", sessions_lf, primary_key="id", row_creation_time="started_at")
db.add_relationship(parent="customers", child="sessions", foreign_key="customer_id")
```

Both `add_table` and `add_relationship` return the database, so they
chain.

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
