# Database.as_of: a database that carries its cutoff time

Design for `Database.as_of(cutoff_time)`, a method that returns a copy of a
database whose tables are seen at a cutoff time.

## Problem

The cutoff time is a parameter of the read path. `Database._get_table(name,
cutoff_time)` is the one place that filters rows by `row_creation_time` and
restores the columns in `row_update_times`, but about fifteen compiler
functions (`_read_table`, `_add_directs`, `_build_aggregation`, ...) carry
`cutoff_time` only to hand it to `_read_table`. `get_table` and `get_keys`
repeat the same parameter, and the sklearn target check does too.

A database that carries its cutoff time removes that threading and gives
callers one way to read a database at a time: `db.as_of(t)`.

## Decisions

1. **`as_of` always returns a copy.** The original database is never changed.
   A copy has its own `_tables`, `_schemas` and `_relationships` containers.
   The tables are lazy and immutable, so the copy costs nothing.
2. **A copy shares the raw tables and holds its own cutoff time.** Filtering
   stays lazy, inside `_get_table`. Calling `as_of` on a copy builds a new
   copy from the same raw tables with the new cutoff time. Filters do not
   stack.
3. **`as_of(None)` returns a copy with no cutoff time.** No `"now"` alias.
4. **`as_of` validates the cutoff time once**, with `check_cutoff_time`. A
   wrong type or a timezone mismatch fails at `as_of`, not at `collect()`.
5. **`apply_features` and `FeatureList.apply` keep `cutoff_time`.** They
   always call `database.as_of(cutoff_time)`, so the argument wins over the
   database's own cutoff time, and `None` gives a copy with no cutoff time.
6. **Overwriting warns.** If the database has a cutoff time that differs from
   the one passed, they issue `OverwrittenCutoffTimeWarning`. An equal cutoff
   time does not warn.
7. **`get_table` and `get_keys` lose `cutoff_time`.** `db.as_of(t).get_table(name)`
   replaces it. This breaks the public API.
8. **`validate` reads the tables as they stand at the cutoff time.** On a copy
   with a cutoff time it checks the rows that copy shows. `validate` needs no
   change.

## Architecture

### Database

```python
def as_of(self, cutoff_time: datetime | None) -> Database: ...

@property
def cutoff_time(self) -> datetime | None: ...
```

- `as_of` calls `check_cutoff_time` when `cutoff_time` is not None, then
  returns a copy with `_cutoff_time` set.
- `_get_table(name)` reads `self._cutoff_time`. Without one it returns the raw
  table. Otherwise it applies the row filter and `_restore_updated_columns`,
  as today.

### Compiler

`compile_features(features, database, cutoff_time)` builds
`database = database.as_of(cutoff_time)` once at its top. `_read_table` and
every function that only passed `cutoff_time` through lose the parameter.

The cutoff time stays a value where something is measured against it:
`_require_cutoff_time`, the `NeedsCutoffTime` primitives and the `when`
callables. They read `database.cutoff_time` instead of a parameter.

### Warning

```python
class OverwrittenCutoffTimeWarning(UserWarning):
    """Warns that a cutoff time replaced the cutoff time of a database."""
```

It has its own class so it can be filtered independently, like
`UnmatchedPrimitiveWarning` and `MissingPrimaryKeyWarning`.

### sklearn

The target check in `sklearn/_tables.py` reads
`database.as_of(cutoff_time)` instead of passing `cutoff_time` to
`_get_table`.

## Documentation

- `docs/guide/databases.md`: an "As of a cutoff time" section.
- Cutoff docstrings in `api.py`, `feature_list.py`, `compiler.py` and
  `database.py`.

## Testing

- `as_of` filters rows and restores updated columns, with the same results as
  today's `get_table(name, cutoff_time)`.
- A second `as_of` replaces the first cutoff time.
- `as_of(None)` gives a copy with no cutoff time.
- The original database keeps its cutoff time.
- `as_of` fails on a timezone mismatch and on a non-datetime.
- `validate` on a copy with a cutoff time checks only the rows the copy shows.
- `apply_features` warns when the database has a different cutoff time, and
  stays silent when it has none or an equal one.
- The existing suite stays green.

## Out of scope

- A `"now"` alias.
- Per-row cutoff times. `cutoff_time` remains one value per database.
