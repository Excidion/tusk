# Table validation: opt-in data checks over a database

Status: draft — pending review
Date: 2026-08-21

## Problem

A `Database` is pure schema. `add_table` reads column names and dtypes from
the frame and takes every declaration on trust: naming a column as
`primary_key` asserts that it identifies a row, and nothing ever confirms it.

When the assertion is false, tusk does not fail — it produces wrong numbers.
A duplicated primary key fans out every join that lands on that table, so a
parent row is counted once per duplicate and `COUNT`, `SUM` and `MEAN` are
inflated by a factor the user cannot see. The feature matrix comes back with
more rows than the target table has, keyed by a column that does not key it.
The symptom surfaces far from the cause, in aggregate values, and reads as a
bug in tusk rather than a defect in the input.

Cheap frames make this worse, not better. `pl.scan_parquet` and a duckdb
relation are exactly the inputs a user has not eyeballed, and exactly the ones
where a stale extract or a botched upstream join is most likely.

What is missing is a way to spend real computation, deliberately, to confirm
the declarations hold before they are relied on.

## Decision

Add opt-in **validation**: checks that run actual queries against the data,
selected by name, reachable from two entry points that share one selector
vocabulary.

Three properties drive the design:

- **Off by default on `add_table`.** Today `add_table` is O(schema); it never
  touches a row. Validating by default would make every call materialize a
  full scan — a silent performance cliff on remote or large tables, and a
  regression for every existing caller. Validation is a thing the user asks
  for.
- **Checks are a registry, not a signature.** Names map to functions in one
  dict. Adding a check is a new function plus a new entry; no call signature
  anywhere changes, and both entry points pick it up for free.
- **One selector type in both places.** `add_table(validate=…)` and
  `Database.validate(…)` accept the same values and mean the same thing by
  them.

### Public surface

| Addition | Meaning |
| --- | --- |
| `src/tusk/validation.py` | New module. Checks, registry, selector resolution. |
| `validation.CHECKS` | `Mapping[str, Callable[[nw.LazyFrame, TableSchema], None]]`, the registry. |
| `validation.check_unique_primary_key(frame, schema)` | The first check. |
| `validation.validate_table(frame, schema, checks=True)` | Runs the selected checks against one table. |
| `Database.validate(checks=True) -> Database` | Runs the selected checks against every table, in insertion order. |
| `Database.add_table(…, validate=False)` | Runs the selected checks against the table being added. |
| `exceptions.ValidationError(TuskError)` | A check found a violation in the data. |

Nothing is added to `tusk.__all__`. `ValidationError` is reachable as
`tusk.exceptions.ValidationError`, matching every other exception; the
`validation` module is an implementation surface that `Database` fronts.

### The selector type

```python
Checks = bool | str | Iterable[str]
```

| Value | Selects |
| --- | --- |
| `True` | Every check in the registry. |
| `False` | None. |
| `"unique_primary_key"` | That one check. |
| `["unique_primary_key", …]` | Those checks. |
| `[]` | None, identically to `False`. |

Resolution is centralized in one function so both entry points cannot drift:

```python
def resolve_checks(selector: Checks) -> tuple[Check, ...]
```

Two decisions inside it:

- **Registry order, not argument order.** A selected list is deduplicated and
  run in the order `CHECKS` declares, so behaviour never depends on how the
  caller happened to type the list. With checks that raise on first failure,
  argument order would otherwise decide *which* error a user sees for data
  that violates two checks at once.
- **An unknown name raises `ValueError`**, listing the registered names:

  ```
  ValueError: unknown check 'uniqe_primary_key'; available checks: 'unique_primary_key'
  ```

  A typo is a caller bug, not a data defect. Keeping it out of
  `ValidationError` means `except ValidationError` catches only genuine data
  failures and never silently swallows a misspelled check name. This mirrors
  the split the codebase already draws: `SchemaError` for a malformed
  database, `PrimitiveError` for an unknown primitive name.

`Database.validate(False)` is legal under the shared type and is a silent
no-op. Special-casing it would mean two selector vocabularies instead of one,
which is the thing this design exists to avoid.

### The first check: `unique_primary_key`

Confirms the declared `primary_key` column holds no repeated value.

A table whose `primary_key` is `None` is **skipped, not failed**. `add_table`
already emits `MissingPrimaryKeyWarning` for that case and documents the
reduced capabilities that follow; re-reporting it as a validation failure
would make `db.validate()` unusable on any database that deliberately holds a
keyless table.

The check runs in two phases so the common case stays cheap:

1. **Detect** with one aggregate, no grouping materialized:

   ```python
   frame.select(total=nw.len(), distinct=nw.col(pk).n_unique())
   ```

   Equal counts pass, and that is the entire cost of a successful check.

2. **Explain**, only on failure, with a second query that names offenders:

   ```python
   frame.group_by(pk).agg(nw.len().alias(...)).filter(... > 1).sort(pk).head(5)
   ```

Paying for the sample only when there is something to report keeps the happy
path — the one that runs on every table of a healthy database — to a single
scan.

The message carries the table, the column, both counts, and the sample:

```
ValidationError: primary_key 'id' of 'customers' is not unique:
4 rows, 3 distinct values, e.g. [7, 12, 19]
```

Both queries were confirmed to work through narwhals on polars lazy frames
and on a duckdb relation, and `collect()[col].to_list()` extracts the sample
identically on both.

#### Null semantics

`n_unique` counts null as one distinct value. Two consequences, both
deliberate and both documented:

- **Repeated nulls fail.** Two null keys collapse to one distinct value, so
  the counts disagree and the check reports them. Correct: a column with two
  null keys does not identify rows.
- **A single null passes.** One null is one distinct value among distinct
  others. A lone null primary key is a real defect, but it is a *nullability*
  defect, not a *uniqueness* one. It belongs to a `non_null_primary_key` check
  in a later round, not folded into this one where it would be invisible in
  the name and unskippable.

### Placement inside `add_table`

`validate` is **keyword-only** — it goes behind a `*` marker, after
`row_creation_time`. The four existing parameters keep their positions and
their positional callability; nothing that grows the signature later becomes
reachable by position.

Validation runs after the existing schema checks and the
`MissingPrimaryKeyWarning`, and **before** `self._frames` and `self._schemas`
are assigned.

This ordering is the point, not an accident: a table that fails validation
must not be left sitting in the database. A caller who catches
`ValidationError` and continues gets a `Database` that is unchanged rather
than one silently holding the table that just failed. `table_names` is the
observable that pins this.

The schema checks must still come first — a `primary_key` that is not a
column of the frame is a `SchemaError`, and running a query against a column
that does not exist would report that as something else.

### Documentation

- **`docs/guide/databases.md`** gains a "Validation" section: what a check
  costs, why it is off by default, the selector forms, and the null semantics
  of the first check. The page's opening claim about pure schema is amended in
  the same pass (hazard 5).
- **`docs/api/validation.md`** is a new mkdocstrings page for the module,
  added to `nav` in `zensical.toml` after `api/database.md`.
- **`Database.validate` and the `validate` parameter** carry their own
  docstrings, which is what `api/database.md` renders.

## Out of scope

Deliberately deferred, to be designed when the second check is specified:

- **Further checks.** `non_null_primary_key`, foreign-key referential
  integrity, `row_creation_time` nullability and monotonicity. The registry
  exists so each is additive; none is specified here.
- **Collect-all reporting.** Every check raises on first violation. A report
  object listing every violation across every table is plausibly useful for
  `Database.validate()` over a large database, but it needs a return type
  designed, and it forces `add_table(validate=True)` to decide what to do with
  a non-empty report. Fail-fast matches how `add_table` already reports
  problems.
- **Caching validation results.** A validated database does not remember that
  it was validated; calling `validate()` twice runs the queries twice.
- **Validation inside `deep_feature_synthesis`.** The entry points stay
  computation-only; validation is the user's explicit call.

## Hazards

1. **`interrogate` is set to `fail-under = 100` and `pydoclint` runs in
   pre-commit.** Every new function needs a complete Google-style docstring
   with `Args`, `Returns` and `Raises` sections that match the signature.
   The registry's values are functions with docstrings of their own.
2. **The `validation` dependency group in `pyproject.toml` is unrelated.** It
   holds featuretools and pandas for the differential tests. The new module is
   `src/tusk/validation.py`; nothing about that dependency group changes, and
   the collision is prose-level only.
3. **`n_unique` treats null as a value.** The null semantics above are a
   consequence of that, not a separate implementation decision. A future
   `non_null_primary_key` check must not "fix" it here.
4. **`collect()` returns different native types per backend** — a polars
   `DataFrame` on polars, a pyarrow `Table` on duckdb. The sample must be read
   through the narwhals `DataFrame` (`df[col].to_list()`), never from the
   native object.
5. **`docs/guide/databases.md` currently states a database "is pure schema:
   adding a table reads its column names and dtypes, nothing else."** This
   change makes that conditionally untrue and the sentence must be amended,
   not merely appended to.
6. **`add_table` is called in roughly twenty places** across `tests/`,
   `benchmarks/`, `README.md` and `docs/`. A keyword-only parameter defaulting
   to `False` leaves every one of them working unchanged; that is what makes
   the default the safe one.

## Verification

1. `uv run pytest` green, including `tests/test_backend_duckdb.py`.
2. New `tests/test_validation.py` covers, at minimum:
   - a duplicated primary key raises `ValidationError` naming the table, the
     column and at least one offending value;
   - a unique primary key passes;
   - `add_table` with the default `validate=False` does not raise on a frame
     with a duplicated key, and does not scan it;
   - `validate=True`, `validate="unique_primary_key"` and
     `validate=["unique_primary_key"]` all raise on that same frame;
   - a failed `add_table` leaves `table_names` unchanged;
   - a table with no `primary_key` is skipped by the check rather than failed;
   - `db.validate()` finds an offender in a non-first table and returns `self`
     when everything passes;
   - an unknown check name raises `ValueError` listing the available names,
     from both entry points;
   - repeated nulls in the primary key fail; a single null passes.
3. The uniqueness check is exercised against the duckdb backend, not polars
   alone — it is the first code in tusk to run an aggregate purely for a
   side effect, and SQL generation is where that is most likely to differ.
4. `uv run --group docs zensical build` succeeds with no broken mkdocstrings
   identifiers, with `api/validation.md` added to `nav` in `zensical.toml`.
5. `uv run pre-commit run --all-files` green, covering `ruff`, `ty`,
   `interrogate` and `pydoclint`.
