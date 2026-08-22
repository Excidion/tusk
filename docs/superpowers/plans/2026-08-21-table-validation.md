# Table Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in data validation — checks that run real queries against a table to confirm its declarations hold — selected by name through one vocabulary shared by `Database.add_table(validate=…)` and `Database.validate(…)`, with `unique_primary_key` as the first check.

**Architecture:** A new `src/tusk/validation.py` owns everything: the check functions, a `CHECKS` registry mapping names to them, and `resolve_checks()` which turns a `bool | str | Iterable[str]` selector into the checks to run. `Database` gains a `validate()` method and an `add_table(validate=…)` parameter, both thin delegations to `validate_table()`. Three tasks: the module standalone, the `Database` wiring, then backend portability and the guide.

**Tech Stack:** Python 3.10+, narwhals, polars/pyarrow/duckdb backends, pytest, uv, ruff + ty + interrogate + pydoclint via pre-commit, zensical for docs.

**Spec:** `docs/superpowers/specs/2026-08-21-table-validation-design.md`

## Global Constraints

- **Standard test command:** `uv run --group validation pytest -q`. The validation group must be present or `tests/differential/` fails to *collect* (it imports pandas/featuretools) even though the marker deselects it. Baseline before any work: `152 passed, 42 deselected`. Note: that dependency group is named `validation` for the featuretools differential tests and has **nothing** to do with the feature being built here. Do not touch `pyproject.toml`.
- **`validate` defaults to `False` on `add_table`, and is keyword-only.** Every existing call site — roughly twenty across `tests/`, `benchmarks/`, `README.md` and `docs/` — must keep working untouched. If a task makes you edit an existing `add_table` call, you have made a mistake.
- **`Database.validate(checks=True)` defaults to running every check.**
- **The import direction is `database` → `validation`, never the reverse at runtime.** `validation.py` needs `TableSchema` only as a type annotation, so it imports it under `if TYPE_CHECKING:`. `database.py` imports `validate_table` and `Checks` normally. Getting this backwards produces a circular import at `import tusk`.
- **Checks raise on first violation.** No collect-all report object. `ValidationError` for a data defect; plain `ValueError` for an unknown check name.
- **A table whose `primary_key` is `None` is skipped, never failed.**
- **Pre-commit gates every commit** with ruff, ty, interrogate (`fail-under = 100`) and pydoclint. Every new function needs a complete Google-style docstring whose `Args:` names match the signature, plus `Returns:` and `Raises:` where they apply. A missing docstring section fails at commit time, not at test time.
- **Out of scope, do not build:** any second check (`non_null_primary_key`, foreign-key integrity), a collect-all report type, caching of validation results, and calling validation from `deep_feature_synthesis` or `apply_features`.

---

### Task 1: The validation module

**Files:**
- Create: `src/tusk/validation.py`
- Modify: `src/tusk/exceptions.py`
- Test: `tests/test_validation.py` (create)
- Create: `docs/api/validation.md`
- Modify: `zensical.toml`

**Interfaces:**
- Consumes: `tusk.database.TableSchema` (fields `name`, `primary_key`, `row_creation_time`, `dtypes`), `tusk.exceptions.TuskError`.
- Produces:
  - `tusk.exceptions.ValidationError(TuskError)`
  - `tusk.validation.Checks = bool | str | Iterable[str]`
  - `tusk.validation.Check = Callable[[nw.LazyFrame, TableSchema], None]`
  - `tusk.validation.CHECKS: Mapping[str, Check]`, currently `{"unique_primary_key": check_unique_primary_key}`
  - `tusk.validation.check_unique_primary_key(frame: nw.LazyFrame, schema: TableSchema) -> None`
  - `tusk.validation.resolve_checks(selector: Checks) -> tuple[Check, ...]`
  - `tusk.validation.validate_table(frame: nw.LazyFrame, schema: TableSchema, checks: Checks = True) -> None`

- [ ] **Step 1: Write the failing test**

Create `tests/test_validation.py`:

```python
import narwhals as nw
import polars as pl
import pytest

from tusk.database import TableSchema
from tusk.exceptions import TuskError, ValidationError
from tusk.validation import (
    CHECKS,
    check_unique_primary_key,
    resolve_checks,
    validate_table,
)


def frame(values):
    """A one-column lazy frame keyed by 'id'."""
    return nw.from_native(pl.LazyFrame({"id": values}))


def schema(primary_key="id"):
    return TableSchema("customers", primary_key, None, {"id": nw.Int64})


def test_duplicate_primary_key_is_reported():
    with pytest.raises(ValidationError) as excinfo:
        check_unique_primary_key(frame([7, 7, 12, 12, 19, 19, 1]), schema())
    message = str(excinfo.value)
    assert "primary_key 'id'" in message
    assert "'customers'" in message
    assert "7 rows" in message
    assert "4 distinct" in message
    assert "7" in message and "12" in message and "19" in message


def test_unique_primary_key_passes():
    check_unique_primary_key(frame([1, 2, 3]), schema())


def test_empty_table_passes():
    check_unique_primary_key(frame([]), schema())


def test_table_without_a_primary_key_is_skipped():
    # add_table already warned about this; failing here would make
    # db.validate() unusable on any database holding a keyless table.
    check_unique_primary_key(frame([7, 7]), schema(primary_key=None))


def test_repeated_nulls_are_not_unique():
    with pytest.raises(ValidationError, match="not unique"):
        check_unique_primary_key(frame([1, None, None]), schema())


def test_a_single_null_passes_the_uniqueness_check():
    # A lone null key is a nullability defect, not a uniqueness one.
    check_unique_primary_key(frame([1, 2, None]), schema())


def test_validation_error_is_a_tusk_error():
    assert issubclass(ValidationError, TuskError)


def test_validate_table_runs_the_registry_by_default():
    with pytest.raises(ValidationError):
        validate_table(frame([7, 7]), schema())


def test_validate_table_runs_nothing_when_false():
    validate_table(frame([7, 7]), schema(), False)


def test_selector_forms_all_select_the_check():
    for selector in (True, "unique_primary_key", ["unique_primary_key"]):
        assert resolve_checks(selector) == (CHECKS["unique_primary_key"],)


def test_empty_selectors_select_nothing():
    assert resolve_checks(False) == ()
    assert resolve_checks([]) == ()


def test_a_repeated_name_runs_the_check_once():
    assert resolve_checks(["unique_primary_key", "unique_primary_key"]) == (
        CHECKS["unique_primary_key"],
    )


def test_checks_run_in_registry_order_not_argument_order(monkeypatch):
    calls = []
    ordered = {
        "first": lambda f, s: calls.append("first"),
        "second": lambda f, s: calls.append("second"),
    }
    monkeypatch.setattr("tusk.validation.CHECKS", ordered)
    validate_table(frame([1]), schema(), ["second", "first"])
    assert calls == ["first", "second"]


def test_an_unknown_check_name_raises_value_error():
    with pytest.raises(ValueError) as excinfo:
        resolve_checks("uniqe_primary_key")
    message = str(excinfo.value)
    assert "uniqe_primary_key" in message
    assert "unique_primary_key" in message


def test_an_unknown_check_name_is_not_a_validation_error():
    # `except ValidationError` must never swallow a typo.
    with pytest.raises(ValueError) as excinfo:
        resolve_checks(["unique_primary_key", "nope"])
    assert not isinstance(excinfo.value, TuskError)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --group validation pytest tests/test_validation.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'tusk.validation'`

- [ ] **Step 3: Add the exception**

Append to `src/tusk/exceptions.py`:

```python
class ValidationError(TuskError):
    """Raised when a validation check finds a defect in a table's data.

    Distinct from :class:`SchemaError`, which reports a malformed database
    before any row is read. A ``ValidationError`` means the declarations are
    well-formed but the data contradicts them.

    An unknown check *name* is a caller mistake rather than a data defect and
    raises :class:`ValueError`, so ``except ValidationError`` never silently
    swallows a typo.
    """
```

- [ ] **Step 4: Write the module**

Create `src/tusk/validation.py`:

```python
"""Checks that confirm a table's declarations hold against its data.

The rest of the schema layer takes declarations on trust: naming a column as
``primary_key`` asserts that it identifies a row, and nothing confirms it.
These checks spend real queries to find out. Nothing here runs unless the user
asks, through :meth:`tusk.Database.validate` or ``add_table(validate=…)``.

Adding a check is a new function plus a new entry in :data:`CHECKS`; no call
signature changes and both entry points pick it up.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING

import narwhals as nw

from tusk.exceptions import ValidationError

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from tusk.database import TableSchema

Check = Callable[["nw.LazyFrame", "TableSchema"], None]
"""A check: given a table's frame and schema, raise or return."""

Checks = bool | str | Iterable[str]
"""Selects checks: every check, none, one by name, or several by name."""

_SAMPLE_SIZE = 5
_COUNT = "__tusk_count"


def check_unique_primary_key(frame: nw.LazyFrame, schema: TableSchema) -> None:
    """Confirm the declared primary key holds no repeated value.

    Runs in two phases so the passing case stays cheap: one aggregate compares
    the row count to the distinct-value count, and only a disagreement pays for
    a second query naming offending values.

    A table with no ``primary_key`` is skipped rather than failed --
    ``add_table`` already warned about it, and failing here would make
    :meth:`tusk.Database.validate` unusable on any database that deliberately
    holds a keyless table.

    Nulls count as one distinct value, so repeated nulls fail this check while
    a single null passes it. A lone null key is a nullability defect, not a
    uniqueness one.

    Args:
        frame: The table's lazy frame.
        schema: The table's schema, naming the column to check.

    Raises:
        ValidationError: If the primary key column holds a repeated value.
    """
    key = schema.primary_key
    if key is None:
        return

    counts = frame.select(total=nw.len(), distinct=nw.col(key).n_unique()).collect()
    total, distinct = counts["total"].item(), counts["distinct"].item()
    if total == distinct:
        return

    duplicates = (
        frame.group_by(key)
        .agg(nw.len().alias(_COUNT))
        .filter(nw.col(_COUNT) > 1)
        .sort(key)
        .head(_SAMPLE_SIZE)
        .collect()
    )
    raise ValidationError(
        f"primary_key {key!r} of {schema.name!r} is not unique: "
        f"{total} rows, {distinct} distinct values, "
        f"e.g. {duplicates[key].to_list()}"
    )


CHECKS: Mapping[str, Check] = {
    "unique_primary_key": check_unique_primary_key,
}
"""Every available check, by name. Iteration order is the order checks run."""


def resolve_checks(selector: Checks) -> tuple[Check, ...]:
    """Turn a selector into the checks it names.

    Selected checks are deduplicated and returned in :data:`CHECKS` order, not
    argument order, so which failure a caller sees never depends on how they
    happened to type the list.

    Args:
        selector: ``True`` for every check, ``False`` (or an empty iterable)
            for none, a string for one by name, or an iterable of names.

    Returns:
        The selected checks, in registry order.

    Raises:
        ValueError: If a name is not in :data:`CHECKS`.
    """
    if isinstance(selector, bool):
        return tuple(CHECKS.values()) if selector else ()

    names = {selector} if isinstance(selector, str) else set(selector)
    unknown = sorted(names - set(CHECKS))
    if unknown:
        listed = ", ".join(repr(name) for name in unknown)
        available = ", ".join(repr(name) for name in CHECKS)
        raise ValueError(f"unknown check {listed}; available checks: {available}")
    return tuple(check for name, check in CHECKS.items() if name in names)


def validate_table(
    frame: nw.LazyFrame, schema: TableSchema, checks: Checks = True
) -> None:
    """Run the selected checks against one table.

    Args:
        frame: The table's lazy frame.
        schema: The table's schema.
        checks: Which checks to run. See :func:`resolve_checks`.

    Raises:
        ValidationError: If a check finds a defect. Raised by the first check
            that fails; later checks do not run.
        ValueError: If ``checks`` names a check that does not exist.
    """
    for check in resolve_checks(checks):
        check(frame, schema)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --group validation pytest tests/test_validation.py -q`
Expected: PASS, 15 tests.

Then the full suite: `uv run --group validation pytest -q`
Expected: `167 passed, 42 deselected`.

- [ ] **Step 6: Add the API docs page**

Create `docs/api/validation.md`:

```markdown
# Validation

::: tusk.validation
```

In `zensical.toml`, add `"api/validation.md",` to the `API reference` nav list, immediately after `"api/database.md",`.

- [ ] **Step 7: Verify the docs build**

Run: `uv run --group docs zensical build`
Expected: succeeds with no mkdocstrings warnings about unresolved identifiers.

- [ ] **Step 8: Commit**

```bash
git add src/tusk/validation.py src/tusk/exceptions.py tests/test_validation.py docs/api/validation.md zensical.toml
git commit -m "feat: add validation checks with a name-selected registry"
```

---

### Task 2: Wire validation into `Database`

**Files:**
- Modify: `src/tusk/database.py`
- Test: `tests/test_validation.py` (append)

**Interfaces:**
- Consumes: `tusk.validation.validate_table`, `tusk.validation.Checks` from Task 1.
- Produces:
  - `Database.add_table(name, table, primary_key=None, row_creation_time=None, *, validate: Checks = False) -> Database`
  - `Database.validate(checks: Checks = True) -> Database`

- [ ] **Step 1: Write the failing test**

Add `import tusk` to the import block at the **top** of
`tests/test_validation.py` (ruff enforces E402 and import sorting, so it cannot
go mid-file), then append:

```python
@pytest.fixture
def spy(monkeypatch):
    """Replace the registry with a recorder, so plumbing is observable."""
    calls = []
    monkeypatch.setattr(
        "tusk.validation.CHECKS",
        {"unique_primary_key": lambda f, s: calls.append(s.name)},
    )
    return calls


def dupes():
    return pl.LazyFrame({"id": [7, 7, 12], "v": [1.0, 2.0, 3.0]})


def test_add_table_does_not_validate_by_default(spy):
    db = tusk.Database("x").add_table("t", dupes(), primary_key="id")
    assert spy == []
    assert db.table_names == ("t",)


def test_add_table_validates_when_asked(spy):
    tusk.Database("x").add_table("t", dupes(), primary_key="id", validate=True)
    assert spy == ["t"]


def test_add_table_accepts_every_selector_form():
    for selector in (True, "unique_primary_key", ["unique_primary_key"]):
        with pytest.raises(ValidationError, match="not unique"):
            tusk.Database("x").add_table(
                "t", dupes(), primary_key="id", validate=selector
            )


def test_a_failed_add_table_leaves_the_database_unchanged():
    db = tusk.Database("x")
    with pytest.raises(ValidationError):
        db.add_table("t", dupes(), primary_key="id", validate=True)
    assert db.table_names == ()
    with pytest.raises(tusk.exceptions.SchemaError):
        db.frame("t")


def test_schema_errors_still_precede_validation():
    # A primary_key that is not a column must stay a SchemaError; querying a
    # column that does not exist would report it as something else.
    with pytest.raises(tusk.exceptions.SchemaError, match="nope"):
        tusk.Database("x").add_table("t", dupes(), primary_key="nope", validate=True)


def test_database_validate_runs_every_table_in_insertion_order(spy):
    db = (
        tusk.Database("x")
        .add_table("a", pl.LazyFrame({"id": [1]}), primary_key="id")
        .add_table("b", pl.LazyFrame({"id": [2]}), primary_key="id")
    )
    assert db.validate() is db
    assert spy == ["a", "b"]


def test_database_validate_reports_the_offending_table():
    db = (
        tusk.Database("x")
        .add_table("a", pl.LazyFrame({"id": [1, 2]}), primary_key="id")
        .add_table("b", dupes(), primary_key="id")
    )
    with pytest.raises(ValidationError, match="'b'"):
        db.validate()


def test_database_validate_false_is_a_no_op(spy):
    db = tusk.Database("x").add_table("t", dupes(), primary_key="id")
    assert db.validate(False) is db
    assert spy == []


def test_database_validate_skips_a_keyless_table():
    with pytest.warns(tusk.exceptions.MissingPrimaryKeyWarning):
        db = tusk.Database("x").add_table("t", dupes())
    assert db.validate() is db


def test_unknown_check_names_reach_both_entry_points():
    with pytest.raises(ValueError, match="unknown check"):
        tusk.Database("x").add_table(
            "t", pl.LazyFrame({"id": [1]}), primary_key="id", validate="nope"
        )
    db = tusk.Database("x").add_table("t", pl.LazyFrame({"id": [1]}), primary_key="id")
    with pytest.raises(ValueError, match="unknown check"):
        db.validate("nope")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --group validation pytest tests/test_validation.py -q`
Expected: FAIL with `TypeError: add_table() got an unexpected keyword argument 'validate'`

- [ ] **Step 3: Import the module in `src/tusk/database.py`**

Add below the existing `from tusk.exceptions import …` line:

```python
from tusk.validation import Checks, validate_table
```

`validation.py` imports `TableSchema` only under `TYPE_CHECKING`, so this direction does not cycle. Do not add an import of `tusk.database` to `validation.py`.

- [ ] **Step 4: Add the parameter to `add_table`**

Change the signature to:

```python
    def add_table(
        self,
        name: str,
        table: Any,
        primary_key: str | None = None,
        row_creation_time: str | None = None,
        *,
        validate: Checks = False,
    ) -> Database:
```

Add to its docstring, after the `row_creation_time` entry in `Args:`:

```
            validate: Which validation checks to run against the data before
                registering the table. ``False`` (the default) runs none and
                reads no rows; ``True`` runs every check; a name or list of
                names runs those. See :func:`tusk.validation.validate_table`.
```

And add to its `Raises:` section:

```
            ValidationError: If a requested check finds a defect in the data.
            ValueError: If ``validate`` names a check that does not exist.
```

- [ ] **Step 5: Validate before mutating**

Replace the final three lines of `add_table`:

```python
        self._frames[name] = lazy
        self._schemas[name] = TableSchema(name, primary_key, row_creation_time, dtypes)
        return self
```

with:

```python
        schema = TableSchema(name, primary_key, row_creation_time, dtypes)
        validate_table(lazy, schema, validate)

        self._frames[name] = lazy
        self._schemas[name] = schema
        return self
```

The order is the point: a table that fails validation must not be left in the database. Do not move this above the existing `SchemaError` checks — a `primary_key` that is not a column must stay a `SchemaError`.

- [ ] **Step 6: Add `Database.validate`**

Insert immediately after `add_relationship`:

```python
    def validate(self, checks: Checks = True) -> Database:
        """Run validation checks against every table in the database.

        Tables are checked in insertion order and the first failure raises, so
        a database with several defects reports the earliest one.

        Unlike the rest of the schema layer, this reads rows: each check runs
        real queries against the data.

        Args:
            checks: Which checks to run. ``True`` (the default) runs every
                check; ``False`` runs none; a name or list of names runs
                those. See :func:`tusk.validation.validate_table`.

        Returns:
            This database, to allow chaining.

        Raises:
            ValidationError: If a check finds a defect in a table's data.
            ValueError: If ``checks`` names a check that does not exist.
        """
        for name, schema in self._schemas.items():
            validate_table(self._frames[name], schema, checks)
        return self
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run --group validation pytest tests/test_validation.py -q`
Expected: PASS, 25 tests.

Then the full suite: `uv run --group validation pytest -q`
Expected: `177 passed, 42 deselected`. No existing test changes — if one broke, the default is not `False` or the parameter is not keyword-only.

- [ ] **Step 8: Commit**

```bash
git add src/tusk/database.py tests/test_validation.py
git commit -m "feat: validate tables from add_table and Database.validate"
```

---

### Task 3: Backend portability and the guide

**Files:**
- Modify: `tests/test_backend_duckdb.py`
- Modify: `docs/guide/databases.md`

**Interfaces:**
- Consumes: everything from Tasks 1 and 2. Produces no new API.

- [ ] **Step 1: Write the failing test**

The uniqueness check is the first code in tusk to run an aggregate purely for a side effect, and a SQL-generating backend is where that is most likely to differ. Append to `tests/test_backend_duckdb.py`:

```python
def test_uniqueness_check_runs_on_duckdb(duck_db):
    db, con = duck_db
    assert db.validate() is db

    con.execute("CREATE TABLE dupes AS SELECT * FROM (VALUES (1),(1),(2)) t(id)")
    with pytest.raises(tusk.exceptions.ValidationError, match="not unique"):
        db.add_table("dupes", con.table("dupes"), primary_key="id", validate=True)
    assert "dupes" not in db.table_names
```

Check the `duck_db` fixture's return before writing this — it yields a
`(Database, connection)` tuple. Unpack it exactly as the existing tests in that
file do.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --group validation pytest tests/test_backend_duckdb.py -q`
Expected: FAIL — the test is new, so it fails only if the check is broken on duckdb. If it passes immediately, that is the correct outcome for a portability test; record that and move on rather than manufacturing a failure.

- [ ] **Step 3: Fix any backend divergence**

If the test fails, the likely cause is the `_COUNT` alias colliding with a
generated SQL identifier, or `n_unique` lowering differently. Both queries were
confirmed working on duckdb through narwhals during design, so investigate
before changing the check — do not weaken the check to make a backend pass.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --group validation pytest tests/test_backend_duckdb.py -q`
Expected: PASS.

- [ ] **Step 5: Amend the guide's opening claim**

`docs/guide/databases.md` currently says a database "is pure schema: adding a
table reads its column names and dtypes, nothing else." That is now only true
by default. Replace that sentence with:

```markdown
By default it is pure schema: adding a table reads its column names and
dtypes, nothing else. No row is read unless you ask for
[validation](#validation).
```

- [ ] **Step 6: Document validation in the guide**

Append to the end of `docs/guide/databases.md`:

````markdown
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
````

- [ ] **Step 7: Verify everything**

Run each and confirm the expected result before committing:

```bash
uv run --group validation pytest -q                 # 178 passed, 42 deselected
uv run --group validation pytest -q -m differential # unchanged from baseline
uv run --group docs zensical build                  # no unresolved identifiers
uv run pre-commit run --all-files                   # ruff, ty, interrogate, pydoclint green
```

- [ ] **Step 8: Commit**

```bash
git add tests/test_backend_duckdb.py docs/guide/databases.md
git commit -m "test: cover validation on duckdb and document it"
```
