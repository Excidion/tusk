# Database.as_of Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `Database.as_of(cutoff_time)`, a copy of a database that carries its cutoff time, and use it to remove the `cutoff_time` parameter threading from the compiler and the read methods.

**Architecture:** A database copy shares the raw lazy tables and holds its own `_cutoff_time`. `_get_table(name)` filters lazily from it. `compile_features` builds `database.as_of(cutoff_time)` once and the compiler reads `database.cutoff_time` wherever a value is needed. `apply_features` and `FeatureList.apply` keep `cutoff_time` and warn when it replaces a different cutoff time.

**Tech Stack:** Python, narwhals, polars (tests), pytest.

**Spec:** `agents/superpowers/specs/2026-10-17-database-as-of-design.md`

## Global Constraints

- Work on a new branch `database-as-of`, not on `main`.
- `as_of` always returns a copy. The original database is never changed.
- Calling `as_of` on a copy builds a new copy from the same raw tables. Filters do not stack.
- `as_of(None)` returns a copy with no cutoff time. There is no `"now"` alias.
- `as_of` validates the cutoff time once, with `check_cutoff_time`.
- `get_table` and `get_keys` lose `cutoff_time`. This breaks the public API.
- `apply_features` and `FeatureList.apply` always use the `cutoff_time` they are given, `None` included. They warn with `OverwrittenCutoffTimeWarning` when the database has a different, non-null cutoff time. An equal cutoff time does not warn.
- `validate` keeps calling `_get_table`, so on a copy with a cutoff time it checks the rows that copy shows.
- Docs, docstrings and identifiers follow `agents/GLOSSARY.md` (table, null, build, compute, cutoff time). Docstrings say what a thing does, not why. Exception and warning class docstrings may say why.
- Code style follows `agents/style/CODESTYLE.md`: no comments that explain what, descriptive names, small functions.
- Run tests with `uv run pytest`. Run lint with `just lint`.

---

### Task 1: Database carries a cutoff time

**Files:**
- Modify: `src/tusk/database.py` (`__init__` ~100-107, `get_table`/`_get_table` ~379-430, `get_keys` ~436-470, new `as_of`, `cutoff_time`)
- Modify: `src/tusk/sklearn/_tables.py:80-85`
- Modify: `src/tusk/compiler.py:70-72,198` (the `compile_features` body and `_read_table`)
- Test: `tests/test_database.py`

**Interfaces:**
- Produces: `Database.as_of(cutoff_time: datetime | None) -> Database`
- Produces: `Database.cutoff_time -> datetime | None` (read-only property)
- Produces: `Database._get_table(name: str) -> nw.LazyFrame` (filtered by `self._cutoff_time`)
- Produces: `Database.get_table(name: str) -> Any` and `Database.get_keys(name: str, sort: bool = True) -> Any`, both without `cutoff_time`

- [ ] **Step 1: Replace and add tests in `tests/test_database.py`**

Replace `test_get_keys_returns_the_keys_that_exist_at_the_cutoff_time` (it passes `cutoff_time` to `get_keys`) with:

```python
def test_as_of_filters_every_table_at_the_cutoff_time(db):
    # Sessions 10, 20 and 30 start on 2024-03-04, -05 and -06.
    keys = db.as_of(dt.datetime(2024, 3, 5)).get_keys("sessions").collect()
    assert sorted(keys["id"].to_list()) == [10, 20]


def test_as_of_leaves_the_original_database_unchanged(db):
    db.as_of(dt.datetime(2024, 3, 5))
    assert db.cutoff_time is None
    assert len(db.get_keys("sessions").collect()) == 3


def test_as_of_on_a_copy_builds_a_copy_with_the_new_cutoff_time(db):
    later = db.as_of(dt.datetime(2024, 3, 5)).as_of(dt.datetime(2024, 3, 6))
    assert later.cutoff_time == dt.datetime(2024, 3, 6)
    assert len(later.get_keys("sessions").collect()) == 3


def test_as_of_none_returns_a_copy_without_a_cutoff_time(db):
    unfiltered = db.as_of(dt.datetime(2024, 3, 5)).as_of(None)
    assert unfiltered.cutoff_time is None
    assert len(unfiltered.get_keys("sessions").collect()) == 3


def test_as_of_rejects_a_cutoff_time_that_is_not_a_datetime(db):
    with pytest.raises(TypeError, match="datetime"):
        db.as_of(dt.date(2024, 3, 5))


def test_as_of_rejects_a_cutoff_time_with_the_wrong_time_zone_awareness(db):
    with pytest.raises(ValidationError, match="tz-aware"):
        db.as_of(dt.datetime(2024, 3, 5, tzinfo=dt.UTC))


def test_validate_checks_only_the_rows_the_cutoff_time_shows():
    table = pl.LazyFrame(
        {"id": [1, 1], "at": [dt.datetime(2024, 1, 1), dt.datetime(2024, 6, 1)]},
    )
    db = tusk.Database("x").add_table(
        "t",
        table,
        primary_key="id",
        row_creation_time="at",
    )
    db.as_of(dt.datetime(2024, 3, 1)).validate(tables="unique_primary_key")
    with pytest.raises(ValidationError):
        db.validate(tables="unique_primary_key")
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_database.py -v`
Expected: the new tests FAIL with `AttributeError: 'Database' object has no attribute 'as_of'`.

- [ ] **Step 3: Implement in `src/tusk/database.py`**

Add `import copy` to the imports. In `__init__` add `self._cutoff_time: datetime | None = None` after `self._backend`. Add, next to `relationships`:

```python
    @property
    def cutoff_time(self) -> datetime | None:
        """The cutoff time the tables are seen at, or None for every row."""
        return self._cutoff_time

    def as_of(self, cutoff_time: datetime | None) -> Database:
        """Return a copy of the database seen at a cutoff time.

        Rows created after the cutoff time are dropped from every table.
        Columns updated after the cutoff time hold the value they held
        before. A table with no `row_creation_time` keeps every row, and a
        table declaring no `row_update_times` keeps every value. The copy
        reads the tables as added, so calling `as_of` on a copy gives the new
        cutoff time alone.

        It raises `TypeError` if `cutoff_time` is not a `datetime`. It raises
        [`ValidationError`][tusk.exceptions.ValidationError] if `cutoff_time`
        disagrees with the database's Datetime columns in tz awareness.

        Args:
            cutoff_time: The cutoff time. None returns a copy that holds every
                row.

        Returns:
            A new database. The original database is unchanged.
        """
        check_cutoff_time(self, cutoff_time)
        copied = copy.copy(self)
        copied._tables = dict(self._tables)
        copied._schemas = dict(self._schemas)
        copied._relationships = list(self._relationships)
        copied._cutoff_time = cutoff_time
        return copied
```

Replace `get_table` and `_get_table`:

```python
    def get_table(self, name: str) -> Any:
        """Return a table as it stands at the database's cutoff time.

        Rows created after the cutoff time are dropped. Columns updated after
        the cutoff time hold the value they held before. A table with no
        `row_creation_time` keeps every row, and a table declaring no
        `row_update_times` keeps every value. Without a cutoff time the table
        is returned as added.

        Args:
            name: The table's name.

        Returns:
            The table on the caller's backend, as the backend's native lazy
            table, if the backend supports lazy tables.

        Raises:
            SchemaError: If the table is unknown.
        """
        return self._get_table(name).to_native()

    def _get_table(self, name: str) -> nw.LazyFrame:
        """Return a table as a narwhals LazyFrame, as it stands at the cutoff time.

        Args:
            name: The table's name.

        Returns:
            The table as a narwhals LazyFrame.

        Raises:
            SchemaError: If the table is unknown.
        """
        try:
            table = self._tables[name]
        except KeyError:
            raise SchemaError(f"unknown table {name!r}") from None
        if self._cutoff_time is None:
            return table

        schema = self.get_schema(name)
        if schema.row_creation_time is not None:
            table = table.filter(nw.col(schema.row_creation_time) <= self._cutoff_time)
        return _restore_updated_columns(table, schema, self._cutoff_time)
```

Change `get_keys` to `def get_keys(self, name: str, sort: bool = True) -> Any:`. Its docstring says "Return a table's primary key column at the cutoff time." Drop the `cutoff_time` Args entry, the sentence about `cutoff_time` tz awareness and `TypeError`, and call `self._get_table(name).select(primary_key)`. The `Args` entry for `sort` stays.

- [ ] **Step 4: Point the other readers at the new methods**

- `src/tusk/sklearn/_tables.py:81`: change `database._get_table(target_table, cutoff_time)` to `database.as_of(cutoff_time)._get_table(target_table)`.
- `src/tusk/compiler.py`: add `database = database.as_of(cutoff_time)` as the first statement of `compile_features`, and change `_read_table`'s `database._get_table(table_name, cutoff_time)` to `database._get_table(table_name)`. The other functions keep their `cutoff_time` parameters for now (Task 2).

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest`
Expected: PASS. If a test still calls `get_table`/`get_keys` with a cutoff time, change it to `db.as_of(cutoff).get_keys(...)`.

- [ ] **Step 6: Commit**

```bash
git switch -c database-as-of
git add src tests
git commit -m "feat: add Database.as_of and carry the cutoff time on the database"
```

---

### Task 2: The compiler reads the cutoff time from the database

**Files:**
- Modify: `src/tusk/compiler.py`

**Interfaces:**
- Consumes: `Database.cutoff_time` from Task 1.
- Produces: `_read_table(database, table_name, needed)`, `_add_aggregations(table, database, table_name, relationship, batch)`, `_join_condition_aggregations(table, child, database, table_name, relationship, batch)`, `_build_aggregation(feature, database, relationship)`, `_add_directs(table, database, relationship, batch)`, `_apply(table, feature, database)`. Each loses `cutoff_time`.
- `_build_condition_mask(schema, condition, cutoff_time)` and `_build_expressions(primitive, inputs, cutoff_time)` keep the parameter: they use it as a value and take no database. Their callers pass `database.cutoff_time`.

This task is a refactor. The existing tests are the safety net, including `tests/test_compiler_aggregation.py::test_a_hand_built_cutoff_time_aggregation_computes` and the `when` condition tests.

- [ ] **Step 1: Run the compiler tests to record the baseline**

Run: `uv run pytest tests/test_compiler_single_table.py tests/test_compiler_aggregation.py tests/test_feature_list.py -q`
Expected: PASS.

- [ ] **Step 2: Remove the parameter**

In `src/tusk/compiler.py`, for each function below delete the `cutoff_time: datetime | None` parameter and its `Args:` line in the docstring, then update the call sites:

| Function | Call-site change |
| --- | --- |
| `compile_features` | `_read_table(database, target, closure)` |
| `_read_table` | `_add_aggregations(table, database, table_name, relationship, batch)`, `_add_directs(table, database, relationship, batch)`, `_apply(table, feature, database)` |
| `_add_aggregations` | `_read_table(database, relationship.child, child_needed)`, `_build_condition_mask(child_schema, condition, database.cutoff_time)`, and `_join_condition_aggregations(...)` without its last argument |
| `_join_condition_aggregations` | `_build_aggregation(feature, database, relationship)` |
| `_build_aggregation` | `_build_expressions(primitive, inputs, database.cutoff_time)` |
| `_add_directs` | `_read_table(database, relationship.parent, parent_needed)` |
| `_apply` | `_build_expressions(feature.primitive, inputs, database.cutoff_time)` |

`compile_features` keeps its `cutoff_time` parameter and the `_require_cutoff_time(closure, cutoff_time)` call. `datetime` stays imported for them.

- [ ] **Step 3: Run the whole suite and lint**

Run: `uv run pytest && just lint`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/tusk/compiler.py
git commit -m "refactor: read the cutoff time from the database in the compiler"
```

---

### Task 3: Warn when applying features replaces a cutoff time

**Files:**
- Modify: `src/tusk/exceptions.py` (new class after `ImplicitEarlierValueWarning`)
- Modify: `src/tusk/feature_list.py` (`FeatureList.apply`, new `_warn_if_replacing_cutoff_time`)
- Test: `tests/test_feature_list.py`

**Interfaces:**
- Consumes: `Database.as_of`, `Database.cutoff_time` from Task 1.
- Produces: `tusk.exceptions.OverwrittenCutoffTimeWarning`.

- [ ] **Step 1: Write the failing tests in `tests/test_feature_list.py`**

Add `import warnings` to the imports and `OverwrittenCutoffTimeWarning` to the `tusk.exceptions` import. Read the `features` fixture first. If it needs a cutoff time to apply, give the calls below one.

```python
def test_apply_warns_when_it_replaces_the_cutoff_time_of_the_database(features, db):
    with pytest.warns(OverwrittenCutoffTimeWarning, match="2024-03-05"):
        features.apply(
            db.as_of(dt.datetime(2024, 3, 5)),
            cutoff_time=dt.datetime(2024, 3, 6),
        )


def test_apply_with_no_cutoff_time_replaces_the_one_of_the_database(features, db):
    with pytest.warns(OverwrittenCutoffTimeWarning):
        matrix = features.apply(db.as_of(dt.datetime(2024, 1, 1)))
    assert matrix.collect().height == features.apply(db).collect().height


@pytest.mark.parametrize("database_cutoff_time", [None, dt.datetime(2024, 3, 6)])
def test_apply_does_not_warn_when_nothing_differs(features, db, database_cutoff_time):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        features.apply(
            db.as_of(database_cutoff_time),
            cutoff_time=dt.datetime(2024, 3, 6),
        )
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_feature_list.py -k "replace or warn" -v`
Expected: FAIL with `ImportError: cannot import name 'OverwrittenCutoffTimeWarning'`.

- [ ] **Step 3: Add the warning class to `src/tusk/exceptions.py`**

```python
class OverwrittenCutoffTimeWarning(UserWarning):
    """Warning issued when applying features replaces a database's cutoff time.

    A database from `as_of` carries a cutoff time. `apply_features` and
    `FeatureList.apply` always use the `cutoff_time` they are given, `None`
    included. Dropping a cutoff time the caller set on the database without
    saying so would change which rows the feature matrix holds. This warning
    has its own class. A caller can filter it independently of the others.
    """
```

- [ ] **Step 4: Implement in `src/tusk/feature_list.py`**

In `src/tusk/feature_list.py`, add `import warnings` and `OverwrittenCutoffTimeWarning` to the imports. Add a `Warns:` entry for `OverwrittenCutoffTimeWarning` to the docstring of `FeatureList.apply`, and call the helper first in its body:

```python
        _warn_if_replacing_cutoff_time(database, cutoff_time)
        return compile_features(self, database, cutoff_time).to_native()
```

`compile_features` is unchanged by this task. Add the helper below the `FeatureList` class (callers above callees):

```python
def _warn_if_replacing_cutoff_time(
    database: Database,
    cutoff_time: datetime | None,
) -> None:
    """Warn if `cutoff_time` differs from the database's own cutoff time.

    Args:
        database: The database to read.
        cutoff_time: The cutoff time that will replace the database's, or None.

    Warns:
        OverwrittenCutoffTimeWarning: If the database has a cutoff time that
            differs from `cutoff_time`.
    """
    if database.cutoff_time not in (None, cutoff_time):
        warnings.warn(
            f"cutoff_time {cutoff_time!r} replaces the database's cutoff time "
            f"{database.cutoff_time!r}",
            OverwrittenCutoffTimeWarning,
            stacklevel=3,
        )
```

- [ ] **Step 5: Run the whole suite and lint**

Run: `uv run pytest && just lint`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src tests
git commit -m "feat: warn when apply replaces the cutoff time of a database"
```

---

### Task 4: Documentation

**Files:**
- Modify: `docs/guide/databases.md` (new section "As of a cutoff time", before "Row update times", ~line 307)
- Modify: `docs/guide/sklearn.md:73,85,107`
- Modify: `docs/guide/deep-feature-synthesis.md` (warnings list, ~line 86)
- Modify: `src/tusk/api.py`, `src/tusk/feature_list.py`, `src/tusk/compiler.py` (cutoff docstrings)

- [ ] **Step 1: Add the section to `docs/guide/databases.md`**

```markdown
## As of a cutoff time

`db.as_of(cutoff_time)` returns a copy of the database whose tables hold only
what was known at the cutoff time. Rows created after it are dropped, and
columns updated after it hold the value they held before.

```python
june = db.as_of(datetime(2024, 6, 1))
june.get_table("orders")   # the orders table on 2024-06-01
june.cutoff_time           # datetime(2024, 6, 1)
db.cutoff_time             # None: the original is unchanged
```

The copy shares the tables with the original, so it costs nothing until you
collect. Calling `as_of` on a copy gives the new cutoff time alone, and
`db.as_of(None)` returns a copy with every row.

`apply_features` and `deep_feature_synthesis` take a `cutoff_time` and call
`as_of` for you. If the database already has a different cutoff time, they use
the one you pass and issue
[`OverwrittenCutoffTimeWarning`][tusk.exceptions.OverwrittenCutoffTimeWarning].

[`validate`][tusk.database.Database.validate] on a copy with a cutoff time
checks the rows that copy shows.
```

- [ ] **Step 2: Update `docs/guide/sklearn.md`**

Line 73: `X_train = db.as_of(train_cutoff_time).get_keys("customers")`. Line 85: describe `Database.get_keys` as returning the primary key column of a table as the database stands at its cutoff time, and say to compute the labels from `db.as_of(train_cutoff_time).get_keys(...)`. Line 107: "get `X` from `db.as_of(cutoff_time).get_keys` at the same cutoff time."

- [ ] **Step 3: Add the new warning to the warnings list in `docs/guide/deep-feature-synthesis.md`**

Add one bullet: [`OverwrittenCutoffTimeWarning`][tusk.exceptions.OverwrittenCutoffTimeWarning]: the `cutoff_time` you passed replaced a different cutoff time set on the database.

- [ ] **Step 4: Update the cutoff docstrings**

In `api.py`, `feature_list.py` and `compiler.py`, add to each `cutoff_time` entry: "It replaces the database's own cutoff time, if it has one." Add `OverwrittenCutoffTimeWarning` under `Warns:` in `apply` and `apply_features` (create the section if absent).

- [ ] **Step 5: Build the docs and run lint**

Run: `just docs && just lint`
Expected: the docs build with no broken cross-reference warnings. Lint passes.

- [ ] **Step 6: Commit**

```bash
git add docs src
git commit -m "docs: document Database.as_of"
```

---

### Task 5: Final verification

- [ ] **Step 1: Run everything**

Run: `uv run pytest && just lint && just test-differential`
Expected: PASS.

- [ ] **Step 2: Report the line count**

Run: `git diff main --stat -- src` and `git diff main --numstat -- src docs tests`
Report the source, test and docs lines added and removed, and the number of function signatures that lost `cutoff_time`.
