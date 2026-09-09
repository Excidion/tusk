# Row Update Times Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a table declare which of its columns were rewritten by a later update, so a cutoff hides post-cutoff column values instead of only post-cutoff rows.

**Architecture:** `add_table` gains a `row_update_times` mapping that lands on `TableSchema`. `compiler.base_frame` — the single place a cutoff is already applied — appends one `with_columns` that swaps each updated column for its declared pre-update value wherever the update happened after the cutoff. Because `base_frame` runs before every join and aggregation, everything downstream inherits the swap. Six new `TABLE_CHECKS` entries police the declaration; five read no rows and join the new `add_table` default.

**Tech Stack:** Python 3.10+, narwhals (backend-agnostic expressions), polars in tests, pytest, ruff + pydoclint + interrogate via pre-commit.

**Spec:** `agents/superpowers/specs/2026-09-09-row-update-times-design.md`

## Global Constraints

- Plans and specs live under `agents/superpowers/`, never `docs/` (`agents/AGENTS.md`).
- Every public **and private** function in `src/` needs a Google-style docstring with `Args:`, `Returns:` and `Raises:` sections that match the signature exactly. `interrogate` demands 100% coverage and `pydoclint` cross-checks the sections. Tests are exempt (`per-file-ignores`).
- Docstrings say *what*, never *how* or *why*; a `Why` belongs in a comment, and only when naming cannot carry it (`agents/style/CODESTYLE.md`).
- ruff lint set is `E, F, I, UP, B, D`. `B007` means an unused loop variable is an error — use every name you bind.
- Callers go above callees in a module; detail increases downward.
- Run tests with `uv run pytest`. Never pass `--active` to `uv`: it retargets the main checkout's venv at this worktree.
- Differential tests are opt-in: `uv run --group validation pytest -m differential`. Verified against featuretools 1.31.0.
- Full gate before the branch is done: `just check` (lint, test, test-differential).

## Vocabulary

The spec and this plan use *mask* for the swap operation. In user-facing strings — docstrings, error messages, warnings, docs — say "update" and "pre-update value", never "mask" or "fallback". `row_update_times` is the declaration; a column named by an inner mapping is an **updated column**; the value it maps to is its **pre-update value**.

---

### Task 1: The declaration on TableSchema and add_table

**Files:**
- Modify: `src/tusk/exceptions.py` (append a warning class)
- Modify: `src/tusk/database.py:17-30` (TableSchema), `:73-152` (add_table), end of module (helpers)
- Test: `tests/test_database.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `TableSchema.row_update_times: Mapping[str, Mapping[str, Any]]`, fifth field, defaulting to `{}`. Fully normalized: plain nested dicts, self-entries already inserted.
  - `TableSchema.column_updates -> tuple[tuple[str, str, Any], ...]` — property yielding `(update_time, column, pre_update_value)` triples in declaration order. **Every later task iterates this, not the raw mapping.**
  - `Database.add_table(..., row_update_times=None, ...)` — fifth positional parameter, after `row_creation_time` and before the keyword-only `validate`.
  - `tusk.exceptions.ImplicitRowUpdateTimeMaskWarning`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_database.py`:

```python
def test_row_update_times_land_on_the_schema():
    db = tusk.Database("d").add_table(
        "orders",
        pl.LazyFrame(
            {
                "id": [1],
                "status": ["delivered"],
                "updated_at": [dt.datetime(2024, 9, 1)],
            },
        ),
        primary_key="id",
        row_update_times={"updated_at": {"status": "pending", "updated_at": None}},
    )
    assert db.schema("orders").row_update_times == {
        "updated_at": {"status": "pending", "updated_at": None},
    }


def test_column_updates_flattens_the_declaration():
    db = tusk.Database("d").add_table(
        "orders",
        pl.LazyFrame(
            {
                "id": [1],
                "status": ["delivered"],
                "updated_at": [dt.datetime(2024, 9, 1)],
            },
        ),
        primary_key="id",
        row_update_times={"updated_at": {"status": "pending", "updated_at": None}},
    )
    assert db.schema("orders").column_updates == (
        ("updated_at", "status", "pending"),
        ("updated_at", "updated_at", None),
    )


def test_a_table_without_row_update_times_has_an_empty_mapping():
    db = tusk.Database("d").add_table(
        "orders", pl.LazyFrame({"id": [1]}), primary_key="id"
    )
    assert db.schema("orders").row_update_times == {}
    assert db.schema("orders").column_updates == ()


def test_an_update_time_that_does_not_list_itself_is_completed_and_warned():
    with pytest.warns(ImplicitRowUpdateTimeMaskWarning, match="'updated_at'"):
        db = tusk.Database("d").add_table(
            "orders",
            pl.LazyFrame(
                {
                    "id": [1],
                    "status": ["delivered"],
                    "updated_at": [dt.datetime(2024, 9, 1)],
                },
            ),
            primary_key="id",
            row_update_times={"updated_at": {"status": "pending"}},
        )
    assert db.schema("orders").row_update_times["updated_at"]["updated_at"] is None


def test_an_update_time_that_lists_itself_warns_nothing():
    with warnings.catch_warnings():
        warnings.simplefilter("error", ImplicitRowUpdateTimeMaskWarning)
        tusk.Database("d").add_table(
            "orders",
            pl.LazyFrame(
                {"id": [1], "updated_at": [dt.datetime(2024, 9, 1)]},
            ),
            primary_key="id",
            row_update_times={"updated_at": {"updated_at": dt.datetime(2024, 1, 1)}},
        )


def test_only_the_update_time_nothing_covers_is_warned_about():
    # 'first' updates 'second', so 'second' is already covered and only
    # 'first' is missing a pre-update value of its own.
    with pytest.warns(ImplicitRowUpdateTimeMaskWarning) as caught:
        tusk.Database("d").add_table(
            "orders",
            pl.LazyFrame(
                {
                    "id": [1],
                    "first": [dt.datetime(2024, 9, 1)],
                    "second": [dt.datetime(2024, 9, 2)],
                },
            ),
            primary_key="id",
            row_update_times={"first": {"second": None}, "second": {}},
        )
    assert len(caught) == 1
    assert "'first'" in str(caught[0].message)


def test_an_unknown_update_time_column_is_rejected():
    with pytest.raises(SchemaError, match="row_update_time 'nope'"):
        tusk.Database("d").add_table(
            "orders",
            pl.LazyFrame({"id": [1]}),
            primary_key="id",
            row_update_times={"nope": {}},
        )


def test_an_unknown_updated_column_is_rejected():
    with pytest.raises(SchemaError, match="'nope'"):
        tusk.Database("d").add_table(
            "orders",
            pl.LazyFrame({"id": [1], "updated_at": [dt.datetime(2024, 9, 1)]}),
            primary_key="id",
            row_update_times={"updated_at": {"nope": 1}},
        )


def test_a_flat_row_update_times_mapping_is_rejected():
    with pytest.raises(SchemaError, match="row_update_time 'updated_at'"):
        tusk.Database("d").add_table(
            "orders",
            pl.LazyFrame({"id": [1], "updated_at": [dt.datetime(2024, 9, 1)]}),
            primary_key="id",
            row_update_times={"updated_at": ["status"]},
        )


def test_a_non_mapping_row_update_times_is_rejected():
    with pytest.raises(SchemaError, match="row_update_times"):
        tusk.Database("d").add_table(
            "orders",
            pl.LazyFrame({"id": [1]}),
            primary_key="id",
            row_update_times=["updated_at"],
        )


def test_the_declaration_is_copied_not_aliased():
    declared = {"updated_at": {"status": "pending", "updated_at": None}}
    db = tusk.Database("d").add_table(
        "orders",
        pl.LazyFrame(
            {
                "id": [1],
                "status": ["delivered"],
                "updated_at": [dt.datetime(2024, 9, 1)],
            },
        ),
        primary_key="id",
        row_update_times=declared,
    )
    declared["updated_at"]["status"] = "mutated"
    assert db.schema("orders").row_update_times["updated_at"]["status"] == "pending"
```

Add `import warnings` and extend the existing imports at the top of the file so `ImplicitRowUpdateTimeMaskWarning` and `SchemaError` resolve:

```python
from tusk.exceptions import (
    ImplicitRowUpdateTimeMaskWarning,
    MissingPrimaryKeyWarning,
    SchemaError,
)
```

(Keep whatever else that import line already names.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_database.py -k row_update or column_updates or update_time -v`
Expected: collection error — `ImportError: cannot import name 'ImplicitRowUpdateTimeMaskWarning'`.

- [ ] **Step 3: Add the warning class**

Append to `src/tusk/exceptions.py`:

```python
class ImplicitRowUpdateTimeMaskWarning(UserWarning):
    """Warns that a row update time was given a null pre-update value.

    An update time listed under no update time is read straight from the
    table, so ``MAX(updated_at)`` returns a timestamp from after the cutoff.
    tusk fills the gap rather than serving that leak, but a null is only its
    guess at what the column held before. Its own class, so it can be filtered
    independently.
    """
```

- [ ] **Step 4: Extend TableSchema**

In `src/tusk/database.py`, change the dataclass import:

```python
from dataclasses import dataclass, field
```

Then extend `TableSchema`:

```python
@dataclass(frozen=True)
class TableSchema:
    """Everything phase 1 knows about a table.

    Attributes:
        name: Table name within the database.
        primary_key: Column uniquely identifying a row, if declared.
        row_creation_time: Column recording when a row became knowable.
        dtypes: Mapping of column name to narwhals dtype.
        row_update_times: Mapping of each column recording an update time to
            the columns that update rewrote, each mapped to the value it held
            before the update.
    """

    name: str
    primary_key: str | None
    row_creation_time: str | None
    dtypes: Mapping[str, Any]
    row_update_times: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    @property
    def column_updates(self) -> tuple[tuple[str, str, Any], ...]:
        """Every update as a (row update time, column, pre-update value) triple."""
        return tuple(
            (update_time, column, value)
            for update_time, updated in self.row_update_times.items()
            for column, value in updated.items()
        )
```

The default keeps the four-argument constructor calls in `tests/test_validation.py` working.

- [ ] **Step 5: Wire it through add_table**

Add the parameter to `Database.add_table`, between `row_creation_time` and the `*`:

```python
        row_update_times: Mapping[str, Mapping[str, Any]] | None = None,
```

Add to its `Args:` block, after `row_creation_time`:

```
            row_update_times: Maps each column recording when a row was
                updated to the columns that update rewrote, each mapped to the
                value it held before. Under a ``cutoff_time``, a row updated
                after the cutoff serves those earlier values instead. An update
                time column that no update time lists is added to its own
                mapping with a null value.
```

Add to its `Warns:` block:

```
            ImplicitRowUpdateTimeMaskWarning: If a ``row_update_times`` key is
                absent from every mapping, so tusk gave it a null value.
```

In the body, immediately after the existing `for column, label in (...)` missing-column loop and before the `if primary_key is None:` warning:

```python
        row_update_times = _normalize_row_update_times(row_update_times, name)
        _reject_unknown_updated_columns(row_update_times, dtypes, name)
        incomplete = _row_update_times_without_a_value(row_update_times)
        _warn_about_incomplete_row_update_times(incomplete, name)
        row_update_times = _insert_own_values(row_update_times, incomplete)
```

and pass it to the schema:

```python
        schema = TableSchema(
            name, primary_key, row_creation_time, dtypes, row_update_times
        )
```

- [ ] **Step 6: Add the helpers**

Append to `src/tusk/database.py`, after `_reject_composite`:

```python
def _normalize_row_update_times(
    row_update_times: Mapping[str, Mapping[str, Any]] | None,
    table: str,
) -> dict[str, dict[str, Any]]:
    """Copy a row update time declaration into plain nested dicts.

    Args:
        row_update_times: The declaration, or None.
        table: Table name, used in the message.

    Returns:
        The declaration as nested dicts, empty when None was given.

    Raises:
        SchemaError: If it is not a mapping of column name to mapping.
    """
    if row_update_times is None:
        return {}
    if not isinstance(row_update_times, Mapping):
        raise SchemaError(
            f"row_update_times of {table!r} must map each update time column "
            f"to the columns it updates; got {row_update_times!r}",
        )
    normalized = {}
    for update_time, updated in row_update_times.items():
        if not isinstance(updated, Mapping):
            raise SchemaError(
                f"row_update_time {update_time!r} of {table!r} must map each "
                f"column it updates to that column's pre-update value; "
                f"got {updated!r}",
            )
        normalized[update_time] = dict(updated)
    return normalized


def _reject_unknown_updated_columns(
    row_update_times: Mapping[str, Mapping[str, Any]],
    dtypes: Mapping[str, Any],
    table: str,
) -> None:
    """Raise if a row update time declaration names a column the table lacks.

    Args:
        row_update_times: The normalized declaration.
        dtypes: The table's columns.
        table: Table name, used in the message.

    Raises:
        SchemaError: If an update time or an updated column is not a column of
            the table.
    """
    for update_time, updated in row_update_times.items():
        if update_time not in dtypes:
            raise SchemaError(
                f"row_update_time {update_time!r} is not a column of {table!r}",
            )
        unknown = [column for column in updated if column not in dtypes]
        if unknown:
            raise SchemaError(
                f"{unknown[0]!r}, listed under row_update_time "
                f"{update_time!r}, is not a column of {table!r}",
            )


def _row_update_times_without_a_value(
    row_update_times: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Return the update time columns that no update time gives a value.

    Args:
        row_update_times: The normalized declaration.

    Returns:
        Their names, in declaration order.
    """
    updated = {column for values in row_update_times.values() for column in values}
    return [name for name in row_update_times if name not in updated]


def _warn_about_incomplete_row_update_times(incomplete: list[str], table: str) -> None:
    """Warn that tusk will read each named column as null before its update.

    Args:
        incomplete: Update time columns nothing gives a value, as built by
            :func:`_row_update_times_without_a_value`.
        table: Table name, used in the message.
    """
    for update_time in incomplete:
        warnings.warn(
            f"row_update_time {update_time!r} of {table!r} does not say what "
            f"it held before the update, so tusk reads it as null; list it "
            f"under itself to choose a value",
            ImplicitRowUpdateTimeMaskWarning,
            stacklevel=3,
        )


def _insert_own_values(
    row_update_times: Mapping[str, Mapping[str, Any]],
    incomplete: list[str],
) -> dict[str, dict[str, Any]]:
    """Give each update time column named in ``incomplete`` a null value.

    Args:
        row_update_times: The normalized declaration.
        incomplete: Update time columns nothing gives a value.

    Returns:
        The declaration with a null entry added for each of them.
    """
    completed = {key: dict(values) for key, values in row_update_times.items()}
    for update_time in incomplete:
        completed[update_time][update_time] = None
    return completed
```

Import the warning at the top of `database.py`:

```python
from tusk.exceptions import (
    ImplicitRowUpdateTimeMaskWarning,
    MissingPrimaryKeyWarning,
    SchemaError,
)
```

- [ ] **Step 7: Run the tests**

Run: `uv run pytest tests/test_database.py -v`
Expected: PASS, and every pre-existing test in the file still passes.

- [ ] **Step 8: Run the whole suite and lint**

Run: `uv run pytest && just lint`
Expected: 546+ passed; pre-commit clean.

- [ ] **Step 9: Commit**

```bash
git add src/tusk/exceptions.py src/tusk/database.py tests/test_database.py
git commit -m "feat: declare row update times on a table"
```

---

### Task 2: Masking in the compiler

**Files:**
- Modify: `src/tusk/compiler.py:151-180` (`base_frame`, plus two new helpers below it)
- Modify: `tests/conftest.py` (new `updating_db` fixture)
- Test: `tests/test_compiler_single_table.py`, `tests/test_compiler_aggregation.py`

**Interfaces:**
- Consumes: `TableSchema.row_update_times` and `TableSchema.column_updates` from Task 1.
- Produces: `base_frame` returns a frame whose updated columns already hold their pre-update values. No new public name; every later task relies on the behaviour, not on a symbol.

- [ ] **Step 1: Add the fixture**

Append to `tests/conftest.py`:

```python
@pytest.fixture
def updating_db():
    """A shop whose orders have a status and a parent that get rewritten.

    Against a cutoff of 2024-06-01: order 10 was updated after it, order 11
    never, order 12 before it.
    """
    customers = pl.LazyFrame(
        {
            "id": [1, 2],
            "signed_up_at": [dt.datetime(2024, 1, 1)] * 2,
        },
    )
    orders = pl.LazyFrame(
        {
            "id": [10, 11, 12],
            "customer_id": [1, 1, 2],
            "amount": [1.0, 2.0, 4.0],
            "placed_at": [dt.datetime(2024, 3, 1)] * 3,
            "status": ["delivered", "pending", "delivered"],
            "updated_at": [
                dt.datetime(2024, 9, 1),
                None,
                dt.datetime(2024, 4, 1),
            ],
        },
    )
    return (
        tusk.Database("shop")
        .add_table(
            "customers",
            customers,
            primary_key="id",
            row_creation_time="signed_up_at",
        )
        .add_table(
            "orders",
            orders,
            primary_key="id",
            row_creation_time="placed_at",
            row_update_times={
                "updated_at": {"status": "pending", "updated_at": None},
            },
        )
        .add_relationship(parent="customers", child="orders", foreign_key="customer_id")
    )
```

`updated_at` is listed under itself so the fixture raises no `ImplicitRowUpdateTimeMaskWarning`; Task 1 already covers the warning.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_compiler_single_table.py`:

```python
def test_a_column_updated_after_the_cutoff_serves_its_earlier_value(updating_db):
    status = IdentityFeature("orders", "status", nw.String())
    got = (
        compile_features(
            FeatureList([status]), updating_db, cutoff_time=dt.datetime(2024, 6, 1)
        )
        .collect()
        .to_native()
        .sort("id")
    )
    # 10 was updated after the cutoff, 11 never, 12 before it.
    assert got["status"].to_list() == ["pending", "pending", "delivered"]


def test_a_null_update_time_keeps_its_value():
    # Order 11's status is 'pending' either way, so prove the null path on a
    # row whose two answers differ: give it a delivered status and no update.
    orders = pl.LazyFrame(
        {
            "id": [11],
            "placed_at": [dt.datetime(2024, 3, 1)],
            "status": ["delivered"],
            "updated_at": [None],
        },
        schema={
            "id": pl.Int64,
            "placed_at": pl.Datetime,
            "status": pl.String,
            "updated_at": pl.Datetime,
        },
    )
    db = tusk.Database("shop").add_table(
        "orders",
        orders,
        primary_key="id",
        row_creation_time="placed_at",
        row_update_times={"updated_at": {"status": "pending", "updated_at": None}},
    )
    status = IdentityFeature("orders", "status", nw.String())
    got = (
        compile_features(FeatureList([status]), db, cutoff_time=dt.datetime(2024, 6, 1))
        .collect()
        .to_native()
    )
    assert got["status"].to_list() == ["delivered"]


def test_an_update_time_updates_itself(updating_db):
    updated = IdentityFeature("orders", "updated_at", nw.Datetime())
    got = (
        compile_features(
            FeatureList([updated]), updating_db, cutoff_time=dt.datetime(2024, 6, 1)
        )
        .collect()
        .to_native()
        .sort("id")
    )
    assert got["updated_at"].to_list() == [None, None, dt.datetime(2024, 4, 1)]


def test_an_update_time_updating_itself_does_not_disturb_the_other_columns(
    updating_db,
):
    """Every mask reads the frame as it was, so order within the call is moot.

    If the masks were applied one after another, masking ``updated_at`` first
    would null it and every later condition would then read null as
    'never updated' and keep the post-cutoff status.
    """
    status = IdentityFeature("orders", "status", nw.String())
    updated = IdentityFeature("orders", "updated_at", nw.Datetime())
    got = (
        compile_features(
            FeatureList([status, updated]),
            updating_db,
            cutoff_time=dt.datetime(2024, 6, 1),
        )
        .collect()
        .to_native()
        .sort("id")
    )
    assert got["status"].to_list() == ["pending", "pending", "delivered"]
    assert got["updated_at"].to_list() == [None, None, dt.datetime(2024, 4, 1)]


def test_no_cutoff_updates_nothing(updating_db):
    status = IdentityFeature("orders", "status", nw.String())
    got = (
        compile_features(FeatureList([status]), updating_db)
        .collect()
        .to_native()
        .sort("id")
    )
    assert got["status"].to_list() == ["delivered", "pending", "delivered"]


def test_an_update_survives_a_stacked_transform(updating_db):
    """A rewound column feeds a transform of a transform, not just a read."""
    updated = IdentityFeature("orders", "updated_at", nw.Datetime())
    month = TransformFeature(resolve("month"), (updated,))
    doubled = TransformFeature(resolve("add_numeric"), (month, month))
    got = (
        compile_features(
            FeatureList([doubled]),
            updating_db,
            cutoff_time=dt.datetime(2024, 6, 1),
        )
        .collect()
        .to_native()
        .sort("id")
    )
    # Orders 10 and 11 have no known update at the cutoff; order 12's is April.
    assert got["ADD_NUMERIC__MONTH__updated_at__MONTH__updated_at"].to_list() == [
        None,
        None,
        8,
    ]


def test_a_timeless_table_survives_the_masking_step(db):
    """with_columns() rejects an empty expression list, so the no-op path
    must return the frame untouched rather than call it with nothing."""
    age = IdentityFeature("customers", "age", nw.Int64())
    got = (
        compile_features(FeatureList([age]), db, cutoff_time=dt.datetime(2024, 6, 1))
        .collect()
        .to_native()
        .sort("id")
    )
    assert got["age"].to_list() == [30, 40, 50]
```

Append to `tests/test_compiler_aggregation.py`:

```python
def test_an_update_survives_an_aggregation(updating_db):
    status = IdentityFeature("orders", "status", nw.String())
    delivered = AggregationFeature(
        resolve("count"), (status,), "customers", "orders", "customer_id"
    )
    got = (
        compile_features(
            FeatureList([delivered]),
            updating_db,
            cutoff_time=dt.datetime(2024, 6, 1),
        )
        .collect()
        .to_native()
        .sort("id")
    )
    assert got["COUNT__orders__status"].to_list() == [2, 1]


def test_an_updated_foreign_key_stops_a_child_reaching_its_parent():
    customers = pl.LazyFrame(
        {"id": [1, 2], "signed_up_at": [dt.datetime(2024, 1, 1)] * 2},
    )
    orders = pl.LazyFrame(
        {
            "id": [10, 11],
            "customer_id": [1, 2],
            "amount": [1.0, 2.0],
            "placed_at": [dt.datetime(2024, 3, 1)] * 2,
            "moved_at": [dt.datetime(2024, 9, 1), None],
        },
    )
    db = (
        tusk.Database("shop")
        .add_table(
            "customers", customers, primary_key="id", row_creation_time="signed_up_at"
        )
        .add_table(
            "orders",
            orders,
            primary_key="id",
            row_creation_time="placed_at",
            row_update_times={"moved_at": {"customer_id": None, "moved_at": None}},
        )
        .add_relationship(parent="customers", child="orders", foreign_key="customer_id")
    )
    amount = IdentityFeature("orders", "amount", nw.Float64())
    total = AggregationFeature(
        resolve("sum"), (amount,), "customers", "orders", "customer_id"
    )
    got = (
        compile_features(
            FeatureList([total]), db, cutoff_time=dt.datetime(2024, 6, 1)
        )
        .collect()
        .to_native()
        .sort("id")
    )
    # Order 10 was moved to customer 1 after the cutoff, so at the cutoff it
    # belonged to nobody and customer 1 has no orders.
    assert got["SUM__orders__amount"].to_list() == [None, 2.0]
```

Both files already import `dt`, `nw`, `pl`, `pytest`, `tusk`, `compile_features`, `FeatureList` and `IdentityFeature`. `test_compiler_single_table.py` also needs `TransformFeature` and `resolve` (it has both). `test_compiler_aggregation.py` needs `AggregationFeature` and `resolve` — check its import block and add whichever is missing.

Two values here depend on conventions the existing files already fix, so read them rather than trusting the numbers above: if `SUM` over an empty group returns `0.0` instead of `None`, and if `MONTH` of a null returns something other than `None`, assert whatever the existing empty-group and null-input tests in those files assert. Match the file; do not invent a convention.

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_compiler_single_table.py tests/test_compiler_aggregation.py -v`
Expected: the new tests FAIL on the raw post-cutoff values (`['delivered', 'pending', 'delivered'] != ['pending', 'pending', 'delivered']`). `test_no_cutoff_updates_nothing` and `test_a_timeless_table_survives_the_masking_step` already pass.

- [ ] **Step 4: Implement the masking**

In `src/tusk/compiler.py`, replace `base_frame` and add two helpers directly below it:

```python
def base_frame(
    database: Database,
    table: str,
    cutoff_time: datetime | None,
) -> nw.LazyFrame:
    """Return a table's frame as it stood at the cutoff.

    Rows created after the cutoff are dropped and columns updated after it
    hold the value they held before. Tables without a ``row_creation_time``
    are timeless and keep every row -- documented rather than warned, per spec
    section 8 -- and a table declaring no ``row_update_times`` keeps every
    value, so a cutoff on a database that declares neither is a silent no-op.

    The target table is filtered like any other, so a cutoff can leave the
    feature matrix with fewer rows than the target table has. That matches
    featuretools: a row that did not exist yet at the cutoff has no features to
    compute.

    Args:
        database: The database holding the frames.
        table: Table name.
        cutoff_time: The cutoff, or None.

    Returns:
        The frame as it stood at the cutoff.
    """
    frame = database.frame(table)
    if cutoff_time is None:
        return frame

    schema = database.schema(table)
    if schema.row_creation_time is not None:
        frame = frame.filter(nw.col(schema.row_creation_time) <= cutoff_time)
    return _restore_updated_columns(frame, schema, cutoff_time)


def _restore_updated_columns(
    frame: nw.LazyFrame,
    schema: TableSchema,
    cutoff_time: datetime,
) -> nw.LazyFrame:
    """Give every column updated after the cutoff the value it held before.

    Args:
        frame: The table's frame, already filtered to the cutoff.
        schema: The table's schema, naming the updates.
        cutoff_time: The cutoff.

    Returns:
        The frame, with one replaced column per declared update.
    """
    # One with_columns for all of them: every expression in a single call
    # reads the frame as it was, so an update time that replaces itself
    # cannot corrupt the conditions of the columns it also replaces.
    replacements = [
        nw.when(_was_updated_by(update_time, cutoff_time))
        .then(nw.col(column))
        .otherwise(nw.lit(value, dtype=schema.dtypes[column]))
        .alias(column)
        for update_time, column, value in schema.column_updates
    ]
    if not replacements:
        return frame
    return frame.with_columns(*replacements)


def _was_updated_by(update_time: str, cutoff_time: datetime) -> nw.Expr:
    """Build the test for a row's update having already happened.

    A null update time is a row that was never updated, so the value it holds
    is the one it has always held.

    Args:
        update_time: Column recording when the row was updated.
        cutoff_time: The cutoff.

    Returns:
        A boolean expression, true where the stored values are the ones that
        held at the cutoff.
    """
    updated = nw.col(update_time)
    return updated.is_null() | (updated <= cutoff_time)
```

`TableSchema` may not be imported in `compiler.py` yet; check the `TYPE_CHECKING` block at the top and add it to the `from tusk.database import ...` line if it is missing.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_compiler_single_table.py tests/test_compiler_aggregation.py -v`
Expected: PASS.

- [ ] **Step 6: Run the whole suite and lint**

Run: `uv run pytest && just lint`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/tusk/compiler.py tests/conftest.py tests/test_compiler_single_table.py tests/test_compiler_aggregation.py
git commit -m "feat: serve pre-update values for columns updated after the cutoff"
```

---

### Task 3: The three schema-shape checks

**Files:**
- Modify: `src/tusk/validation.py` (three checks after `check_dtype_row_creation_time`, one helper, three `TABLE_CHECKS` entries)
- Test: `tests/test_validation.py`

**Interfaces:**
- Consumes: `TableSchema.row_update_times`, `TableSchema.column_updates`.
- Produces: `check_dtype_row_update_times`, `check_unmasked_primary_key`, `check_unmasked_row_creation_time` — all `(frame: nw.LazyFrame, schema: TableSchema) -> None`, registered in `TABLE_CHECKS` under `"datetime_row_update_times"`, `"unmasked_primary_key"`, `"unmasked_row_creation_time"`. Also `_updating_row_update_time(schema, column) -> str | None`, reused by Task 4's tests only indirectly.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_validation.py`:

```python
def updating_schema(row_update_times, primary_key="id", row_creation_time=None):
    """A schema over id / status / created_at / updated_at."""
    return TableSchema(
        "orders",
        primary_key,
        row_creation_time,
        {
            "id": nw.Int64(),
            "status": nw.String(),
            "created_at": nw.Datetime(),
            "updated_at": nw.Datetime(),
        },
        row_update_times,
    )


def test_a_date_row_update_time_is_reported():
    schema = TableSchema(
        "orders",
        "id",
        None,
        {"id": nw.Int64(), "updated_on": nw.Date()},
        {"updated_on": {"updated_on": None}},
    )
    with pytest.raises(ValidationError, match="row_update_time 'updated_on'"):
        check_dtype_row_update_times(frame([1]), schema)


def test_a_datetime_row_update_time_passes():
    check_dtype_row_update_times(
        frame([1]), updating_schema({"updated_at": {"updated_at": None}})
    )


def test_a_table_without_row_update_times_passes_the_dtype_check():
    check_dtype_row_update_times(frame([1]), updating_schema({}))


def test_an_updated_primary_key_is_reported():
    schema = updating_schema({"updated_at": {"id": None, "updated_at": None}})
    with pytest.raises(ValidationError, match="primary_key 'id'"):
        check_unmasked_primary_key(frame([1]), schema)


def test_an_untouched_primary_key_passes():
    check_unmasked_primary_key(
        frame([1]), updating_schema({"updated_at": {"status": "pending"}})
    )


def test_a_table_without_a_primary_key_passes_the_unmasked_check():
    check_unmasked_primary_key(
        frame([1]),
        updating_schema({"updated_at": {"status": "pending"}}, primary_key=None),
    )


def test_an_updated_row_creation_time_is_reported():
    schema = updating_schema(
        {"updated_at": {"created_at": None, "updated_at": None}},
        row_creation_time="created_at",
    )
    with pytest.raises(ValidationError, match="row_creation_time 'created_at'"):
        check_unmasked_row_creation_time(frame([1]), schema)


def test_an_untouched_row_creation_time_passes():
    check_unmasked_row_creation_time(
        frame([1]),
        updating_schema(
            {"updated_at": {"status": "pending"}}, row_creation_time="created_at"
        ),
    )


def test_a_table_without_a_row_creation_time_passes_the_unmasked_check():
    check_unmasked_row_creation_time(
        frame([1]), updating_schema({"updated_at": {"status": "pending"}})
    )
```

Extend the `from tusk.validation import (...)` block with `check_dtype_row_update_times`, `check_unmasked_primary_key`, `check_unmasked_row_creation_time`.

Note: the module's existing `frame()` helper builds a one-column `id` frame. These checks read only the schema, so the frame's shape is irrelevant — that is why every call passes `frame([1])`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_validation.py -v`
Expected: collection error — `ImportError: cannot import name 'check_dtype_row_update_times'`.

- [ ] **Step 3: Implement the checks**

In `src/tusk/validation.py`, insert after `check_dtype_row_creation_time`:

```python
def check_dtype_row_update_times(frame: nw.LazyFrame, schema: TableSchema) -> None:
    """Confirm every declared row update time is a Datetime, not a Date.

    A table with no ``row_update_times`` is skipped. Reads the schema only.

    Args:
        frame: The table's lazy frame. Unused.
        schema: The table's schema, naming the columns to check.

    Raises:
        ValidationError: If a row update time column is not a Datetime.
    """
    for column in schema.row_update_times:
        dtype = schema.dtypes[column]
        if dtype == nw.Datetime:
            continue
        raise ValidationError(
            f"row_update_time {column!r} of {schema.name!r} is {dtype}, "
            f"expected Datetime",
        )


def check_unmasked_primary_key(frame: nw.LazyFrame, schema: TableSchema) -> None:
    """Confirm no row update time rewrites the primary key.

    A table with no ``primary_key`` is skipped. Reads the schema only.

    Args:
        frame: The table's lazy frame. Unused.
        schema: The table's schema, naming the column to check.

    Raises:
        ValidationError: If a row update time lists the primary key.
    """
    update_time = _updating_row_update_time(schema, schema.primary_key)
    if update_time is None:
        return

    raise ValidationError(
        f"primary_key {schema.primary_key!r} of {schema.name!r} is listed "
        f"under row_update_time {update_time!r}; the primary key names the "
        f"feature matrix's rows, so it cannot be served an earlier value",
    )


def check_unmasked_row_creation_time(
    frame: nw.LazyFrame, schema: TableSchema
) -> None:
    """Confirm no row update time rewrites the row creation time.

    A table with no ``row_creation_time`` is skipped. Reads the schema only.

    Args:
        frame: The table's lazy frame. Unused.
        schema: The table's schema, naming the column to check.

    Raises:
        ValidationError: If a row update time lists the row creation time.
    """
    update_time = _updating_row_update_time(schema, schema.row_creation_time)
    if update_time is None:
        return

    raise ValidationError(
        f"row_creation_time {schema.row_creation_time!r} of {schema.name!r} "
        f"is listed under row_update_time {update_time!r}; every visible row "
        f"was created at or before the cutoff already",
    )


def _updating_row_update_time(schema: TableSchema, column: str | None) -> str | None:
    """Return the row update time that rewrites a column.

    Args:
        schema: The table's schema.
        column: The column to look for, or None.

    Returns:
        The first row update time listing it, or None when nothing does.
    """
    if column is None:
        return None
    for update_time, updated, _ in schema.column_updates:
        if updated == column:
            return update_time
    return None
```

`_updating_row_update_time` goes at the bottom of the module's check section, above `TABLE_CHECKS`, so callers stay above it.

Register them in `TABLE_CHECKS`:

```python
TABLE_CHECKS = {
    "non_null_primary_key": check_non_null_primary_key,
    "unique_primary_key": check_unique_primary_key,
    "datetime_row_creation_time": check_dtype_row_creation_time,
    "datetime_row_update_times": check_dtype_row_update_times,
    "unmasked_primary_key": check_unmasked_primary_key,
    "unmasked_row_creation_time": check_unmasked_row_creation_time,
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_validation.py -v`
Expected: PASS.

Note: `validate_table(..., checks=True)` runs the whole registry, so any existing test relying on the registry's exact contents may need its expectation widened. Read the failure before changing anything — the fix is to update the expectation, not to skip a check.

- [ ] **Step 5: Run the whole suite and lint**

Run: `uv run pytest && just lint`

- [ ] **Step 6: Commit**

```bash
git add src/tusk/validation.py tests/test_validation.py
git commit -m "feat: check row update time dtypes and untouched key columns"
```

---

### Task 4: singly_masked_columns

**Files:**
- Modify: `src/tusk/validation.py` (one check, one `TABLE_CHECKS` entry)
- Test: `tests/test_validation.py`

**Interfaces:**
- Consumes: `TableSchema.column_updates`, the `updating_schema` test helper from Task 3.
- Produces: `check_singly_masked_columns(frame, schema) -> None`, registered as `"singly_masked_columns"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_validation.py`:

```python
def test_a_column_updated_by_two_update_times_is_reported():
    schema = TableSchema(
        "orders",
        "id",
        None,
        {
            "id": nw.Int64(),
            "status": nw.String(),
            "shipped_at": nw.Datetime(),
            "updated_at": nw.Datetime(),
        },
        {
            "shipped_at": {"status": "pending"},
            "updated_at": {"status": "unknown"},
        },
    )
    with pytest.raises(ValidationError) as excinfo:
        check_singly_masked_columns(frame([1]), schema)
    message = str(excinfo.value)
    assert "'status'" in message
    assert "'shipped_at'" in message
    assert "'updated_at'" in message


def test_one_update_time_over_many_columns_passes():
    check_singly_masked_columns(
        frame([1]),
        updating_schema({"updated_at": {"status": "pending", "updated_at": None}}),
    )


def test_two_update_times_over_different_columns_pass():
    schema = TableSchema(
        "orders",
        "id",
        None,
        {
            "id": nw.Int64(),
            "status": nw.String(),
            "note": nw.String(),
            "shipped_at": nw.Datetime(),
            "updated_at": nw.Datetime(),
        },
        {
            "shipped_at": {"status": "pending", "shipped_at": None},
            "updated_at": {"note": None, "updated_at": None},
        },
    )
    check_singly_masked_columns(frame([1]), schema)


def test_an_update_time_updated_by_another_one_is_reported():
    # 'first' rewrites 'second', and add_table also gives 'second' an entry of
    # its own, so 'second' ends up with two.
    schema = TableSchema(
        "orders",
        "id",
        None,
        {"id": nw.Int64(), "first": nw.Datetime(), "second": nw.Datetime()},
        {"first": {"second": None, "first": None}, "second": {"second": None}},
    )
    with pytest.raises(ValidationError, match="'second'"):
        check_singly_masked_columns(frame([1]), schema)
```

Add `check_singly_masked_columns` to the `from tusk.validation import (...)` block.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_validation.py -k singly -v`
Expected: collection error — `ImportError: cannot import name 'check_singly_masked_columns'`.

- [ ] **Step 3: Implement the check**

In `src/tusk/validation.py`, insert after `check_unmasked_row_creation_time`:

```python
def check_singly_masked_columns(frame: nw.LazyFrame, schema: TableSchema) -> None:
    """Confirm no column is rewritten by two row update times.

    Two update times over one column give it two earlier values and two
    moments to switch between them, and nothing chooses. The entry
    ``add_table`` adds for an update time that says nothing about itself
    counts like any other. Reads the schema only.

    Args:
        frame: The table's lazy frame. Unused.
        schema: The table's schema, naming the updates.

    Raises:
        ValidationError: If one column appears under two row update times.
    """
    seen: dict[str, str] = {}
    for update_time, column, _ in schema.column_updates:
        if column in seen:
            raise ValidationError(
                f"{column!r} of {schema.name!r} is listed under both "
                f"row_update_time {seen[column]!r} and row_update_time "
                f"{update_time!r}; give it one row_update_time",
            )
        seen[column] = update_time
```

Register it in `TABLE_CHECKS`, after `"datetime_row_update_times"`:

```python
    "singly_masked_columns": check_singly_masked_columns,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_validation.py -v`
Expected: PASS.

- [ ] **Step 5: Run the whole suite and lint**

Run: `uv run pytest && just lint`

- [ ] **Step 6: Commit**

```bash
git add src/tusk/validation.py tests/test_validation.py
git commit -m "feat: reject a column updated by two row update times"
```

---

### Task 5: matching_fallback_dtypes

**Files:**
- Modify: `src/tusk/validation.py` (one check, one private helper, one `TABLE_CHECKS` entry, one import)
- Test: `tests/test_validation.py`

**Interfaces:**
- Consumes: `TableSchema.column_updates`.
- Produces: `check_matching_fallback_dtypes(frame, schema) -> None`, registered as `"matching_fallback_dtypes"`; `_fits_dtype(value, dtype) -> bool`.

Background you need: narwhals dtype instances expose `is_integer()`, `is_float()`, `is_boolean()`, `is_temporal()` and compare equal to their class (`nw.Datetime() == nw.Datetime` is `True`, `nw.Date() == nw.Datetime` is `False`). Both the class and an instance answer these, so a schema built with `nw.Int64` and one built with `nw.Int64()` behave identically. In Python `bool` is a subclass of `int` and `datetime` is a subclass of `date`, so those two pairs must be tested in the narrow-first order below.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_validation.py`:

```python
@pytest.mark.parametrize(
    ("value", "dtype"),
    [
        (None, nw.Int64()),
        (None, nw.String()),
        (1, nw.Int64()),
        (1, nw.Float64()),
        (1.5, nw.Float64()),
        (True, nw.Boolean()),
        ("pending", nw.String()),
        ("pending", nw.Categorical()),
        (dt.datetime(2024, 1, 1), nw.Datetime()),
        (dt.date(2024, 1, 1), nw.Date()),
    ],
)
def test_a_fitting_pre_update_value_passes(value, dtype):
    schema = TableSchema(
        "orders",
        "id",
        None,
        {"id": nw.Int64(), "column": dtype, "updated_at": nw.Datetime()},
        {"updated_at": {"column": value, "updated_at": None}},
    )
    check_matching_fallback_dtypes(frame([1]), schema)


@pytest.mark.parametrize(
    ("value", "dtype"),
    [
        ("pending", nw.Int64()),
        (1.5, nw.Int64()),
        (True, nw.Int64()),
        (1, nw.String()),
        (1, nw.Boolean()),
        (dt.date(2024, 1, 1), nw.Datetime()),
        (dt.datetime(2024, 1, 1), nw.Date()),
    ],
)
def test_a_misfitting_pre_update_value_is_reported(value, dtype):
    schema = TableSchema(
        "orders",
        "id",
        None,
        {"id": nw.Int64(), "column": dtype, "updated_at": nw.Datetime()},
        {"updated_at": {"column": value, "updated_at": None}},
    )
    with pytest.raises(ValidationError) as excinfo:
        check_matching_fallback_dtypes(frame([1]), schema)
    message = str(excinfo.value)
    assert "'column'" in message
    assert "'updated_at'" in message


def test_a_table_without_row_update_times_passes_the_value_check():
    check_matching_fallback_dtypes(frame([1]), updating_schema({}))
```

Add `check_matching_fallback_dtypes` to the `from tusk.validation import (...)` block. `dt` is already imported at the top of the file.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_validation.py -k pre_update -v`
Expected: collection error — `ImportError: cannot import name 'check_matching_fallback_dtypes'`.

- [ ] **Step 3: Implement the check**

Change the datetime import at the top of `src/tusk/validation.py`:

```python
from datetime import date, datetime
```

Insert after `check_singly_masked_columns`:

```python
def check_matching_fallback_dtypes(frame: nw.LazyFrame, schema: TableSchema) -> None:
    """Confirm every declared pre-update value fits the column it replaces.

    Reads the schema only. A null fits every column.

    Args:
        frame: The table's lazy frame. Unused.
        schema: The table's schema, naming the updates.

    Raises:
        ValidationError: If a pre-update value does not fit its column's dtype.
    """
    for update_time, column, value in schema.column_updates:
        dtype = schema.dtypes[column]
        if _fits_dtype(value, dtype):
            continue
        raise ValidationError(
            f"{value!r}, listed for {column!r} of {schema.name!r} under "
            f"row_update_time {update_time!r}, is not a {dtype} value",
        )
```

and, alongside `_updating_row_update_time` at the bottom of the check section:

```python
def _fits_dtype(value: Any, dtype: Any) -> bool:
    """Decide whether a Python value can stand in for a narwhals dtype.

    A null fits every column. A dtype family tusk does not recognise accepts
    every value, because refusing one it cannot judge is worse than letting
    the backend judge it at collect time.

    Args:
        value: The declared pre-update value.
        dtype: The narwhals dtype of the column it replaces.

    Returns:
        True if the value fits the dtype.
    """
    # bool is a subclass of int and datetime is a subclass of date, so both
    # narrow types have to be tested before the wide ones.
    if value is None:
        return True
    if isinstance(value, bool):
        return dtype == nw.Boolean
    if isinstance(value, int):
        return dtype.is_integer() or dtype.is_float()
    if isinstance(value, float):
        return dtype.is_float()
    if isinstance(value, str):
        return dtype in (nw.String, nw.Categorical, nw.Enum)
    if isinstance(value, datetime):
        return dtype == nw.Datetime
    if isinstance(value, date):
        return dtype == nw.Date
    return True
```

`Any` is already imported from `typing` — confirm, and add it if the `TYPE_CHECKING` refactor left it out.

Register it in `TABLE_CHECKS`, after `"singly_masked_columns"`:

```python
    "matching_fallback_dtypes": check_matching_fallback_dtypes,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_validation.py -v`
Expected: PASS.

- [ ] **Step 5: Run the whole suite and lint**

Run: `uv run pytest && just lint`

- [ ] **Step 6: Commit**

```bash
git add src/tusk/validation.py tests/test_validation.py
git commit -m "feat: check a pre-update value fits the column it replaces"
```

---

### Task 6: ordered_row_times

**Files:**
- Modify: `src/tusk/validation.py` (one check, one `TABLE_CHECKS` entry)
- Test: `tests/test_validation.py`

**Interfaces:**
- Consumes: `TableSchema.row_update_times`.
- Produces: `check_ordered_row_times(frame, schema) -> None`, registered as `"ordered_row_times"`. **This one scans rows** and stays out of Task 7's default tuple.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_validation.py`:

```python
def times_frame(created, updated):
    """A frame of creation and update timestamps."""
    return nw.from_native(
        pl.LazyFrame(
            {"created_at": created, "updated_at": updated},
            schema={"created_at": pl.Datetime, "updated_at": pl.Datetime},
        ),
    )


def times_schema():
    """A schema over created_at and updated_at, the latter updating itself."""
    return TableSchema(
        "orders",
        None,
        "created_at",
        {"created_at": nw.Datetime(), "updated_at": nw.Datetime()},
        {"updated_at": {"updated_at": None}},
    )


def test_an_update_before_the_creation_is_reported():
    got = times_frame(
        [dt.datetime(2024, 3, 1)] * 2,
        [dt.datetime(2024, 1, 1), dt.datetime(2024, 5, 1)],
    )
    with pytest.raises(ValidationError) as excinfo:
        check_ordered_row_times(got, times_schema())
    message = str(excinfo.value)
    assert "'updated_at'" in message
    assert "'created_at'" in message
    assert "1 rows" in message


def test_updates_at_or_after_the_creation_pass():
    check_ordered_row_times(
        times_frame(
            [dt.datetime(2024, 3, 1)] * 2,
            [dt.datetime(2024, 3, 1), dt.datetime(2024, 5, 1)],
        ),
        times_schema(),
    )


def test_a_null_update_time_passes_the_order_check():
    check_ordered_row_times(
        times_frame([dt.datetime(2024, 3, 1)], [None]), times_schema()
    )


def test_a_table_without_a_row_creation_time_passes_the_order_check():
    schema = TableSchema(
        "orders",
        None,
        None,
        {"created_at": nw.Datetime(), "updated_at": nw.Datetime()},
        {"updated_at": {"updated_at": None}},
    )
    check_ordered_row_times(
        times_frame([dt.datetime(2024, 3, 1)], [dt.datetime(2024, 1, 1)]), schema
    )


def test_a_table_without_row_update_times_passes_the_order_check():
    schema = TableSchema(
        "orders",
        None,
        "created_at",
        {"created_at": nw.Datetime(), "updated_at": nw.Datetime()},
        {},
    )
    check_ordered_row_times(
        times_frame([dt.datetime(2024, 3, 1)], [dt.datetime(2024, 1, 1)]), schema
    )


def test_the_order_check_is_not_in_the_add_table_default():
    # It scans, so add_table must not pay for it.
    from tusk.validation import DEFAULT_TABLE_CHECKS

    assert "ordered_row_times" not in DEFAULT_TABLE_CHECKS
```

Add `check_ordered_row_times` to the `from tusk.validation import (...)` block.

Leave `test_the_order_check_is_not_in_the_add_table_default` failing until Task 7 introduces `DEFAULT_TABLE_CHECKS`; note that in the commit message.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_validation.py -k order -v`
Expected: collection error — `ImportError: cannot import name 'check_ordered_row_times'`.

- [ ] **Step 3: Implement the check**

In `src/tusk/validation.py`, insert after `check_matching_fallback_dtypes`:

```python
def check_ordered_row_times(frame: nw.LazyFrame, schema: TableSchema) -> None:
    """Confirm no row was updated before it was created.

    Scans the table. A table with no ``row_creation_time`` or no
    ``row_update_times`` is skipped, as is a row whose update time is null:
    a row that was never updated has no order to check.

    Args:
        frame: The table's lazy frame.
        schema: The table's schema, naming the columns to compare.

    Raises:
        ValidationError: If an update time is before the row creation time.
    """
    created = schema.row_creation_time
    if created is None or not schema.row_update_times:
        return

    early = frame.select(
        (nw.col(update_time) < nw.col(created)).sum().alias(update_time)
        for update_time in schema.row_update_times
    ).collect()

    for update_time in schema.row_update_times:
        rows = early[update_time].item()
        if not rows:
            continue
        raise ValidationError(
            f"row_update_time {update_time!r} of {schema.name!r} is before "
            f"row_creation_time {created!r} in {rows} rows",
        )
```

Register it in `TABLE_CHECKS`, last:

```python
    "ordered_row_times": check_ordered_row_times,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_validation.py -v`
Expected: every new test PASSES except `test_the_order_check_is_not_in_the_add_table_default`, which fails on `ImportError: cannot import name 'DEFAULT_TABLE_CHECKS'`. Task 7 fixes it.

- [ ] **Step 5: Commit**

```bash
git add src/tusk/validation.py tests/test_validation.py
git commit -m "feat: add an opt-in check that updates follow row creation

test_the_order_check_is_not_in_the_add_table_default stays red until
DEFAULT_TABLE_CHECKS lands in the next commit."
```

---

### Task 7: The new add_table default

**Files:**
- Modify: `src/tusk/validation.py` (a `DEFAULT_TABLE_CHECKS` tuple below `TABLE_CHECKS`)
- Modify: `src/tusk/database.py` (`add_table`'s `validate` default and docstring)
- Test: `tests/test_validation.py`, `tests/test_database.py`

**Interfaces:**
- Consumes: every check name registered in Tasks 3-6.
- Produces: `tusk.validation.DEFAULT_TABLE_CHECKS: tuple[str, ...]`, the new default of `Database.add_table(validate=...)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_validation.py`:

```python
def test_every_default_check_is_registered():
    assert set(DEFAULT_TABLE_CHECKS) <= set(validation.TABLE_CHECKS)


def test_no_default_check_reads_rows():
    """Every default check must answer from the schema alone.

    A frame that raises on any access proves it: if a check touched the data,
    this test would see the exception instead of a clean pass.
    """

    class Unreadable:
        def __getattr__(self, name):
            raise AssertionError(f"a default check read the frame: {name}")

    schema = TableSchema(
        "orders",
        "id",
        "created_at",
        {
            "id": nw.Int64(),
            "status": nw.String(),
            "created_at": nw.Datetime(),
            "updated_at": nw.Datetime(),
        },
        {"updated_at": {"status": "pending", "updated_at": None}},
    )
    validate_table(Unreadable(), schema, DEFAULT_TABLE_CHECKS)
```

Append to `tests/test_database.py`:

```python
def test_add_table_rejects_a_date_row_update_time_by_default():
    with pytest.raises(ValidationError, match="row_update_time 'updated_on'"):
        tusk.Database("d").add_table(
            "orders",
            pl.LazyFrame({"id": [1], "updated_on": [dt.date(2024, 9, 1)]}),
            primary_key="id",
            row_update_times={"updated_on": {"updated_on": None}},
        )


def test_add_table_rejects_an_updated_primary_key_by_default():
    with pytest.raises(ValidationError, match="primary_key 'id'"):
        tusk.Database("d").add_table(
            "orders",
            pl.LazyFrame({"id": [1], "updated_at": [dt.datetime(2024, 9, 1)]}),
            primary_key="id",
            row_update_times={"updated_at": {"id": None, "updated_at": None}},
        )


def test_add_table_rejects_a_misfitting_pre_update_value_by_default():
    with pytest.raises(ValidationError, match="'status'"):
        tusk.Database("d").add_table(
            "orders",
            pl.LazyFrame(
                {
                    "id": [1],
                    "status": [1],
                    "updated_at": [dt.datetime(2024, 9, 1)],
                },
            ),
            primary_key="id",
            row_update_times={"updated_at": {"status": "pending", "updated_at": None}},
        )


def test_add_table_still_does_not_scan_by_default():
    # unique_primary_key and ordered_row_times both read rows; neither may run.
    tusk.Database("d").add_table(
        "orders",
        pl.LazyFrame(
            {
                "id": [1, 1],
                "created_at": [dt.datetime(2024, 3, 1)] * 2,
                "updated_at": [dt.datetime(2024, 1, 1)] * 2,
            },
        ),
        primary_key="id",
        row_creation_time="created_at",
        row_update_times={"updated_at": {"updated_at": None}},
    )
```

Extend the imports: `DEFAULT_TABLE_CHECKS` and `validate_table` in `tests/test_validation.py`, `ValidationError` in `tests/test_database.py`. Now that `DEFAULT_TABLE_CHECKS` is imported at module level, drop the function-local `from tusk.validation import DEFAULT_TABLE_CHECKS` that Task 6 put inside `test_the_order_check_is_not_in_the_add_table_default`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_validation.py tests/test_database.py -v`
Expected: `ImportError: cannot import name 'DEFAULT_TABLE_CHECKS'`, and the three `add_table` rejection tests fail because nothing raises.

- [ ] **Step 3: Add the default tuple**

In `src/tusk/validation.py`, directly below `TABLE_CHECKS`:

```python
# Every check here answers from the declared schema, so add_table can run all
# of them without reading a row. A check that scans belongs in TABLE_CHECKS
# and not here.
DEFAULT_TABLE_CHECKS = (
    "datetime_row_creation_time",
    "datetime_row_update_times",
    "singly_masked_columns",
    "unmasked_primary_key",
    "unmasked_row_creation_time",
    "matching_fallback_dtypes",
)
```

- [ ] **Step 4: Point add_table at it**

In `src/tusk/database.py`, extend the validation import:

```python
from tusk.validation import (
    DEFAULT_TABLE_CHECKS,
    validate_database,
    validate_relationship,
    validate_table,
)
```

Change `add_table`'s parameter:

```python
        validate: bool | str | Iterable[str] = DEFAULT_TABLE_CHECKS,
```

and its `Args:` entry:

```
            validate: Pick a string or list of strings from
                [here](validation/#tusk.validation.TABLE_CHECKS)
                to enable specific checks.
                `True` runs every check, `False` runs none. The default runs
                every check that reads no rows.
```

Add `ValidationError` to `add_table`'s `Raises:` block if it is not already named there.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_validation.py tests/test_database.py -v`
Expected: PASS, including `test_the_order_check_is_not_in_the_add_table_default` from Task 6.

- [ ] **Step 6: Run the whole suite and lint**

Run: `uv run pytest && just lint`
Expected: all green. If a pre-existing test built a table that trips a new default check, that table was always wrong — fix the fixture, do not pass `validate=False`.

- [ ] **Step 7: Commit**

```bash
git add src/tusk/validation.py src/tusk/database.py tests/test_validation.py tests/test_database.py
git commit -m "feat: run every row-free table check from add_table"
```

---

### Task 8: The schema diagram

**Files:**
- Modify: `src/tusk/plotting.py:294-319` (`describe_comments`), `:44-58` (`from_database` docstring)
- Modify: `src/tusk/database.py` (`Database.plot` docstring)
- Test: `tests/test_plotting.py`

**Interfaces:**
- Consumes: `TableSchema.row_update_times`, `TableSchema.column_updates`.
- Produces: two new comment phrases. No new symbol.

Background: `render_attributes` keeps a column under `columns="structural"` exactly when `describe_role` returns something non-empty, and `describe_role` is markers plus comments. Adding a comment therefore pulls its column into the structural view with no change to the selection code.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_plotting.py`:

```python
@pytest.fixture
def updating_diagram_db():
    """A table whose status is rewritten by an update time."""
    return tusk.Database("shop").add_table(
        "orders",
        pl.LazyFrame(
            {
                "id": [1],
                "status": ["delivered"],
                "note": ["hi"],
                "placed_at": [dt.datetime(2024, 3, 1)],
                "updated_at": [dt.datetime(2024, 9, 1)],
            },
        ),
        primary_key="id",
        row_creation_time="placed_at",
        row_update_times={"updated_at": {"status": "pending", "updated_at": None}},
    )


def test_an_update_time_is_labelled(updating_diagram_db):
    source = SchemaDiagram.from_database(updating_diagram_db, columns=True).source
    assert 'Datetime[us] updated_at "row update time"' in source


def test_an_updated_column_names_its_update_time(updating_diagram_db):
    source = SchemaDiagram.from_database(updating_diagram_db, columns=True).source
    assert 'String status "updated by updated_at"' in source


def test_an_update_time_is_not_labelled_as_updated_by_itself(updating_diagram_db):
    source = SchemaDiagram.from_database(updating_diagram_db, columns=True).source
    assert "updated by updated_at; updated by updated_at" not in source
    assert 'updated_at "row update time; updated by updated_at"' not in source


def test_the_structural_view_keeps_update_times_and_updated_columns(
    updating_diagram_db,
):
    source = SchemaDiagram.from_database(
        updating_diagram_db, columns="structural"
    ).source
    assert "updated_at" in source
    assert "status" in source
    assert "note" not in source
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_plotting.py -k updat -v`
Expected: FAIL — the phrases are absent, and `"note"` is in the structural source only if something is badly wrong (it should already be absent, so that assertion passes from the start).

- [ ] **Step 3: Extend describe_comments**

In `src/tusk/plotting.py`, replace the body of `describe_comments` and widen its docstring's first paragraph:

```python
def describe_comments(
    column: str,
    schema: TableSchema,
    parents: dict[str, list[str]],
) -> list[str]:
    """Describe everything about a column that Mermaid has no marker for.

    Mermaid knows only PK, FK and UK, and its ``classDef`` styling cannot
    target an individual attribute, so the tables a foreign key points at, the
    ``row_creation_time``, and every ``row_update_times`` role all have to
    travel in the comment slot.

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
    if column in schema.row_update_times:
        comments.append("row update time")
    comments.extend(
        f"updated by {update_time}"
        for update_time, updated, _ in schema.column_updates
        # An update time carries its own label already, so naming it as its
        # own updater would only repeat itself.
        if updated == column and update_time != column
    )
    return comments
```

- [ ] **Step 4: Update the two docstrings**

In `src/tusk/plotting.py`, `SchemaDiagram.from_database`:

```
            columns: True lists every column, False lists none, and
                ``"structural"`` lists only the primary key, the foreign keys,
                the ``row_creation_time``, the ``row_update_times``, and the
                columns they update.
```

Make the identical change to `Database.plot`'s `Args:` entry in `src/tusk/database.py`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_plotting.py -v`
Expected: PASS.

- [ ] **Step 6: Run the whole suite and lint**

Run: `uv run pytest && just lint`

- [ ] **Step 7: Commit**

```bash
git add src/tusk/plotting.py src/tusk/database.py tests/test_plotting.py
git commit -m "feat: show row update times in the schema diagram"
```

---

### Task 9: Documentation

**Files:**
- Modify: `docs/guide/databases.md`
- Modify: `docs/guide/deep-feature-synthesis.md:52-80` (the cutoff section)
- Modify: `src/tusk/api.py:53-56` and `:114-118`, `src/tusk/feature_list.py:71`
- Test: `uv run --group docs zensical build --clean`

**Interfaces:** none. Documentation only.

- [ ] **Step 1: Add the guide section**

Append to `docs/guide/databases.md`, after whatever section currently follows the `row_creation_time` material:

````markdown
## Row update times

`row_creation_time` decides whether a **row** is visible at a cutoff. It says
nothing about a **column**. A table that keeps only each row's current state
plus a timestamp of the last edit

| id | placed_at | status | updated_at |
| -- | --------- | ------ | ---------- |
| 1 | 2024-01-05 | delivered | 2024-09-01 |

is fully visible at `cutoff_time=2024-06-01`, and `status` comes back holding a
value written three months after it. `row_update_times` closes that:

```python
db.add_table(
    "orders",
    orders,
    primary_key="id",
    row_creation_time="placed_at",
    row_update_times={
        "updated_at": {"status": "pending", "updated_at": None},
    },
)
```

The outer key names the column recording when the row was last edited. The
inner mapping names the columns that edit rewrote, each mapped to **the value
it held before**. Under a cutoff, a row whose `updated_at` falls after it
serves those earlier values instead of its current ones.

tusk cannot work the earlier value out for you. The table kept only the current
one; `"pending"` above is your knowledge about your own data, not something
tusk can recover. `None` is a legitimate answer — it says the column's earlier
value is unknown — but it is an answer you choose, not a default you fall into.

A null `updated_at` means the row was never edited, so its stored value is its
original one and is visible at every cutoff.

### The update time updates itself

`MAX(orders.updated_at)` on a customer answers "when will this customer's order
next be touched" unless `updated_at` gets an earlier value of its own. tusk
therefore adds `"updated_at": None` to the mapping when you leave it out, and
warns with
[`ImplicitRowUpdateTimeMaskWarning`][tusk.exceptions.ImplicitRowUpdateTimeMaskWarning]
so you can pick a different value. Listing it yourself silences the warning.

### What cannot be updated

The primary key and the `row_creation_time` may not appear in a mapping. The
primary key names the feature matrix's rows, and every visible row was created
at or before the cutoff already, so neither has an earlier value that means
anything. Both are refused by a check that runs by default.

Foreign keys **may** be updated, and doing so is one of the more useful cases:
an order moved to another customer after the cutoff, with its foreign key given
an earlier value of `None`, correctly stops contributing to either customer's
aggregations at that cutoff.

### What tusk cannot see

A column that is overwritten in place and named in no `row_update_times` is
indistinguishable, to tusk, from one that is never touched. It will be read at
face value and it will leak. Nothing in a schema reveals which columns are
rewritten; only you know that.
````

- [ ] **Step 2: Extend the DFS guide**

In `docs/guide/deep-feature-synthesis.md`, after the paragraph beginning "Tables with no `row_creation_time` are timeless", add:

```markdown
A cutoff also rewinds individual columns. A table that declares
[`row_update_times`](databases.md#row-update-times) serves the earlier value of
every column whose update happened after the cutoff, so a status set in
September is not visible from a June cutoff. This happens before any join or
aggregation, so aggregated and stacked features see the rewound values too.
```

- [ ] **Step 3: Amend the cutoff docstrings**

In `src/tusk/api.py`, both `cutoff_time` `Args:` entries currently read "Only rows whose ``row_creation_time`` is at or before this…". Append one sentence to each:

```
                Columns declared in a table's ``row_update_times`` also serve
                the value they held before, wherever the update happened after
                the cutoff.
```

Make the same addition to the `cutoff_time` entry in `src/tusk/feature_list.py`.

- [ ] **Step 4: Build the docs**

Run: `uv run --group docs zensical build --clean`
Expected: build succeeds with no warning about a broken cross-reference. If `[tusk.exceptions.ImplicitRowUpdateTimeMaskWarning]` does not resolve, check how the other exception cross-references in `docs/` are spelled and match them.

- [ ] **Step 5: Run the whole suite and lint**

Run: `uv run pytest && just lint`

- [ ] **Step 6: Commit**

```bash
git add docs src/tusk/api.py src/tusk/feature_list.py
git commit -m "docs: document row update times"
```

---

### Task 10: Differential test against featuretools

**Files:**
- Create: `tests/differential/test_row_update_times.py`
- Test: `uv run --group validation pytest -m differential tests/differential/test_row_update_times.py`

**Interfaces:**
- Consumes: the whole feature, plus `_as_tusk` from `tests/differential/__init__.py`.
- Produces: nothing.

Background: featuretools spells this `secondary_time_index`, taking a mapping of column to list of columns on `EntitySet.add_dataframe` (verified present in featuretools 1.31.0). It always nulls those columns, so the only comparable case is one where every pre-update value is `None`. Model the file's header and imports on `tests/differential/test_aggregations.py`.

- [ ] **Step 1: Write the test**

```python
"""Cross-check row update times against featuretools' secondary time index.

featuretools nulls a secondary-time-index column whose timestamp is after the
cutoff. tusk serves a declared earlier value, so the two agree only where
every declared value is null -- which is the case this file pins.

Run with: uv run --group validation pytest -m differential

Verified against featuretools 1.31.0.
"""

import datetime as dt

import pytest

import tusk
from differential import _as_tusk

pd = pytest.importorskip("pandas")
pl = pytest.importorskip("polars")
featuretools = pytest.importorskip("featuretools")

pytestmark = pytest.mark.differential

CUTOFF = dt.datetime(2024, 6, 1)

CUSTOMERS = {
    "id": [1, 2],
    "signed_up_at": [dt.datetime(2024, 1, 1)] * 2,
}
ORDERS = {
    "id": [10, 11, 12],
    "customer_id": [1, 1, 2],
    "amount": [1.0, 2.0, 4.0],
    "placed_at": [dt.datetime(2024, 3, 1)] * 3,
    "updated_at": [
        dt.datetime(2024, 9, 1),
        dt.datetime(2024, 4, 1),
        dt.datetime(2024, 9, 1),
    ],
}


def _featuretools_matrix():
    """The MEAN of a secondary-time-indexed column, as featuretools computes it."""
    entityset = featuretools.EntitySet("shop")
    entityset.add_dataframe(
        dataframe=pd.DataFrame(CUSTOMERS),
        dataframe_name="customers",
        index="id",
        time_index="signed_up_at",
    )
    entityset.add_dataframe(
        dataframe=pd.DataFrame(ORDERS),
        dataframe_name="orders",
        index="id",
        time_index="placed_at",
        secondary_time_index={"updated_at": ["amount"]},
    )
    entityset.add_relationship("customers", "id", "orders", "customer_id")
    matrix, _ = featuretools.dfs(
        entityset=entityset,
        target_dataframe_name="customers",
        agg_primitives=["mean"],
        trans_primitives=[],
        max_depth=1,
        cutoff_time=CUTOFF,
    )
    return matrix.sort_index()


def _tusk_matrix():
    """The same MEAN, as tusk computes it."""
    database = (
        tusk.Database("shop")
        .add_table(
            "customers",
            pl.LazyFrame(CUSTOMERS),
            primary_key="id",
            row_creation_time="signed_up_at",
        )
        .add_table(
            "orders",
            pl.LazyFrame(ORDERS),
            primary_key="id",
            row_creation_time="placed_at",
            row_update_times={"updated_at": {"amount": None, "updated_at": None}},
        )
        .add_relationship(parent="customers", child="orders", foreign_key="customer_id")
    )
    matrix, _ = tusk.deep_feature_synthesis(
        database=database,
        target_table="customers",
        agg_primitives=["mean"],
        trans_primitives=[],
        max_depth=1,
        cutoff_time=CUTOFF,
    )
    return matrix.collect().to_native().sort("id")


def test_a_null_pre_update_value_matches_a_secondary_time_index():
    expected = _featuretools_matrix()
    got = _tusk_matrix()
    column = _as_tusk("MEAN(orders.amount)")
    assert got[column].to_list() == pytest.approx(
        expected["MEAN(orders.amount)"].tolist(), nan_ok=True
    )
```

- [ ] **Step 2: Run it**

Run: `uv run --group validation pytest -m differential tests/differential/test_row_update_times.py -v`
Expected: PASS.

If it fails, read the difference before touching `src/`. The two known-benign mismatches are the null-versus-NaN spelling of an empty mean (handled by `nan_ok=True`) and row ordering (handled by `sort_index()` / `sort("id")`). A genuine value disagreement is a bug in Task 2 — take it to `superpowers:systematic-debugging`, not to a loosened assertion.

- [ ] **Step 3: Run the full gate**

Run: `just check`
Expected: lint, the default suite, and the whole differential suite all green.

- [ ] **Step 4: Commit**

```bash
git add tests/differential/test_row_update_times.py
git commit -m "test: cross-check row update times against featuretools"
```

---

## Done criteria

- `just check` is green.
- `uv run --group docs zensical build --clean` is green.
- `tusk.Database.add_table` accepts `row_update_times` and a cutoff rewinds every declared column before any join.
- Six new names in `TABLE_CHECKS`; five of them in `DEFAULT_TABLE_CHECKS`.
- Issue #15 is answered in full: the fallback-per-column requirement, the warning when an update time says nothing about itself, and the check for a column updated under two update times.
