# Standalone and Ordered Transforms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship sixteen transform primitives, guard `natural_log` against negative inputs, test the ❓ transform rows against featuretools, and bring the coverage table up to date.

**Architecture:** Each primitive is a frozen dataclass in `src/tusk/primitives/transform.py` that builds one narwhals expression; ordered primitives set `order_dependent = True` and the existing compiler wraps them in `.over(order_by=(row_creation_time, primary_key))`. One shared test module holds a hand-built eight-row table and every expected value, so the polars suite, the duckdb suite and the differential suite all check the same rows.

**Tech Stack:** Python 3.10+, narwhals 2.24, polars and duckdb in tests, featuretools 1.31.0 in the opt-in differential suite, pytest, ruff + ty + pydoclint + interrogate via pre-commit, roborev reviewing every commit.

**Spec:** `agents/superpowers/specs/2026-09-14-standalone-and-ordered-transforms-design.md` (roadmap: `agents/superpowers/specs/2026-09-13-primitive-parity-roadmap-design.md`)

## Global Constraints

- Work in the worktree `/home/excidion/dev/tusk/.claude/worktrees/standalone-ordered-transforms` on branch `worktree-standalone-ordered-transforms`.
- Plans and specs live under `agents/superpowers/`, never `docs/` (`agents/AGENTS.md`).
- Every public **and private** function in `src/` needs a Google-style docstring with `Args:` and `Returns:` sections matching the signature; `interrogate` demands 100% and `pydoclint` cross-checks. Tests are exempt.
- Docstrings say *what*; a comment says *why*, and only when naming cannot (`agents/style/CODESTYLE.md`). Callers above callees.
- Every primitive must give the same value on lazy polars and duckdb.
- Run tests with `uv run pytest`. Never pass `--active` to `uv`.
- Differential tests: `uv run --group validation pytest -m differential`.
- Full gate before the branch is done: `just check`.
- Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_018ztwcZ16kzzv4xDHNBhfTP
  ```
- roborev reviews every commit in the background (post-commit hook). After each commit, run `roborev wait`. Exit 0 means the review passed. Exit 1 means it failed: read it with `roborev show`, fix the findings with the `roborev-fix` skill, commit, and run `roborev wait` again. A task is done only when its last commit's review passes.

## The shared table

Eight rows. `occurred_at` is the `row_creation_time`, so the ordered primitives
see the rows in the order **3, 1, 5, 2, 8, 4, 7, 6**. Values below are listed by
`id` 1–8.

| Column | By id 1–8 | Covers |
| --- | --- | --- |
| `occurred_at` | 2024-02-29 13:45:30, 2024-03-02 08:00:05, 2023-12-31 23:59:59, 2024-03-10, 2024-03-01 06:30, 2024-03-20 18:15:45, 2024-03-15 12:00, 2024-03-05 09:09:09 | order differs from id order |
| `value` | null, 0, 4, 9, −1, 2, 2, 0 | null, zero, negative, ties, zero after zero |
| `flag` | T, F, F, F, F, T, F, T | a false before the first true |
| `maybe_flag` | null, T, T, F, F, T, F, null | a null flag |
| `due_at` | 2024-02-29 13:45:30, null, 1900-03-01 23:59:59, 2000-12-31, 2023-06-15 07:08:09, 2100-01-01 12:30, 2024-12-31 23:00:01, 2019-01-01 | null, leap and non-leap century years, day 366 |
| `label` | a, null, b, a, b, a, null, b | null labels for `cum_count` |

Every expected value in this plan was probed on lazy polars and duckdb (identical
except the last floating-point digit of `percentile`) and on featuretools 1.31.0.

---

### Task 1: Shared cases and the elementwise numeric primitives

**Files:**
- Create: `tests/transform_cases.py`
- Modify: `src/tusk/primitives/transform.py` (change `NaturalLog`; add classes after it)
- Modify: `src/tusk/primitives/__init__.py`
- Test: `tests/test_primitives_transform.py`, `tests/test_backend_duckdb.py`

**Interfaces:**
- Produces:
  - `tests/transform_cases.py`: `ROWS: pd.DataFrame`; `EXPECTED: dict[str, tuple[str, nw.dtypes.DType, list]]` keyed by feature column → `(primitive name, output dtype, value per id 1..8)`; `rows_database(table) -> tusk.Database`; `feature_values(matrix, column) -> list`; `assert_values_match(got: list, expected: list) -> None`.
  - Classes `IsNull`, `Negate`, `SquareRoot`, `Sine`, `Cosine` in `tusk.primitives`.

- [ ] **Step 1: Create the shared cases module**

`tests/transform_cases.py`:

```python
"""One table and the value every standalone and ordered transform gives on it.

Shared by the polars suite, the duckdb suite and the differential suite, so
all three check the same rows. ``occurred_at`` is the row creation time and
orders the rows 3, 1, 5, 2, 8, 4, 7, 6, which differs from id order.
"""

import datetime as dt
import math

import narwhals as nw
import pandas as pd
import pyarrow as pa
import pytest

import tusk

ROWS = pd.DataFrame(
    {
        "id": range(1, 9),
        "occurred_at": pd.to_datetime(
            [
                dt.datetime(2024, 2, 29, 13, 45, 30),
                dt.datetime(2024, 3, 2, 8, 0, 5),
                dt.datetime(2023, 12, 31, 23, 59, 59),
                dt.datetime(2024, 3, 10),
                dt.datetime(2024, 3, 1, 6, 30),
                dt.datetime(2024, 3, 20, 18, 15, 45),
                dt.datetime(2024, 3, 15, 12),
                dt.datetime(2024, 3, 5, 9, 9, 9),
            ],
        ),
        "value": [None, 0.0, 4.0, 9.0, -1.0, 2.0, 2.0, 0.0],
        "flag": [True, False, False, False, False, True, False, True],
        "maybe_flag": pd.array(
            [None, True, True, False, False, True, False, None],
            dtype="boolean",
        ),
        "due_at": pd.to_datetime(
            [
                dt.datetime(2024, 2, 29, 13, 45, 30),
                None,
                dt.datetime(1900, 3, 1, 23, 59, 59),
                dt.datetime(2000, 12, 31),
                dt.datetime(2023, 6, 15, 7, 8, 9),
                dt.datetime(2100, 1, 1, 12, 30),
                dt.datetime(2024, 12, 31, 23, 0, 1),
                dt.datetime(2019, 1, 1),
            ],
        ),
        "label": pd.array(
            ["a", None, "b", "a", "b", "a", None, "b"],
            dtype="string",
        ),
    },
)

EXPECTED = {
    "IS_NULL__value": (
        "is_null",
        nw.Boolean,
        [True, False, False, False, False, False, False, False],
    ),
    "NEGATE__value": (
        "negate",
        nw.Float64,
        [None, -0.0, -4.0, -9.0, 1.0, -2.0, -2.0, -0.0],
    ),
    "SQUARE_ROOT__value": (
        "square_root",
        nw.Float64,
        [None, 0.0, 2.0, 3.0, None, math.sqrt(2), math.sqrt(2), 0.0],
    ),
    "NATURAL_LOG__value": (
        "natural_log",
        nw.Float64,
        [None, -math.inf, math.log(4), math.log(9), None, math.log(2), math.log(2), -math.inf],
    ),
    "SINE__value": (
        "sine",
        nw.Float64,
        [None, 0.0, math.sin(4), math.sin(9), math.sin(-1), math.sin(2), math.sin(2), 0.0],
    ),
    "COSINE__value": (
        "cosine",
        nw.Float64,
        [None, 1.0, math.cos(4), math.cos(9), math.cos(-1), math.cos(2), math.cos(2), 1.0],
    ),
}


def rows_database(table):
    """Wrap the shared table in a one-table database.

    Args:
        table: ``ROWS`` as a native lazy frame of any backend.

    Returns:
        A database whose ``rows`` table is ordered by ``occurred_at``.
    """
    return tusk.Database("rows").add_table(
        "rows",
        table,
        primary_key="id",
        row_creation_time="occurred_at",
    )


def feature_values(matrix, column):
    """Materialize one feature column in id order.

    Args:
        matrix: A native feature matrix from ``deep_feature_synthesis``.
        column: The feature column to read.

    Returns:
        The column's values as plain Python objects, null as None.
    """
    return nw.from_native(matrix).collect().sort("id")[column].to_list()


def assert_values_match(got, expected):
    """Compare one feature column against its expected values, row by row.

    A null and a NaN are different values here: ``square_root`` of a negative
    number must be null, not NaN.

    Args:
        got: The column as a list, in id order.
        expected: The expected values; a float is compared approximately.
    """
    assert len(got) == len(expected)
    for actual, wanted in zip(got, expected, strict=True):
        actual = _as_timedelta(actual)
        if wanted is None:
            assert actual is None
        elif isinstance(wanted, float) and math.isnan(wanted):
            assert isinstance(actual, float) and math.isnan(actual)
        elif isinstance(wanted, float):
            assert actual == pytest.approx(wanted)
        else:
            assert actual == wanted


def _as_timedelta(value):
    """Convert duckdb's interval value into a timedelta.

    Args:
        value: One materialized value.

    Returns:
        A timedelta for an interval, else the value unchanged.
    """
    if not isinstance(value, pa.MonthDayNano):
        return value
    assert value.months == 0
    return dt.timedelta(days=value.days, microseconds=value.nanoseconds // 1000)
```

- [ ] **Step 2: Write the failing polars test**

Add to the imports of `tests/test_primitives_transform.py`:

```python
from transform_cases import (
    EXPECTED,
    ROWS,
    assert_values_match,
    feature_values,
    rows_database,
)
```

Append:

```python
@pytest.mark.parametrize("column", sorted(EXPECTED))
def test_transforms_give_the_expected_value_on_every_row(column):
    primitive_name, dtype, expected = EXPECTED[column]
    matrix, _ = tusk.deep_feature_synthesis(
        database=rows_database(pl.from_pandas(ROWS).lazy()),
        target_table="rows",
        agg_primitives=[],
        trans_primitives=[primitive_name],
        max_depth=1,
    )
    assert nw.from_native(matrix).collect_schema()[column] == dtype
    assert_values_match(feature_values(matrix, column), expected)


@pytest.mark.parametrize(
    ("dtype", "values", "expected"),
    [
        (pl.Int8, [-128, 1], [128.0, -1.0]),
        (pl.UInt32, [0, 4294967295], [-0.0, -4294967295.0]),
    ],
)
def test_negate_does_not_wrap_around_an_integer_dtype(dtype, values, expected):
    frame = nw.from_native(pl.LazyFrame({"v": values}, schema={"v": dtype}))
    assert _apply(frame, "negate", "v") == expected
```

- [ ] **Step 3: Write the failing duckdb test**

Add to the imports of `tests/test_backend_duckdb.py`:

```python
from transform_cases import (
    EXPECTED as TRANSFORM_EXPECTED,
    ROWS,
    assert_values_match as assert_transform_values_match,
    feature_values,
    rows_database,
)

from tusk.primitives import Negate, resolve
```

Append:

```python
@pytest.mark.parametrize("column", sorted(TRANSFORM_EXPECTED))
def test_transforms_give_the_polars_values_on_duckdb(column):
    """Every standalone and ordered transform survives translation to SQL, row by row.

    ``square_root`` and ``natural_log`` pin the negative-input guard, which
    polars would otherwise answer with NaN, and the ordered primitives pin
    the ordering by ``occurred_at``.

    Args:
        column: The feature column under test.
    """
    primitive_name, _, expected = TRANSFORM_EXPECTED[column]
    con = duckdb.connect()
    con.register("rows_frame", ROWS)
    matrix, _ = tusk.deep_feature_synthesis(
        database=rows_database(con.sql("SELECT * FROM rows_frame")),
        target_table="rows",
        agg_primitives=[],
        trans_primitives=[primitive_name],
        max_depth=1,
    )
    assert_transform_values_match(feature_values(matrix, column), expected)


def test_negate_does_not_overflow_an_integer_dtype_on_duckdb():
    """duckdb raises on negating TINYINT's minimum and wraps an unsigned value."""
    con = duckdb.connect()
    frame = nw.from_native(
        con.sql(
            "SELECT * FROM (VALUES (-128::TINYINT, 4294967295::UINTEGER)) "
            "t(small, unsigned)",
        ),
    )
    negate = Negate()
    got = frame.select(
        negate.outputs(nw.col("small"))[0].alias("small"),
        negate.outputs(nw.col("unsigned"))[0].alias("unsigned"),
    ).collect()
    assert got["small"].to_list() == [128.0]
    assert got["unsigned"].to_list() == [-4294967295.0]
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run pytest tests/test_primitives_transform.py tests/test_backend_duckdb.py -k "every_row or on_duckdb or negate" -v`
Expected: the `is_null`, `negate`, `square_root`, `sine` and `cosine` cases FAIL with `PrimitiveError` (unknown primitive name); `NATURAL_LOG__value` FAILs on polars at id 5 (NaN, not None) and passes on duckdb.

- [ ] **Step 5: Implement**

In `src/tusk/primitives/transform.py`, replace `NaturalLog` and add the new classes directly after it:

```python
@register
@dataclass(frozen=True)
class NaturalLog(TransformPrimitive):
    """Natural logarithm. A negative input gives null; zero gives negative infinity."""

    name = "natural_log"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the natural-logarithm expression.

        Args:
            expr: A numeric expression.

        Returns:
            A narwhals expression of natural logarithms.
        """
        # polars answers NaN for a negative input, duckdb null
        return nw.when(expr >= 0).then(expr.log())


@register
@dataclass(frozen=True)
class IsNull(TransformPrimitive):
    """Whether the value is null."""

    name = "is_null"
    input_dtypes = (F.ANY,)
    output_dtype = nw.Boolean

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the null-indicator expression.

        Args:
            expr: An expression of any dtype.

        Returns:
            A narwhals boolean expression.
        """
        return expr.is_null()


@register
@dataclass(frozen=True)
class Negate(TransformPrimitive):
    """The value with its sign flipped, as a float."""

    name = "negate"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the negation expression.

        Args:
            expr: A numeric expression.

        Returns:
            A narwhals expression of the negated values.
        """
        # an integer dtype's minimum has no negation inside the same dtype
        return expr.cast(nw.Float64) * -1


@register
@dataclass(frozen=True)
class SquareRoot(TransformPrimitive):
    """Square root. A negative input gives null."""

    name = "square_root"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the square-root expression.

        Args:
            expr: A numeric expression.

        Returns:
            A narwhals expression of square roots.
        """
        # polars answers NaN for a negative input, duckdb null
        return nw.when(expr >= 0).then(expr.sqrt())


@register
@dataclass(frozen=True)
class Sine(TransformPrimitive):
    """Sine of a value in radians."""

    name = "sine"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the sine expression.

        Args:
            expr: A numeric expression in radians.

        Returns:
            A narwhals expression of sines.
        """
        return expr.sin()


@register
@dataclass(frozen=True)
class Cosine(TransformPrimitive):
    """Cosine of a value in radians."""

    name = "cosine"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the cosine expression.

        Args:
            expr: A numeric expression in radians.

        Returns:
            A narwhals expression of cosines.
        """
        return expr.cos()
```

In `src/tusk/primitives/__init__.py`, add `Cosine`, `IsNull`, `Negate`, `Sine` and `SquareRoot` to the `from tusk.primitives.transform import (...)` block and to `__all__`, keeping both alphabetical.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_primitives_transform.py tests/test_backend_duckdb.py -v`
Expected: PASS, including every earlier test in both files.

- [ ] **Step 7: Commit and wait for the review**

```bash
git add tests/transform_cases.py tests/test_primitives_transform.py tests/test_backend_duckdb.py src/tusk/primitives/transform.py src/tusk/primitives/__init__.py
git commit -m "feat: add is_null, negate, square_root, sine and cosine

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_018ztwcZ16kzzv4xDHNBhfTP"
roborev wait
```

Expected: `roborev wait` exits 0. On exit 1, follow the roborev rule in Global Constraints.

---

### Task 2: Datetime parts

**Files:**
- Modify: `src/tusk/primitives/transform.py` (add classes after `IsWeekend`)
- Modify: `src/tusk/primitives/__init__.py`
- Modify: `tests/transform_cases.py` (extend `EXPECTED`)
- Test: `tests/test_primitives_transform.py`, `tests/test_backend_duckdb.py`

**Interfaces:**
- Consumes: `EXPECTED`, `rows_database`, `feature_values`, `assert_values_match` from Task 1; the parametrized tests from Task 1 pick the new entries up.
- Produces: classes `Minute`, `Second`, `DayOfYear`, `IsLeapYear` in `tusk.primitives`.

- [ ] **Step 1: Extend the expected values**

Add to `EXPECTED` in `tests/transform_cases.py`:

```python
    "MINUTE__due_at": ("minute", nw.Int8, [45, None, 59, 0, 8, 30, 0, 0]),
    "SECOND__due_at": ("second", nw.Int8, [30, None, 59, 0, 9, 0, 1, 0]),
    "DAY_OF_YEAR__due_at": (
        "day_of_year",
        nw.Int16,
        [60, None, 60, 366, 166, 1, 366, 1],
    ),
    "IS_LEAP_YEAR__due_at": (
        "is_leap_year",
        nw.Boolean,
        [True, None, False, True, False, False, True, False],
    ),
```

- [ ] **Step 2: Write the failing time-of-day tests**

Append to `tests/test_primitives_transform.py`:

```python
@pytest.mark.parametrize(("name", "expected"), [("minute", [2, None]), ("second", [3, None])])
def test_minute_and_second_read_a_time_column(name, expected):
    frame = nw.from_native(pl.LazyFrame({"at": [dt.time(1, 2, 3), None]}))
    assert _apply(frame, name, "at") == expected
```

Append to `tests/test_backend_duckdb.py`:

```python
@pytest.mark.parametrize(("name", "expected"), [("minute", [2, None]), ("second", [3, None])])
def test_minute_and_second_read_a_time_column_on_duckdb(name, expected):
    con = duckdb.connect()
    frame = nw.from_native(
        con.sql("SELECT * FROM (VALUES (1, TIME '01:02:03'), (2, NULL)) t(id, at)"),
    )
    primitive = resolve(name)
    got = frame.select("id", primitive.outputs(nw.col("at"))[0].alias("o"))
    assert got.collect().sort("id")["o"].to_list() == expected
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_primitives_transform.py tests/test_backend_duckdb.py -k "due_at or time_column" -v`
Expected: FAIL with `PrimitiveError` (unknown primitive name).

- [ ] **Step 4: Implement**

Add after `IsWeekend` in `src/tusk/primitives/transform.py`:

```python
@register
@dataclass(frozen=True)
class Minute(TransformPrimitive):
    """Minute of the hour, 0-59."""

    name = "minute"
    input_dtypes = (F.HAS_TIME,)
    output_dtype = nw.Int8

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the minute-of-hour expression.

        Args:
            expr: A temporal expression.

        Returns:
            A narwhals expression of the minute of the hour.
        """
        return expr.dt.minute()


@register
@dataclass(frozen=True)
class Second(TransformPrimitive):
    """Second of the minute, 0-59."""

    name = "second"
    input_dtypes = (F.HAS_TIME,)
    output_dtype = nw.Int8

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the second-of-minute expression.

        Args:
            expr: A temporal expression.

        Returns:
            A narwhals expression of the second of the minute.
        """
        return expr.dt.second()


@register
@dataclass(frozen=True)
class DayOfYear(TransformPrimitive):
    """Day of the year, 1-366."""

    name = "day_of_year"
    input_dtypes = (F.HAS_DATE,)
    output_dtype = nw.Int16

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the day-of-year expression.

        Args:
            expr: A temporal expression.

        Returns:
            A narwhals expression of the day of the year.
        """
        return expr.dt.ordinal_day()


@register
@dataclass(frozen=True)
class IsLeapYear(TransformPrimitive):
    """Whether the date falls in a leap year. A null date gives null."""

    name = "is_leap_year"
    input_dtypes = (F.HAS_DATE,)
    output_dtype = nw.Boolean

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the leap-year expression.

        Args:
            expr: A temporal expression.

        Returns:
            A narwhals boolean expression.
        """
        year = expr.dt.year().cast(nw.Int32)
        return ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)
```

In `src/tusk/primitives/__init__.py`, add `DayOfYear`, `IsLeapYear`, `Minute` and `Second` to the transform import block and `__all__`, alphabetically.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_primitives_transform.py tests/test_backend_duckdb.py -v`
Expected: PASS.

- [ ] **Step 6: Commit and wait for the review**

```bash
git add tests/transform_cases.py tests/test_primitives_transform.py tests/test_backend_duckdb.py src/tusk/primitives/transform.py src/tusk/primitives/__init__.py
git commit -m "feat: add minute, second, day_of_year and is_leap_year

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_018ztwcZ16kzzv4xDHNBhfTP"
roborev wait
```

Expected: `roborev wait` exits 0. On exit 1, follow the roborev rule in Global Constraints.

---

### Task 3: Percentile

**Files:**
- Modify: `src/tusk/primitives/transform.py` (add the class after `Cosine`)
- Modify: `src/tusk/primitives/__init__.py`
- Modify: `tests/transform_cases.py` (extend `EXPECTED`)
- Test: `tests/test_primitives_transform.py`

**Interfaces:**
- Consumes: the shared cases from Task 1.
- Produces: class `Percentile` in `tusk.primitives`.

- [ ] **Step 1: Extend the expected values**

Add to `EXPECTED`:

```python
    "PERCENTILE__value": (
        "percentile",
        nw.Float64,
        [None, 5 / 14, 12 / 14, 1.0, 2 / 14, 9 / 14, 9 / 14, 5 / 14],
    ),
```

- [ ] **Step 2: Write the failing group test**

Append to `tests/test_primitives_transform.py`:

```python
def test_percentile_ranks_within_each_group():
    """Inside groupby_trans_primitives the rank is taken per foreign key."""
    database = (
        tusk.Database("groups")
        .add_table("parents", pl.LazyFrame({"id": [1, 2]}), primary_key="id")
        .add_table(
            "children",
            pl.LazyFrame(
                {
                    "id": [1, 2, 3, 4, 5],
                    "parent_id": [1, 1, 1, 2, 2],
                    "amount": [1.0, 3.0, 3.0, 5.0, None],
                },
            ),
            primary_key="id",
        )
        .add_relationship(parent="parents", child="children", foreign_key="parent_id")
    )
    matrix, _ = tusk.deep_feature_synthesis(
        database=database,
        target_table="children",
        agg_primitives=[],
        trans_primitives=[],
        groupby_trans_primitives=["percentile"],
        max_depth=1,
    )
    assert_values_match(
        feature_values(matrix, "PERCENTILE__amount__by__parent_id"),
        [1 / 3, 2.5 / 3, 2.5 / 3, 1.0, None],
    )
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_primitives_transform.py tests/test_backend_duckdb.py -k percentile -v`
Expected: FAIL with `PrimitiveError` (unknown primitive name).

- [ ] **Step 4: Implement**

Add after `Cosine` in `src/tusk/primitives/transform.py`:

```python
@register
@dataclass(frozen=True)
class Percentile(TransformPrimitive):
    """Rank of the value among the known values, from above 0 to 1.

    Tied values share their average rank. A null stays null and is not
    counted.
    """

    name = "percentile"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the percentile-rank expression.

        Args:
            expr: A numeric expression.

        Returns:
            A narwhals expression of percentile ranks.
        """
        return expr.rank("average") / expr.count()
```

In `src/tusk/primitives/__init__.py`, add `Percentile` to the transform import block and `__all__`, alphabetically.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_primitives_transform.py tests/test_backend_duckdb.py -v`
Expected: PASS.

- [ ] **Step 6: Commit and wait for the review**

```bash
git add tests/transform_cases.py tests/test_primitives_transform.py src/tusk/primitives/transform.py src/tusk/primitives/__init__.py
git commit -m "feat: add percentile

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_018ztwcZ16kzzv4xDHNBhfTP"
roborev wait
```

Expected: `roborev wait` exits 0. On exit 1, follow the roborev rule in Global Constraints.

---

### Task 4: Ordered numeric transforms

**Files:**
- Modify: `src/tusk/primitives/transform.py` (add classes after `TimeSincePrevious`)
- Modify: `src/tusk/primitives/__init__.py`
- Modify: `tests/transform_cases.py` (extend `EXPECTED`)
- Test: `tests/test_primitives_transform.py`

**Interfaces:**
- Consumes: the shared cases from Task 1.
- Produces: classes `CumMean`, `SameAsPrevious`, `AbsoluteDiff`, `PercentChange` in `tusk.primitives`.

- [ ] **Step 1: Extend the expected values**

In time order (ids 3, 1, 5, 2, 8, 4, 7, 6) `value` is 4, null, −1, 0, 0, 9, 2, 2. Add to `EXPECTED`:

```python
    "CUM_MEAN__value": (
        "cum_mean",
        nw.Float64,
        [None, 1.0, 4.0, 2.4, 1.5, 16 / 7, 7 / 3, 0.75],
    ),
    "SAME_AS_PREVIOUS__value": (
        "same_as_previous",
        nw.Boolean,
        [None, False, None, False, None, True, False, True],
    ),
    "ABSOLUTE_DIFF__value": (
        "absolute_diff",
        nw.Float64,
        [None, 1.0, None, 9.0, None, 0.0, 7.0, 0.0],
    ),
    "PERCENT_CHANGE__value": (
        "percent_change",
        nw.Float64,
        [None, -1.0, None, math.inf, None, 0.0, -7 / 9, math.nan],
    ),
```

- [ ] **Step 2: Write the failing flag test**

Append to `tests/test_primitives_transform.py`:

```python
@pytest.mark.parametrize(
    "name",
    ["cum_mean", "same_as_previous", "absolute_diff", "percent_change"],
)
def test_ordered_numeric_transforms_are_flagged(name):
    primitive = resolve(name)
    assert isinstance(primitive, TransformPrimitive)
    assert primitive.order_dependent is True
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_primitives_transform.py tests/test_backend_duckdb.py -k "CUM_MEAN or SAME_AS or ABSOLUTE_DIFF or PERCENT_CHANGE or ordered_numeric" -v`
Expected: FAIL with `PrimitiveError` (unknown primitive name).

- [ ] **Step 4: Implement**

Add after `TimeSincePrevious` in `src/tusk/primitives/transform.py`:

```python
@register
@dataclass(frozen=True)
class CumMean(TransformPrimitive):
    """Running mean of the known values in row-creation order. A null row is null."""

    name = "cum_mean"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64
    order_dependent = True

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the cumulative-mean expression.

        Args:
            expr: A numeric expression.

        Returns:
            A narwhals expression of the running mean.
        """
        return expr.cum_sum() / expr.cum_count()


@register
@dataclass(frozen=True)
class SameAsPrevious(TransformPrimitive):
    """Whether the value equals the previous row's in row-creation order.

    The first row, and a row whose own or previous value is null, is null.
    """

    name = "same_as_previous"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Boolean
    order_dependent = True

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the equal-to-previous expression.

        Args:
            expr: A numeric expression.

        Returns:
            A narwhals boolean expression.
        """
        return expr == expr.shift(1)


@register
@dataclass(frozen=True)
class AbsoluteDiff(TransformPrimitive):
    """Size of the change from the previous row in row-creation order.

    The first row, and a row whose own or previous value is null, is null.
    """

    name = "absolute_diff"
    input_dtypes = (F.NUMERIC,)
    order_dependent = True

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the absolute-difference expression.

        Args:
            expr: A numeric expression.

        Returns:
            A narwhals expression of absolute differences.
        """
        return expr.diff().abs()


@register
@dataclass(frozen=True)
class PercentChange(TransformPrimitive):
    """Relative change from the previous row in row-creation order.

    The first row, and a row whose own or previous value is null, is null. A
    zero previous value gives infinity, or NaN when the value is zero too.
    """

    name = "percent_change"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64
    order_dependent = True

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the relative-change expression.

        Args:
            expr: A numeric expression.

        Returns:
            A narwhals expression of relative changes, where 0.5 is a 50% rise.
        """
        return expr / expr.shift(1) - 1
```

In `src/tusk/primitives/__init__.py`, add `AbsoluteDiff`, `CumMean`, `PercentChange` and `SameAsPrevious` to the transform import block and `__all__`, alphabetically.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_primitives_transform.py tests/test_backend_duckdb.py -v`
Expected: PASS.

- [ ] **Step 6: Commit and wait for the review**

```bash
git add tests/transform_cases.py tests/test_primitives_transform.py src/tusk/primitives/transform.py src/tusk/primitives/__init__.py
git commit -m "feat: add cum_mean, same_as_previous, absolute_diff and percent_change

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_018ztwcZ16kzzv4xDHNBhfTP"
roborev wait
```

Expected: `roborev wait` exits 0. On exit 1, follow the roborev rule in Global Constraints.

---

### Task 5: Cumulative time since the last true or false

**Files:**
- Modify: `src/tusk/primitives/transform.py` (add classes after `PercentChange`, and a module-level helper at the end of the file)
- Modify: `src/tusk/primitives/__init__.py`
- Modify: `tests/transform_cases.py` (extend `EXPECTED`)
- Test: `tests/test_backend_duckdb.py`

**Interfaces:**
- Consumes: the shared cases from Task 1.
- Produces: classes `CumulativeTimeSinceLastTrue`, `CumulativeTimeSinceLastFalse` in `tusk.primitives`.

- [ ] **Step 1: Extend the expected values**

`td` is `dt.timedelta`. Add to `EXPECTED`:

```python
    "CUMULATIVE_TIME_SINCE_LAST_TRUE__occurred_at__flag": (
        "cumulative_time_since_last_true",
        nw.Duration,
        [
            dt.timedelta(0),
            dt.timedelta(days=1, seconds=65675),
            None,
            dt.timedelta(days=4, seconds=53451),
            dt.timedelta(seconds=60270),
            dt.timedelta(0),
            dt.timedelta(days=10, seconds=10251),
            dt.timedelta(0),
        ],
    ),
    "CUMULATIVE_TIME_SINCE_LAST_FALSE__occurred_at__flag": (
        "cumulative_time_since_last_false",
        nw.Duration,
        [
            dt.timedelta(days=59, seconds=49531),
            dt.timedelta(0),
            dt.timedelta(0),
            dt.timedelta(0),
            dt.timedelta(0),
            dt.timedelta(days=5, seconds=22545),
            dt.timedelta(0),
            dt.timedelta(days=3, seconds=4144),
        ],
    ),
    "CUMULATIVE_TIME_SINCE_LAST_TRUE__occurred_at__maybe_flag": (
        "cumulative_time_since_last_true",
        nw.Duration,
        [
            dt.timedelta(days=59, seconds=49531),
            dt.timedelta(0),
            dt.timedelta(0),
            dt.timedelta(days=7, seconds=57595),
            dt.timedelta(days=60, seconds=23401),
            dt.timedelta(0),
            dt.timedelta(days=13, seconds=14395),
            dt.timedelta(days=3, seconds=4144),
        ],
    ),
    "CUMULATIVE_TIME_SINCE_LAST_FALSE__occurred_at__maybe_flag": (
        "cumulative_time_since_last_false",
        nw.Duration,
        [
            None,
            dt.timedelta(days=1, seconds=5405),
            None,
            dt.timedelta(0),
            dt.timedelta(0),
            dt.timedelta(days=5, seconds=22545),
            dt.timedelta(0),
            dt.timedelta(days=4, seconds=9549),
        ],
    ),
```

- [ ] **Step 2: Write the failing Date test**

Append to `tests/test_backend_duckdb.py`:

```python
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "cumulative_time_since_last_true",
            [dt.timedelta(0), dt.timedelta(days=4), dt.timedelta(0)],
        ),
        (
            "cumulative_time_since_last_false",
            [None, dt.timedelta(0), dt.timedelta(days=4)],
        ),
    ],
)
def test_cumulative_time_since_measures_a_date_column_on_duckdb(name, expected):
    """duckdb subtracts two DATEs into a day count, so the primitive casts first.

    Args:
        name: The primitive under test.
        expected: The elapsed time per row, in id order.
    """
    con = duckdb.connect()
    frame = nw.from_native(
        con.sql(
            "SELECT * FROM (VALUES (1, DATE '2024-01-01', TRUE), "
            "(2, DATE '2024-01-05', FALSE), (3, DATE '2024-01-09', TRUE)) "
            "t(id, d, f)",
        ),
    )
    primitive = resolve(name)
    elapsed = primitive.outputs(nw.col("d"), nw.col("f"))[0].over(order_by="id")
    got = frame.select("id", elapsed.alias("o")).collect().sort("id")["o"].to_list()
    assert_transform_values_match(got, expected)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_primitives_transform.py tests/test_backend_duckdb.py -k "CUMULATIVE or date_column" -v`
Expected: FAIL with `PrimitiveError` (unknown primitive name).

- [ ] **Step 4: Implement**

Add after `PercentChange` in `src/tusk/primitives/transform.py`:

```python
@register
@dataclass(frozen=True)
class CumulativeTimeSinceLastTrue(TransformPrimitive):
    """Time elapsed since the latest row whose flag is true, in row-creation order.

    Null until the first true flag, and on a row whose datetime is null. A
    null flag is not true.
    """

    name = "cumulative_time_since_last_true"
    input_dtypes = (F.HAS_DATE, F.BOOLEAN)
    output_dtype = nw.Duration
    order_dependent = True

    def build(self, moment: nw.Expr, flag: nw.Expr) -> nw.Expr:
        """Build the time-since-last-true expression.

        Args:
            moment: The datetime of each row.
            flag: The boolean flag of each row.

        Returns:
            A narwhals expression of the elapsed time.
        """
        return _time_since_last_match(moment, flag)


@register
@dataclass(frozen=True)
class CumulativeTimeSinceLastFalse(TransformPrimitive):
    """Time elapsed since the latest row whose flag is false, in row-creation order.

    Null until the first false flag, and on a row whose datetime is null. A
    null flag is not false.
    """

    name = "cumulative_time_since_last_false"
    input_dtypes = (F.HAS_DATE, F.BOOLEAN)
    output_dtype = nw.Duration
    order_dependent = True

    def build(self, moment: nw.Expr, flag: nw.Expr) -> nw.Expr:
        """Build the time-since-last-false expression.

        Args:
            moment: The datetime of each row.
            flag: The boolean flag of each row.

        Returns:
            A narwhals expression of the elapsed time.
        """
        return _time_since_last_match(moment, ~flag)
```

Append at the very end of `src/tusk/primitives/transform.py`:

```python
def _time_since_last_match(moment: nw.Expr, is_match: nw.Expr) -> nw.Expr:
    """Build the time elapsed since the latest row that matches.

    Args:
        moment: The datetime of each row.
        is_match: Where the row matches; a null does not match.

    Returns:
        A narwhals expression of the elapsed time, to be ordered by the caller.
    """
    # duckdb subtracts two Dates into a day count instead of an interval
    timestamps = moment.cast(nw.Datetime)
    latest_match = nw.when(is_match).then(timestamps).fill_null(strategy="forward")
    return timestamps - latest_match
```

In `src/tusk/primitives/__init__.py`, add `CumulativeTimeSinceLastFalse` and `CumulativeTimeSinceLastTrue` to the transform import block and `__all__`, alphabetically.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_primitives_transform.py tests/test_backend_duckdb.py -v`
Expected: PASS.

- [ ] **Step 6: Commit and wait for the review**

```bash
git add tests/transform_cases.py tests/test_backend_duckdb.py src/tusk/primitives/transform.py src/tusk/primitives/__init__.py
git commit -m "feat: add cumulative_time_since_last_true and _false

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_018ztwcZ16kzzv4xDHNBhfTP"
roborev wait
```

Expected: `roborev wait` exits 0. On exit 1, follow the roborev rule in Global Constraints.

---

### Task 6: Differential helpers and the general transforms

**Files:**
- Create: `tests/differential/transform_matrices.py`
- Create: `tests/differential/test_general_transforms.py`

**Interfaces:**
- Consumes: `ROWS` from `tests/transform_cases.py`; every primitive from Tasks 1 and 3.
- Produces: in `differential.transform_matrices`: `featuretools_values(primitive_name: str, feature_name: str) -> list`, `tusk_values(primitive_name: str, column: str) -> list`, `assert_agree(ours: list, theirs: list) -> None`.

- [ ] **Step 1: Create the helpers**

`tests/differential/transform_matrices.py`:

```python
"""Build one transform column over the shared table in tusk and in featuretools.

Both sides order the rows by ``occurred_at``: tusk as the row creation time,
featuretools as the time index.
"""

import datetime as dt

import pandas as pd
import polars as pl
import pytest
from transform_cases import ROWS, feature_values, rows_database

import tusk

featuretools = pytest.importorskip("featuretools")

LOGICAL_TYPES = {
    "value": "Double",
    "flag": "Boolean",
    "maybe_flag": "BooleanNullable",
    "label": "Categorical",
}


def featuretools_values(primitive_name, feature_name):
    """Run one transform primitive through featuretools and read one column.

    Args:
        primitive_name: The primitive's featuretools name.
        feature_name: The featuretools feature column to read.

    Returns:
        The column's values in id order, missing values as featuretools
        stores them.
    """
    entityset = featuretools.EntitySet("rows").add_dataframe(
        dataframe_name="rows",
        dataframe=ROWS,
        index="id",
        time_index="occurred_at",
        logical_types=LOGICAL_TYPES,
    )
    matrix, _ = featuretools.dfs(
        entityset=entityset,
        target_dataframe_name="rows",
        agg_primitives=[],
        trans_primitives=[primitive_name],
        max_depth=1,
    )
    return matrix.sort_index()[feature_name].tolist()


def tusk_values(primitive_name, column):
    """Run one transform primitive through tusk on polars and read one column.

    Args:
        primitive_name: The primitive's tusk name.
        column: The tusk feature column to read.

    Returns:
        The column's values in id order, null as None.
    """
    matrix, _ = tusk.deep_feature_synthesis(
        database=rows_database(pl.from_pandas(ROWS).lazy()),
        target_table="rows",
        agg_primitives=[],
        trans_primitives=[primitive_name],
        max_depth=1,
    )
    return feature_values(matrix, column)


def assert_agree(ours, theirs):
    """Assert that tusk and featuretools give the same value on every row.

    A tusk null agrees with any featuretools missing value (NaN, NaT, NA),
    since featuretools stores a missing float as NaN.

    Args:
        ours: tusk's column in id order.
        theirs: featuretools' column in id order.
    """
    assert len(ours) == len(theirs)
    for our, their in zip(ours, theirs, strict=True):
        if pd.isna(their):
            assert our is None or pd.isna(our)
        elif isinstance(their, dt.timedelta):
            # pytest.approx refuses a timedelta without an explicit tolerance
            assert our == their
        else:
            assert our == pytest.approx(their)
```

- [ ] **Step 2: Write the tests**

`tests/differential/test_general_transforms.py`:

```python
"""Cross-check tusk's general transform primitives against featuretools.

Each test builds one primitive on both sides over the shared table in
``tests/transform_cases.py`` and compares the one resulting column.

Run with: uv run --group validation pytest -m differential

Verified against featuretools 1.31.0.
"""

import math

import pytest

from differential.transform_matrices import assert_agree, featuretools_values, tusk_values

pytestmark = pytest.mark.differential


@pytest.mark.parametrize(
    ("ours", "column", "theirs", "feature_name"),
    [
        ("is_null", "IS_NULL__value", "is_null", "IS_NULL(value)"),
        ("negate", "NEGATE__value", "negate", "-(value)"),
        ("sine", "SINE__value", "sine", "SINE(value)"),
        ("cosine", "COSINE__value", "cosine", "COSINE(value)"),
        ("percentile", "PERCENTILE__value", "percentile", "PERCENTILE(value)"),
        ("absolute", "ABSOLUTE__value", "absolute", "ABSOLUTE(value)"),
    ],
)
def test_general_transforms_match_featuretools(ours, column, theirs, feature_name):
    """Covers a null, zero, a negative number and ties in ``value``."""
    assert_agree(tusk_values(ours, column), featuretools_values(theirs, feature_name))


@pytest.mark.parametrize(
    ("ours", "column", "theirs", "feature_name"),
    [
        ("square_root", "SQUARE_ROOT__value", "square_root", "SQUARE_ROOT(value)"),
        ("natural_log", "NATURAL_LOG__value", "natural_logarithm", "NATURAL_LOGARITHM(value)"),
    ],
)
def test_square_root_and_natural_log_give_null_where_featuretools_gives_nan(
    ours,
    column,
    theirs,
    feature_name,
):
    """Row id 5 holds -1: tusk answers null, featuretools NaN. Every other row agrees."""
    ours_values = tusk_values(ours, column)
    theirs_values = featuretools_values(theirs, feature_name)
    negative_row = 4
    assert ours_values[negative_row] is None
    assert math.isnan(theirs_values[negative_row])
    del ours_values[negative_row]
    del theirs_values[negative_row]
    assert_agree(ours_values, theirs_values)
```

- [ ] **Step 3: Run the tests**

Run: `uv run --group validation pytest -m differential tests/differential/test_general_transforms.py -v`
Expected: 8 PASS. A failure here is a real disagreement: stop and report it rather than change the expected values.

- [ ] **Step 4: Commit and wait for the review**

```bash
git add tests/differential/transform_matrices.py tests/differential/test_general_transforms.py
git commit -m "test: cross-check the general transforms against featuretools

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_018ztwcZ16kzzv4xDHNBhfTP"
roborev wait
```

Expected: `roborev wait` exits 0. On exit 1, follow the roborev rule in Global Constraints.

---

### Task 7: Differential datetime parts

**Files:**
- Modify: `tests/differential/test_datetime_transforms.py`

**Interfaces:**
- Consumes: `featuretools_values`, `tusk_values`, `assert_agree` from Task 6; the primitives from Task 2.

- [ ] **Step 1: Write the tests**

Add to the imports of `tests/differential/test_datetime_transforms.py`:

```python
from differential.transform_matrices import assert_agree, featuretools_values, tusk_values
```

Append:

```python
@pytest.mark.parametrize(
    "name",
    ["minute", "second", "day_of_year", "year", "month", "day", "hour"],
)
def test_datetime_parts_match_featuretools(name):
    """``due_at`` holds a null, 1900, 2000, 2100 and a day 366."""
    assert_agree(
        tusk_values(name, f"{name.upper()}__due_at"),
        featuretools_values(name, f"{name.upper()}(due_at)"),
    )


def test_is_leap_year_of_a_null_date_is_null_rather_than_false():
    """Row id 2 has no ``due_at``: tusk answers null, featuretools False."""
    ours = tusk_values("is_leap_year", "IS_LEAP_YEAR__due_at")
    theirs = featuretools_values("is_leap_year", "IS_LEAP_YEAR(due_at)")
    assert ours == [True, None, False, True, False, False, True, False]
    assert theirs == [True, False, False, True, False, False, True, False]
```

- [ ] **Step 2: Run the tests**

Run: `uv run --group validation pytest -m differential tests/differential/test_datetime_transforms.py -v`
Expected: 9 PASS (8 new, 1 existing). A failure is a real disagreement: stop and report it.

- [ ] **Step 3: Commit and wait for the review**

```bash
git add tests/differential/test_datetime_transforms.py
git commit -m "test: cross-check the datetime parts against featuretools

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_018ztwcZ16kzzv4xDHNBhfTP"
roborev wait
```

Expected: `roborev wait` exits 0. On exit 1, follow the roborev rule in Global Constraints.

---

### Task 8: Differential cumulative transforms

**Files:**
- Create: `tests/differential/test_cumulative_transforms.py`

**Interfaces:**
- Consumes: `featuretools_values`, `tusk_values`, `assert_agree` from Task 6; `ROWS` from Task 1; the primitives from Tasks 4 and 5.

- [ ] **Step 1: Write the tests**

`tests/differential/test_cumulative_transforms.py`:

```python
"""Cross-check tusk's cumulative transform primitives against featuretools.

Both sides order the shared table by ``occurred_at``: ids 3, 1, 5, 2, 8, 4, 7,
6. In that order ``value`` is 4, null, -1, 0, 0, 9, 2, 2. featuretools
forward-fills a null before ``same_as_previous``, ``absolute_diff`` and
``percent_change``; tusk does not, which is where those three diverge.

Run with: uv run --group validation pytest -m differential

Verified against featuretools 1.31.0.
"""

import math

import pytest
from transform_cases import ROWS

from differential.transform_matrices import assert_agree, featuretools_values, tusk_values

featuretools = pytest.importorskip("featuretools")

pytestmark = pytest.mark.differential


@pytest.mark.parametrize("name", ["cum_sum", "cum_min", "cum_max", "diff"])
def test_cumulative_transforms_match_featuretools(name):
    """A null row stays null and the running value carries past it on both sides."""
    assert_agree(
        tusk_values(name, f"{name.upper()}__value"),
        featuretools_values(name, f"{name.upper()}(value)"),
    )


@pytest.mark.parametrize(
    "name",
    ["cumulative_time_since_last_true", "cumulative_time_since_last_false"],
)
def test_cumulative_time_since_matches_featuretools_in_seconds(name):
    """tusk returns a Duration, featuretools float seconds; the amounts agree."""
    ours = tusk_values(name, f"{name.upper()}__occurred_at__flag")
    ours_in_seconds = [None if value is None else value.total_seconds() for value in ours]
    assert_agree(ours_in_seconds, featuretools_values(name, f"{name.upper()}(occurred_at, flag)"))


def test_cum_mean_divides_by_known_values_where_featuretools_divides_by_rows():
    ours = tusk_values("cum_mean", "CUM_MEAN__value")
    theirs = featuretools_values("cum_mean", "CUM_MEAN(value)")
    assert_agree(ours, [None, 1.0, 4.0, 2.4, 1.5, 16 / 7, 7 / 3, 0.75])
    assert_agree(theirs, [None, 0.75, 4.0, 2.0, 1.0, 2.0, 2.0, 0.6])


def test_same_as_previous_does_not_forward_fill():
    """The first row (id 3), the null (id 1) and the row after it (id 5) diverge."""
    ours = tusk_values("same_as_previous", "SAME_AS_PREVIOUS__value")
    theirs = featuretools_values("same_as_previous", "SAME_AS_PREVIOUS(value)")
    assert ours == [None, False, None, False, None, True, False, True]
    assert theirs == [True, False, False, False, False, True, False, True]


def test_absolute_diff_does_not_forward_fill():
    """The null (id 1) and the row after it (id 5) diverge."""
    ours = tusk_values("absolute_diff", "ABSOLUTE_DIFF__value")
    theirs = featuretools_values("absolute_diff", "ABSOLUTE_DIFF(value)")
    assert_agree(ours, [None, 1.0, None, 9.0, None, 0.0, 7.0, 0.0])
    assert theirs[0] == 0.0
    assert theirs[4] == 5.0
    assert_agree(theirs, [0.0, 1.0, None, 9.0, 5.0, 0.0, 7.0, 0.0])


def test_percent_change_does_not_forward_fill():
    """The null (id 1) and the row after it (id 5) diverge.

    Zero after zero (id 8) is NaN on both sides.
    """
    ours = tusk_values("percent_change", "PERCENT_CHANGE__value")
    theirs = featuretools_values("percent_change", "PERCENT_CHANGE(value)")
    assert ours[0] is None
    assert ours[4] is None
    assert theirs[0] == 0.0
    assert theirs[4] == -1.25
    assert math.isnan(ours[7])
    assert math.isnan(theirs[7])
    assert_agree(ours, [None, -1.0, None, math.inf, None, 0.0, -7 / 9, None])
    assert_agree(theirs, [0.0, -1.0, None, math.inf, -1.25, 0.0, -7 / 9, None])


def test_cum_count_counts_known_values_where_featuretools_counts_rows():
    """``label`` is null on ids 2 and 7; featuretools still counts those rows."""
    ours = tusk_values("cum_count", "CUM_COUNT__label")
    theirs = featuretools_values("cum_count", "CUM_COUNT(label)")
    assert ours == [2, 3, 1, 5, 3, 6, 5, 4]
    assert theirs == [2, 4, 1, 6, 3, 8, 7, 5]


def test_diff_datetime_is_time_since_previous():
    """featuretools' DFS builds no diff_datetime here, so it is called directly.

    Both lists are in time order.
    """
    ordered = ROWS.sort_values("occurred_at")
    theirs = featuretools.primitives.DiffDatetime()(ordered["occurred_at"]).tolist()
    ours_by_id = tusk_values("time_since_previous", "TIME_SINCE_PREVIOUS__occurred_at")
    ours = [ours_by_id[row_id - 1] for row_id in ordered["id"]]
    assert_agree(ours, theirs)
```

- [ ] **Step 2: Run the tests**

Run: `uv run --group validation pytest -m differential tests/differential/test_cumulative_transforms.py -v`
Expected: 13 PASS. A failure is a real disagreement: stop and report it.

- [ ] **Step 3: Commit and wait for the review**

```bash
git add tests/differential/test_cumulative_transforms.py
git commit -m "test: cross-check the cumulative transforms against featuretools

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_018ztwcZ16kzzv4xDHNBhfTP"
roborev wait
```

Expected: `roborev wait` exits 0. On exit 1, follow the roborev rule in Global Constraints.

---

### Task 9: Documentation

**Files:**
- Modify: `docs/api/primitives.md`
- Modify: `docs/guide/primitives.md`
- Modify: `docs/guide/primitive-coverage.md`

**Interfaces:**
- Consumes: every class from Tasks 1–5 and every test name from Tasks 6–8.

- [ ] **Step 1: API reference**

In `docs/api/primitives.md`, under `## Transform primitives`, insert after `::: tusk.primitives.IsWeekend`:

```markdown
::: tusk.primitives.Minute

::: tusk.primitives.Second

::: tusk.primitives.DayOfYear

::: tusk.primitives.IsLeapYear
```

and after `::: tusk.primitives.NaturalLog`:

```markdown
::: tusk.primitives.IsNull

::: tusk.primitives.Negate

::: tusk.primitives.SquareRoot

::: tusk.primitives.Sine

::: tusk.primitives.Cosine

::: tusk.primitives.Percentile
```

Under `## Order-dependent transform primitives`, append after `::: tusk.primitives.TimeSincePrevious`:

```markdown
::: tusk.primitives.CumMean

::: tusk.primitives.SameAsPrevious

::: tusk.primitives.AbsoluteDiff

::: tusk.primitives.PercentChange

::: tusk.primitives.CumulativeTimeSinceLastTrue

::: tusk.primitives.CumulativeTimeSinceLastFalse
```

- [ ] **Step 2: Guide**

In `docs/guide/primitives.md`, section `## What can go in groupby_trans_primitives`, replace the first paragraph with:

```markdown
Only **group-aware** primitives — ones whose expression reduces or scans across
the group defined by a foreign key. The order-dependent built-ins (`cum_sum`,
`cum_count`, `cum_min`, `cum_max`, `cum_mean`, `diff`, `absolute_diff`,
`same_as_previous`, `percent_change`, `time_since_previous`,
`cumulative_time_since_last_true`, `cumulative_time_since_last_false`) all
qualify, and so does `percentile`, which ranks each value within its group.
These are the primitives you'll normally pass here.
```

Insert this section directly before `## Comparing two columns`:

```markdown
## Negative inputs to `square_root` and `natural_log`

A negative number has no real square root or logarithm, so both primitives
give null for it on every backend. Zero is not negative: `square_root` gives
`0.0` and `natural_log` gives negative infinity.
```

- [ ] **Step 3: Coverage table**

In `docs/guide/primitive-coverage.md`, replace each row below in place (match on the featuretools primitive name), with `C`, `D` and `G` standing for:

- `C` = `https://github.com/Excidion/tusk/blob/main/tests/differential/test_cumulative_transforms.py`
- `D` = `https://github.com/Excidion/tusk/blob/main/tests/differential/test_datetime_transforms.py`
- `G` = `https://github.com/Excidion/tusk/blob/main/tests/differential/test_general_transforms.py`

Write the full URL in every link; the letters are only shorthand in this plan.

```markdown
| Cumulative transform | [`cum_count`][tusk.primitives.CumCount] | [`cum_count`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.CumCount.html) | ⚠️ | [`test_cum_count_counts_known_values_where_featuretools_counts_rows`](C) | featuretools counts every row; tusk counts only non-null values. |
| Cumulative transform | [`cum_max`][tusk.primitives.CumMax] | [`cum_max`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.CumMax.html) | ✅ | [`test_cumulative_transforms_match_featuretools`](C) |  |
| Cumulative transform | [`cum_min`][tusk.primitives.CumMin] | [`cum_min`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.CumMin.html) | ✅ | [`test_cumulative_transforms_match_featuretools`](C) |  |
| Cumulative transform | [`cum_sum`][tusk.primitives.CumSum] | [`cum_sum`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.CumSum.html) | ✅ | [`test_cumulative_transforms_match_featuretools`](C) |  |
| Cumulative transform | [`diff`][tusk.primitives.Diff] | [`diff`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Diff.html) | ✅ | [`test_cumulative_transforms_match_featuretools`](C) |  |
| Cumulative transform | [`cum_mean`][tusk.primitives.CumMean] | [`cum_mean`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.CumMean.html) | ⚠️ | [`test_cum_mean_divides_by_known_values_where_featuretools_divides_by_rows`](C) | A null is left out of the divisor; featuretools divides by the number of rows. |
| Cumulative transform | [`cumulative_time_since_last_false`][tusk.primitives.CumulativeTimeSinceLastFalse] | `cumulative_time_since_last_false` | ⚠️ | [`test_cumulative_time_since_matches_featuretools_in_seconds`](C) | Returns a `Duration`; featuretools returns float seconds. Takes any datetime column; featuretools takes only the time index. |
| Cumulative transform | [`cumulative_time_since_last_true`][tusk.primitives.CumulativeTimeSinceLastTrue] | `cumulative_time_since_last_true` | ⚠️ | [`test_cumulative_time_since_matches_featuretools_in_seconds`](C) | Returns a `Duration`; featuretools returns float seconds. Takes any datetime column; featuretools takes only the time index. |
| Cumulative transform | — | [`diff_datetime`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.DiffDatetime.html) | ⚠️ | [`test_diff_datetime_is_time_since_previous`](C) | Not a separate tusk primitive; `time_since_previous` computes it. |
| Datetime transform | [`day`][tusk.primitives.Day] | [`day`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Day.html) | ✅ | [`test_datetime_parts_match_featuretools`](D) |  |
| Datetime transform | [`hour`][tusk.primitives.Hour] | [`hour`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Hour.html) | ✅ | [`test_datetime_parts_match_featuretools`](D) |  |
| Datetime transform | [`month`][tusk.primitives.Month] | [`month`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Month.html) | ✅ | [`test_datetime_parts_match_featuretools`](D) |  |
| Datetime transform | [`year`][tusk.primitives.Year] | [`year`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Year.html) | ✅ | [`test_datetime_parts_match_featuretools`](D) |  |
| Datetime transform | [`day_of_year`][tusk.primitives.DayOfYear] | [`day_of_year`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.DayOfYear.html) | ✅ | [`test_datetime_parts_match_featuretools`](D) |  |
| Datetime transform | [`is_leap_year`][tusk.primitives.IsLeapYear] | [`is_leap_year`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.IsLeapYear.html) | ⚠️ | [`test_is_leap_year_of_a_null_date_is_null_rather_than_false`](D) | A null date is null in tusk and `False` in featuretools. |
| Datetime transform | [`minute`][tusk.primitives.Minute] | [`minute`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Minute.html) | ✅ | [`test_datetime_parts_match_featuretools`](D) |  |
| Datetime transform | [`second`][tusk.primitives.Second] | [`second`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Second.html) | ✅ | [`test_datetime_parts_match_featuretools`](D) |  |
| General transform | [`absolute`][tusk.primitives.Absolute] | [`absolute`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Absolute.html) | ✅ | [`test_general_transforms_match_featuretools`](G) |  |
| General transform | [`natural_log`][tusk.primitives.NaturalLog] | [`natural_logarithm`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.NaturalLogarithm.html) | ⚠️ | [`test_square_root_and_natural_log_give_null_where_featuretools_gives_nan`](G) | A negative input is null in tusk and NaN in featuretools. |
| General transform | [`absolute_diff`][tusk.primitives.AbsoluteDiff] | [`absolute_diff`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.AbsoluteDiff.html) | ⚠️ | [`test_absolute_diff_does_not_forward_fill`](C) | A null and the row after it are null in tusk; featuretools fills in the previous known value first. |
| General transform | [`cosine`][tusk.primitives.Cosine] | [`cosine`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Cosine.html) | ✅ | [`test_general_transforms_match_featuretools`](G) |  |
| General transform | [`is_null`][tusk.primitives.IsNull] | [`is_null`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.IsNull.html) | ✅ | [`test_general_transforms_match_featuretools`](G) |  |
| General transform | [`negate`][tusk.primitives.Negate] | [`negate`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Negate.html) | ✅ | [`test_general_transforms_match_featuretools`](G) | Always returns a float, so an integer column's smallest value can be negated. |
| General transform | [`percent_change`][tusk.primitives.PercentChange] | [`percent_change`](https://docs.featuretools.com/en/latest/generated/featuretools.primitives.PercentChange.html) | ⚠️ | [`test_percent_change_does_not_forward_fill`](C) | A null and the row after it are null in tusk; featuretools fills in the previous known value first. |
| General transform | [`percentile`][tusk.primitives.Percentile] | [`percentile`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Percentile.html) | ✅ | [`test_general_transforms_match_featuretools`](G) |  |
| General transform | [`same_as_previous`][tusk.primitives.SameAsPrevious] | [`same_as_previous`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.SameAsPrevious.html) | ⚠️ | [`test_same_as_previous_does_not_forward_fill`](C) | The first row, a null and the row after it are null in tusk; featuretools fills in the previous known value first and answers `False` on the first row. |
| General transform | [`sine`][tusk.primitives.Sine] | [`sine`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Sine.html) | ✅ | [`test_general_transforms_match_featuretools`](G) |  |
| General transform | [`square_root`][tusk.primitives.SquareRoot] | [`square_root`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.SquareRoot.html) | ⚠️ | [`test_square_root_and_natural_log_give_null_where_featuretools_gives_nan`](G) | A negative input is null in tusk and NaN in featuretools. |
```

- [ ] **Step 4: Build the docs**

Run: `just docs`
Expected: the build finishes without a warning about an unresolved `tusk.primitives.*` reference.

- [ ] **Step 5: Commit and wait for the review**

```bash
git add docs/api/primitives.md docs/guide/primitives.md docs/guide/primitive-coverage.md
git commit -m "docs: document the standalone and ordered transforms

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_018ztwcZ16kzzv4xDHNBhfTP"
roborev wait
```

Expected: `roborev wait` exits 0. On exit 1, follow the roborev rule in Global Constraints.

---

### Task 10: Branch gate

**Files:** none unless a check fails.

- [ ] **Step 1: Run everything CI runs**

Run: `just check`
Expected: lint, the default suite and the differential suite all pass.

- [ ] **Step 2: Confirm every commit's review passed**

Run: `for sha in $(git rev-list main..HEAD); do roborev wait "$sha" || echo "FAILED $sha"; done`
Expected: no `FAILED` line. For any failed commit, run `roborev show <sha>`, fix the findings with the `roborev-fix` skill, commit, and repeat this step.

- [ ] **Step 3: Review the branch diff**

Run: `git diff main...HEAD --stat` and read the full `git diff main...HEAD`.
Expected: only the files named in Tasks 1–9 (plus the spec and this plan) changed, and nothing outside the spec's scope.
