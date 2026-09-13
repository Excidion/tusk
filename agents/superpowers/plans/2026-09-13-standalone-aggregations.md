# Standalone Aggregations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship fifteen aggregation primitives that fit inside one lazy `group_by().agg()`, align tusk's DFS defaults with featuretools', and bring the coverage table up to date.

**Architecture:** Each primitive is a frozen dataclass in `src/tusk/primitives/aggregation.py` that builds a narwhals reduction; empty groups are filled from `default_value` by the existing compiler. One shared test module holds a hand-built parent/child fixture and every expected value, so the polars suite, the duckdb suite and the differential suite all check the same groups: mixed, constant, all-null, varied and childless.

**Tech Stack:** Python 3.10+, narwhals 2.24, polars and duckdb in tests, featuretools 1.31.0 in the opt-in differential suite, pytest, ruff + ty + pydoclint + interrogate via pre-commit.

**Spec:** `agents/superpowers/specs/2026-09-13-standalone-aggregations-design.md` (roadmap: `agents/superpowers/specs/2026-09-13-primitive-parity-roadmap-design.md`)

## Global Constraints

- Plans and specs live under `agents/superpowers/`, never `docs/` (`agents/AGENTS.md`).
- Every public **and private** function in `src/` needs a Google-style docstring with `Args:` and `Returns:` sections matching the signature; `interrogate` demands 100% and `pydoclint` cross-checks. Tests are exempt.
- Docstrings say *what*; a comment says *why*, and only when naming cannot (`agents/style/CODESTYLE.md`). Callers above callees.
- Every primitive must give the same value on lazy polars and duckdb.
- Names: `n_true`, not `num_true`. Every other primitive keeps its featuretools name.
- Run tests with `uv run pytest`. Never pass `--active` to `uv`.
- Differential tests: `uv run --group validation pytest -m differential`.
- Full gate before the branch is done: `just check`.
- Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  ```

## Expected values

These come from probing the planned expressions on both backends and
featuretools on the fixture below. Parents 1–5; parent 5 has no children.

| Parent | `value` | `flag` | `label` | `seen_at` |
| --- | --- | --- | --- | --- |
| 1 | 1, 2, 6, null | T, F, T, null | a, a, b, null | 2024-01-01 05:00, 2024-01-01 09:00, 2024-03-02, null |
| 2 | 4, 4 | T, T | null, null | 2023-03-02, 2024-03-02 |
| 3 | null ×3 | null ×3 | null ×3 | null ×3 |
| 4 | 1, 5, 2, 8 | F, F, T, F | x, y, z, w | 2024-01-31, 2024-02-01, 2025-02-01, 2024-02-29 |

---

### Task 1: Boolean aggregations and the shared test cases

**Files:**
- Create: `tests/aggregation_cases.py`
- Modify: `src/tusk/primitives/aggregation.py` (append classes after `TimeSinceLastFalse`, before `_time_since_last_selected`)
- Modify: `src/tusk/primitives/__init__.py`
- Test: `tests/test_primitives_aggregation.py`

**Interfaces:**
- Produces:
  - `tests/aggregation_cases.py`: `PARENTS: pd.DataFrame`, `CHILDREN: pd.DataFrame`, `EXPECTED: dict[str, tuple[str, nw.dtypes.DType, list]]` keyed by primitive name → `(column name, output dtype, value per parent id 1..5)`, and `assert_values_match(got: list, expected: list) -> None`.
  - Classes `All`, `Any`, `NTrue` in `tusk.primitives`.

- [ ] **Step 1: Create the shared cases module**

`tests/aggregation_cases.py`:

```python
"""One parent/child fixture and the value every standalone aggregation gives on it.

Shared by the polars suite, the duckdb suite and the differential suite, so
all three check the same groups. Parent 1 mixes known values with a null,
parent 2 is constant, parent 3 is all null, parent 4 varies and parent 5 has
no children.
"""

import datetime as dt

import narwhals as nw
import pandas as pd
import pytest

PARENTS = pd.DataFrame({"id": [1, 2, 3, 4, 5]})

CHILDREN = pd.DataFrame(
    {
        "id": range(1, 14),
        "parent_id": [1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4],
        "value": [1.0, 2.0, 6.0, None, 4.0, 4.0, None, None, None, 1.0, 5.0, 2.0, 8.0],
        "flag": pd.array(
            [True, False, True, None, True, True, None, None, None]
            + [False, False, True, False],
            dtype="boolean",
        ),
        "label": pd.array(
            ["a", "a", "b", None, None, None, None, None, None, "x", "y", "z", "w"],
            dtype="string",
        ),
        "seen_at": pd.to_datetime(
            [
                dt.datetime(2024, 1, 1, 5),
                dt.datetime(2024, 1, 1, 9),
                dt.datetime(2024, 3, 2),
                None,
                dt.datetime(2023, 3, 2),
                dt.datetime(2024, 3, 2),
                None,
                None,
                None,
                dt.datetime(2024, 1, 31),
                dt.datetime(2024, 2, 1),
                dt.datetime(2025, 2, 1),
                dt.datetime(2024, 2, 29),
            ],
        ),
    },
)

EXPECTED = {
    "all": ("ALL__children__flag", nw.Boolean, [False, True, True, False, None]),
    "any": ("ANY__children__flag", nw.Boolean, [True, True, False, True, False]),
    "n_true": ("N_TRUE__children__flag", nw.Int64, [2, 2, 0, 1, 0]),
}


def assert_values_match(got, expected):
    """Compare one feature column against its expected values, parent by parent.

    Args:
        got: The column as a list, in parent id order.
        expected: The expected values; a float is compared approximately.
    """
    assert len(got) == len(expected)
    for actual, wanted in zip(_normalized(got), expected, strict=True):
        if isinstance(wanted, float):
            assert actual == pytest.approx(wanted)
        else:
            assert actual == wanted


def _normalized(values):
    """Make a materialized column comparable with plain Python values.

    Args:
        values: A materialized column.

    Returns:
        The values, with every backend's missing value as None and every
        duration as a timedelta.
    """
    return [_normalized_value(value) for value in values]


def _normalized_value(value):
    """Make one materialized value comparable with a plain Python one.

    Args:
        value: A value from a pandas column.

    Returns:
        None for any missing value, a timedelta for a duration, else the value.
    """
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timedelta):
        return value.to_pytimedelta()
    return value
```

- [ ] **Step 2: Write the failing test**

Append to `tests/test_primitives_aggregation.py`, and add `import tusk` plus
`from aggregation_cases import CHILDREN, EXPECTED, PARENTS, assert_values_match`
to its imports:

```python
@pytest.mark.parametrize("primitive_name", sorted(EXPECTED))
def test_standalone_aggregations_on_every_kind_of_group(primitive_name):
    column, dtype, expected = EXPECTED[primitive_name]
    database = (
        tusk.Database("cases")
        .add_table("parents", pl.from_pandas(PARENTS).lazy(), primary_key="id")
        .add_table("children", pl.from_pandas(CHILDREN).lazy(), primary_key="id")
        .add_relationship(parent="parents", child="children", foreign_key="parent_id")
    )
    matrix, _ = tusk.deep_feature_synthesis(
        database=database,
        target_table="parents",
        agg_primitives=[primitive_name],
        trans_primitives=[],
        max_depth=1,
    )
    assert nw.from_native(matrix).collect_schema()[column] == dtype
    got = matrix.collect().sort("id").to_pandas()[column].tolist()
    assert_values_match(got, expected)
```

- [ ] **Step 3: Run it to verify it fails**

Run: `uv run pytest tests/test_primitives_aggregation.py -k standalone -v`
Expected: 3 FAIL with `PrimitiveError` (unknown primitive name).

- [ ] **Step 4: Implement the three primitives**

Append to `src/tusk/primitives/aggregation.py` after `TimeSinceLastFalse`:

```python
@register
@dataclass(frozen=True)
class All(AggregationPrimitive):
    """Whether every known value of a boolean column is true."""

    name = "all"
    input_dtypes = (F.BOOLEAN,)
    output_dtype = nw.Boolean

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the every-value-true expression, ignoring nulls.

        Args:
            expr: The boolean column.

        Returns:
            A narwhals expression.
        """
        return expr.all()


@register
@dataclass(frozen=True)
class Any(AggregationPrimitive):
    """Whether any known value of a boolean column is true."""

    name = "any"
    input_dtypes = (F.BOOLEAN,)
    output_dtype = nw.Boolean
    default_value = False

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the some-value-true expression, ignoring nulls.

        Args:
            expr: The boolean column.

        Returns:
            A narwhals expression.
        """
        return expr.any()


@register
@dataclass(frozen=True)
class NTrue(AggregationPrimitive):
    """Number of rows where a boolean column is true."""

    name = "n_true"
    input_dtypes = (F.BOOLEAN,)
    output_dtype = nw.Int64
    default_value = 0

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the true-count expression; a null is not true.

        Args:
            expr: The boolean column.

        Returns:
            A narwhals expression.
        """
        # duckdb's SUM returns a Decimal whatever it sums
        return expr.fill_null(False).cast(nw.Int64).sum().cast(nw.Int64)
```

In `src/tusk/primitives/__init__.py`, add `All`, `Any` and `NTrue` to the
`tusk.primitives.aggregation` import and to `__all__`, keeping both sorted.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_primitives_aggregation.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/aggregation_cases.py tests/test_primitives_aggregation.py src/tusk/primitives/aggregation.py src/tusk/primitives/__init__.py
git commit -m "feat: add the all, any and n_true aggregations"
```

---

### Task 2: Spread and moment statistics

**Files:**
- Modify: `src/tusk/primitives/aggregation.py`, `src/tusk/primitives/__init__.py`
- Modify: `tests/aggregation_cases.py` (extend `EXPECTED`)

**Interfaces:**
- Consumes: `EXPECTED`, `assert_values_match` and the parametrized test from Task 1.
- Produces: `Skew`, `Kurtosis`, `Variance`, `MaxMinDelta` in `tusk.primitives`; private `_where_the_column_varies(expr, moment) -> nw.Expr` at the bottom of `aggregation.py`.

- [ ] **Step 1: Extend the expected values**

Add to `EXPECTED` in `tests/aggregation_cases.py`:

```python
    "skew": (
        "SKEW__children__value",
        nw.Float64,
        [0.5951700641394972, None, None, 0.3651483716701108, None],
    ),
    "kurtosis": (
        "KURTOSIS__children__value",
        nw.Float64,
        [-1.5, None, None, -1.4266666666666667, None],
    ),
    "variance": ("VARIANCE__children__value", nw.Float64, [7.0, 0.0, None, 10.0, None]),
    "max_min_delta": (
        "MAX_MIN_DELTA__children__value",
        nw.Float64,
        [5.0, 0.0, None, 7.0, None],
    ),
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_primitives_aggregation.py -k standalone -v`
Expected: the 4 new cases FAIL with `PrimitiveError`.

- [ ] **Step 3: Implement**

Append after `NTrue`:

```python
@register
@dataclass(frozen=True)
class Skew(AggregationPrimitive):
    """Skewness of a numeric column, without bias correction."""

    name = "skew"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the skewness expression; a constant group is null.

        Args:
            expr: The column to reduce.

        Returns:
            A narwhals expression.
        """
        return _where_the_column_varies(expr, expr.skew())


@register
@dataclass(frozen=True)
class Kurtosis(AggregationPrimitive):
    """Excess kurtosis of a numeric column, without bias correction."""

    name = "kurtosis"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the excess-kurtosis expression; a constant group is null.

        Args:
            expr: The column to reduce.

        Returns:
            A narwhals expression.
        """
        return _where_the_column_varies(expr, expr.kurtosis())


@register
@dataclass(frozen=True)
class Variance(AggregationPrimitive):
    """Sample variance of a numeric column."""

    name = "variance"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the variance expression.

        Args:
            expr: The column to reduce.

        Returns:
            A narwhals expression.
        """
        return expr.var()


@register
@dataclass(frozen=True)
class MaxMinDelta(AggregationPrimitive):
    """Difference between the largest and smallest value of a numeric column."""

    name = "max_min_delta"
    input_dtypes = (F.NUMERIC,)

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the largest-minus-smallest expression.

        Args:
            expr: The column to reduce.

        Returns:
            A narwhals expression.
        """
        return expr.max() - expr.min()
```

Append at the very bottom of the module, below `_time_since_last_selected`:

```python
def _where_the_column_varies(expr: nw.Expr, moment: nw.Expr) -> nw.Expr:
    """Keep a standardized moment only for a group whose values vary.

    Args:
        expr: The column the moment was computed from.
        moment: The moment's expression.

    Returns:
        The moment, or null where the column's standard deviation is zero or
        unknown.
    """
    # A constant group divides zero by zero, which polars answers with NaN
    # and duckdb with 0.0 or null.
    return nw.when(expr.std() > 0).then(moment)
```

Add `Kurtosis`, `MaxMinDelta`, `Skew` and `Variance` to
`src/tusk/primitives/__init__.py`, sorted.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_primitives_aggregation.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/aggregation_cases.py src/tusk/primitives/aggregation.py src/tusk/primitives/__init__.py
git commit -m "feat: add the skew, kurtosis, variance and max_min_delta aggregations"
```

---

### Task 3: Uniqueness aggregations

**Files:**
- Modify: `src/tusk/primitives/aggregation.py` (`NUnique.build` and new classes), `src/tusk/primitives/__init__.py`
- Modify: `tests/aggregation_cases.py`

**Interfaces:**
- Consumes: Task 1's test cases.
- Produces: `IsUnique`, `HasNoDuplicates`, `PercentUnique` in `tusk.primitives`; private `_n_distinct_known(expr) -> nw.Expr` at the bottom of `aggregation.py`, which Task 4 reuses.

- [ ] **Step 1: Extend the expected values**

```python
    "is_unique": ("IS_UNIQUE__children__label", nw.Boolean, [False, None, None, True, None]),
    "has_no_duplicates": (
        "HAS_NO_DUPLICATES__children__label",
        nw.Boolean,
        [False, False, False, True, True],
    ),
    "percent_unique": (
        "PERCENT_UNIQUE__children__label",
        nw.Float64,
        [0.5, 0.0, 0.0, 1.0, None],
    ),
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_primitives_aggregation.py -k standalone -v`
Expected: the 3 new cases FAIL with `PrimitiveError`.

- [ ] **Step 3: Extract the distinct-known helper**

Replace the body of `NUnique.build` with `return _n_distinct_known(expr)`, and
move its explanation into the helper. Append at the bottom of the module:

```python
def _n_distinct_known(expr: nw.Expr) -> nw.Expr:
    """Build the number of distinct known values in a group.

    Args:
        expr: The column to count distinct values of.

    Returns:
        A narwhals Int64 expression that counts no null as a value.
    """
    # n_unique counts null as one more value on every backend. drop_nulls()
    # cannot remove it first: it changes length, which a lazy group_by().agg()
    # rejects.
    return (expr.n_unique() - expr.is_null().any().cast(nw.Int64)).cast(nw.Int64)
```

Trim `NUnique.build`'s docstring to what it returns; the *why* now lives in the
helper's comment.

- [ ] **Step 4: Run the existing n_unique tests**

Run: `uv run pytest tests/test_primitives_aggregation.py -k "n_unique or group_one" -v`
Expected: PASS, unchanged behaviour.

- [ ] **Step 5: Implement the three primitives**

Append after `MaxMinDelta`:

```python
@register
@dataclass(frozen=True)
class IsUnique(AggregationPrimitive):
    """Whether no known value of a column repeats; nulls are not values."""

    name = "is_unique"
    input_dtypes = (F.ANY,)
    output_dtype = nw.Boolean

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the no-known-value-repeats expression.

        Args:
            expr: The column to test.

        Returns:
            A narwhals expression, null for a group without a known value.
        """
        known = expr.count()
        return nw.when(known > 0).then(_n_distinct_known(expr) == known)


@register
@dataclass(frozen=True)
class HasNoDuplicates(AggregationPrimitive):
    """Whether no value of a column repeats; two nulls are a repeat."""

    name = "has_no_duplicates"
    input_dtypes = (F.ANY,)
    output_dtype = nw.Boolean
    default_value = True

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the no-value-repeats expression.

        Args:
            expr: The column to test.

        Returns:
            A narwhals expression.
        """
        return expr.n_unique() == nw.len()


@register
@dataclass(frozen=True)
class PercentUnique(AggregationPrimitive):
    """Distinct known values of a column as a fraction of its rows."""

    name = "percent_unique"
    input_dtypes = (F.ANY,)
    output_dtype = nw.Float64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the distinct-fraction expression; a null counts as a row.

        Args:
            expr: The column to count distinct values of.

        Returns:
            A narwhals expression.
        """
        return _n_distinct_known(expr) / nw.len()
```

Add `HasNoDuplicates`, `IsUnique` and `PercentUnique` to `__init__.py`, sorted.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_primitives_aggregation.py -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/aggregation_cases.py src/tusk/primitives/aggregation.py src/tusk/primitives/__init__.py
git commit -m "feat: add the is_unique, has_no_duplicates and percent_unique aggregations"
```

---

### Task 4: Datetime aggregations

**Files:**
- Modify: `src/tusk/primitives/aggregation.py`, `src/tusk/primitives/__init__.py`
- Modify: `tests/aggregation_cases.py`
- Test: `tests/test_primitives_aggregation.py`

**Interfaces:**
- Consumes: `_n_distinct_known` from Task 3.
- Produces: `FirstLastTimeDelta`, `NUniqueDays`, `NUniqueDaysOfCalendarYear`, `NUniqueDaysOfMonth`, `NUniqueMonths` in `tusk.primitives`.

- [ ] **Step 1: Extend the expected values**

```python
    "first_last_time_delta": (
        "FIRST_LAST_TIME_DELTA__children__seen_at",
        nw.Duration,
        [
            dt.timedelta(days=60, hours=19),
            dt.timedelta(days=366),
            None,
            dt.timedelta(days=367),
            None,
        ],
    ),
    "n_unique_days": ("N_UNIQUE_DAYS__children__seen_at", nw.Int64, [2, 2, 0, 4, 0]),
    "n_unique_days_of_calendar_year": (
        "N_UNIQUE_DAYS_OF_CALENDAR_YEAR__children__seen_at",
        nw.Int64,
        [2, 1, 0, 3, 0],
    ),
    "n_unique_days_of_month": (
        "N_UNIQUE_DAYS_OF_MONTH__children__seen_at",
        nw.Int64,
        [2, 1, 0, 3, 0],
    ),
    "n_unique_months": ("N_UNIQUE_MONTHS__children__seen_at", nw.Int64, [2, 2, 0, 3, 0]),
```

- [ ] **Step 2: Add a test for Date input**

`HAS_DATE` also matches `Date`, where duckdb's subtraction differs. Append to
`tests/test_primitives_aggregation.py`:

```python
def test_first_last_time_delta_of_dates_is_a_duration():
    lf = nw.from_native(
        pl.LazyFrame({"g": [1, 1], "d": [dt.date(2024, 1, 1), dt.date(2024, 3, 2)]}),
    )
    got = _agg(lf, resolve("first_last_time_delta"), "d")
    assert got["o0"][0] == dt.timedelta(days=61)
```

Add `import datetime as dt` to the module's imports.

- [ ] **Step 3: Run it to verify it fails**

Run: `uv run pytest tests/test_primitives_aggregation.py -k "standalone or first_last" -v`
Expected: the 5 new cases and the Date test FAIL with `PrimitiveError`.

- [ ] **Step 4: Implement**

Append after `PercentUnique`:

```python
@register
@dataclass(frozen=True)
class FirstLastTimeDelta(AggregationPrimitive):
    """Time between a group's earliest and latest datetime."""

    name = "first_last_time_delta"
    input_dtypes = (F.HAS_DATE,)
    output_dtype = nw.Duration

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the latest-minus-earliest expression.

        Args:
            expr: The datetime column to reduce.

        Returns:
            A narwhals expression.
        """
        # duckdb subtracts two Dates into a day count instead of an interval
        timestamps = expr.cast(nw.Datetime)
        return timestamps.max() - timestamps.min()


@register
@dataclass(frozen=True)
class NUniqueDays(AggregationPrimitive):
    """Number of distinct calendar dates in a datetime column."""

    name = "n_unique_days"
    input_dtypes = (F.HAS_DATE,)
    output_dtype = nw.Int64
    default_value = 0

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the distinct-date count; nulls are not values.

        Args:
            expr: The datetime column to reduce.

        Returns:
            A narwhals expression.
        """
        return _n_distinct_known(expr.dt.date())


@register
@dataclass(frozen=True)
class NUniqueDaysOfCalendarYear(AggregationPrimitive):
    """Number of distinct month-and-day pairs in a datetime column."""

    name = "n_unique_days_of_calendar_year"
    input_dtypes = (F.HAS_DATE,)
    output_dtype = nw.Int64
    default_value = 0

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the distinct month-and-day count; nulls are not values.

        Args:
            expr: The datetime column to reduce.

        Returns:
            A narwhals expression.
        """
        # month and day come back as Int8, which month * 100 overflows
        month = expr.dt.month().cast(nw.Int32)
        day = expr.dt.day().cast(nw.Int32)
        return _n_distinct_known(month * 100 + day)


@register
@dataclass(frozen=True)
class NUniqueDaysOfMonth(AggregationPrimitive):
    """Number of distinct days of the month in a datetime column."""

    name = "n_unique_days_of_month"
    input_dtypes = (F.HAS_DATE,)
    output_dtype = nw.Int64
    default_value = 0

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the distinct day-of-month count; nulls are not values.

        Args:
            expr: The datetime column to reduce.

        Returns:
            A narwhals expression.
        """
        return _n_distinct_known(expr.dt.day())


@register
@dataclass(frozen=True)
class NUniqueMonths(AggregationPrimitive):
    """Number of distinct calendar months, year included, in a datetime column."""

    name = "n_unique_months"
    input_dtypes = (F.HAS_DATE,)
    output_dtype = nw.Int64
    default_value = 0

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the distinct year-and-month count; nulls are not values.

        Args:
            expr: The datetime column to reduce.

        Returns:
            A narwhals expression.
        """
        # month comes back as Int8, which year * 12 would overflow
        year = expr.dt.year().cast(nw.Int32)
        month = expr.dt.month().cast(nw.Int32)
        return _n_distinct_known(year * 12 + month)
```

Add `FirstLastTimeDelta`, `NUniqueDays`, `NUniqueDaysOfCalendarYear`,
`NUniqueDaysOfMonth` and `NUniqueMonths` to `__init__.py`, sorted.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_primitives_aggregation.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/aggregation_cases.py tests/test_primitives_aggregation.py src/tusk/primitives/aggregation.py src/tusk/primitives/__init__.py
git commit -m "feat: add the first_last_time_delta and distinct-date aggregations"
```

---

### Task 5: Defaults follow featuretools

**Files:**
- Modify: `src/tusk/primitives/aggregation.py:21-29`, `src/tusk/primitives/transform.py:19`
- Test: `tests/test_primitives_aggregation.py:120-121`, `tests/test_primitives_transform.py:123`

**Interfaces:**
- Consumes: `Skew` from Task 2.
- Produces: `AGG_DEFAULTS`, `TRANS_DEFAULTS` with the values below.

- [ ] **Step 1: Update the failing assertions**

In `tests/test_primitives_aggregation.py`:

```python
def test_defaults_are_the_documented_set():
    assert AGG_DEFAULTS == (
        "count",
        "sum",
        "mean",
        "min",
        "max",
        "std",
        "n_unique",
        "skew",
        "percent_true",
    )
```

In `tests/test_primitives_transform.py`, the assertion becomes
`assert TRANS_DEFAULTS == ("year", "month", "day", "weekday")`.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_primitives_aggregation.py tests/test_primitives_transform.py -k defaults -v`
Expected: 2 FAIL.

- [ ] **Step 3: Change the defaults**

```python
AGG_DEFAULTS: tuple[str, ...] = (
    "count",
    "sum",
    "mean",
    "min",
    "max",
    "std",
    "n_unique",
    "skew",
    "percent_true",
)
```

```python
TRANS_DEFAULTS: tuple[str, ...] = ("year", "month", "day", "weekday")
```

- [ ] **Step 4: Run the whole suite**

Run: `uv run pytest`
Expected: all PASS. Tests over default output use subset checks, so extra
default features change no assertion. If one fails, it pins the old default
set: extend its expectation with the `SKEW__`, `PERCENT_TRUE__` or `DAY__`
columns it now also generates, and say so in the commit message.

- [ ] **Step 5: Commit**

```bash
git add src/tusk/primitives/aggregation.py src/tusk/primitives/transform.py tests/test_primitives_aggregation.py tests/test_primitives_transform.py
git commit -m "feat: match featuretools' default primitives"
```

---

### Task 6: The same values on duckdb

**Files:**
- Test: `tests/test_backend_duckdb.py`

**Interfaces:**
- Consumes: `PARENTS`, `CHILDREN`, `EXPECTED`, `assert_values_match` from `tests/aggregation_cases.py`.

- [ ] **Step 1: Write the test**

Add `from aggregation_cases import CHILDREN, EXPECTED, PARENTS, assert_values_match`
to the imports, and append:

```python
@pytest.mark.parametrize("primitive_name", sorted(EXPECTED))
def test_standalone_aggregations_give_the_polars_values_on_duckdb(primitive_name):
    """Every standalone aggregation survives translation to SQL, group by group.

    The constant group pins the skew and kurtosis guard, which duckdb would
    otherwise answer with 0.0 or null where polars answers NaN. ``n_true``
    pins the cast after SUM, and ``first_last_time_delta`` the interval
    subtraction. Materialized with ``.df()``; see
    ``test_time_since_holds_the_elapsed_time_on_duckdb`` for why.

    Args:
        primitive_name: The aggregation under test.
    """
    column, _, expected = EXPECTED[primitive_name]
    con = duckdb.connect()
    con.register("parents_frame", PARENTS)
    con.register("children_frame", CHILDREN)
    database = (
        tusk.Database("cases")
        .add_table(
            "parents",
            nw.from_native(con.sql("SELECT * FROM parents_frame")),
            primary_key="id",
        )
        .add_table(
            "children",
            nw.from_native(con.sql("SELECT * FROM children_frame")),
            primary_key="id",
        )
        .add_relationship(parent="parents", child="children", foreign_key="parent_id")
    )
    matrix, _ = tusk.deep_feature_synthesis(
        database=database,
        target_table="parents",
        agg_primitives=[primitive_name],
        trans_primitives=[],
        max_depth=1,
    )
    got = matrix.df().sort_values("id")[column].tolist()
    assert_values_match(got, expected)
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_backend_duckdb.py -k standalone -v`
Expected: 15 PASS. The expressions were already probed on duckdb; a failure
here is a real backend divergence. Fix the primitive, never the expectation.

- [ ] **Step 3: Commit**

```bash
git add tests/test_backend_duckdb.py
git commit -m "test: check the standalone aggregations on duckdb"
```

---

### Task 7: Differential tests against featuretools

**Files:**
- Modify: `tests/differential/test_aggregations.py`

**Interfaces:**
- Consumes: `PARENTS`, `CHILDREN`, `assert_values_match` from `tests/aggregation_cases.py`; the existing `_featuretools_matrix` and `_tusk_matrix`.
- Produces: the test names the coverage table links to in Task 8.

- [ ] **Step 1: Let featuretools take logical types**

Give `_featuretools_matrix` a keyword parameter `logical_types=None`, pass it
to the children's `es.add_dataframe(..., logical_types=logical_types)`, and
document it in `Args:`: "Woodwork logical types for the child's columns, or
None to let woodwork infer them." In `_tusk_matrix`'s docstring, drop the
sentence claiming tusk and featuretools spell names the same way.

- [ ] **Step 2: Add the helpers and the featuretools declarations**

Add `import math` and
`from aggregation_cases import CHILDREN, PARENTS, assert_values_match` to the
imports. Append:

```python
FEATURETOOLS_LOGICAL_TYPES = {
    "value": "Double",
    "flag": "BooleanNullable",
    "label": "Categorical",
}


def _ours_and_theirs(
    tusk_name,
    featuretools_name,
    featuretools_column,
    *,
    children=CHILDREN,
    logical_types=FEATURETOOLS_LOGICAL_TYPES,
    featuretools_time_index=None,
):
    """Compute one aggregation on both sides, as plain values per parent id.

    Woodwork infers whole-number floats as ``IntegerNullable``, which
    ``kurtosis`` rejects, so the child's columns are declared explicitly.
    featuretools' ``first_last_time_delta`` only reads its time index.

    Args:
        tusk_name: tusk's name for the primitive.
        featuretools_name: featuretools' name for the primitive.
        featuretools_column: featuretools' name for the feature column.
        children: The child table.
        logical_types: Woodwork logical types for the child's columns.
        featuretools_time_index: The child column featuretools treats as its
            time index, or None.

    Returns:
        tusk's and featuretools' values, each a dict from parent id to a
        value, with durations in seconds and missing values as None.
    """
    tusk_column = _as_tusk(featuretools_column).replace(
        featuretools_name.upper(),
        tusk_name.upper(),
        1,
    )
    ours = _tusk_matrix(PARENTS, children, tusk_name)[tusk_column]
    theirs = _featuretools_matrix(
        PARENTS,
        children,
        featuretools_name,
        logical_types=logical_types,
        time_index=featuretools_time_index,
    )[featuretools_column]
    return _plain_by_parent(ours), _plain_by_parent(theirs)


def _plain_by_parent(column):
    """Key a feature column by parent id with plain Python values.

    Args:
        column: A feature column indexed by parent id.

    Returns:
        A dict from parent id to a value; durations become seconds and every
        missing value becomes None.
    """
    return {parent: _plain(value) for parent, value in column.items()}


def _plain(value):
    """Turn one materialized value into a plain Python one.

    Args:
        value: A value from a pandas column.

    Returns:
        None for any missing value, seconds for a duration, and the Python
        scalar for a numpy one, so ``is False`` works on a boolean.
    """
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    if isinstance(value, np.generic):
        return value.item()
    return value
```

- [ ] **Step 3: Write the tests where the values agree**

```python
@pytest.mark.parametrize(
    ("tusk_name", "featuretools_name", "featuretools_column"),
    [
        ("all", "all", "ALL(children.flag)"),
        ("n_true", "num_true", "NUM_TRUE(children.flag)"),
        ("variance", "variance", "VARIANCE(children.value)"),
        ("n_unique_days", "n_unique_days", "N_UNIQUE_DAYS(children.seen_at)"),
        (
            "n_unique_days_of_calendar_year",
            "n_unique_days_of_calendar_year",
            "N_UNIQUE_DAYS_OF_CALENDAR_YEAR(children.seen_at)",
        ),
        (
            "n_unique_days_of_month",
            "n_unique_days_of_month",
            "N_UNIQUE_DAYS_OF_MONTH(children.seen_at)",
        ),
        ("n_unique_months", "n_unique_months", "N_UNIQUE_MONTHS(children.seen_at)"),
    ],
)
def test_standalone_aggregations_match_featuretools_on_every_parent_row(
    tusk_name,
    featuretools_name,
    featuretools_column,
):
    """tusk and featuretools agree on a mixed, constant, all-null and empty group."""
    ours, theirs = _ours_and_theirs(tusk_name, featuretools_name, featuretools_column)
    assert_values_match(list(ours.values()), list(theirs.values()))
```

- [ ] **Step 4: Write the tests where they diverge**

Each asserts agreement where it holds, then both sides where it does not.

```python
def test_any_of_an_empty_group_is_false_rather_than_null():
    """No row is true, so tusk says False; featuretools reports a missing value."""
    ours, theirs = _ours_and_theirs("any", "any", "ANY(children.flag)")
    assert_values_match([ours[p] for p in (1, 2, 3, 4)], [theirs[p] for p in (1, 2, 3, 4)])
    assert ours[5] is False
    assert theirs[5] is None


def test_skew_differs_from_featuretools_only_by_its_bias_correction():
    """featuretools' skew is tusk's scaled by pandas' sample correction.

    The correction for n known values is sqrt(n * (n - 1)) / (n - 2).
    Groups too small or too constant to have a skew are missing on both sides.
    """
    ours, theirs = _ours_and_theirs("skew", "skew", "SKEW(children.value)")
    known_counts = CHILDREN.groupby("parent_id")["value"].count()
    for parent in (1, 4):
        n = known_counts[parent]
        correction = math.sqrt(n * (n - 1)) / (n - 2)
        assert theirs[parent] == pytest.approx(ours[parent] * correction)
        assert theirs[parent] != pytest.approx(ours[parent])
    for parent in (2, 3, 5):
        assert ours[parent] is None
        assert theirs[parent] is None


def test_kurtosis_matches_featuretools_only_on_a_fully_known_group():
    """featuretools answers 0 wherever tusk skips a null or finds no spread."""
    ours, theirs = _ours_and_theirs("kurtosis", "kurtosis", "KURTOSIS(children.value)")
    assert ours[4] == pytest.approx(theirs[4])
    assert ours[1] == pytest.approx(-1.5)
    assert theirs[1] == 0.0
    for parent in (2, 3, 5):
        assert ours[parent] is None
        assert theirs[parent] == 0.0


@pytest.mark.parametrize(
    ("tusk_name", "featuretools_column", "children", "featuretools_time_index"),
    [
        ("max_min_delta", "MAX_MIN_DELTA(children.value)", CHILDREN, None),
        (
            "first_last_time_delta",
            "FIRST_LAST_TIME_DELTA(children.seen_at)",
            CHILDREN[CHILDREN["seen_at"].notna()],
            "seen_at",
        ),
    ],
)
def test_a_delta_without_a_known_value_is_null_rather_than_zero(
    tusk_name,
    featuretools_column,
    children,
    featuretools_time_index,
):
    """A group with nothing to subtract is null in tusk and 0 in featuretools.

    featuretools only reads ``first_last_time_delta`` from its time index,
    which may not hold nulls, so that case drops the null datetimes; parent 3
    then has no children at all.
    """
    ours, theirs = _ours_and_theirs(
        tusk_name,
        tusk_name,
        featuretools_column,
        children=children,
        featuretools_time_index=featuretools_time_index,
    )
    assert_values_match([ours[p] for p in (1, 2, 4)], [theirs[p] for p in (1, 2, 4)])
    for parent in (3, 5):
        assert ours[parent] is None
        assert theirs[parent] == 0.0


def test_is_unique_is_null_rather_than_false_without_a_known_value():
    """No known value means nothing to compare; featuretools answers False."""
    ours, theirs = _ours_and_theirs("is_unique", "is_unique", "IS_UNIQUE(children.label)")
    assert_values_match([ours[p] for p in (1, 4)], [theirs[p] for p in (1, 4)])
    for parent in (2, 3, 5):
        assert ours[parent] is None
        assert theirs[parent] is False


def test_is_unique_ignores_nulls_beside_a_known_value():
    """tusk skips the repeated null; featuretools counts it as a repeat."""
    children = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "parent_id": [1, 1, 1],
            "label": pd.array(["d", None, None], dtype="string"),
        },
    )
    ours, theirs = _ours_and_theirs(
        "is_unique",
        "is_unique",
        "IS_UNIQUE(children.label)",
        children=children,
        logical_types={"label": "Categorical"},
    )
    assert ours[1] is True
    assert theirs[1] is False


def test_has_no_duplicates_counts_repeated_nulls_as_duplicates():
    """Two nulls repeat in tusk; featuretools drops nulls before looking."""
    ours, theirs = _ours_and_theirs(
        "has_no_duplicates",
        "has_no_duplicates",
        "HAS_NO_DUPLICATES(children.label)",
    )
    assert_values_match([ours[p] for p in (1, 4, 5)], [theirs[p] for p in (1, 4, 5)])
    for parent in (2, 3):
        assert ours[parent] is False
        assert theirs[parent] is True


def test_percent_unique_of_an_empty_group_is_null_rather_than_zero():
    """A fraction of no rows is undefined in tusk; featuretools reports 0."""
    ours, theirs = _ours_and_theirs(
        "percent_unique",
        "percent_unique",
        "PERCENT_UNIQUE(children.label)",
    )
    assert_values_match([ours[p] for p in (1, 2, 3, 4)], [theirs[p] for p in (1, 2, 3, 4)])
    assert ours[5] is None
    assert theirs[5] == 0.0
```

- [ ] **Step 5: Run the differential suite**

Run: `uv run --group validation pytest -m differential tests/differential/test_aggregations.py -v`
Expected: all PASS. The featuretools values were probed on this fixture; a
failure is a real divergence to report, not an expectation to change.

- [ ] **Step 6: Commit**

```bash
git add tests/differential/test_aggregations.py
git commit -m "test: cross-check the standalone aggregations against featuretools"
```

---

### Task 8: Documentation and the coverage table

**Files:**
- Modify: `docs/api/primitives.md`, `docs/guide/primitives.md`, `docs/guide/primitive-coverage.md`

- [ ] **Step 1: API reference**

In `docs/api/primitives.md`, after `::: tusk.primitives.TimeSinceLastFalse`,
add one block per class, each separated by a blank line: `All`, `Any`,
`NTrue`, `Skew`, `Kurtosis`, `Variance`, `MaxMinDelta`, `FirstLastTimeDelta`,
`IsUnique`, `HasNoDuplicates`, `PercentUnique`, `NUniqueDays`,
`NUniqueDaysOfCalendarYear`, `NUniqueDaysOfMonth`, `NUniqueMonths`, e.g.
`::: tusk.primitives.All`.

- [ ] **Step 2: Primitives guide**

In `docs/guide/primitives.md`:

- The **Aggregation** list becomes: `count`, `sum`, `mean`, `min`, `max`, `std`, `median`, `variance`, `skew`, `kurtosis`, `max_min_delta`, `n_unique`, `is_unique`, `has_no_duplicates`, `percent_unique`, `all`, `any`, `n_true`, `percent_true`, `quantiles`, `first_last_time_delta`, `n_unique_days`, `n_unique_days_of_calendar_year`, `n_unique_days_of_month`, `n_unique_months`, `time_since_first`, `time_since_last`, `time_since_last_true`, `time_since_last_false`.
- **Defaults** becomes: "Passing `agg_primitives=None` or `trans_primitives=None` selects the primitives featuretools uses by default, where tusk has them: `count`, `sum`, `mean`, `min`, `max`, `std`, `n_unique`, `skew`, `percent_true` for aggregation, and `year`, `month`, `day`, `weekday` for transforms."
- Add these rows to the **Empty groups** table, after `PERCENT_TRUE`:

```markdown
| `N_TRUE`, `N_UNIQUE_DAYS`, `N_UNIQUE_DAYS_OF_CALENDAR_YEAR`, `N_UNIQUE_DAYS_OF_MONTH`, `N_UNIQUE_MONTHS` | `0` | Zero rows hold nothing to count. Nulls are not values, so a group of only nulls is also `0`. |
| `ANY` | `false` | No row is true. |
| `HAS_NO_DUPLICATES` | `true` | No value repeats. Two nulls do repeat, so a group of several nulls is `false`. |
| `ALL` | `null` | A group of only nulls is `true`, as polars and duckdb answer, but a group with no rows is left unknown. |
| `IS_UNIQUE` | `null` | Nothing to compare. Nulls are not values, so a group of only nulls is `null` too. |
| `VARIANCE`, `SKEW`, `KURTOSIS`, `MAX_MIN_DELTA`, `FIRST_LAST_TIME_DELTA`, `PERCENT_UNIQUE` | `null` | Undefined over an empty set. `SKEW` and `KURTOSIS` are also `null` for a group whose values do not vary. |
```

- [ ] **Step 3: Coverage table rows**

In `docs/guide/primitive-coverage.md`, replace each row below in place. The
test links point at
`https://github.com/Excidion/tusk/blob/main/tests/differential/test_aggregations.py`;
`<doc>` is `https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives`.

```markdown
| Aggregation | [`all`][tusk.primitives.All] | [`all`](<doc>.All.html) | ✅ | [`test_standalone_aggregations_match_featuretools_on_every_parent_row`](<test link>) |  |
| Aggregation | [`any`][tusk.primitives.Any] | [`any`](<doc>.Any.html) | ⚠️ | [`test_any_of_an_empty_group_is_false_rather_than_null`](<test link>) | An empty group is `False` in tusk and null in featuretools. Nulls are skipped on both sides. |
| Aggregation | [`first_last_time_delta`][tusk.primitives.FirstLastTimeDelta] | `first_last_time_delta` | ⚠️ | [`test_a_delta_without_a_known_value_is_null_rather_than_zero`](<test link>) | Takes any datetime column and returns a `Duration`; featuretools takes only the time index and returns float seconds. A group without a known datetime is null, not zero. |
| Aggregation | [`has_no_duplicates`][tusk.primitives.HasNoDuplicates] | `has_no_duplicates` | ⚠️ | [`test_has_no_duplicates_counts_repeated_nulls_as_duplicates`](<test link>) | Two nulls are a repeat in tusk; featuretools drops nulls first. See `is_unique` for the test that ignores nulls. |
| Aggregation | [`is_unique`][tusk.primitives.IsUnique] | `is_unique` | ⚠️ | [`test_is_unique_is_null_rather_than_false_without_a_known_value`](<test link>) / [`test_is_unique_ignores_nulls_beside_a_known_value`](<test link>) | Nulls are not values in tusk, so a group without a known value is null; featuretools counts nulls as values. |
| Aggregation | [`kurtosis`][tusk.primitives.Kurtosis] | `kurtosis` | ⚠️ | [`test_kurtosis_matches_featuretools_only_on_a_fully_known_group`](<test link>) | tusk skips nulls; featuretools answers `0` for a group holding a null. A constant or empty group is null in tusk and `0` in featuretools. |
| Aggregation | [`max_min_delta`][tusk.primitives.MaxMinDelta] | `max_min_delta` | ⚠️ | [`test_a_delta_without_a_known_value_is_null_rather_than_zero`](<test link>) | A group without a known value is null in tusk and `0` in featuretools. |
| Aggregation | [`n_true`][tusk.primitives.NTrue] | [`num_true`](<doc>.NumTrue.html) | ✅ | [`test_standalone_aggregations_match_featuretools_on_every_parent_row`](<test link>) |  |
| Aggregation | [`n_unique_days`][tusk.primitives.NUniqueDays] | `n_unique_days` | ✅ | [`test_standalone_aggregations_match_featuretools_on_every_parent_row`](<test link>) |  |
| Aggregation | [`n_unique_days_of_calendar_year`][tusk.primitives.NUniqueDaysOfCalendarYear] | `n_unique_days_of_calendar_year` | ✅ | [`test_standalone_aggregations_match_featuretools_on_every_parent_row`](<test link>) |  |
| Aggregation | [`n_unique_days_of_month`][tusk.primitives.NUniqueDaysOfMonth] | `n_unique_days_of_month` | ✅ | [`test_standalone_aggregations_match_featuretools_on_every_parent_row`](<test link>) |  |
| Aggregation | [`n_unique_months`][tusk.primitives.NUniqueMonths] | `n_unique_months` | ✅ | [`test_standalone_aggregations_match_featuretools_on_every_parent_row`](<test link>) |  |
| Aggregation | [`percent_unique`][tusk.primitives.PercentUnique] | `percent_unique` | ⚠️ | [`test_percent_unique_of_an_empty_group_is_null_rather_than_zero`](<test link>) | An empty group is null in tusk and `0` in featuretools. |
| Aggregation | [`skew`][tusk.primitives.Skew] | [`skew`](<doc>.Skew.html) | ⚠️ | [`test_skew_differs_from_featuretools_only_by_its_bias_correction`](<test link>) | Without bias correction, as narwhals computes it; featuretools applies pandas' sample correction. A constant group is null on both sides. |
```

Write every `<doc>` and `<test link>` out in full; they are shorthand here only.

Move the `variance` row from General transform to Aggregation, after
`trend`, as:

```markdown
| Aggregation | [`variance`][tusk.primitives.Variance] | [`variance`](<doc>.Variance.html) | ✅ | [`test_standalone_aggregations_match_featuretools_on_every_parent_row`](<test link>) |  |
```

- [ ] **Step 4: ⛔ reclassifications**

Change these rows' status to ⛔ and their comment as shown:

| Row | Comment |
| --- | --- |
| `count_greater_than`, `count_less_than`, `count_inside_range`, `count_outside_range` | `Threshold argument; a useful threshold depends on the data.` |
| `mode`, `n_most_common`, `n_most_common_frequency` | keep `Length-changing inside a lazy \`group_by().agg()\`.` |
| `exponential_weighted_average`, `exponential_weighted_std`, `exponential_weighted_variance` | `duckdb has no exponentially weighted window.` |
| `date_to_time_zone` | `duckdb keeps the time zone on the connection, not on the column.` |

- [ ] **Step 5: Build the docs**

Run: `just docs`
Expected: builds without warnings about unresolved `tusk.primitives.*` references.

- [ ] **Step 6: Commit**

```bash
git add docs/api/primitives.md docs/guide/primitives.md docs/guide/primitive-coverage.md
git commit -m "docs: document the standalone aggregations and reclassify out-of-scope rows"
```

---

### Task 9: Gate and pull request

- [ ] **Step 1: Run everything CI runs**

Run: `just check`
Expected: lint, tests and differential tests all pass. Fix any finding at its
source; do not skip hooks.

- [ ] **Step 2: Push and open the pull request**

```bash
git push -u origin worktree-standalone-aggregations
gh pr create --title "Standalone aggregation primitives" --body "$(cat <<'EOF'
Adds fifteen aggregation primitives that fit inside one lazy `group_by().agg()`: `all`, `any`, `n_true`, `skew`, `kurtosis`, `variance`, `max_min_delta`, `first_last_time_delta`, `is_unique`, `has_no_duplicates`, `percent_unique`, `n_unique_days`, `n_unique_days_of_calendar_year`, `n_unique_days_of_month`, `n_unique_months`.

- Every primitive is checked on polars and duckdb over a mixed, constant, all-null and empty group, and against featuretools 1.31.0. Seven rows are ✅; eight are ⚠️ with the divergence asserted on both sides.
- `skew` and `kurtosis` are null for a constant group, which is the only way polars and duckdb agree there.
- Defaults now match featuretools': `skew`, `percent_true` and `day` join them.
- Rows no roadmap phase will implement are marked ⛔ with the reason.

Phase 1 of `agents/superpowers/specs/2026-09-13-primitive-parity-roadmap-design.md`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
