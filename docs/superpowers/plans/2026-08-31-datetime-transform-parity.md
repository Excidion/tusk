# Datetime Transform Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give tusk a `time_since` transform measured against the cutoff time, and a `Duration` dtype family with extractor primitives that turn a duration into a number.

**Architecture:** `DtypeFamily.TEMPORAL` splits into narrower `DATETIME` and `DURATION` families so DFS stacking lands on the right primitives. A `NeedsCutoffTime` mixin carries the cutoff time as state on the primitive instance; the compiler validates it is present and binds it with `dataclasses.replace` at apply time, so `build()` stays pure and no other primitive changes.

**Tech Stack:** Python 3.10+, narwhals (backend-agnostic expressions), polars (test backend), duckdb (SQL portability tests), pytest, featuretools 1.31.0 (differential reference only).

**Spec:** `docs/superpowers/specs/2026-08-30-datetime-transform-parity-design.md`

## Global Constraints

- **One commit per task.** Each task's commit covers its source, tests, and any doc row it changes.
- **`CODESTYLE.md` is binding.** Newspaper ordering (callers above callees, high-level above detail); descriptive pronounceable names, no abbreviations; docstrings say *what* a thing does; comments explain *why* only and never restate the code; no commented-out code.
- **One word per concept:** `cutoff_time` everywhere — the parameter, the field, the prose. Never `reference_time`, `cutoff`, or `clock`.
- **`time_since` is `cutoff_time - value`**, so a past timestamp yields a positive duration and a timestamp after the cutoff yields a negative one.
- **All four extractors truncate toward zero**, matching narwhals' own convention. Never use `//`, which floors toward negative infinity and would disagree with the native methods on negative durations.
- **No years, no months.** `age`, `total_years` and `total_months` are out of scope; `age` keeps its ❌ coverage row.
- **Nothing joins `TRANS_DEFAULTS`.** Every new primitive is opt-in through `trans_primitives=[...]`.
- **Verification before each commit:** `uv run pytest` and `uv run ruff check .`, plus `uv run --group validation pytest -m differential` for any task touching the differential suite. Report the output.

## File Structure

| File | Responsibility |
| --- | --- |
| `src/tusk/dtypes.py` | Modify: add `DATETIME` and `DURATION` families and their `matches()` branches |
| `src/tusk/primitives/base.py` | Modify: add the `NeedsCutoffTime` mixin |
| `src/tusk/primitives/transform.py` | Modify: narrow seven primitives to `DATETIME`; add `TimeSince` and four extractors; change `TimeSincePrevious` to return `Duration` |
| `src/tusk/compiler.py` | Modify: validate the cutoff time is present, and bind it into row-wise primitives |
| `tests/test_dtypes.py` | Family membership, including the regression that a `Duration` is not a `DATETIME` |
| `tests/test_primitives_transform.py` | Unit tests for the new primitives and their truncation semantics |
| `tests/test_sklearn_encoders.py` | `dtype_selector` over the new families |
| `tests/test_backend_duckdb.py` | SQL translation of `time_since` and the extractors |
| `tests/differential/test_datetime_transforms.py` | Create: per-primitive comparison against featuretools |
| `docs/guide/primitive-coverage.md` | Row updates |

---

### Task 1: Split the temporal dtype family

Fixes a live bug: `Duration.is_temporal()` is `True`, so a `Duration` column currently attracts `year`/`month`/`hour`/`weekday`/`is_weekend` and DFS crashes with `InvalidOperationError: 'year' operation not supported for dtype 'duration[μs]'`.

**Files:**
- Modify: `src/tusk/dtypes.py`
- Modify: `src/tusk/primitives/transform.py` (seven `input_dtypes` declarations)
- Test: `tests/test_dtypes.py`, `tests/test_sklearn_encoders.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `DtypeFamily.DATETIME` and `DtypeFamily.DURATION`, with string values `"datetime"` and `"duration"`. Later tasks declare `input_dtypes = (F.DATETIME,)` or `(F.DURATION,)`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_dtypes.py`:

```python
def test_duration_is_temporal_but_not_datetime():
    """A Duration is temporal, yet must not match the calendar family.

    Calendar primitives declare DATETIME. Before the split they declared
    TEMPORAL, which a Duration satisfies, so DFS generated YEAR(duration)
    and polars raised InvalidOperationError.
    """
    duration = nw.Duration()
    assert matches(duration, DtypeFamily.TEMPORAL)
    assert matches(duration, DtypeFamily.DURATION)
    assert not matches(duration, DtypeFamily.DATETIME)


def test_datetime_and_date_are_datetime_but_not_duration():
    for dtype in (nw.Datetime(), nw.Date()):
        assert matches(dtype, DtypeFamily.TEMPORAL)
        assert matches(dtype, DtypeFamily.DATETIME)
        assert not matches(dtype, DtypeFamily.DURATION)
```

In `tests/test_sklearn_encoders.py`:

```python
def test_dtype_selector_separates_datetime_from_duration():
    """The narrow families are the supported way to route a duration column."""
    frame = pl.DataFrame(
        {
            "when": [dt.datetime(2024, 1, 1)],
            "elapsed": [dt.timedelta(hours=3)],
            "n": [1],
        },
    )
    assert dtype_selector("datetime")(frame) == ["when"]
    assert dtype_selector("duration")(frame) == ["elapsed"]
    assert dtype_selector("temporal")(frame) == ["when", "elapsed"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_dtypes.py tests/test_sklearn_encoders.py -v`
Expected: FAIL with `AttributeError: DATETIME` / `ValueError: 'datetime' is not a valid DtypeFamily`.

- [ ] **Step 3: Add the families**

In `src/tusk/dtypes.py`, add to the `DtypeFamily` enum, after `TEMPORAL`:

```python
    DATETIME = "datetime"
    DURATION = "duration"
```

Extend the enum's class docstring to say why three temporal families exist: `TEMPORAL` is the union kept for `dtype_selector`, while primitives declare the narrow family they can actually compute.

In `matches()`, add these branches before the `raise`:

```python
    if family is DtypeFamily.DATETIME:
        return dtype == nw.Datetime or dtype == nw.Date
    if family is DtypeFamily.DURATION:
        return dtype == nw.Duration
```

- [ ] **Step 4: Narrow the calendar primitives**

In `src/tusk/primitives/transform.py`, change `input_dtypes = (F.TEMPORAL,)` to `input_dtypes = (F.DATETIME,)` in all seven classes: `Year`, `Month`, `Weekday`, `Day`, `Hour`, `IsWeekend`, `TimeSincePrevious`.

Verify none remain: `grep -n "F.TEMPORAL" src/tusk/primitives/transform.py` must print nothing.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_dtypes.py tests/test_sklearn_encoders.py -v`
Expected: PASS.

- [ ] **Step 6: Run the full suite and linter**

Run: `uv run pytest` then `uv run ruff check .`
Expected: all pass. If a test asserted `F.TEMPORAL` on a calendar primitive, update it to `F.DATETIME` — that assertion was pinning the bug.

- [ ] **Step 7: Commit**

```bash
git add src/tusk/dtypes.py src/tusk/primitives/transform.py tests/test_dtypes.py tests/test_sklearn_encoders.py
git commit -m "fix: stop calendar primitives matching Duration columns"
```

---

### Task 2: The cutoff-time mixin and `time_since`

**Files:**
- Modify: `src/tusk/primitives/base.py`
- Modify: `src/tusk/primitives/transform.py`
- Modify: `src/tusk/compiler.py`
- Test: `tests/test_primitives_transform.py`

**Interfaces:**
- Consumes: `DtypeFamily.DATETIME` from Task 1.
- Produces: `NeedsCutoffTime`, a frozen dataclass mixin with one field `cutoff_time: datetime | None = None`, importable from `tusk.primitives.base`. Also the registered primitive name `"time_since"`, whose output column for input `x` is `TIME_SINCE__x` with dtype `Duration`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_primitives_transform.py`:

```python
def test_time_since_measures_from_the_cutoff_time():
    """A past timestamp gives a positive duration, a future one negative."""
    frame = nw.from_native(
        pl.LazyFrame(
            {"t": [dt.datetime(2024, 1, 1), dt.datetime(2024, 6, 1), None]},
        ),
    )
    primitive = TimeSince(cutoff_time=dt.datetime(2024, 3, 1))
    got = frame.select(primitive.outputs(nw.col("t"))[0].alias("o")).collect()
    assert got.to_native()["o"].to_list() == [
        dt.timedelta(days=60),
        dt.timedelta(days=-92),
        None,
    ]


def test_time_since_needs_a_datetime_input():
    assert TimeSince().input_dtypes == (DtypeFamily.DATETIME,)
    assert isinstance(TimeSince(), NeedsCutoffTime)


def test_deep_feature_synthesis_rejects_time_since_without_a_cutoff_time(db):
    """The clock is the cutoff time, so there is no answer without one."""
    with pytest.raises(ValidationError, match="time_since needs a cutoff_time"):
        tusk.deep_feature_synthesis(
            database=db,
            target_table="customers",
            agg_primitives=[],
            trans_primitives=["time_since"],
            max_depth=1,
        )


def test_deep_feature_synthesis_computes_time_since_with_a_cutoff_time(db):
    matrix, _ = tusk.deep_feature_synthesis(
        database=db,
        target_table="customers",
        agg_primitives=[],
        trans_primitives=["time_since"],
        max_depth=1,
        cutoff_time=dt.datetime(2024, 3, 1),
    )
    got = matrix.collect().sort("id").to_native()
    # customers.signed_up_at is 2024-01-01 for all three rows (tests/conftest.py)
    assert got["TIME_SINCE__signed_up_at"].to_list() == [dt.timedelta(days=60)] * 3
```

The `db` fixture is in `tests/conftest.py`; its `customers` table has `signed_up_at = 2024-01-01` for ids 1, 2, 3.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_primitives_transform.py -k time_since -v`
Expected: FAIL with `NameError: TimeSince` / `ImportError: NeedsCutoffTime`.

- [ ] **Step 3: Add the mixin**

In `src/tusk/primitives/base.py`, after the imports add `from dataclasses import dataclass` and `from datetime import datetime`, then at the end of the module:

```python
@dataclass(frozen=True)
class NeedsCutoffTime:
    """A primitive whose value is measured against the cutoff time.

    Attributes:
        cutoff_time: The moment values are measured against. The compiler
            fills this in; a primitive built by hand carries None until then.
    """

    cutoff_time: datetime | None = None
```

The mixin is the marker: `isinstance(primitive, NeedsCutoffTime)` is what drives both validation and binding, so no parallel boolean flag can disagree with the type.

- [ ] **Step 4: Add the primitive**

In `src/tusk/primitives/transform.py`, import `NeedsCutoffTime` alongside `TransformPrimitive`, then add:

```python
@register
@dataclass(frozen=True)
class TimeSince(NeedsCutoffTime, TransformPrimitive):
    """Time elapsed from a datetime to the cutoff time."""

    name = "time_since"
    input_dtypes = (F.DATETIME,)
    output_dtype = nw.Duration

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the elapsed-time expression.

        Args:
            expr: A datetime expression.

        Returns:
            A narwhals expression of the duration since each value.
        """
        return nw.lit(self.cutoff_time) - expr
```

Place it in newspaper order relative to its neighbours. Do not add it to `TRANS_DEFAULTS`.

- [ ] **Step 5: Validate and bind in the compiler**

In `src/tusk/compiler.py`, import `dataclasses`, `ValidationError`, and `NeedsCutoffTime`.

In `compile_features`, after the existing `_reject_colliding_names(features)` call:

```python
    _require_cutoff_time(_closure(features), cutoff_time)
```

Add the helper below `_reject_colliding_names`, keeping callers above callees:

```python
def _require_cutoff_time(features: set[Feature], cutoff_time: datetime | None) -> None:
    """Fail if a primitive measures against a cutoff time that was not given.

    Args:
        features: The transitive closure of features to compile.
        cutoff_time: The cutoff, or None.

    Raises:
        ValidationError: If any primitive needs the cutoff time and none was
            given.
    """
    if cutoff_time is not None:
        return
    needed = sorted(
        {
            primitive.name
            for feature in features
            if isinstance(primitive := getattr(feature, "primitive", None), NeedsCutoffTime)
        },
    )
    if needed:
        raise ValidationError(
            f"{', '.join(needed)} needs a cutoff_time; pass one to "
            "deep_feature_synthesis",
        )
```

`getattr` is required because `IdentityFeature` carries no primitive.

Change the `_apply` call site in `_table_frame` to pass the cutoff through:

```python
        frame = _apply(frame, feature, database, cutoff_time)
```

Then in `_apply`, add the parameter (documenting it in the docstring's `Args`) and bind before building:

```python
    primitive = feature.primitive
    if isinstance(primitive, NeedsCutoffTime):
        primitive = dataclasses.replace(primitive, cutoff_time=cutoff_time)
    exprs = list(primitive.outputs(*inputs))
```

Leave the existing `getattr(feature.primitive, "order_dependent", False)` check reading from `feature.primitive`; only the expression build uses the bound copy.

Binding for aggregation features is not part of this plan — no aggregation primitive uses `NeedsCutoffTime` yet.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_primitives_transform.py -k time_since -v`
Expected: PASS.

- [ ] **Step 7: Run the full suite and linter**

Run: `uv run pytest` then `uv run ruff check .`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/tusk/primitives/base.py src/tusk/primitives/transform.py src/tusk/compiler.py tests/test_primitives_transform.py
git commit -m "feat: add time_since, measured against the cutoff time"
```

---

### Task 3: Duration extractors

**Files:**
- Modify: `src/tusk/primitives/transform.py`
- Test: `tests/test_primitives_transform.py`

**Interfaces:**
- Consumes: `DtypeFamily.DURATION` from Task 1.
- Produces: registered names `"total_seconds"`, `"total_minutes"`, `"total_hours"`, `"total_days"`. Each takes one `DURATION` input and returns `Int64`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_primitives_transform.py`:

```python
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("total_seconds", [90, 82800, 90000]),
        ("total_minutes", [1, 1380, 1500]),
        ("total_hours", [0, 23, 25]),
        ("total_days", [0, 0, 1]),
    ],
)
def test_extractors_truncate_to_whole_units(name, expected):
    """90.5s, 23h and 25h through each extractor.

    Every extractor answers "how many whole units have elapsed", so 90.5
    seconds is 1 whole minute and 23 hours is 0 whole days.
    """
    frame = nw.from_native(
        pl.LazyFrame(
            {
                "d": [
                    dt.timedelta(seconds=90, microseconds=500000),
                    dt.timedelta(hours=23),
                    dt.timedelta(hours=25),
                ],
            },
        ),
    )
    primitive = resolve(name)
    got = frame.select(primitive.outputs(nw.col("d"))[0].alias("o")).collect()
    assert got.to_native()["o"].to_list() == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("total_seconds", [-90, -82800]),
        ("total_minutes", [-1, -1380]),
        ("total_hours", [0, -23]),
        ("total_days", [0, 0]),
    ],
)
def test_extractors_truncate_toward_zero_on_negative_durations(name, expected):
    """time_since is negative after the cutoff, so this case is reachable.

    narwhals' own total_seconds and total_minutes truncate toward zero rather
    than flooring, so the derived hours and days must too: a floor would make
    -90.5 seconds into -1 whole hours while total_seconds called it -90.
    """
    frame = nw.from_native(
        pl.LazyFrame(
            {
                "d": [
                    dt.timedelta(seconds=-90, microseconds=-500000),
                    dt.timedelta(hours=-23),
                ],
            },
        ),
    )
    primitive = resolve(name)
    got = frame.select(primitive.outputs(nw.col("d"))[0].alias("o")).collect()
    assert got.to_native()["o"].to_list() == expected


def test_extractors_take_a_duration_input():
    for name in ("total_seconds", "total_minutes", "total_hours", "total_days"):
        assert resolve(name).input_dtypes == (DtypeFamily.DURATION,)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_primitives_transform.py -k extractors -v`
Expected: FAIL with `PrimitiveError: unknown primitive 'total_seconds'`.

- [ ] **Step 3: Implement the extractors**

In `src/tusk/primitives/transform.py`:

```python
@register
@dataclass(frozen=True)
class TotalSeconds(TransformPrimitive):
    """Whole seconds in a duration."""

    name = "total_seconds"
    input_dtypes = (F.DURATION,)
    output_dtype = nw.Int64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the whole-seconds expression.

        Args:
            expr: A duration expression.

        Returns:
            A narwhals expression of whole seconds.
        """
        return expr.dt.total_seconds()


@register
@dataclass(frozen=True)
class TotalMinutes(TransformPrimitive):
    """Whole minutes in a duration."""

    name = "total_minutes"
    input_dtypes = (F.DURATION,)
    output_dtype = nw.Int64

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the whole-minutes expression.

        Args:
            expr: A duration expression.

        Returns:
            A narwhals expression of whole minutes.
        """
        return expr.dt.total_minutes()


@register
@dataclass(frozen=True)
class TotalHours(TransformPrimitive):
    """Whole hours in a duration."""

    name = "total_hours"
    input_dtypes = (F.DURATION,)
    output_dtype = nw.Int64
    seconds_per_hour = 3600

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the whole-hours expression.

        narwhals has no ``total_hours``, and its own conversions truncate
        toward zero rather than flooring, so dividing and casting matches
        them where ``//`` would disagree on negative durations.

        Args:
            expr: A duration expression.

        Returns:
            A narwhals expression of whole hours.
        """
        return (expr.dt.total_seconds() / self.seconds_per_hour).cast(nw.Int64)


@register
@dataclass(frozen=True)
class TotalDays(TransformPrimitive):
    """Whole days in a duration."""

    name = "total_days"
    input_dtypes = (F.DURATION,)
    output_dtype = nw.Int64
    seconds_per_day = 86400

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the whole-days expression.

        narwhals has no ``total_days``, and its own conversions truncate
        toward zero rather than flooring, so dividing and casting matches
        them where ``//`` would disagree on negative durations.

        Args:
            expr: A duration expression.

        Returns:
            A narwhals expression of whole days.
        """
        return (expr.dt.total_seconds() / self.seconds_per_day).cast(nw.Int64)
```

Do not add any of these to `TRANS_DEFAULTS`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_primitives_transform.py -k extractors -v`
Expected: PASS.

- [ ] **Step 5: Verify stacking works end to end**

Add:

```python
def test_extractors_stack_on_time_since(db):
    """The composition that replaces featuretools' unit= parameter."""
    matrix, _ = tusk.deep_feature_synthesis(
        database=db,
        target_table="customers",
        agg_primitives=[],
        trans_primitives=["time_since", "total_days"],
        max_depth=2,
        cutoff_time=dt.datetime(2024, 3, 1),
    )
    got = matrix.collect().sort("id").to_native()
    assert got["TOTAL_DAYS__TIME_SINCE__signed_up_at"].to_list() == [60, 60, 60]
```

Run: `uv run pytest tests/test_primitives_transform.py -k stack -v`
Expected: PASS.

- [ ] **Step 6: Run the full suite and linter**

Run: `uv run pytest` then `uv run ruff check .`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/tusk/primitives/transform.py tests/test_primitives_transform.py
git commit -m "feat: add duration extractors for seconds, minutes, hours and days"
```

---

### Task 4: `TimeSincePrevious` returns a duration

**Files:**
- Modify: `src/tusk/primitives/transform.py`
- Test: `tests/test_primitives_transform.py`

**Interfaces:**
- Consumes: `DtypeFamily.DATETIME` from Task 1, and the extractors from Task 3 for the stacking test.
- Produces: `TIME_SINCE_PREVIOUS__x` changes dtype from `Float64` to `Duration`.

- [ ] **Step 1: Update the existing test to the new contract**

`TimeSincePrevious` currently returns `Float64` seconds via `expr.diff().dt.total_seconds().cast(nw.Float64)`. Find its existing tests with `grep -rn "time_since_previous\|TimeSincePrevious" tests/` and change their expectations from float seconds to `datetime.timedelta` values. Add:

```python
def test_time_since_previous_returns_a_duration():
    """One type for every elapsed-time primitive in tusk.

    featuretools returns float seconds here; tusk returns a Duration so the
    extractors stack on it the same way they stack on time_since.
    """
    assert TimeSincePrevious().output_dtype == nw.Duration
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_primitives_transform.py -k time_since_previous -v`
Expected: FAIL — `output_dtype` is still `nw.Float64`.

- [ ] **Step 3: Change the primitive**

In `src/tusk/primitives/transform.py`, in `TimeSincePrevious`:

```python
    output_dtype = nw.Duration

    def build(self, expr: nw.Expr) -> nw.Expr:
        """Build the elapsed-time expression.

        Args:
            expr: A datetime expression.

        Returns:
            A narwhals expression of the duration since the previous row.
        """
        return expr.diff()
```

Update the class docstring from "Seconds elapsed since the previous row in row-creation order" to "Time elapsed since the previous row in row-creation order."

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_primitives_transform.py -k time_since_previous -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite and linter**

Run: `uv run pytest` then `uv run ruff check .`
Expected: all pass. Any other test asserting float seconds from this primitive needs updating to a `timedelta`.

- [ ] **Step 6: Commit**

```bash
git add src/tusk/primitives/transform.py tests/test_primitives_transform.py
git commit -m "refactor: time_since_previous returns a Duration"
```

---

### Task 5: Differential suite, duckdb portability, and coverage rows

**Files:**
- Create: `tests/differential/test_datetime_transforms.py`
- Modify: `tests/test_backend_duckdb.py`
- Modify: `docs/guide/primitive-coverage.md`

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces: nothing later tasks depend on.

- [ ] **Step 1: Write the differential test**

Create `tests/differential/test_datetime_transforms.py`, modelled on `tests/differential/test_aggregations.py` but self-contained (do not import from it). Carry `pytestmark = pytest.mark.differential` and gate `numpy`, `pandas`, `polars` and `featuretools` behind `pytest.importorskip`, exactly as that file does.

Use whole-second timestamps only, so the extractor's truncation is not in play and any difference is a real disagreement about elapsed time.

```python
def test_time_since_matches_featuretools_on_every_row(rows):
    """tusk's TOTAL_SECONDS(TIME_SINCE(x)) equals featuretools' TIME_SINCE(x).

    Covers a null datetime, which must stay null on both sides, and a
    timestamp after the cutoff, where the duration is negative.
    """
```

The featuretools side is a single-entry `trans_primitives=["time_since"]` with a `cutoff_time`; the tusk side is `trans_primitives=["time_since", "total_seconds"]` at `max_depth=2`, comparing tusk's `TOTAL_SECONDS__TIME_SINCE__x` against featuretools' `TIME_SINCE(x)` as floats. Assert the fixture's own invariants before comparing — that some value is null and some timestamp is after the cutoff — so the coverage claim cannot outlive the coverage.

- [ ] **Step 2: Run it**

Run: `uv run --group validation pytest -m differential -k datetime -v`
Expected: PASS. If tusk and featuretools disagree, **stop and report the values** rather than changing tusk — a disagreement is a question for the maintainer, not a bug to fix.

- [ ] **Step 3: Add the duckdb portability tests**

In `tests/test_backend_duckdb.py`, following the shape of the existing `test_percent_true_holds_the_null_rule_on_duckdb`, add one test that synthesizes `time_since` and `total_days` over the existing `transactions.occurred_at` data with a fixed `cutoff_time`, and asserts the values. This checks narwhals' SQL translation of both the datetime subtraction and the divide-and-cast.

If duckdb rejects either expression or returns different values, **do not work around it and do not weaken the test** — report the exact error or values.

- [ ] **Step 4: Run the duckdb tests**

Run: `uv run pytest tests/test_backend_duckdb.py -v`
Expected: PASS.

- [ ] **Step 5: Update the coverage table**

In `docs/guide/primitive-coverage.md`, matching the exact link format of neighbouring rows (mkdocstrings `[...][tusk.primitives.X]` for tusk, absolute `https://github.com/Excidion/tusk/blob/main/...` URLs for tests):

| Row | Change |
| --- | --- |
| `time_since` | tusk column becomes `` [`time_since`][tusk.primitives.TimeSince] ``, Status ❌ → ✅, Test links the new differential test, Comment cleared |
| `time_since_previous` | Status ❓ → ⚠️, Comment: "Returns a `Duration`; featuretools returns float seconds." |
| four new ➕ rows | `total_seconds`, `total_minutes`, `total_hours`, `total_days` under Datetime transform, featuretools column `—`, Test linking the unit tests, Comment: "Unit test only, since featuretools has no counterpart." |

Leave the `age` row unchanged at ❌ with its existing comment.

- [ ] **Step 6: Verify the docs reference resolves**

Run: `grep -n "TimeSince\|TotalSeconds\|TotalMinutes\|TotalHours\|TotalDays" docs/api/primitives.md`

If the new classes are not listed there, add them alongside the existing primitive entries — a `[...][tusk.primitives.X]` link to a class mkdocstrings does not document renders as broken text.

- [ ] **Step 7: Run everything**

Run: `uv run pytest`, `uv run --group validation pytest -m differential`, `uv run ruff check .`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add tests/differential/test_datetime_transforms.py tests/test_backend_duckdb.py docs/guide/primitive-coverage.md docs/api/primitives.md
git commit -m "test: cross-check time_since against featuretools and duckdb"
```

---

## Self-Review

**Spec coverage.** Dtype split → Task 1. `dtype_selector` over the new families → Task 1. `NeedsCutoffTime`, validation, binding → Task 2. `time_since` → Task 2. Four extractors and their floor semantics → Task 3. `TimeSincePrevious` → Task 4. Differential parity, duckdb portability, coverage rows → Task 5. The regression test for the `Duration` crash → Task 1. Depth documentation is covered by the coverage-table comments and the `time_since` docstring; no separate task.

**Deviation from the spec, deliberate.** The spec says the extractors "floor"; verification showed narwhals' `total_seconds`/`total_minutes` truncate toward zero rather than flooring, and the two differ on negative durations, which `time_since` produces. The plan uses truncation throughout for internal consistency and pins it with `test_extractors_truncate_toward_zero_on_negative_durations`. The spec's own example (90.5s → 90, 23h → 0 days) is unaffected.

**Placeholders.** None: every code step carries the code, every test step the assertions, every run step the command and expected result.

**Type consistency.** `NeedsCutoffTime.cutoff_time` is `datetime | None` in Task 2's definition, its `_require_cutoff_time` signature, and the `_apply` binding. Family names `DATETIME`/`DURATION` are identical in Tasks 1, 2 and 3. Registered names `time_since`, `total_seconds`, `total_minutes`, `total_hours`, `total_days` match between definition, tests and coverage rows. Output column names follow `generate_name`'s `__`-joined form throughout.
