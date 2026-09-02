# Datetime Transform Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give tusk a `time_since` transform measured against the cutoff time, returning a `Duration`, and the dtype families needed for a duration to be routed rather than mis-matched.

**Architecture:** `DtypeFamily.TEMPORAL` splits into narrower `HAS_DATE`, `HAS_TIME` and `DURATION` families, named for the operation a primitive needs and overlapping on `Datetime`, so each primitive matches only columns it can compute over. `NeedsCutoffTime` primitives store nothing: the compiler passes the cutoff time to `outputs()`, which forwards it to `build()`, so one `FeatureList` stays applicable at several cutoff times.

**Tech Stack:** Python 3.10+, narwhals (backend-agnostic expressions), polars (test backend), duckdb (SQL portability tests), pytest, featuretools 1.31.0 (differential reference only).

**Spec:** `superpowers/specs/2026-08-30-datetime-transform-parity-design.md`

## Global Constraints

- **One commit per task.** Each task's commit covers its source, tests, and any doc row it changes.
- **`CODESTYLE.md` is binding.** Newspaper ordering (callers above callees, high-level above detail); descriptive pronounceable names, no abbreviations; docstrings say *what* a thing does; comments explain *why* only and never restate the code; no commented-out code.
- **One word per concept:** `cutoff_time` everywhere — the parameter, the field, the prose. Never `reference_time`, `cutoff`, or `clock`.
- **`time_since` is `cutoff_time - value`**, so a past timestamp yields a positive duration and a timestamp after the cutoff yields a negative one.
- **A duration stays a duration.** tusk builds features, not encodings; converting one to a number is the caller's choice. No extractor primitives, and no `age`, `total_years` or `total_months`.
- **Nothing joins `TRANS_DEFAULTS`.** Every new primitive is opt-in through `trans_primitives=[...]`.
- **Verification before each commit:** `uv run pytest` and `uv run ruff check .`, plus `uv run --group validation pytest -m differential` for any task touching the differential suite. Report the output.

## File Structure

| File | Responsibility |
| --- | --- |
| `src/tusk/dtypes.py` | Modify: add `HAS_DATE`, `HAS_TIME` and `DURATION` families and their `matches()` branches |
| `src/tusk/primitives/base.py` | Modify: add the `NeedsCutoffTime` mixin |
| `src/tusk/primitives/transform.py` | Modify: narrow the calendar primitives; add `TimeSince`; change `TimeSincePrevious` to return `Duration` |
| `src/tusk/compiler.py` | Modify: validate the cutoff time is present, and bind it into row-wise primitives |
| `tests/test_dtypes.py` | Family membership, including the regression that a `Duration` is not a `DATETIME` |
| `tests/test_primitives_transform.py` | Unit tests for the new primitives |
| `tests/test_sklearn_encoders.py` | `dtype_selector` over the new families |
| `tests/test_backend_duckdb.py` | SQL translation of `time_since` |
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
- Produces: `NeedsCutoffTime`, a `Primitive` subclass whose `outputs()` takes `cutoff_time` as a keyword argument and forwards it to `build()`, importable from `tusk.primitives.base`. Nothing is stored on the instance. Also the registered primitive name `"time_since"`, whose output column for input `x` is `TIME_SINCE__x` with dtype `Duration`.

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

- [ ] **Step 3: Add the base class**

In `src/tusk/primitives/base.py`, after the imports add `from datetime import datetime`, then at the end of the module:

```python
class NeedsCutoffTime(Primitive):
    """A primitive that measures against the cutoff time.

    The cutoff time describes the question being asked rather than the
    feature, so it is never stored on a primitive: the compiler passes it in
    at the moment the expression is built. That is what lets one
    :class:`~tusk.FeatureList` be applied at several cutoff times.
    """

    def outputs(self, *inputs: nw.Expr, cutoff_time: datetime) -> tuple[nw.Expr, ...]:
        """Normalize :meth:`build` to a tuple of expressions.

        Args:
            *inputs: One expression per declared input.
            cutoff_time: The moment the values are measured against.

        Returns:
            One expression per output column.
        """
        return _as_tuple(self.build(*inputs, cutoff_time=cutoff_time))

    @abstractmethod
    def build(
        self,
        *inputs: nw.Expr,
        cutoff_time: datetime,
    ) -> nw.Expr | Sequence[nw.Expr]:
        """Build this primitive's narwhals expression.

        Args:
            *inputs: One expression per declared input.
            cutoff_time: The moment the values are measured against.

        Returns:
            A single expression, or a sequence for multi-output primitives.
        """
```

`isinstance(primitive, NeedsCutoffTime)` is what drives both validation and binding, so no parallel boolean flag can disagree with the type. It subclasses `Primitive` rather than being a plain mixin because a mixin left the type checker reconciling two conflicting `outputs` signatures at the compiler's `isinstance` branch.

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

    def build(self, expr: nw.Expr, *, cutoff_time: datetime) -> nw.Expr:
        """Build the elapsed-time expression.

        Args:
            expr: A datetime expression.
            cutoff_time: The moment the values are measured against.

        Returns:
            A narwhals expression of the duration since each value.
        """
        return nw.lit(cutoff_time) - expr
```

Place it in newspaper order relative to its neighbours. Do not add it to `TRANS_DEFAULTS`.

- [ ] **Step 5: Validate and bind in the compiler**

In `src/tusk/compiler.py`, import `ValidationError` and `NeedsCutoffTime`.

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
            if isinstance(
                primitive := getattr(feature, "primitive", None), NeedsCutoffTime
            )
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

Then in `_apply`, add the parameter (documenting it in the docstring's `Args`) and thread it through a helper that passes `cutoff_time` as a keyword to `outputs()` only when the primitive needs it — nothing is ever assigned onto the primitive itself:

```python
    inputs = [nw.col(b.name) for b in feature.base_features]
    exprs = list(_build_expressions(feature.primitive, inputs, cutoff_time))
```

```python
def _build_expressions(
    primitive: Primitive,
    inputs: Sequence[nw.Expr],
    cutoff_time: datetime | None,
) -> tuple[nw.Expr, ...]:
    """Build a primitive's output expressions, threading the cutoff time when needed."""
    if isinstance(primitive, NeedsCutoffTime) and cutoff_time is not None:
        return primitive.outputs(*inputs, cutoff_time=cutoff_time)
    return primitive.outputs(*inputs)
```

Leave the existing `getattr(feature.primitive, "order_dependent", False)` check reading from `feature.primitive`; it needs no cutoff time.

Binding for aggregation features is not part of this task — no aggregation primitive uses `NeedsCutoffTime` yet. It is added in a later commit, once `_add_aggregations` also threads `cutoff_time` through `_build_expressions`.

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

### Task 3: `TimeSincePrevious` returns a duration

**Files:**
- Modify: `src/tusk/primitives/transform.py`
- Test: `tests/test_primitives_transform.py`

**Interfaces:**
- Consumes: `DtypeFamily.DATETIME` from Task 1.
- Produces: `TIME_SINCE_PREVIOUS__x` changes dtype from `Float64` to `Duration`.

- [ ] **Step 1: Update the existing test to the new contract**

`TimeSincePrevious` currently returns `Float64` seconds via `expr.diff().dt.total_seconds().cast(nw.Float64)`. Find its existing tests with `grep -rn "time_since_previous\|TimeSincePrevious" tests/` and change their expectations from float seconds to `datetime.timedelta` values. Add:

```python
def test_time_since_previous_returns_a_duration():
    """One type for every elapsed-time primitive in tusk.

    featuretools returns float seconds here; tusk returns a Duration, the
    same type time_since returns, and leaves the choice of unit to the
    caller.
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

### Task 4: Differential suite, duckdb portability, and coverage rows

**Files:**
- Create: `tests/differential/test_datetime_transforms.py`
- Modify: `tests/test_backend_duckdb.py`
- Modify: `docs/guide/primitive-coverage.md`

**Interfaces:**
- Consumes: `time_since` and `TimeSincePrevious` from Tasks 2 and 3.
- Produces: nothing later tasks depend on.

- [ ] **Step 1: Write the differential test**

Create `tests/differential/test_datetime_transforms.py`, modelled on `tests/differential/test_aggregations.py` but self-contained (do not import from it). Carry `pytestmark = pytest.mark.differential` and gate `numpy`, `pandas`, `polars` and `featuretools` behind `pytest.importorskip`, exactly as that file does.

featuretools' `TIME_SINCE` returns float seconds; tusk's returns a `Duration`. Convert tusk's column to seconds **in the test** — `.dt.total_seconds()` on the collected frame — not through a primitive. tusk ships no duration-to-number primitive by design.

Use whole-second timestamps so the conversion is exact and any difference is a real disagreement about elapsed time.

```python
def test_time_since_matches_featuretools_on_every_row(rows):
    """tusk's TIME_SINCE, converted to seconds, equals featuretools' TIME_SINCE.

    Covers a null datetime, which must stay null on both sides, and a
    timestamp after the cutoff time, where the elapsed time is negative.
    """
```

Both sides use a single-entry `trans_primitives=["time_since"]` with the same `cutoff_time`. Assert the fixture's own invariants before comparing — that some value is null and some timestamp falls after the cutoff time — so the coverage claim cannot outlive the coverage.

- [ ] **Step 2: Run it**

Run: `uv run --group validation pytest -m differential -k datetime -v`
Expected: PASS. If tusk and featuretools disagree on a value, **stop and report the numbers** rather than changing tusk — a disagreement is a question for the maintainer, not a bug to fix.

- [ ] **Step 3: Add the duckdb portability test**

In `tests/test_backend_duckdb.py`, following the shape of the existing `test_percent_true_holds_the_null_rule_on_duckdb`, add one test that synthesizes `time_since` over the existing `transactions.occurred_at` data with a fixed `cutoff_time` and asserts the resulting durations. This checks narwhals' SQL translation of a datetime literal minus a datetime column.

If duckdb rejects the expression or returns different values, **do not work around it and do not weaken the test** — report the exact error or values.

- [ ] **Step 4: Run the duckdb tests**

Run: `uv run pytest tests/test_backend_duckdb.py -v`
Expected: PASS.

- [ ] **Step 5: Update the coverage table**

In `docs/guide/primitive-coverage.md`, matching the exact link format of neighbouring rows (mkdocstrings `[...][tusk.primitives.X]` for tusk, absolute `https://github.com/Excidion/tusk/blob/main/...` URLs for tests):

| Row | Change |
| --- | --- |
| `time_since` | tusk column becomes `` [`time_since`][tusk.primitives.TimeSince] ``, Status ❌ → ⚠️, Test links the new differential test, Comment: "Returns a `Duration`; featuretools returns float seconds." |
| `time_since_previous` | Status ❓ → ⚠️, Test links the new differential test if it covers this primitive, otherwise its unit test, Comment: "Returns a `Duration`; featuretools returns float seconds." |

Leave the `age` row unchanged at ❌ with its existing comment. Add no rows for duration-to-number primitives — tusk ships none.

- [ ] **Step 6: Verify the docs reference resolves**

Run: `grep -n "TimeSince" docs/api/primitives.md`

If `TimeSince` is not listed there, add it alongside the existing primitive entries — a `[...][tusk.primitives.TimeSince]` link to a class mkdocstrings does not document renders as broken text.

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

**Spec coverage.** Dtype split, including `HAS_TIME` for `hour` → Task 1. `dtype_selector` over the new families → Task 1. `NeedsCutoffTime`, validation, and passing the cutoff time to `build()` → Task 2. `time_since` → Task 2. `TimeSincePrevious` → Task 3. Differential parity, duckdb portability, coverage rows → Task 4. The regression tests for the `Duration` and `Date` crashes → Task 1.

**Deviations from the original plan, both maintainer decisions.** The four duration extractors were dropped: tusk builds features, not encodings, and deep feature synthesis does not stack a transform on another transform's output, so composition could not have replaced featuretools' `unit=` parameter anyway. `age` and `total_years` were dropped because a year is not a well-defined duration.

`NeedsCutoffTime` shipped as a `Primitive` subclass carrying no state, not the frozen-dataclass-mixin-with-a-`cutoff_time`-field this plan specified. Storing the value on the instance made it a constructor argument nobody wanted, guarded by a check that only fired where it ran; a plain mixin also left the type checker reconciling two conflicting `outputs` signatures at the compiler's `isinstance` branch. The compiler instead threads `cutoff_time` through as a keyword argument to `outputs()`/`build()`, both here and — once `8989b22` extended `_add_aggregations` the same way — on the aggregation path this plan said was out of scope.

**Placeholders.** None: every code step carries the code, every test step the assertions, every run step the command and expected result.

**Type consistency.** Family names `HAS_DATE`/`HAS_TIME`/`DURATION` are identical across tasks. `time_since` and `time_since_previous` are the only registered names this plan adds or changes, and both return `nw.Duration`. Output column names follow `generate_name`'s `__`-joined form throughout.
