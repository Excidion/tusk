# Datetime transform parity: elapsed time as a first-class dtype

Design for the `time_since` family and the duration dtype it needs.

## Problem

`docs/guide/primitive-coverage.md` marks `time_since` ❌ with the comment
"Needs a reference clock". Six aggregations (`time_since_first`,
`time_since_last`, `time_since_last_true`, `time_since_last_false`,
`time_since_last_min`, `time_since_last_max`) and one more transform (`age`)
carry the same comment. None can be built until tusk decides what "now" means
and how a primitive reaches it.

A second, independent problem surfaces the moment elapsed time becomes a
value: tusk has no way to talk about a duration. `DtypeFamily.TEMPORAL` matches
`dtype.is_temporal()`, and `Duration.is_temporal()` is `True`, so a `Duration`
column attracts every calendar primitive. This is a live bug today, not a
consequence of this design:

```
InvalidOperationError: `year` operation not supported for dtype `duration[μs]`
```

Any user whose table holds a `Duration` column hits it with default primitives.

## Decisions

These were settled with the maintainer before this document was written.

1. **The cutoff time is the clock.** tusk already takes `cutoff_time` as a row
   filter; it becomes the reference point too, as it is in featuretools
   (`uses_calc_time`). Requesting a cutoff-sensitive primitive without a
   `cutoff_time` is a `ValidationError`, not a fall back to the wall clock —
   the same database and cutoff must always give the same matrix.
2. **Elapsed time is a `Duration`, not a number.** `time_since` returns a
   duration; separate extractor primitives turn a duration into a number.
   Composition through DFS replaces featuretools' `unit=` parameter.
3. **`TEMPORAL` splits** into `DATETIME` and `DURATION` so that stacking lands
   on the right primitives.
4. **`age` and `total_years` are mathematically honest**, not
   featuretools-identical. Both are ⚠️ divergences.
5. **One word per concept:** `cutoff_time` everywhere — the parameter, the
   field, and the prose.

## Architecture

### Dtype families

`DtypeFamily` is public API: `tusk.sklearn.dtype_selector` takes one, and users
pass its string values. `TEMPORAL` is therefore kept as the union rather than
removed, and two narrower families are added beneath it.

| Family | Matches | Purpose |
| --- | --- | --- |
| `TEMPORAL` | `dtype.is_temporal()` — unchanged | `dtype_selector("temporal")` keeps working |
| `DATETIME` | `Datetime`, `Date` | Calendar primitives |
| `DURATION` | `Duration` | Elapsed-time extractors |

`matches()` gains the two branches. Every primitive currently declaring
`F.TEMPORAL` narrows to `F.DATETIME`: `day`, `hour`, `month`, `year`,
`weekday`, `is_weekend`, `time_since_previous`. That narrowing is what fixes
the crash above — a `Duration` column stops matching them.

`dtype_selector("datetime")` and `dtype_selector("duration")` follow with no
new code, because the selector is generic over `DtypeFamily`. They are the
supported way for a user to route a raw duration column into their own encoder
in a `ColumnTransformer`, and are tested and documented as such rather than
left to chance.

### Reaching the cutoff time

`Primitive.build` takes expressions only. Threading a context parameter through
every primitive to serve seven of them is the wrong trade, so the cutoff time
travels as state on the primitive instance.

```py
@dataclass(frozen=True)
class NeedsCutoffTime:
    """A primitive whose value is measured against the cutoff time."""

    cutoff_time: datetime | None = None
```

A cutoff-sensitive aggregation and a cutoff-sensitive transform need different
bases, so this is a mixin rather than a layer in the chain:

```py
class TimeSince(NeedsCutoffTime, TransformPrimitive): ...
```

`NeedsCutoffTime` is a predicate rather than a noun, bending `CODESTYLE.md`'s
class-naming rule. That is deliberate: the name states the requirement that
actually bites a user, and it reads correctly at the point of use.

The mixin is the marker. `isinstance(primitive, NeedsCutoffTime)` drives both
validation and binding, so there is no parallel boolean flag that can disagree
with the type.

**Validation** happens in `tusk.api`, before anything compiles.
`deep_feature_synthesis` and `calculate_feature_matrix` raise `ValidationError`
when a `NeedsCutoffTime` primitive is requested and `cutoff_time` is None:

```
time_since needs a cutoff_time; pass one to deep_feature_synthesis
```

**Binding** happens in the compiler, which substitutes the value with
`dataclasses.replace(primitive, cutoff_time=cutoff_time)` at apply time.
`build()` stays pure and every other primitive is untouched.

### Why `total_microseconds` and not `total_seconds`

narwhals' `total_seconds()` truncates to whole seconds and returns `Int64`, and
`total_minutes()` floors to whole minutes:

```py
timedelta(seconds=90, microseconds=500000)
  .dt.total_seconds() -> 90      # 0.5s lost
  .dt.total_minutes() -> 1       # a floor, not a fraction
```

Deriving the extractors from those would silently lose sub-second precision and
make `TOTAL_MINUTES` a floor where featuretools gives a fraction. Every
extractor therefore derives from `total_microseconds()`, which is exact at
polars' resolution, and divides in floating point.

A backend storing durations at nanosecond resolution loses sub-microsecond
precision here. That is accepted: it is below any plausible feature-engineering
signal, and it is three orders of magnitude better than the alternative.

`Int64` microseconds overflow after roughly 292 000 years, so the range is not
a practical concern.

## Primitives

| Registry name | Input | Output | Expression |
| --- | --- | --- | --- |
| `time_since` | `DATETIME` | `Duration` | `lit(cutoff_time) - expr` |
| `age` | `DATETIME` | `Float64` | years between `expr` and the cutoff time |
| `total_seconds` | `DURATION` | `Float64` | `total_microseconds() / 1e6` |
| `total_minutes` | `DURATION` | `Float64` | `total_microseconds() / 6e7` |
| `total_hours` | `DURATION` | `Float64` | `total_microseconds() / 3.6e9` |
| `total_days` | `DURATION` | `Float64` | `total_microseconds() / 8.64e10` |
| `total_years` | `DURATION` | `Float64` | `total_microseconds() / (365.25 * 8.64e10)` |

`time_since` is `cutoff_time - value`, so a past timestamp yields a positive
duration and a timestamp after the cutoff yields a negative one. This is
featuretools' sign convention.

`TimeSincePrevious` changes from `Float64` seconds to `Duration`, so that every
elapsed-time primitive in tusk carries the same type. It also stops truncating,
since it no longer routes through `total_seconds()`.

### The year constant

`total_years` is the one approximate member of an otherwise exact set, and says
so in its docstring. `age` uses the same constant, and is defined so that

```
AGE(x) == TOTAL_YEARS(TIME_SINCE(x))
```

holds exactly. That equality is asserted as a test; it is what stops the two
code paths drifting apart.

featuretools is internally inconsistent here, which is the argument for not
copying it: `Age` divides whole days by **365**, while its own
`convert_time_units` treats a year as **31 540 000 s ≈ 365.05 days**. featuretools
also floors to whole days before dividing, so its value steps once per day
rather than varying continuously. tusk uses exact elapsed time and 365.25 days.

### `age` applies to every datetime column

featuretools restricts `Age` to columns tagged `date_of_birth`. tusk matches on
narwhals dtypes alone and has no semantic tags — a deliberate trade recorded in
`tusk/dtypes.py` — so `age` applies to any `DATETIME` column, and DFS will
generate `AGE(signed_up_at)` as readily as `AGE(date_of_birth)`.

On a non-birthdate column `AGE(x)` is exactly `TOTAL_YEARS(TIME_SINCE(x))`,
so the output is meaningful, merely redundantly named. Since `age` is opt-in
rather than a default, the noise only appears for a user who asked for it.

### Defaults

None of these primitives join `TRANS_DEFAULTS`. Adding `time_since` would make
every default DFS call require a `cutoff_time`, breaking existing callers. The
extractors stay out for symmetry and to keep the default matrix width stable.
All are opt-in through `trans_primitives=[...]`.

### Depth

`time_since` at `max_depth=1` produces a raw `Duration` column, which
`dtype_selector("numeric")` will not select and scikit-learn cannot consume. A
usable number needs `max_depth=2` and an extractor in `trans_primitives`. This
is documented rather than warned about: tusk returns what was asked for, and
`dtype_selector("duration")` exists precisely so the column can be routed.

## Testing

Differential tests go in a new `tests/differential/test_datetime_transforms.py`,
following the per-group harness established in
`tests/differential/test_aggregations.py`.

**Parity**
- `time_since`: tusk's `TOTAL_SECONDS(TIME_SINCE(x))` against featuretools'
  `TIME_SINCE(x)` (float seconds), over a fixed cutoff, covering a null
  datetime and a timestamp after the cutoff (negative duration).

**Divergence, pinned**
- `age`: agreement with featuretools rounded to whole years, *and* the exact
  ratio `tusk == featuretools * 365 / 365.25` on midnight-aligned data. The
  rounding assertion alone would pass even if the constant changed, so both are
  asserted.
- `time_since_previous`: returns `Duration` where featuretools returns float
  seconds.

**Invariant**
- `AGE(x) == TOTAL_YEARS(TIME_SINCE(x))` on the same data.

**Regression**
- A `Duration` column no longer attracts `year`, `month`, `hour`, `weekday` or
  `is_weekend`. This test fails on `main` with `InvalidOperationError`.

**Validation**
- `time_since` without a `cutoff_time` raises `ValidationError`, from both
  `deep_feature_synthesis` and `calculate_feature_matrix`.

**Precision**
- `TOTAL_SECONDS` of a 90.5-second duration is `90.5`, not `90`, and
  `TOTAL_MINUTES` of it is `1.5083...`, not `1`. These are the assertions that
  would fail against a `total_seconds()`-based implementation.

**Selectors**
- `dtype_selector("temporal")` selects both `Datetime` and `Duration` columns;
  `dtype_selector("datetime")` and `dtype_selector("duration")` each select
  only their own.

**Portability**
- The extractors and `time_since` on duckdb, as the aggregation work did, to
  check the SQL translation.

## Coverage table

| Primitive | Before | After |
| --- | --- | --- |
| `time_since` | ❌ Needs a reference clock | ✅ |
| `age` | ❌ Needs a reference clock | ⚠️ exact seconds and 365.25 days, against featuretools' floored days and 365 |
| `time_since_previous` | ❓ | ⚠️ returns a `Duration`, not float seconds |
| `total_seconds`, `total_minutes`, `total_hours`, `total_days`, `total_years` | — | ➕ five new rows |

The six `time_since_*` aggregations keep their ❌ and their comment. They are
out of scope here, but `NeedsCutoffTime` is the mechanism they will use, which
is why this work comes first.

## Out of scope

- The six cutoff-sensitive aggregations.
- `total_months`, and any other approximate unit beyond `total_years`.
- The remaining datetime transforms (`minute`, `second`, `quarter`,
  `is_month_start`, `season`, and the rest). Each gets its own judgement call
  with the maintainer.
- `week` and `nth_week_of_month`, which narwhals cannot express — it exposes no
  ISO week number.
