# Datetime transform parity: elapsed time as a first-class dtype

Design for `time_since` and the duration dtype it needs.

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

These were settled with the maintainer.

1. **The cutoff time is the clock.** tusk already takes `cutoff_time` as a row
   filter; it becomes the reference point too, as it is in featuretools
   (`uses_calc_time`). Requesting a cutoff-sensitive primitive without a
   `cutoff_time` is a `ValidationError`, not a fall back to the wall clock —
   the same database and cutoff must always give the same matrix.
2. **A primitive never stores the cutoff time.** It describes the question
   being asked, not the feature, so it is passed in when the expression is
   built. That is what lets one `FeatureList` be applied at several cutoff
   times.
3. **Elapsed time is a `Duration`, and stays one.** tusk builds features, not
   encodings; converting a duration to a number is the caller's choice, as it
   is for a string. No extractor primitives.
4. **`TEMPORAL` splits** into `DATETIME`, `TIMESTAMP` and `DURATION` so each
   primitive matches only the columns it can actually compute over.
5. **No years anywhere.** A year is not a well-defined duration, so `age` and
   `total_years` are both out of scope. `age` keeps its ❌ row.
6. **One word per concept:** `cutoff_time` everywhere — the parameter, the
   argument, and the prose.

## Architecture

### Dtype families

`DtypeFamily` is public API: `tusk.sklearn.dtype_selector` takes one, and users
pass its string values. `TEMPORAL` is therefore kept as the union rather than
removed, and two narrower families are added beneath it.

| Family | Matches | Purpose |
| --- | --- | --- |
| `TEMPORAL` | `dtype.is_temporal()` — unchanged | `dtype_selector("temporal")` keeps working |
| `DATETIME` | `Datetime`, `Date` | Calendar primitives |
| `TIMESTAMP` | `Datetime` | Time-of-day primitives, e.g. `hour` |
| `DURATION` | `Duration` | Routed downstream by `dtype_selector` |

`matches()` gains the three branches. Every primitive currently declaring
`F.TEMPORAL` narrows: `day`, `month`, `year`, `weekday`, `is_weekend` and
`time_since_previous` to `F.DATETIME`, and `hour` to `F.TIMESTAMP`. That
narrowing is what fixes the crash above — a `Duration` column stops matching
any of them, and `hour` additionally stops matching a `Date` column, which
raises the same way (`'hour' operation not supported for dtype 'date'`).

`dtype_selector("datetime")` and `dtype_selector("duration")` follow with no
new code, because the selector is generic over `DtypeFamily`. They are the
supported way for a user to route a raw duration column into their own encoder
in a `ColumnTransformer`, and are tested and documented as such rather than
left to chance.

### Reaching the cutoff time

`Primitive.build` takes expressions only, and threading a context parameter
through every primitive to serve one of them is the wrong trade. Instead the
mixin declares its own call shape:

```py
class NeedsCutoffTime(Primitive):
    def outputs(self, *inputs: nw.Expr, cutoff_time: datetime) -> tuple[nw.Expr, ...]: ...
    def build(self, *inputs: nw.Expr, cutoff_time: datetime) -> nw.Expr | Sequence[nw.Expr]: ...


class TimeSince(NeedsCutoffTime, TransformPrimitive): ...
```

Nothing is stored and nothing is copied. Omitting the argument is an ordinary
`TypeError`, so there is no unbound state to guard against, and one
`FeatureList` stays applicable at several cutoff times by construction rather
than by the compiler being careful.

`NeedsCutoffTime` is a predicate rather than a noun, bending `CODESTYLE.md`'s
class-naming rule. That is deliberate: the name states the requirement that
actually bites a user, and it reads correctly at the point of use. It subclasses
`Primitive` because a plain mixin left the type checker reconciling two
conflicting `outputs` signatures at the compiler's `isinstance` branch.

Two alternatives were priced and rejected. Storing the value on the instance
made it a constructor argument nobody wanted, guarded by a check that only
fired where it ran — `features_only=True` and a direct `outputs()` call both
skipped it. Declaring the cutoff time as an expression input would have put it
into the feature graph, where synthesis matches every input against real
columns and names features from their bases, requiring it to be excluded again
from matching, naming, dtype and column emission.

**Validation** happens in `compile_features`, the single choke point every path
reaches — `deep_feature_synthesis`, `apply_features`, `FeatureList.apply`, and
direct `Feature` construction alike:

```
time_since needs a cutoff_time; pass one when applying the features
```

**The compiler passes the cutoff time on both the row-wise and the aggregation
path**, so a `NeedsCutoffTime` aggregation computes rather than being refused.

## Primitives

| Registry name | Input | Output | Expression |
| --- | --- | --- | --- |
| `time_since` | `DATETIME` | `Duration` | `lit(cutoff_time) - expr` |

`time_since` is `cutoff_time - value`, so a past timestamp yields a positive
duration and a timestamp after the cutoff yields a negative one. This is
featuretools' sign convention.

`TimeSincePrevious` changes from `Float64` seconds to `Duration`, so that every
elapsed-time primitive in tusk carries the same type. It also stops truncating,
since it no longer routes through `total_seconds()`.

### A duration is the feature; a number is an encoding

An earlier draft of this spec added `total_seconds`, `total_minutes`,
`total_hours` and `total_days`, on the argument that composing them with
`time_since` replaces featuretools' `unit=` parameter. That is dropped.

tusk builds features, not encodings. A `Duration` is the feature: it is the
elapsed time, losing nothing. Turning it into a number means choosing a unit
and a rounding rule, and that choice belongs to whoever is fitting the model —
exactly as it does for a string column, which tusk also hands over unencoded.
`dtype_selector("duration")` is the supported route for doing it downstream.

This also removes a dependency on behaviour tusk does not have. Deep feature
synthesis does **not** stack a transform primitive on another transform's
output within a table, at any `max_depth`:

```
trans_primitives=["year", "absolute"], max_depth=1, 2 and 3
  -> ABSOLUTE__v, YEAR__t   (never ABSOLUTE__YEAR__t)
```

`_Context.build` calls `_transforms(table, features, depth_limit)` once, with
`features` holding identity, aggregation and direct features, and appends the
result afterwards, so a transform never sees another transform. The
`feature.depth <= depth_limit` guard inside `_transforms` suggests stacking was
intended, but it was never wired. Aggregation-on-aggregation stacking across
tables does work and is documented; transform-on-transform does not.

Wiring it up is a change to the synthesis engine with its own consequences for
default output width, and is not part of this work.

### Defaults

`time_since` does not join `TRANS_DEFAULTS`. Adding it would make every default
DFS call require a `cutoff_time`, breaking existing callers. It is opt-in
through `trans_primitives=[...]`.

### Depth

`time_since` produces a `Duration` column at any depth. `dtype_selector("numeric")`
will not select it and scikit-learn cannot consume it directly, which is the
same position a `String` column is in. `dtype_selector("duration")` exists so it
can be routed to an encoder of the caller's choosing.

## Testing

Differential tests go in a new `tests/differential/test_datetime_transforms.py`,
following the per-group harness established in
`tests/differential/test_aggregations.py`.

**Parity**
- `time_since`: featuretools returns float seconds, tusk returns a `Duration`.
  The test converts tusk's column to seconds *in the test* — not through a
  primitive — and compares, over a fixed cutoff time, covering a null datetime
  and a timestamp after the cutoff time (negative duration).

**Divergence, pinned**
- `time_since` and `time_since_previous` both return a `Duration` where
  featuretools returns float seconds. Same information, different
  representation.

**Regression**
- A `Duration` column no longer attracts `year`, `month`, `weekday` or
  `is_weekend`, and a `Date` column no longer attracts `hour`. Both fail on
  `main`.

**Validation**
- `time_since` without a `cutoff_time` raises `ValidationError`, from
  `deep_feature_synthesis` and from `FeatureList.apply`.
- One `FeatureList` applied at two different cutoff times gives two different
  correct answers. This is what the cutoff time never being stored on a
  primitive buys, and nothing else in the suite pins it.

**Selectors**
- `dtype_selector("temporal")` selects `Datetime`, `Date` and `Duration`
  columns; `dtype_selector("datetime")`, `("timestamp")` and `("duration")`
  each select only their own.

**Portability**
- `time_since` on duckdb, as the aggregation work did, to check the SQL
  translation of the datetime subtraction.

## Coverage table

| Primitive | Before | After |
| --- | --- | --- |
| `time_since` | ❌ Needs a reference clock | ⚠️ returns a `Duration`, not float seconds |
| `time_since_previous` | ❓ | ⚠️ returns a `Duration`, not float seconds |

`age` keeps its ❌ row and its "Needs a reference clock" comment.

The six `time_since_*` aggregations keep their ❌ and their comment. They are
out of scope here, but `NeedsCutoffTime` is the mechanism they will use, which
is why this work comes first.

## Out of scope

- `age` and `total_years`. A year is not a well-defined duration: featuretools
  is itself inconsistent about it, with `Age` dividing whole days by **365**
  while its own `convert_time_units` treats a year as **31 540 000 s ≈ 365.05
  days**. tusk ships no primitive whose answer depends on picking one.
- `total_months`, for the same reason.
- The six cutoff-sensitive aggregations.
- The remaining datetime transforms (`minute`, `second`, `quarter`,
  `is_month_start`, `season`, and the rest). Each gets its own judgement call
  with the maintainer.
- `week` and `nth_week_of_month`, which narwhals cannot express — it exposes no
  ISO week number.
