# Holiday primitives

Five transform primitives that read a date against a user-supplied holiday
calendar, covering featuretools' `is_federal_holiday`, `date_to_holiday` and
`distance_to_holiday`, plus two tusk-only day counts.

## Decisions

Settled with the maintainer.

1. **Holidays are a primitive parameter.** No `holidays` package dependency and
   no holiday table in the `Database`. Each primitive takes a required
   `holidays` field, a mapping of `datetime.date` to holiday name, e.g.
   `IsHoliday(holidays={date(2024, 12, 25): "Christmas"})`. This follows the
   existing pattern of `Quantiles(qs=...)`.
2. **Explicit dates only.** No recurring month-day keys. A fixed holiday is
   expanded over the needed years by the user; moving holidays (Easter) and
   one-offs work the same way.
3. **No `holiday=` filter argument.** featuretools' `distance_to_holiday`
   takes the name of one holiday. In tusk the user passes only that holiday's
   dates.
4. **No 365-day tolerance.** featuretools answers NaN for `distance_to_holiday`
   when the nearest holiday is more than 365 days away. tusk measures against
   the dates it is given, however far.
5. **One calendar per run, no calendar name in feature names.** Names are
   `IS_HOLIDAY__<column>` etc. Two holiday calendars on the same column in one
   run produce equal names, which `_reject_colliding_names` in
   `src/tusk/compiler.py` already rejects; nothing new is needed.
6. **Not in `TRANS_DEFAULTS`.** They cannot be built without arguments.

## Primitives

All live in `src/tusk/primitives/transform.py`, subclass `TransformPrimitive`,
take one `F.HAS_DATE` input (`Date` or `Datetime`), and are `@register`ed.
A `Datetime` is compared by its calendar date (`expr.dt.date()` or a cast to
`nw.Date`; whichever works on polars and duckdb). A null input gives null.

| Name | Class | Output | Value |
| --- | --- | --- | --- |
| `is_holiday` | `IsHoliday` | `Boolean` | The date is a key of `holidays`. |
| `holiday_name` | `HolidayName` | `String` | The name for the date; null on any other day. |
| `days_to_holiday` | `DaysToHoliday` | `Int64` | Signed days to the nearest holiday, `holiday - date`, so positive means ahead. On a tie between a past and a future holiday, the past one wins (negative), matching pandas' `merge_asof(direction="nearest")` that featuretools uses. |
| `days_until_holiday` | `DaysUntilHoliday` | `Int64` | Days to the next holiday on or after the date; 0 on a holiday. Null after the last given holiday. |
| `days_since_holiday` | `DaysSinceHoliday` | `Int64` | Days since the last holiday on or before the date; 0 on a holiday. Null before the first given holiday. |

`days_to_holiday` is never null for a non-null date, since the calendar is
never empty.

## Parameter handling

- Field: `holidays: Mapping[date, str]`, no default.
- `__post_init__` normalizes it to a tuple of `(date, name)` pairs sorted by
  date (via `object.__setattr__`, as the dataclass is frozen), so the
  primitive is hashable and two primitives built from equal mappings are
  equal. Features are kept in sets, so hashability is required.
- An empty mapping raises `PrimitiveError` with an actionable message.
- A key that is a `datetime` rather than a `date`, or not a `date` at all,
  raises `PrimitiveError`. (`datetime` is a subclass of `date`; reject it
  explicitly so a time of day never silently disappears.)
- The shared normalization lives in one place (a small base class or helper
  in `transform.py`); the five primitives do not repeat it.

## Resolving by name

`resolve("is_holiday")` currently calls `_REGISTRY[spec]()`, which fails with
a bare `TypeError` for a primitive with a required field. `resolve` in
`src/tusk/primitives/registry.py` checks the class's dataclass fields first
and, if any has neither a default nor a default factory, raises:

`PrimitiveError: primitive 'is_holiday' needs arguments; pass IsHoliday(holidays=...) instead of its name`

The check is generic (it names the required fields), so custom primitives with
required fields get the same message. It is tested on its own.

## Building the expressions

Plain narwhals expressions, so every backend runs them.

- `is_holiday`: `date.is_in(dates)`.
- `holiday_name`: `replace_strict(mapping, default=None, return_dtype=nw.String)`
  (verified to work on duckdb), or a `when` chain if it misbehaves on dates.
- Day counts: a `when` chain over the sorted dates picks the next (or last)
  holiday date; the difference is taken in days. Cast both sides to
  `nw.Datetime` before subtracting, because duckdb subtracts two `Date`s into
  an integer instead of an interval (see `_time_since_last_match`).
  `days_to_holiday` compares `days_until` and `days_since` and prefers the
  past holiday on a tie.
- Expression size grows linearly with the number of holidays. The docstrings
  say so; a few hundred dates is the intended scale.

## Tests

- **Shared cases** (`tests/transform_cases.py`): add `EXPECTED` entries for all
  five on `due_at` (a `Datetime` with a null) and on a new `Date` column if
  `Date` is not covered yet, with a small holiday calendar that includes: a
  date on a holiday, dates between two holidays (including an exact tie),
  dates before the first and after the last holiday. `EXPECTED` currently maps
  a column to a primitive *name*; extend it so an entry can carry a primitive
  instance, and make every consumer (`test_primitives_transform.py`,
  `test_backend_duckdb.py`, the differential helpers) accept either. These run
  on polars and duckdb.
- **Parameter tests**: normalization (dict order does not matter, equal
  mappings give equal and equally hashed primitives), empty mapping, a
  `datetime` key, and the `resolve` error by name.
- **Differential** (`tests/differential/test_holiday_transforms.py`): build the
  holiday dict from featuretools' own `HolidayUtil("US")`
  (`featuretools.primitives.utils` or wherever it lives in 1.31.0) and compare:
  - `is_holiday` vs `is_federal_holiday`
  - `holiday_name` vs `date_to_holiday`
  - `days_to_holiday` vs `distance_to_holiday(holiday=...)`, with tusk given
    only that holiday's dates, on test dates within 365 days of a holiday,
    including a tie.
  The differential helpers take a primitive name today; they need to accept a
  featuretools primitive instance (already supported) and a tusk instance.
- **Coverage table** (`docs/guide/primitive-coverage.md`): the three holiday
  rows become ✅ if their differential tests agree, or ⚠️ with a comment
  (`distance_to_holiday` is ⚠️ if tolerance or the `holiday=` argument is
  judged a divergence; decide from the test results and say so in the PR).
  Add ➕ rows for `days_until_holiday` and `days_since_holiday`. Add all five
  to `docs/api/primitives.md` and export them from `tusk.primitives`.
- **Docs**: a short usage example (building the mapping, including a dict
  comprehension over years for a fixed holiday) in the guide page where
  parameterized primitives are documented.
