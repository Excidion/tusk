# Standalone and ordered transforms

Phase 2 of `2026-09-13-primitive-parity-roadmap-design.md`: sixteen transform
primitives that each fit a single narwhals expression, tests for the ❓
transform rows, and the ⚠️ pointer for a featuretools transform tusk answers
with an existing primitive.

## Decisions

Settled with the maintainer, on top of the roadmap's shared rules.

1. **No forward fill.** featuretools forward-fills nulls before
   `same_as_previous`, `absolute_diff` and `percent_change`. narwhals cannot
   nest a forward fill inside a shift in one expression (`Cannot nest over
   statements`), and a primitive builds one expression. A null input or a null
   previous value gives null, as it does for `diff`.
2. **`absolute_diff` ships as its own primitive.** tusk never applies a
   `trans_primitives` entry to another `trans_primitives` output on the same
   table, so `absolute` stacked on `diff` is not a feature a user gets from
   `trans_primitives`. `diff_datetime` becomes ⚠️ pointing at
   `time_since_previous`, which computes the same thing.
3. **A negative input gives null for `square_root` and `natural_log`.**
   Unguarded, polars answers NaN and duckdb null. `when(expr >= 0)` makes both
   backends null; zero stays `0.0` and `-inf` respectively. featuretools
   answers NaN, so both rows are ⚠️. This changes `natural_log`, whose
   negative inputs were backend-defined until now.
4. **`negate` and `absolute_diff` return `Float64`.** Negating an unsigned
   integer fails on polars and wraps around on duckdb; negating `Int8`'s
   minimum wraps on polars and raises on duckdb. `absolute_diff`'s `diff()`
   has the same failure mode: an `Int8` column from -128 to 127 wraps on
   polars and raises on duckdb, and a `UInt8` decrease raises on duckdb.
   Casting to `Float64` first gives one answer everywhere for both.
5. **No new defaults.** featuretools' default transforms are `age`, `day`,
   `year`, `month`, `weekday`, `haversine`, `num_words` and `num_characters`;
   none of them belongs to this phase.

## Primitives

All live in `src/tusk/primitives/transform.py`. "Ordered" primitives set
`order_dependent = True` and are wrapped by the compiler in
`.over(..., order_by=(row_creation_time, primary_key))`.

| Name | Input | Output | Expression | Ordered |
| --- | --- | --- | --- | --- |
| `is_null` | `ANY` | `Boolean` | `expr.is_null()` | |
| `negate` | `NUMERIC` | `Float64` | `expr.cast(Float64) * -1` | |
| `square_root` | `NUMERIC` | `Float64` | `when(expr >= 0).then(expr.sqrt())` | |
| `sine` | `NUMERIC` | `Float64` | `expr.sin()` | |
| `cosine` | `NUMERIC` | `Float64` | `expr.cos()` | |
| `minute` | `HAS_TIME` | `Int8` | `expr.dt.minute()` | |
| `second` | `HAS_TIME` | `Int8` | `expr.dt.second()` | |
| `day_of_year` | `HAS_DATE` | `Int16` | `expr.dt.ordinal_day()` | |
| `is_leap_year` | `HAS_DATE` | `Boolean` | `(year % 4 == 0) & (year % 100 != 0) \| (year % 400 == 0)`, `year` cast to `Int32` | |
| `percentile` | `NUMERIC` | `Float64` | `expr.rank("average") / expr.count()` | |
| `cum_mean` | `NUMERIC` | `Float64` | `expr.cum_sum() / expr.cum_count()` | ✓ |
| `same_as_previous` | `NUMERIC` | `Boolean` | `expr == expr.shift(1)` | ✓ |
| `absolute_diff` | `NUMERIC` | `Float64` | `expr.cast(Float64).diff().abs()` | ✓ |
| `percent_change` | `NUMERIC` | `Float64` | `expr / expr.shift(1) - 1` | ✓ |
| `cumulative_time_since_last_true` | `(HAS_DATE, BOOLEAN)` | `Duration` | `moment - when(flag).then(moment).fill_null(strategy="forward")` | ✓ |
| `cumulative_time_since_last_false` | `(HAS_DATE, BOOLEAN)` | `Duration` | `moment - when(~flag).then(moment).fill_null(strategy="forward")` | ✓ |

`natural_log` changes to `when(expr >= 0).then(expr.log())`.

### Null and edge behavior

- `percentile` ranks non-null values with ties averaged and divides by the
  number of non-null values; a null stays null. It ranks across every row it
  is given, so inside `groupby_trans_primitives` it ranks within the group.
- `cum_mean` skips nulls in both the running sum and the running count; a null
  row is null.
- `same_as_previous`, `absolute_diff` and `percent_change` are null on the
  first row and on any row whose own or previous value is null.
  `percent_change` over a zero previous value is `inf`, `-inf` or NaN, the
  same on both backends.
- `cumulative_time_since_last_*` are null until the first row whose flag is
  true (false), and on a row whose datetime is null. A null flag is not a
  match, so it keeps the previous matching row.
- `is_leap_year` of a null date is null.

### Backend details verified by probe

Against narwhals 2.24.0, polars 1.43.2 and duckdb 1.5.5:

- Every expression above gives identical values on lazy polars and duckdb.
- `cumulative_time_since_last_*` cast a `Date` moment to `Datetime` before
  subtracting. duckdb's `Date - Date` is an integer number of days, not an
  interval.
- duckdb returns the elapsed time as an `INTERVAL`, which narwhals reports as
  `Unknown` and `DuckDBPyRelation.pl()` cannot import; duckdb tests read such
  columns with `.df()`, as the `time_since` test already does.

## Existing rows

| Row | Change |
| --- | --- |
| `diff_datetime` | ⚠️, pointing at `time_since_previous`. |
| `cum_count` | ⚠️: featuretools counts every row, tusk only non-null values. |
| `cum_sum`, `cum_min`, `cum_max`, `diff`, `absolute`, `year`, `month`, `day`, `hour` | ❓ to ✅, once the differential test confirms agreement. `diff` shares `absolute_diff`'s pre-cast integer overflow (decision 4); fixing it is a later phase's follow-up. |
| `natural_log` | ❓ to ⚠️: a negative input is null, featuretools NaN. |

## Testing

- `tests/transform_cases.py`: a shared eight-row table, `ROWS`, and `EXPECTED`,
  a value per column per new primitive. It covers a null, a zero, a negative,
  a boolean flag and a nullable boolean flag, and a tied `value` column, so
  the cases reach negative `square_root` and `natural_log` inputs, ties in
  `percentile`, nulls in every ordered primitive and a null date in
  `is_leap_year`.
- `tests/test_primitives_transform.py`: `EXPECTED` checked on polars, plus
  `negate` and `absolute_diff` on `Int8` and an unsigned integer dtype (where
  `ROWS`'s `value` column, already a float, cannot reach the wraparound),
  `minute` and `second` on a standalone `Time` column, and `percentile` and
  `cumulative_time_since_last_true` each grouped by foreign key, so a
  group's rank or match does not leak into its neighbour's.
- `tests/test_backend_duckdb.py`: the same `EXPECTED`, integer-dtype, `Time`-
  column and grouped checks on duckdb. This is what pins decisions 3 and 4;
  `test_cumulative_time_since_measures_a_date_column_on_duckdb` pins the
  `Date` cast.
- Differential tests against featuretools 1.31.0. Agreement is asserted where
  values agree, and each divergence is asserted on both sides.
  - `tests/differential/test_datetime_transforms.py`: `minute`, `second`,
    `day_of_year`, `is_leap_year`, `year`, `month`, `day`, `hour`.
  - `tests/differential/test_general_transforms.py` (new): `is_null`,
    `negate`, `square_root`, `sine`, `cosine`, `percentile`, `absolute`,
    `natural_log`.
  - `tests/differential/test_cumulative_transforms.py` (new): `cum_mean`,
    `same_as_previous`, `absolute_diff`, `percent_change`,
    `cumulative_time_since_last_true`, `cumulative_time_since_last_false`,
    `cum_sum`, `cum_min`, `cum_max`, `diff`, `cum_count`; `diff_datetime`
    against `time_since_previous`. The datetime column is featuretools' time
    index, which the `cumulative_time_since_last_*` primitives require and
    which orders the rows on both sides.

Expected coverage, to be confirmed by the differential tests:

| Status | Rows |
| --- | --- |
| ✅ | `is_null`, `negate`, `sine`, `cosine`, `minute`, `second`, `day_of_year`, `percentile`, `cum_sum`, `cum_min`, `cum_max`, `diff` (integer overflow, see decision 4), `absolute`, `year`, `month`, `day`, `hour` |
| ⚠️ | `square_root`, `natural_log` (negative input), `is_leap_year` (null date), `cum_mean` (null divisor), `same_as_previous` (first row, no fill), `absolute_diff`, `percent_change` (no fill), `cumulative_time_since_last_true`, `cumulative_time_since_last_false` (`Duration`, any datetime), `cum_count` (null rows), `diff_datetime` (pointer) |

## Documentation

- `docs/api/primitives.md`: sixteen entries in the transform and
  order-dependent transform sections.
- `docs/guide/primitives.md`: the shipped list; `percentile` and the new
  ordered primitives in "What can go in `groupby_trans_primitives`"; the
  negative-input rule for `square_root` and `natural_log`.
- `docs/guide/primitive-coverage.md`: every row named above, with its test
  link, and the stale "narwhals has none" comment on the
  `cumulative_time_since_last_*` rows removed.

## Out of scope

- Every primitive in roadmap phases 3 to 6, including `is_weekend`.
- Forward-filling nulls, which would need a primitive to build in two stages.
