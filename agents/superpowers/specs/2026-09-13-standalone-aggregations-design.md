# Standalone aggregations

Phase 1 of `2026-09-13-primitive-parity-roadmap-design.md`: fifteen
aggregation primitives that fit inside a single lazy `group_by().agg()` and
need no new machinery.

## Decisions

Settled with the maintainer, on top of the roadmap's shared rules.

1. **`is_unique` ships; `has_no_duplicates` does not.** `is_unique` is
   `n_unique == len`: a null counts as a value, so two nulls are a repeat,
   and it declares no `default_value`, leaving an empty group null rather
   than false. `has_no_duplicates` would have answered the same question
   with the opposite null rule; once nulls count as values everywhere, that
   rule collapses into `is_unique` and the separate primitive is redundant.
2. **`skew` is narwhals' biased estimator**, not pandas' bias-corrected one,
   and is marked ⚠️.
3. **A constant group gives null `skew` and `kurtosis`.** Unguarded, polars
   answers NaN and duckdb 0.0 or null. The moment ratio is `0/0`, so null on
   every backend.
4. **`first_last_time_delta` is latest minus earliest of any datetime column**,
   returned as a `Duration`. featuretools takes only the time index and
   returns float seconds.
5. **Empty and all-null groups** take the values in the table below, chosen
   one primitive at a time.
6. **Defaults follow featuretools.** `skew` and `percent_true` join
   `AGG_DEFAULTS`; `day` joins `TRANS_DEFAULTS`.

## Primitives

All live in `src/tusk/primitives/aggregation.py`. "Empty" is a parent with no
child rows, filled after the left join from `default_value`. "All null" is a
non-empty group whose every input value is null.

| Name | Input | Output | Expression | Empty | All null | Nulls in a mixed group |
| --- | --- | --- | --- | --- | --- | --- |
| `all_true` | `BOOLEAN` | `Boolean` | `expr.all()` | null | `True` | ignored |
| `any_true` | `BOOLEAN` | `Boolean` | `expr.any()` | `False` | `False` | ignored |
| `n_true` | `BOOLEAN` | `Int64` | `expr.fill_null(False).cast(Int64).sum().cast(Int64)` | `0` | `0` | not true |
| `skew` | `NUMERIC` | `Float64` | `when(expr.std() > 0).then(expr.skew())` | null | null | ignored |
| `kurtosis` | `NUMERIC` | `Float64` | `when(expr.std() > 0).then(expr.kurtosis())` | null | null | ignored |
| `variance` | `NUMERIC` | `Float64` | `expr.var()` | null | null | ignored |
| `max_min_delta` | `NUMERIC` | input's | `expr.max() - expr.min()` | null | null | ignored |
| `first_last_time_delta` | `HAS_DATE` | `Duration` | `latest - earliest`, both cast to `Datetime` | null | null | ignored |
| `is_unique` | `ANY` | `Boolean` | `expr.n_unique() == nw.len()` | null (no default) | `True` for one row, `False` for more | values |
| `percent_unique` | `ANY` | `Float64` | `expr.n_unique() / nw.len()` | null | `1 / n` | values in the numerator, rows in the denominator |
| `n_unique_days` | `HAS_DATE` | `Int64` | `expr.dt.date().n_unique()` | `0` | `1` | values |
| `n_unique_days_of_calendar_year` | `HAS_DATE` | `Int64` | `(month * 100 + day).n_unique()` | `0` | `1` | values |
| `n_unique_days_of_month` | `HAS_DATE` | `Int64` | `expr.dt.day().n_unique()` | `0` | `1` | values |
| `n_unique_months` | `HAS_DATE` | `Int64` | `(year * 12 + month).n_unique()` | `0` | `1` | values |

`count` is `expr.count()`, the number of non-null values.

### A null counts as a value

narwhals' `n_unique` already counts null as one more distinct value on both
backends, so an all-null group reports 1 and a group mixing a null with known
values counts the null as one of them. `NUnique.build` used to correct this
by subtracting a null indicator, through a module-level helper,
`_n_distinct_known(expr)`, also used by `is_unique`, `percent_unique` and all
four `n_unique_*` date primitives. The helper is gone: every one of those
primitives now calls `n_unique()` directly, and `n_unique` itself changed to
match -- an all-null group is `1`, not `0`, and a group mixing a null with
known values counts one more distinct value than before.

### Backend details verified by probe

- `n_true` casts *after* the sum. duckdb's `SUM` returns `Decimal` even over an
  `Int64` input.
- `first_last_time_delta` casts to `Datetime` before subtracting. duckdb's
  `Date - Date` is an integer number of days, not an interval.
- The calendar keys cast `month` and `day` to `Int32` first. Both come back as
  `Int8`, and `month * 100` overflows it.
- The `std() > 0` guard makes `skew` and `kurtosis` agree on both backends,
  including single-row groups, where `std` is null.

## Defaults

```py
AGG_DEFAULTS = ("count", "sum", "mean", "min", "max", "std", "n_unique", "skew", "percent_true")
TRANS_DEFAULTS = ("year", "month", "day", "weekday")
```

The two tests pinning today's tuples (`tests/test_primitives_aggregation.py:121`,
`tests/test_primitives_transform.py:123`) change with them.

## Testing

- `tests/test_primitives_aggregation.py`: per primitive, resolution by name,
  input dtypes, output dtype, and values on polars over one fixture holding a
  mixed group, an all-null group, a constant group, a duplicated null and a
  childless parent.
- `tests/test_backend_duckdb.py`: the same fixture on duckdb gives the same
  values. This is what pins decision 3 and the three casts above.
- `tests/differential/test_aggregations.py`: every primitive against
  featuretools 1.31.0. Agreement is asserted where values agree, and each
  divergence is asserted on both sides. The child's numeric column is declared
  `Double` to featuretools: woodwork infers whole numbers as `IntegerNullable`,
  which `kurtosis` rejects. `first_last_time_delta` needs the datetime column
  as featuretools' time index, so it runs on a child without null datetimes.
  `is_unique` is also checked against featuretools' `has_no_duplicates`,
  since tusk ships no separate primitive for it.

Expected coverage, to be confirmed by the differential tests:

| Status | Rows |
| --- | --- |
| ✅ | `all_true`, `n_true`, `variance` |
| ⚠️ | `any_true` (empty group), `skew` (bias), `kurtosis` (featuretools answers 0 for a group holding a null, a constant group or an empty group), `max_min_delta` (empty group), `first_last_time_delta` (`Duration`, any datetime), `is_unique` (nulls, and doubles as featuretools' `has_no_duplicates`), `percent_unique` (nulls, empty group), `n_unique_days`, `n_unique_days_of_calendar_year`, `n_unique_days_of_month`, `n_unique_months` (nulls) |

## Documentation

- `docs/api/primitives.md`: fifteen entries.
- `docs/guide/primitives.md`: the shipped list, the new defaults, and new rows
  in the empty-group table.
- `docs/guide/primitive-coverage.md`: the fifteen rows, `variance` moved from
  General transform to Aggregation where featuretools files it, and the ⛔
  reclassifications listed in the roadmap.

## Out of scope

- Every primitive in roadmap phases 2 to 6.
