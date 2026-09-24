# Value-count aggregations

Make `mode` run on every backend tusk supports by computing it from per-group
value counts, and introduce value-count aggregations as a primitive kind that
later primitives (`entropy`, `max_count`, `min_count`, `median_count`,
`n_most_common`) can reuse. This pull request ships `mode` as the only one; it
exists so the maintainer can judge the added complexity before merging.

## Why

`mode` on main is `expr.drop_nulls().mode(keep="all").min()`. narwhals only
implements that on polars and pandas: duckdb, sqlframe and ibis raise
`NotImplementedError`, and pyarrow and dask reject it inside `group_by().agg()`.
That breaks shared rule 1 of the
[parity roadmap](2026-09-13-primitive-parity-roadmap-design.md) (both backends
or ⛔). `mode(keep="any")` runs on SQL backends but returns an arbitrary tied
value, never featuretools' answer.

## Probe

Twenty shuffled datasets, fifty groups each, every group a two-way tie;
featuretools answers the smaller tied value. Counting each row's value within
its group with a window, flagging the rows holding the group's highest count
with a second window, and taking the smallest flagged value in the
`group_by().agg()`:

| Backend | Result |
| --- | --- |
| polars, pandas, pyarrow, duckdb, sqlframe, ibis | 20/20 match featuretools |
| dask | fails: `when/then` combined with `over` is not co-aligned |

A separate count table joined back onto the parent also covered dask, but
duplicates the compiler's join, default and condition handling. It was
rejected for that cost.

## Design

### Primitive kind

`ValueCountAggregationPrimitive` in `tusk/primitives/base.py`, a subclass of
`GroupRelativeAggregationPrimitive`:

- `build_per_row(values, counts)` receives the input column and, per row, how
  often that row's value occurs in its group. The count is null where the
  value is null.
- `build(per_row)` reduces the per-row column, as for any group-relative
  aggregation.
- A value-count primitive takes exactly one input column.

Its docstring states what the two arguments hold. Why the counts arrive as a
column (narwhals rejects length-changing expressions inside a lazy `agg`, and
SQL cannot nest one window inside another) goes into a comment only where the
code would otherwise look wrong.

### Compiler

`_add_per_row_columns` in `tusk/compiler.py` adds, in a `with_columns` of its
own and before the existing per-row columns, one count column per value-count
feature:

```python
nw.when(~value.is_null()).then(nw.len().over(foreign_key, value.name))
```

named `<feature name>__value_count`. `_build_per_row_column` passes
`(value, count column)` to a value-count primitive's `build_per_row`, then
wraps the result in `.over(foreign_key)` as it does today. Conditions, the
cutoff, defaults and the join are unchanged: both columns are added to the
already-masked child.

### `mode`

```python
class Mode(ValueCountAggregationPrimitive):
    def build_per_row(self, values, counts):
        return nw.when(counts == counts.max()).then(values)

    def build(self, per_row):
        return per_row.min()
```

Behaviour is unchanged from main: nulls are skipped, a tie gives the smallest
value, and an empty or all-null group gives null. `mode` joins `AGG_DEFAULTS`,
per roadmap rule 6, since featuretools' DFS uses it by default.

## Tests

- `mode` rejoins the duckdb parametrization in `tests/test_backend_duckdb.py`;
  `AGGREGATIONS_DUCKDB_CANNOT_RUN` and `test_mode_is_not_implemented_on_duckdb`
  go.
- The shared cases in `tests/aggregation_cases.py` and the featuretools
  comparison stay as they are; parent 4 is the four-way tie.
- A unit test drives a custom `ValueCountAggregationPrimitive` through DFS to
  pin the contract: the counts it receives per row, and null counts for null
  values.
- Tests asserting the exact contents of `AGG_DEFAULTS` or of a default DFS run
  are updated for `mode`.

## Docs

- `docs/api/primitives.md` lists `ValueCountAggregationPrimitive` with the
  other base classes.
- `docs/guide/custom-primitives.md` gets a short section on writing one.
- The `mode` row of `docs/guide/primitive-coverage.md` drops its backend
  limitation.
