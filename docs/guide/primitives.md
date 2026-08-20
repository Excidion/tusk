# Primitives

A primitive builds a narwhals expression. DFS composes primitives into
features; the primitive itself never sees a value, which is what keeps
everything pushed down to the backend.

## What ships with tusk

**Aggregation** — `count`, `sum`, `mean`, `min`, `max`, `std`, `median`,
`n_unique`, `percent_true`, `quantiles`.

**Transform** — `year`, `month`, `day`, `hour`, `weekday`, `is_weekend`,
`absolute`, `natural_log`, `add_numeric`, `subtract_numeric`,
`multiply_numeric`, `divide_numeric`.

**Order-dependent** (require a `row_creation_time` on the table) — `cum_sum`,
`cum_count`, `cum_min`, `cum_max`, `diff`, `time_since_previous`.

Every name above is also an importable class — `from tusk.primitives import
Year, CumSum` — and takes the same form as a user-defined one, so `Year` and
`Count` are the same kind of object as anything you write yourself. See
[Custom primitives](custom-primitives.md).

## Defaults

Passing `agg_primitives=None` or `trans_primitives=None` selects a sensible
default subset: `count`, `sum`, `mean`, `min`, `max`, `std`, `n_unique` for
aggregation, and `year`, `month`, `weekday` for transforms.

Arithmetic primitives are excluded from the defaults because they generate
hundreds of features on wide tables.

## Multi-output primitives

A multi-output primitive such as `quantiles` produces indexed columns —
`QUANTILES__transactions__amount__0`, `__1`, `__2` — and nothing else stacks on
it: there is no single column for another primitive to read. It is a valid
output at any depth, just never an input.

## Empty groups

Aggregating a group with no rows is the most surprising correct behaviour in
the library, so it is worth stating plainly. After the left join, a customer
with no sessions gets:

| Primitive | Value | Why |
|---|---|---|
| `COUNT` | `0` | We know there were zero rows. |
| `N_UNIQUE` | `0` | Zero rows hold zero distinct values. Nulls are not values either, so a group of only nulls is also `0`. |
| `SUM` | `0` | The additive identity. |
| `MEAN`, `MIN`, `MAX`, `STD`, `MEDIAN`, `QUANTILES` | `null` | Genuinely undefined over an empty set: `0/0`, and the min or max of nothing. |

The split is not arbitrary. Reporting `COUNT = 0` asserts we *know* there were
no rows; a null `SUM` beside it would claim the total is unknown, which
contradicts a known-zero count. `MEAN` has no such defence — there is no number
that is the average of nothing — so it stays null. Each value lives on the
primitive as `default_value` rather than as a special case in the compiler.

featuretools agrees on `COUNT` and `SUM`, and also leaves `MEAN`/`MIN`/`MAX`
null. It differs on `N_UNIQUE`, which it leaves as `NaN`; tusk reports `0` for
the reason above.

## What can go in `groupby_trans_primitives`

Only **group-aware** primitives — ones whose expression reduces or scans across
the group defined by a foreign key. The order-dependent built-ins (`cum_sum`,
`cum_count`, `cum_min`, `cum_max`, `diff`, `time_since_previous`) all qualify,
and are the primitives you'll normally pass here.

Every other built-in transform (`absolute`, `month`, `add_numeric`, …) is
**elementwise** rather than group-aware, and narwhals rejects `.over()` on an
elementwise expression:

```
InvalidOperationError: Cannot apply over to elementwise expression
```

Passing one of these in `groupby_trans_primitives` therefore fails — but at
expression-build time, not later at `.collect()`, so you learn immediately
rather than after a long query. The failure surfaces synchronously out of
`deep_feature_synthesis()` only when it compiles, i.e. `features_only=False`; with
`features_only=True` synthesis happily emits the definition and the error waits
until you call `apply_features()` on it.

This leaves the grouped, non-order-dependent path reachable only by
user-defined primitives — that's [the intended extension
point](custom-primitives.md#group-aware-primitives).
