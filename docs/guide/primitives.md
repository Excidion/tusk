# Primitives

A primitive builds a narwhals expression. DFS composes primitives into
features; the primitive itself never sees a value, which is what keeps
everything pushed down to the backend.

## What ships with tusk

See the [API reference](../api/primitives.md) for the full list of
aggregation, transform and order-dependent primitives tusk ships.

Every `time_since_*` primitive needs more than its input columns: it
requires a `cutoff_time` at apply time, since its value is measured against
that moment rather than derived from the rows alone. Order-dependent
primitives require a `row_creation_time` on the table.

See [primitive coverage](primitive-coverage.md) for how these line up against
featuretools.

Every name above is also an importable class — `from tusk.primitives import
Year, CumSum` — and takes the same form as a user-defined one, so `Year` and
`Count` are the same kind of object as anything you write yourself. See
[Custom primitives](custom-primitives.md).

## Defaults

Passing `agg_primitives=None` or `trans_primitives=None` selects the
primitives featuretools uses by default, where tusk has them:
[`AGG_DEFAULTS`][tusk.primitives.aggregation.AGG_DEFAULTS] for aggregation
and [`TRANS_DEFAULTS`][tusk.primitives.transform.TRANS_DEFAULTS] for
transforms.

Arithmetic primitives are excluded from the defaults because they generate
hundreds of features on wide tables.

## Multi-output primitives

A multi-output primitive such as `quantiles` produces indexed columns —
`QUANTILES__orders__quantity__0`, `__1`, `__2` — and nothing else stacks on
it: there is no single column for another primitive to read. It is a valid
output at any depth, just never an input.

## Empty groups

Aggregating a group with no rows is the most surprising correct behaviour in
the library, so it is worth stating plainly. After the left join, a customer
with no orders gets:

| Primitive | Value | Why |
|---|---|---|
| `COUNT` | `0` | We know there were zero rows. |
| `N_UNIQUE` | `0` | Zero rows hold zero distinct values. A null counts as one value, so a group of only nulls is `1`. |
| `SUM` | `0` | The additive identity. |
| `MEAN`, `MIN`, `MAX`, `STD`, `MEDIAN`, `QUANTILES` | `null` | Genuinely undefined over an empty set: `0/0`, and the min or max of nothing. |
| `PERCENT_TRUE` | `null` | Undefined over an empty set, same as `MEAN`. Within a non-empty group a null counts as false, so an all-null group computes to `0.0` rather than falling through to this default. |
| `N_TRUE` | `0` | Zero rows hold nothing to count. |
| `N_UNIQUE_DAYS`, `N_UNIQUE_DAYS_OF_CALENDAR_YEAR`, `N_UNIQUE_DAYS_OF_MONTH`, `N_UNIQUE_MONTHS` | `0` | Zero rows hold zero distinct values. A null date counts as one value, so a group of only nulls is `1`. |
| `ANY_TRUE` | `false` | No row is true. |
| `ALL_TRUE` | `null` | A group of only nulls is `true`, as polars and duckdb answer, but a group with no rows is left unknown. |
| `IS_UNIQUE` | `null` | No rows to compare. Nulls are values, so a group of several nulls is `false`. |
| `VARIANCE`, `SKEW`, `KURTOSIS`, `MAX_MIN_DELTA`, `FIRST_LAST_TIME_DELTA`, `PERCENT_UNIQUE` | `null` | Undefined over an empty set. `SKEW` and `KURTOSIS` are also `null` for a group whose values do not vary. |

The split is not arbitrary. Reporting `COUNT = 0` asserts we *know* there were
no rows; a null `SUM` beside it would claim the total is unknown, which
contradicts a known-zero count. `MEAN` has no such defence — there is no number
that is the average of nothing — so it stays null. Each value lives on the
primitive as `default_value` rather than as a special case in the compiler.

featuretools agrees on `COUNT` and `SUM`, and also leaves `MEAN`/`MIN`/`MAX`
null. It differs on `N_UNIQUE`: it leaves an empty group as `NaN`, where tusk
reports `0` for the reason above, and it ignores a null value entirely, where
tusk counts it as one distinct value.

## Nulls in `and` and `or`

A null is an unknown value, not a third truth value, so an unknown input only
makes the answer unknown when it could have changed it:

| | `true` | `false` | `null` |
|---|---|---|---|
| **`AND`** `true` | `true` | `false` | `null` |
| **`AND`** `false` | `false` | `false` | `false` |
| **`AND`** `null` | `null` | `false` | `null` |
| **`OR`** `true` | `true` | `true` | `true` |
| **`OR`** `false` | `true` | `false` | `null` |
| **`OR`** `null` | `true` | `null` | `null` |

`false AND null` is `false` because nothing the unknown turns out to be makes
the conjunction true. This is what polars, duckdb and every SQL engine answer,
and tusk builds the operator rather than working around it.

featuretools instead propagates the null in every one of those cells. `NOT`
agrees on both sides: the negation of an unknown is unknown.

## Negative inputs to `square_root` and `natural_log`

A negative number has no real square root or logarithm, so both primitives
give null for it on every backend. Zero is not negative: `square_root` gives
`0.0` and `natural_log` gives negative infinity.

## Comparing two columns

The comparison primitives accept a pair of numbers or a pair of datetimes,
never one of each — a primitive may declare several input shapes, and each
shape is matched as a whole:

```py
input_dtypes = ((F.NUMERIC, F.NUMERIC), (F.HAS_DATE, F.HAS_DATE))
```

`equal` and `not_equal` also accept a pair of booleans or a pair of strings.
A null on either side gives a null answer, as in SQL: an unknown value cannot
be shown equal to anything, nor greater than it. See [primitive
coverage](primitive-coverage.md) for how this holds up against featuretools,
column by column.

Labels are a separate case. `Categorical` and `Enum` columns are compared with
`equal_categorical` and `not_equal_categorical`, which compare the labels
themselves rather than their encodings — two `Enum` columns with different
member lists cannot be compared directly on polars at all. A null label
follows the same rule as everything else on this page: it makes the
comparison unknown rather than simply unequal.

`(F.HAS_DATE, F.HAS_DATE)` matches a `Datetime` column or a `Date` column,
regardless of time zone, so a table holding a `Date` column and a tz-aware
or tz-naive `Datetime` column will generate a comparison between
any two of them.
You can guard against this with `db.validate()`.


## What can go in `groupby_trans_primitives`

Only **group-aware** primitives — ones whose expression reduces or scans across
the group defined by a foreign key. The order-dependent built-ins (`cum_sum`,
`cum_count`, `cum_min`, `cum_max`, `cum_mean`, `diff`, `absolute_diff`,
`same_as_previous`, `percent_change`, `time_since_previous`,
`cumulative_time_since_last_true`, `cumulative_time_since_last_false`) all
qualify. These are the primitives you'll normally pass here.

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

No built-in sits on the grouped, non-order-dependent path; a primitive of
that kind needs a user-defined primitive — that's [the intended extension
point](custom-primitives.md#group-aware-primitives).
