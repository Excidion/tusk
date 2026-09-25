# Primitives

A primitive builds a narwhals expression, which DFS composes into features.
The whole computation stays on your backend.

## What ships with tusk

See the [API reference](../api/primitives.md) for the full list of
aggregation, transform, group transform and ordered transform primitives tusk
ships.

Every `time_since_*` primitive needs more than its input columns: it
requires a `cutoff_time` at apply time, since its value is measured against
that moment rather than derived from the rows alone. Ordered transform
primitives require a `row_creation_time` on the table.

See [primitive coverage](primitive-coverage.md) for how these line up against
featuretools.

Every name above is also an importable class, so `from tusk.primitives import
Year, CumSum` works. They take the same form as a user-defined one, so `Year`
and `Count` are the same kind of object as anything you write yourself. See
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

A multi-output primitive such as `quantiles` produces indexed columns:
`QUANTILES__orders__quantity__0`, `__1`, `__2`. A primitive takes one column as
input, so these columns are the end of their chain: they go into the matrix,
and no further primitive builds on them. Use one at any depth.

## Empty groups

Aggregating a group with no rows is the most surprising correct behaviour in
the library, so it is worth stating plainly. After the left join, a customer
with no orders gets:

| Primitive | Value | Why |
|---|---|---|
| `COUNT` | `0` | We know there were zero rows. |
| `N_UNIQUE` | `0` | Zero rows hold zero distinct values. A null counts as one value, so a group of only nulls is `1`. |
| `SUM` | `0` | The additive identity. |
| `MEAN`, `MIN`, `MAX`, `STD`, `MEDIAN`, `QUANTILES`, `MODE` | `null` | Genuinely undefined over an empty set: `0/0`, and the min or max of nothing. |
| `PERCENT_TRUE` | `null` | Undefined over an empty set, same as `MEAN`. Within a non-empty group a null counts as false, so an all-null group computes to `0.0` rather than falling through to this default. |
| `N_TRUE` | `0` | Zero rows hold nothing to count. |
| `N_UNIQUE_DAYS`, `N_UNIQUE_DAYS_OF_CALENDAR_YEAR`, `N_UNIQUE_DAYS_OF_MONTH`, `N_UNIQUE_MONTHS` | `0` | Zero rows hold zero distinct values. A null date counts as one value, so a group of only nulls is `1`. |
| `ANY_TRUE` | `false` | No row is true. |
| `ALL_TRUE` | `null` | A group of only nulls is `true`, as polars and duckdb answer, but a group with no rows is left unknown. |
| `IS_UNIQUE` | `null` | No rows to compare. Nulls are values, so a group of several nulls is `false`. |
| `VARIANCE`, `SKEW`, `KURTOSIS`, `MAX_MIN_DELTA`, `FIRST_LAST_TIME_DELTA`, `PERCENT_UNIQUE`, `ENTROPY` | `null` | Undefined over an empty set. `SKEW` and `KURTOSIS` are also `null` for a group whose values do not vary. |

`COUNT` and `SUM` report `0` because a group with no rows has nothing to count
and nothing to add up. `MEAN` stays null, because there is no number that is
the average of nothing. Each value lives on the primitive as `default_value`,
so your own primitives set theirs the same way.

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

## Comparing two columns

The comparison primitives accept a pair of numbers or a pair of datetimes,
never one of each. A primitive may declare several input shapes, and each
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
themselves rather than their encodings. On polars, two `Enum` columns with
different member lists cannot be compared directly at all. A null label
follows the same rule as everything else on this page: it makes the
comparison unknown rather than simply unequal.

`(F.HAS_DATE, F.HAS_DATE)` matches a `Datetime` column or a `Date` column,
regardless of time zone, so a table holding a `Date` column and a tz-aware
or tz-naive `Datetime` column will generate a comparison between
any two of them.
You can guard against this with `db.validate()`.


## How transforms are applied

`trans_primitives` takes every transform primitive, and synthesis applies each
one by its class:

- A **row-wise transform**
  ([`TransformPrimitive`][tusk.primitives.TransformPrimitive]) runs on each
  row, reading only that row: `absolute`, `month`, `add_numeric`, …
- A **group transform primitive**
  ([`GroupTransformPrimitive`][tusk.primitives.GroupTransformPrimitive]) runs
  within each foreign-key group: `percentile`, which ranks each value within
  its group.
- An **ordered transform primitive**
  ([`OrderedTransformPrimitive`][tusk.primitives.OrderedTransformPrimitive])
  runs within each foreign-key group and reads the group's rows in
  `row_creation_time` order, such as `cum_sum`; the [API
  reference](../api/primitives.md#ordered-transform-primitives) lists them all.

A group or ordered transform gives one feature per parent relationship of the
table, named after the foreign key it groups by, such as
`CUM_SUM__amount__by__session_id`. It never runs across the whole table: a
running total or a rank over every row would mix every entity's rows and
change with whichever rows are in the dataset, such as a test split, which is
a data-leakage risk. It looks only at the rows sharing its row's foreign key,
the same rows an aggregation sees.

A group is not always a single entity. Grouping by a shared parent, such as
drivers when the target is customers, puts several customers' rows in one
group, exactly as an aggregation over drivers does.

A primitive of your own picks its behavior the same way, through the class it
subclasses; see [choosing a transform base
class](custom-primitives.md#choosing-a-transform-base-class).
