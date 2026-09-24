# Custom primitives

A primitive builds a narwhals expression, so it runs on your backend:

```python
from dataclasses import dataclass

import narwhals as nw
from tusk.dtypes import DtypeFamily as F
from tusk.primitives import AggregationPrimitive, register


@register
@dataclass(frozen=True)
class Range(AggregationPrimitive):
    """Difference between the largest and smallest value."""

    name = "range"  # identifier and also the stem of generated column names
    input_dtypes = (F.NUMERIC,)  # one per input; empty means zero-arity, like count
    output_dtype = nw.Float64  # build()'s result is cast to this
    default_value = None  # empty group has no range

    def build(self, expr: nw.Expr) -> nw.Expr:
        return expr.max() - expr.min()
```

Then pass `"range"` or `Range()` to `deep_feature_synthesis()`. Parameters are
ordinary dataclass fields.

`@dataclass(frozen=True)` is required: features deduplicate by value. A
primitive without it is rejected with `PrimitiveError`.

See [empty groups](primitives.md#empty-groups) for how `default_value` is used
downstream.

Say your primitive takes either two numbers or two dates, never one of each.
Declare `input_dtypes` as a tuple of signatures instead of a flat one:

```python
input_dtypes = ((F.NUMERIC, F.NUMERIC), (F.HAS_DATE, F.HAS_DATE))
```

Each signature is matched as a whole, and every one must take the same number
of inputs.

Subclass [`AggregationPrimitive`][tusk.primitives.AggregationPrimitive] for
something that reduces a child table to one row per parent, or
[`OrderedAggregationPrimitive`][tusk.primitives.OrderedAggregationPrimitive]
when it reads the rows in `row_creation_time` order, as `first` does. An
aggregation that measures each row against its own group first, as
`count_above_mean` does, subclasses
[`GroupRelativeAggregationPrimitive`][tusk.primitives.GroupRelativeAggregationPrimitive],
since SQL backends reject an aggregate nested in an aggregate. One that needs
how often each value occurs in its group, as `mode` does, subclasses
[`ValueCountAggregationPrimitive`][tusk.primitives.ValueCountAggregationPrimitive]:
its `build_per_row(values, counts)` receives the column and each row's count.
Subclass
[`TransformPrimitive`][tusk.primitives.TransformPrimitive] for something that
maps a row to a row. A transform that reads the other rows of its foreign-key
group subclasses
[`GroupTransformPrimitive`][tusk.primitives.GroupTransformPrimitive], or
[`OrderedTransformPrimitive`][tusk.primitives.OrderedTransformPrimitive] when
it reads them in `row_creation_time` order; see [choosing a transform base
class](#choosing-a-transform-base-class).

[`@register`][tusk.primitives.register] puts the class in the registry so its
`name` resolves as a string; without it you can still pass an instance.

## Choosing a transform base class

Every transform goes in `trans_primitives`, and the class it subclasses decides
how synthesis applies it:

- [`TransformPrimitive`][tusk.primitives.TransformPrimitive]: the value comes
  from the row's own columns only, as in `year` or `add_numeric`. It runs on
  each row.
- [`GroupTransformPrimitive`][tusk.primitives.GroupTransformPrimitive]: the
  value needs the other rows sharing the row's foreign key, but not their
  order, as in a share of the group's total or `percentile`. It runs within
  each foreign-key group.
- [`OrderedTransformPrimitive`][tusk.primitives.OrderedTransformPrimitive]:
  the value needs the earlier rows of its group in `row_creation_time` order,
  as in `cum_sum` or `diff`. It runs within each foreign-key group, ordered by
  the table's `row_creation_time`, then its primary key, so the table needs a
  `row_creation_time`.

A practical test is to look at what `build()` calls on its input:

- a reduction or window over the column (`sum`, `mean`, `count`, `rank`, …)
  reads other rows, so subclass `GroupTransformPrimitive`;
- `cum_*`, `shift`, `diff` or a forward fill depends on order, so subclass
  `OrderedTransformPrimitive`;
- anything else subclasses `TransformPrimitive`.

Subclass by what your expression does, since the class is what tusk goes on:

- A reduction under `TransformPrimitive`, such as `expr / expr.sum()`,
  computes over every row of the table, on lazy polars and on duckdb alike, so
  every row's value leaks into every other row's and your feature carries the
  whole dataset in it.
- An order-dependent expression under `TransformPrimitive`, such as
  `expr.cum_sum()`, passes synthesis, including with `features_only=True`.
  narwhals then raises `InvalidOperationError: Order-dependent expressions are
  not supported for use in LazyFrame` when the feature matrix's query is
  built, on lazy polars and on duckdb alike, before anything is collected.

A share of the group's total reads the other rows of its group without needing
them in order, so it subclasses `GroupTransformPrimitive`:

```python
from dataclasses import dataclass

import narwhals as nw
from tusk.dtypes import DtypeFamily as F
from tusk.primitives import GroupTransformPrimitive, register


@register
@dataclass(frozen=True)
class ShareOfGroupTotal(GroupTransformPrimitive):
    """Each value's fraction of its group's total."""

    name = "share_of_group_total"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64

    def build(self, expr: nw.Expr) -> nw.Expr:
        return expr / expr.sum()
```

Pass `"share_of_group_total"` in `trans_primitives` and it computes each
transaction's share of its session's total, each session's share of its
customer's total, and so on. Its total never spans the whole table.

See [how transforms are applied](primitives.md#how-transforms-are-applied) for
the built-in transforms of each kind.

## Primitives that measure against the cutoff time

Mix in [`NeedsCutoffTime`][tusk.primitives.NeedsCutoffTime] for a primitive
whose value depends on the moment the feature matrix is built, not just its
input column. `time_since` is the built-in example. `build()` takes
`cutoff_time` as a keyword alongside the usual input expressions:

```python
from dataclasses import dataclass
from datetime import datetime

import narwhals as nw
from tusk.dtypes import DtypeFamily as F
from tusk.primitives import NeedsCutoffTime, TransformPrimitive, register


@register
@dataclass(frozen=True)
class TimeSince(NeedsCutoffTime, TransformPrimitive):
    """Time elapsed from a datetime to the cutoff time."""

    name = "time_since"
    input_dtypes = (F.HAS_DATE,)
    output_dtype = nw.Duration

    def build(self, expr: nw.Expr, *, cutoff_time: datetime) -> nw.Expr:
        return nw.lit(cutoff_time) - expr
```

`cutoff_time` arrives as a build argument, so one `FeatureList` can be built
once and applied at several cutoff times. `deep_feature_synthesis()` and
`FeatureList.apply()` raise `ValidationError` if such a primitive is requested
and no `cutoff_time` is given.
