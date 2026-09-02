# Custom primitives

A primitive builds a narwhals expression and never sees a value, which is what
keeps it pushed down:

```python
from dataclasses import dataclass

import narwhals as nw
from tusk.dtypes import DtypeFamily as F
from tusk.primitives import AggregationPrimitive, register


@register
@dataclass(frozen=True)
class Range(AggregationPrimitive):
    """Difference between the largest and smallest value."""

    name = "range"
    input_dtypes = (F.NUMERIC,)

    def build(self, expr: nw.Expr) -> nw.Expr:
        return expr.max() - expr.min()
```

Then pass `"range"` or `Range()` to `deep_feature_synthesis()`. Parameters are
ordinary dataclass fields.

`@dataclass(frozen=True)` is required: features deduplicate by value. A
primitive without it is rejected with `PrimitiveError`.

There is no second, shorter way to declare one. Every built-in primitive is a
frozen dataclass written out like this, so `Year` and `Count` are the same kind
of object as `Range` — nothing in tusk can reach a definition path your own
code cannot.

## The pieces

| Attribute | Meaning |
| --- | --- |
| `name` | The string DFS resolves, and the uppercased stem of generated column names. |
| `input_dtypes` | One [`DtypeFamily`][tusk.dtypes.DtypeFamily] per argument. An empty tuple means zero-arity, like `count`. |
| `output_dtype` | The narwhals dtype produced. Omit to derive it from the inputs by overriding `return_dtype`. |
| `default_value` | What an empty group gets. See [empty groups](primitives.md#empty-groups). |
| `build` | Takes one expression per input and returns the output expression, or a sequence of them for a multi-output primitive. |

Subclass [`AggregationPrimitive`][tusk.primitives.AggregationPrimitive] for
something that reduces a child table to one row per parent, and
[`TransformPrimitive`][tusk.primitives.TransformPrimitive] for something that
maps a row to a row.

[`@register`][tusk.primitives.register] puts the class in the registry so its
`name` resolves as a string; without it you can still pass an instance.

## Group-aware primitives

A primitive like "share of group total" is group-aware without being
order-dependent, which no built-in covers:

```python
from dataclasses import dataclass

import narwhals as nw
from tusk.dtypes import DtypeFamily as F
from tusk.primitives import TransformPrimitive, register


@register
@dataclass(frozen=True)
class ShareOfGroupTotal(TransformPrimitive):
    """Each value's fraction of its group's total."""

    name = "share_of_group_total"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64

    def build(self, expr: nw.Expr) -> nw.Expr:
        return expr / expr.sum()
```

Pass `"share_of_group_total"` in `groupby_trans_primitives` and it computes
each transaction's share of its session's total, each session's share of its
customer's total, and so on — a genuinely useful feature type that has no other
path into tusk. See [what can go in
`groupby_trans_primitives`](primitives.md#what-can-go-in-groupby_trans_primitives)
for why the built-in elementwise transforms cannot.

## Primitives that measure against the cutoff time

Mix in [`NeedsCutoffTime`][tusk.primitives.NeedsCutoffTime] for a primitive
whose value depends on the moment the feature matrix is built, not just its
input column — `time_since` is the built-in example. `build()` takes
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

The primitive never stores `cutoff_time` — it is a dataclass field-free
argument passed in at build time — which is what lets one `FeatureList` built
once be applied at several cutoff times. `deep_feature_synthesis()` and
`FeatureList.apply()` raise `ValidationError` if such a primitive is requested
and no `cutoff_time` is given.
