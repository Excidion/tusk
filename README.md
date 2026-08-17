# tusk

Deep feature synthesis for [narwhals](https://narwhals-dev.github.io/narwhals/)
lazy dataframes. Tusk generates features across related tables the way
[featuretools](https://featuretools.alteryx.com/) does, but it builds a single
lazy query plan instead of materializing intermediate frames — so synthesis
pushes down into whichever engine holds your data, and works the same on every
backend narwhals supports.

## Install

```bash
uv add tusk
```

## Usage

```python
from datetime import datetime

import tusk
from tusk.primitives import Quantiles

es = tusk.EntitySet("retail")
es.add_dataframe("customers", customers_lf, primary_key="id",
                 row_creation_time="signed_up_at")
es.add_dataframe("sessions", sessions_lf, primary_key="id",
                 row_creation_time="started_at")
es.add_dataframe("transactions", tx_lf, primary_key="id",
                 row_creation_time="occurred_at")

es.add_relationship(parent="customers", child="sessions", foreign_key="customer_id")
es.add_relationship(parent="sessions", child="transactions", foreign_key="session_id")

feature_matrix, features = tusk.dfs(
    entityset=es,
    target_dataframe_name="customers",
    agg_primitives=["mean", "count", Quantiles(qs=(0.25, 0.5, 0.75))],
    trans_primitives=["month", "weekday"],
    max_depth=2,
    cutoff_time=datetime(2026, 1, 1),
)
```

`feature_matrix` comes back in the frame type you put in — lazy in, lazy out —
so nothing is computed until you collect it. `features` is a list of inspectable
definitions you can re-apply to new data:

```python
matrix = tusk.calculate_feature_matrix(features, es_new)
```

## Differences from featuretools

- **Lazy throughout.** Tusk never collects; you decide when to compute.
- **Any narwhals backend**, not just pandas. One entity set uses one backend.
- **`primary_key` and `row_creation_time`** rather than `index` and
  `time_index`. Narwhals has no index concept, and `row_creation_time` names
  what the column means: when the row became knowable.
- **Three-argument relationships.** `add_relationship(parent=, child=,
  foreign_key=)` — the parent side is always the parent's primary key.
- **One global `cutoff_time`**, not per-row cutoff times.
- **`primary_key` is optional**, but a table without one cannot be a
  relationship parent or a DFS target, and order-dependent primitives on it
  have non-deterministic tiebreaks. You get a `MissingPrimaryKeyWarning`.

## Primitives

Aggregation: `count`, `sum`, `mean`, `min`, `max`, `std`, `median`, `n_unique`,
`percent_true`, `quantiles`.

Transform: `year`, `month`, `day`, `hour`, `weekday`, `is_weekend`, `absolute`,
`natural_log`, `add_numeric`, `subtract_numeric`, `multiply_numeric`,
`divide_numeric`.

Order-dependent (require a `row_creation_time`): `cum_sum`, `cum_count`,
`cum_min`, `cum_max`, `diff`, `time_since_previous`.

Passing `agg_primitives=None` or `trans_primitives=None` selects a sensible
default subset. Arithmetic primitives are excluded from the defaults because
they generate hundreds of features on wide tables.

### What can go in `groupby_trans_primitives`

Only **group-aware** primitives — ones whose expression reduces or scans
across the group defined by a foreign key. The order-dependent built-ins
(`cum_sum`, `cum_count`, `cum_min`, `cum_max`, `diff`, `time_since_previous`)
all qualify, and are the primitives you'll normally pass here.

Every other built-in transform (`absolute`, `month`, `add_numeric`, …) is
**elementwise** rather than group-aware, and narwhals rejects `.over()` on an
elementwise expression: `InvalidOperationError: Cannot apply over to
elementwise expression`. Passing one of these in `groupby_trans_primitives`
therefore fails — but synchronously, at expression-build time out of `dfs()`,
not later at `.collect()`, so you learn immediately rather than after a long
query.

This leaves the grouped, non-order-dependent path reachable only by
user-defined primitives — that's the intended extension point. A primitive
like "share of group total" is group-aware without being order-dependent:

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
customer's total, and so on — a genuinely useful feature type that has no
other path into tusk.

### Custom primitives

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

Then pass `"range"` or `Range()` to `dfs()`. Parameters are ordinary dataclass
fields.
