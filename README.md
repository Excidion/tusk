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
so if you passed lazy frames, nothing is computed until you collect it.
`features` is a list of inspectable definitions you can re-apply to new data:

```python
matrix = tusk.calculate_feature_matrix(features, es_new)
```

## Differences from featuretools

- **Lazy in, lazy out.** Feed tusk lazy frames and it returns a lazy frame:
  it builds one query plan and never collects, so you decide when to compute.
  Feed it eager frames and it returns an eager frame, collecting once at the
  end — that single `collect()` is the only one in the library.
- **Any narwhals backend**, not just pandas. One entity set uses one backend.
- **Feature names are SQL identifiers.** Featuretools writes
  `MEAN(transactions.amount)`; tusk writes `MEAN__transactions__amount`. On a
  backend that generates SQL, dots and parentheses parse as table qualifiers
  and function calls rather than as part of a name, so the conventional form
  is unusable there. Every construct joins with `__`:

  | meaning | column name |
  | --- | --- |
  | a parent's column | `customers__age` |
  | an aggregation | `MEAN__transactions__amount` |
  | a zero-arity aggregation | `COUNT__transactions` |
  | stacked to depth 2 | `MEAN__sessions__MEAN__transactions__amount` |
  | a grouped transform | `CUM_SUM__amount__by__session_id` |
  | one output of a multi-output primitive | `QUANTILES__amount__0` |

  The conventional form is kept on `Feature.display_name` for logs, docs and
  error messages. If a source column name collides with a generated one,
  `calculate_feature_matrix` raises rather than silently dropping a column.
- **`primary_key` and `row_creation_time`** rather than `index` and
  `time_index`. Narwhals has no index concept, and `row_creation_time` names
  what the column means: when the row became knowable.
- **Three-argument relationships.** `add_relationship(parent=, child=,
  foreign_key=)` — the parent side is always the parent's primary key.
- **One global `cutoff_time`**, not per-row cutoff times. It filters the
  target table too, so the matrix can have fewer rows than the target — a row
  that did not exist yet at the cutoff has no features to compute. Tables with
  no `row_creation_time` are timeless and pass through unfiltered, so a cutoff
  on an entity set that declares none is silently a no-op. With
  `features_only=True` the cutoff is ignored entirely, since nothing is
  computed and feature definitions do not record it.
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

A multi-output primitive such as `quantiles` produces indexed columns —
`QUANTILES__transactions__amount__0`, `__1`, `__2` — and nothing else stacks
on it: there is no single column for another primitive to read. It is a valid
output at any depth, just never an input.

### Empty groups

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
contradicts a known-zero count. `MEAN` has no such defence — there is no
number that is the average of nothing — so it stays null. Each value lives on
the primitive as `default_value` rather than as a special case in the compiler.

featuretools agrees on `COUNT` and `SUM`, and also leaves `MEAN`/`MIN`/`MAX`
null. It differs on `N_UNIQUE`, which it leaves as `NaN`; tusk reports `0` for
the reason above.

### What can go in `groupby_trans_primitives`

Only **group-aware** primitives — ones whose expression reduces or scans
across the group defined by a foreign key. The order-dependent built-ins
(`cum_sum`, `cum_count`, `cum_min`, `cum_max`, `diff`, `time_since_previous`)
all qualify, and are the primitives you'll normally pass here.

Every other built-in transform (`absolute`, `month`, `add_numeric`, …) is
**elementwise** rather than group-aware, and narwhals rejects `.over()` on an
elementwise expression: `InvalidOperationError: Cannot apply over to
elementwise expression`. Passing one of these in `groupby_trans_primitives`
therefore fails — but at expression-build time, not later at `.collect()`, so
you learn immediately rather than after a long query. The failure surfaces
synchronously out of `dfs()` only when it compiles, i.e. `features_only=False`;
with `features_only=True` synthesis happily emits the definition and the error
waits until you call `calculate_feature_matrix()` on it.

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
