# Tusk: Deep Feature Synthesis for Narwhals Lazy Frames

Design document. Status: approved for planning. Date: 2026-08-17.

## 1. Purpose

Tusk implements the Deep Feature Synthesis (DFS) algorithm over narwhals lazy
frames. It offers a featuretools-shaped interface — an `EntitySet` of related
tables plus a `dfs()` entry point — but computes features by building a single
lazy query plan instead of materializing intermediate frames.

Three goals drive every tradeoff below:

1. **Scale beyond memory.** Feature synthesis pushes down into the backend
   engine. Tusk never calls `collect()` in its own logic.
2. **Backend portability.** Tusk uses the narwhals API only, so the same DFS
   code runs on any narwhals-supported backend.
3. **A lean rewrite.** Featuretools' long tail is deliberately omitted. Every
   feature in this document earns its place.

### Non-goals

- Per-row cutoff times, training windows, and `approximate` (see §12).
- `where` clauses, interesting values, `seed_features`, `primitive_options`.
- Woodwork logical types or semantic tags.
- Python-callable primitives that operate on materialized values.
- Composite primary or foreign keys.

## 2. Public API

```python
import tusk
from tusk.primitives import NthMostCommon

es = tusk.EntitySet("retail")
es.add_dataframe("customers", customers_lf,
                 primary_key="id", row_creation_time="signed_up_at")
es.add_dataframe("sessions", sessions_lf,
                 primary_key="id", row_creation_time="started_at")
es.add_dataframe("transactions", tx_lf,
                 primary_key="id", row_creation_time="occurred_at")

es.add_relationship(parent="customers", child="sessions", foreign_key="customer_id")
es.add_relationship(parent="sessions", child="transactions", foreign_key="session_id")

feature_matrix, features = tusk.dfs(
    entityset=es,
    target_dataframe_name="customers",
    agg_primitives=["mean", "count", NthMostCommon(n=3)],
    trans_primitives=["month", "weekday"],
    groupby_trans_primitives=["cum_sum"],
    max_depth=2,
    cutoff_time=datetime(2026, 1, 1),
    features_only=False,
)

# Re-apply a fitted feature set to new data.
matrix = tusk.calculate_feature_matrix(features, es_new, cutoff_time=...)
```

Parameter names track featuretools where the concept is unchanged, so existing
knowledge transfers. `feature_matrix` is returned in the caller's native frame
type; `features` is a list of inspectable `Feature` objects.

Defaults and return contract:

- `max_depth=2`, matching featuretools.
- `cutoff_time=None` means no time filtering is applied.
- `features_only=True` returns the feature list alone, not a tuple.
- The feature matrix has **one row per target row**, keyed by the target's
  `primary_key`. The target table must therefore have a `primary_key`; a
  positional correspondence would be meaningless on a lazy frame.

`calculate_feature_matrix` exists because the two-phase architecture provides it
for free, and it is the reason users care about `features_only`: fit a feature
set on training data, apply it unchanged to new data.

## 3. Architecture

Two phases with a hard boundary between them.

**Phase 1 — synthesis** (`synthesis.py`) walks the relationship graph using
only schemas: column names, dtypes, keys, relationships. It emits an immutable
graph of `Feature` definitions. No frame is touched. `features_only=True`
returns after this phase.

**Phase 2 — compilation** (`compiler.py`) turns a feature list into narwhals
operations, batching aggressively so that feature count does not drive join
count.

`synthesis.py` must not import `compiler.py`. Phase 1 is therefore testable
with zero dataframes, which is what keeps a single-backend test matrix
defensible.

### Module layout

```
src/tusk/
  __init__.py        # EntitySet, dfs, calculate_feature_matrix
  entityset.py       # EntitySet, TableSchema, Relationship
  dtypes.py          # dtype-family predicates over narwhals dtypes
  features.py        # Feature definitions (phase-1 output)
  synthesis.py       # phase 1
  compiler.py        # phase 2
  primitives/
    base.py          # Primitive ABCs and metadata
    registry.py      # name -> primitive, decorator sugar
    aggregation.py
    transform.py
  exceptions.py
```

Runtime dependency: **narwhals only**.

## 4. Schema model

### `EntitySet.add_dataframe(name, dataframe, primary_key=None, row_creation_time=None)`

`dataframe` accepts a native frame or a narwhals frame. Native inputs are
wrapped with `nw.from_native`; the original type is recorded so results can be
returned in kind.

**`primary_key` is optional.** It is required only when the table is a
relationship parent (it is the join key) or when an order-dependent primitive
needs a deterministic tiebreak. Omitting it raises `MissingPrimaryKeyWarning`
at `add_dataframe()` time, naming exactly what is forfeited:

> `sessions` has no `primary_key`: it cannot be used as a relationship parent,
> and order-dependent primitives (e.g. `cum_sum`) on it will have
> non-deterministic tiebreaks.

`MissingPrimaryKeyWarning` is its own class so legitimate leaf tables can
filter it in one line.

**`row_creation_time`** names the column recording when a row became knowable.
This is the semantic the cutoff filter tests and the order that order-dependent
primitives sort by. The name is deliberate: a transactions table may carry
`occurred_at`, `settled_at`, and `refunded_at`, and only one of them belongs in
this slot. (The temporal-database term of art is `valid_from`, rejected because
it implies a matching `valid_to` that does not exist here.)

**Single-backend rule.** Narwhals cannot join a polars `LazyFrame` to a DuckDB
relation. Every frame added after the first must share the first frame's
backend, enforced at `add_dataframe()` with a `SchemaError`. This also makes
the round-trip return type unambiguous.

### `EntitySet.add_relationship(parent, child, foreign_key)`

`foreign_key` names the **child's** column. The parent side is always the
parent's `primary_key`. Differing column names are the normal case
(`customers.id` ← `sessions.customer_id`).

Three arguments rather than featuretools' four, with identical expressive
power: featuretools also requires the parent column to be the parent's index.
The form covers multiple relationships between the same pair
(`orders.billing_customer_id` and `orders.shipping_customer_id` → `customers`)
and self-references (`employees.manager_id` → `employees`).

Composite keys raise `SchemaError` rather than silently misbehaving.

### Type system

Primitive matching uses **narwhals dtypes only** — numeric, temporal, string,
boolean, categorical — via predicates in `dtypes.py`. No logical types, no
semantic tags, no woodwork. The cost is a lost distinction between an integer
that is a quantity and an integer that is a category; the benefit is zero extra
concepts, zero dependencies, and identical behaviour on every backend.

## 5. Primitive model

A primitive is a **frozen dataclass** that builds narwhals expressions. It
never sees a value. This is what preserves pushdown:

```python
build(*inputs: nw.Expr, **params) -> nw.Expr
```

Featuretools' model — `get_function()` returning a callable over a pandas
Series — is incompatible with every goal in §1 and is not used.

### Metadata

Phase 1 builds the feature graph before touching data, so stacking at depth ≥ 2
requires this metadata statically:

| Field | Purpose |
|---|---|
| `name` | registry key and generated column name |
| `input_dtypes` | which dtype families each input accepts |
| `return_dtype(input_dtypes)` | output dtype, so stacked features validate without executing |
| `number_of_outputs` | multi-output support; property, since it may depend on params |
| `commutative` | avoid emitting both `a - b` and `b - a` |
| `stack_on_self` | blocks e.g. `COUNT(COUNT(...))`; default `True` |
| `default_value` | value for empty groups after a left join |
| `generate_name(base_names)` | naming template; default `NAME(arg, arg)` |

`return_dtype` and `number_of_outputs` are a method and a property rather than
class attributes because both may depend on instance parameters (`max` preserves
its input dtype; `NthMostCommon(n=3)` yields three columns).

### Extension

Built-ins live in a registry keyed by name. Users subclass
`AggregationPrimitive` / `TransformPrimitive` and register. `dfs()` accepts
names or instances interchangeably. Parameters are ordinary dataclass fields —
which is why plain decorated functions were rejected as the primary form: they
collapse the moment a primitive takes parameters. A decorator is provided as
sugar for the zero-parameter case, producing the same instance type.

### Backend capability

Primitives declare **no** capability metadata. If a backend cannot compute
`skew` or a given window function, the user sees that backend's own error at
`collect()`. This is an accepted rough edge, documented rather than papered
over; a per-backend support matrix is a maintenance treadmill this project does
not need.

## 6. Feature definitions

Frozen dataclasses with structural equality, so duplicate sub-features collapse
via a set with no bookkeeping.

| Type | Meaning | Example name |
|---|---|---|
| `IdentityFeature` | a raw column | `amount` |
| `TransformFeature` | primitive over same-table features | `MONTH(signed_up_at)` |
| `AggregationFeature` | primitive over a child's feature, grouped by FK | `MEAN(transactions.amount)` |
| `DirectFeature` | a parent's feature joined onto the child | `customers.age` |
| `GroupByTransformFeature` | transform applied within FK groups | `CUM_SUM(amount) by customer_id` |

Each exposes `.name`, `.dtype`, `.depth`, `.table`, `.base_features`, and
`.output_names`. Multi-output primitives yield indexed names:
`N_MOST_COMMON(product)[0]` … `[2]`.

`.dtype` is derived statically from primitive metadata, never from data.

**Depth rule:** identity features are depth 0; every primitive application adds
1. `SUM(sessions.MEAN(transactions.amount))` is depth 2. Featuretools has
subtler rules here that routinely confuse users; this one is predictable.

## 7. Phase 1: synthesis

```
build(table, budget, path):
    feats = identity features of table
    if budget > 0:
        for child not reachable back along path:
            for base in build(child, budget-1, path+[rel]):
                feats += AggregationFeature(prim, base, rel)   # per matching prim
        for parent not reachable back along path:
            for base in build(parent, budget-1, path+[rel]):
                feats += DirectFeature(base, rel)
    feats += TransformFeature(prim, combo) for matching combos within budget
    return feats
```

Two rules carry the algorithm:

- **Never traverse back along the path you arrived on.** This prevents
  `customers → sessions → customers` from feeding the target's own columns back
  as derived features, and it makes diamond-shaped and self-referential schemas
  terminate.
- **Primary keys and foreign keys are excluded from primitive inputs.**
  `MEAN(customer_id)` is noise. Foreign keys remain usable as groupby keys and
  as inputs to `count` and `n_unique`.

Beyond those, stacking is governed by `stack_on_self` and by dtype-family
matching, which naturally blocks nonsense such as `MONTH(MEAN(...))`.

Identical features produced along different routes deduplicate by structural
equality.

## 8. Phase 2: compilation

`compile(features, entityset, cutoff) -> nw.LazyFrame`. `dfs()` is
`synthesize()` piped into `compile()`; `calculate_feature_matrix()` is
`compile()` with a user-supplied feature list. There is no second code path.

1. **Transitive closure.** Walk `.base_features` to find the tables and
   sub-features actually needed.
2. **Cutoff first.** For each table carrying a `row_creation_time`, apply
   `filter(nw.col(rct) <= cutoff)` before anything else, so it pushes to the
   scan. Tables without one are timeless and pass through unfiltered —
   documented, not warned. Cutoff effects do not cascade across relationships.
3. **Compile bottom-up**, deepest table first. Per table: `with_columns()` for
   transforms, then fold in aggregations from children.
4. **Batch.** All aggregation features sharing a `(child table, relationship
   path)` collapse into one `group_by(fk).agg(...)` and one left join. Direct
   features from a shared parent path collapse into one `select` and one join.
   Thirty features across two child tables produce two joins, not thirty.
5. **Assemble.** Select the target's primary key plus feature columns, aliased
   to `.name`.
6. **Round-trip.** `.to_native()`, or `.collect().to_native()` when the caller
   passed an eager frame.

Tusk calls `collect()` nowhere else.

### Ordering

Order-dependent primitives must **not** rely on a frame-level `.sort()`. On lazy
backends a sort is not guaranteed to survive subsequent operations. Ordering
rides on the window expression instead:

```python
nw.col("amount").cum_sum().over("customer_id", order_by=("occurred_at", "id"))
```

Narwhals *requires* `order_by` on `Expr.over` for order-dependent functions on
lazy backends — `cum_sum`, `cum_count`, `cum_min`, `cum_max`, `cum_prod`,
`diff`, `shift`, `rank`, and all rolling functions. Tusk's rule that
order-dependent primitives require a `row_creation_time` is therefore narwhals'
contract, not a local policy. The `primary_key` tiebreak is what the
`MissingPrimaryKeyWarning` is about; without it, ties resolve arbitrarily.

`.over(order_by=...)` also works without a partition, so ungrouped
order-dependent transforms are supported.

### Correctness rules

- **Left joins everywhere**, so target rows never disappear when a child group
  is empty.
- **Empty groups receive the primitive's `default_value`.** After a left join a
  customer with zero transactions has `null` for `MEAN` (correct) and must be
  `fill_null(0)` for `COUNT` (also correct). This lives on the primitive
  dataclass, not as a special case in the compiler. It is a common source of
  silent bugs in naive reimplementations.

## 9. Built-in primitives

**Aggregation:** `count`, `sum`, `mean`, `min`, `max`, `std`, `median`,
`n_unique`, `percent_true`, `n_most_common` (multi-output).

**Transform:** `year`, `month`, `day`, `hour`, `weekday`, `is_weekend`,
`absolute`, `natural_log`.

**GroupBy transform:** `cum_sum`, `cum_count`, `cum_min`, `cum_max`, `diff`,
`time_since_previous`. Narwhals has no `cum_mean`; if wanted it must be composed
from `cum_sum / cum_count`, so it is left out of v1.

**Multi-input arithmetic** — `add_numeric`, `subtract_numeric`,
`multiply_numeric`, `divide_numeric` — ships but stays **out of the defaults**.
On a twenty-column table these generate hundreds of features, and featuretools
users are routinely burned by it. Available by name, off unless requested.

### Defaults

`agg_primitives=None` resolves to
`["count", "sum", "mean", "min", "max", "std", "n_unique"]`.
`trans_primitives=None` resolves to `["year", "month", "weekday"]`.
`groupby_trans_primitives=None` resolves to `[]`.

A zero-configuration `dfs()` call therefore does something sensible.

## 10. Errors and warnings

`TuskError` is the base class.

- **`SchemaError`** — unknown table or column; composite key; backend mismatch;
  missing `primary_key` on a relationship parent or on the DFS target; a
  relationship graph that cannot terminate.
- **`PrimitiveError`** — unknown primitive name; no columns matching a
  primitive's input dtypes; order-dependent primitive on a table without
  `row_creation_time`.
- **`MissingPrimaryKeyWarning`** — as described in §4.

Every one of these fires during phase 1, from schemas alone, before any query is
built. The single deliberate exception is backend capability (§5): those surface
as the backend's own error at `collect()`.

## 11. Testing

Test-driven throughout. The backend fixture is parametrized over a list that
currently reads `["polars"]`.

### Tier 1 — core suite (permanent, `dev` group)

Hand-built fixtures of a dozen rows, constructed in Python, with every expected
value computed by hand. The `forbid-binary` pre-commit hook rules out committed
parquet or feather fixtures in any case.

- **Phase 1 needs no dataframes.** Given a schema fixture, assert the exact set
  of generated feature names, their depths and dtypes; assert termination on
  diamond and self-referential schemas; assert the never-traverse-back rule
  blocks target-column leakage. Most of the algorithm's risk lives here and all
  of it is testable with zero I/O.
- **Phase 2 against hand-computed values** on a customers/sessions/transactions
  fixture, including empty groups, null-only groups, single-row groups, and
  ties — the cases real data contains only by accident.
- **Laziness is tested behaviorally, not by grepping for `.collect()`.** Build
  an EntitySet over a scan of a nonexistent file, run `dfs()`, assert it returns
  normally, and assert the error appears only at `.collect()`. Anything that
  materializes early fails this test.
- **Batching guard.** Assert that N aggregation features from one child table
  produce exactly one join in `explain()`. Polars-specific and white-box, but it
  is the only way to keep the compiler's central performance property from
  regressing silently.

### Tier 2 — differential vs featuretools (scaffolding, `validation` group)

Generate small synthetic multi-table data, run featuretools on pandas and tusk
on polars, compare overlapping feature values. This is the sharpest available
signal that DFS was reimplemented *correctly* rather than merely plausibly.

This tier is temporary. It exists to validate the algorithm during development
and is removed once the library settles — concretely, once the differential
suite has passed unchanged across two consecutive releases that added
primitives. Removal is a deliberate decision, recorded in the changelog.

### Tier 3 — relbench performance (`benchmark` group)

Run tusk against a relbench dataset and record runtime and peak memory. Relbench
is the only tier that can put evidence behind the scale-beyond-memory goal in
§1, and its schema model maps almost one-to-one onto ours (`pkey_col`,
`time_col`, `fkey_col_to_pkey_table` → `primary_key`, `row_creation_time`,
relationships — to be verified during implementation). Relbench also publishes a
featuretools-DFS baseline to compare against.

Kept until deliberately cut.

### Execution

Tiers 2 and 3 are marked (`@pytest.mark.differential`, `@pytest.mark.benchmark`)
and excluded from the default `pytest` run and from pre-commit. Their
dependencies live in separate `[dependency-groups]` entries so neither the
runtime nor the core test loop carries them.

## 12. Known risks and accepted tradeoffs

- **Portability is asserted, not verified.** A single-backend test matrix cannot
  catch backend-specific assumptions. DuckDB is the cheap addition that would
  catch lazy-discipline violations (no row order, restricted windows); the
  backend fixture is parametrized so adding it is a one-line change.
- **Backend capability errors surface late**, at `collect()`, in the backend's
  own vocabulary (§5).
- **A single global cutoff is not leak-free for ML.** Per-row cutoff times are
  what prevent target leakage in a training set; §13 covers the path there.
- **Multi-input arithmetic primitives can explode feature counts** if requested
  on wide tables. Mitigated by keeping them out of the defaults.

## 13. Future work

Deliberately deferred; none requires restructuring the feature graph.

- Per-row cutoff times, training windows, `approximate`.
- `where` clauses / interesting values, `seed_features`, `primitive_options`.
- Additional backends in the test matrix, starting with DuckDB.
- `tusk.from_relbench(db)` — roughly fifteen lines once the schema model exists.
- Feature serialization to disk.

## 14. Packaging and tooling

- Add `[build-system]` with hatchling and adopt a `src/` layout; the project
  currently has neither.
- Runtime dependency: `narwhals`. Dev adds `pytest` and `polars`.
- Dependency groups: `dev`, `validation` (featuretools, pandas), `benchmark`
  (relbench).
- **Google-style docstrings**, configured in `pydoclint`. Both `interrogate` and
  `pydoclint` are already in pre-commit, so every public symbol needs a complete
  docstring.
