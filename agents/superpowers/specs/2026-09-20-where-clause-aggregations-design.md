# Where-clause aggregations: conditional features and valid time

Design for `add_table(where=..., when=...)` and
`deep_feature_synthesis(where_primitives=...)`, the featuretools equivalent of
`where_primitives` plus interesting values. Closes issue #18.

## Problem

Issue #18 asks how to handle a relationship that is true only between a start
and an end time: a subscription contract that runs from `starts_at` until
`cancelled_at`, a car owned by one customer and then another.

`row_creation_time` cannot express this. It decides when a row becomes
*knowable*, and a cancelled contract stays perfectly knowable — it is the
subscription that ended, not the record. These are two different time axes:

- **Knowledge time** — when the row became, and stopped being, visible to us.
  `row_creation_time` is this today. Its missing partner,
  `row_deletion_time`, would be a hard filter at every depth.
- **Valid time** — when the fact the row asserts was true. A cancelled
  contract is history worth aggregating over, so this must *not* be a hard
  filter.

The issue proposes a `Current` filter primitive, or a family of `CurrentCount`
/ `CurrentSum` / `CurrentMin` aggregations. The maintainer's closing comment
asks whether any combination of depth and primitives ever reaches "the sum of
all prices of all current subscriptions", where `price` lives on a
`subscription_types` table two hops from the target.

**It does not.** A primitive receives expressions for the columns it declares
as inputs; it never sees the join path. `CurrentSum(price)` cannot know which
contract row a price came from, so it cannot mask by contract validity. The
same applies to a row-level `Current` transform.

What does reach it is masking the *child frame's rows* before grouping. By the
time contracts are aggregated onto customers, `subscription_types.price` is
already a direct-feature column on the contracts frame, so masking contract
rows masks the prices with them:

```python
nw.col("subscription_types__price").filter(is_current).sum()
```

This generalizes past validity. The same machinery expresses
`SUM(orders.amount WHERE is_refunded)` and
`MEAN(contracts.price WHERE enterprise)`. Validity is one mask among many, so
the design is a general conditional-aggregation mechanism rather than a
temporal special case.

## Decisions

These were settled with the maintainer.

1. **Valid time is a per-feature choice, not a global filter.** Both "every
   contract ever" and "contracts current at the cutoff" are legitimate
   features and both are generated. Contrast `row_deletion_time`, which would
   have to filter globally; that is why the two cannot share a mechanism or a
   name.

2. **Masks are declared by the user, not discovered from the data.**
   featuretools derives interesting values by scanning for the most common
   values of each categorical column. tusk does not: `synthesize` reads
   schemas only and never touches a frame, and a scan would break that. The
   user names each condition instead. This removes the entire
   interesting-values concept, the data pass that would compute it, and the
   fitted state it would put in the schema.

3. **Two parameters, split by whether the mask needs the cutoff.** `where`
   takes `nw.Expr`; `when` takes callables receiving the cutoff. The
   alternative — one parameter with union-typed values — needs
   `inspect.signature` to tell them apart and a `**_` tail on every lambda.
   The split makes the distinction structural: an `isinstance` check validates
   it, and `_require_cutoff_time` reads which dict a key came from instead of
   sniffing callables.

4. **`when` callables take the cutoff positionally**, as `lambda cutoff: ...`,
   with no `**kwargs` tail. The tail would buy forward compatibility against a
   second run-scoped value — realistically only a future `training_window`.
   That is speculative, tusk is pre-1.0, and if it ever arrives the old
   lambdas fail with a loud `TypeError` at compile time rather than silently.
   Paying boilerplate on every lambda forever against a maybe-never break is
   the speculative generality YAGNI exists to cut.

5. **`NeedsCutoffTime` is untouched.** It is already the primitive-side
   equivalent of `when`, so the two APIs stay conceptually parallel without a
   breaking change to custom primitives.

6. **`where_primitives` defaults to `("count", "sum")`.** featuretools
   defaults to `count` alone, which would leave the issue's own motivating
   feature — current MRR — ungenerated unless explicitly asked for. Adding
   `sum` costs one extra feature per numeric column per clause and makes the
   headline case work by default.

7. **A clause does not consume depth.** It modifies one aggregation rather
   than stacking a primitive application. Counting it would put
   `SUM(contracts.subscription_types.price WHEN current)` out of reach at the
   default `max_depth=2`, defeating the purpose.

8. **Features store the clause's kind and key, never its expression.** A
   `FeatureList` stays a set of definitions that re-applies to new data;
   `features.apply(db_new)` resolves the key against the new database's
   schema. No `nw.Expr` and no lambda is ever serialized or carried.

9. **Clause keys need not be unique across `where` and `when`.** The `WHERE`
   and `WHEN` name tokens already disambiguate, and features record the kind
   regardless because `_require_cutoff_time` needs it. Only `__` is forbidden
   in a key, because it is the name separator.

10. **An empty mask falls back to the primitive's existing
    `default_value`.** "This group had no rows" and "no rows matched the
    clause" produce the same value. This needs no new mechanism: the post-join
    `fill_null(default_value)` already normalizes any backend returning NULL
    for an empty selection.

11. **`valid_from` / `valid_until` sugar is deferred.** It would desugar into
    a reserved `current` clause and buy interval validation, a drawable
    interval in `db.plot()`, and the `[from, until)` closure decided once
    rather than by each user. It is worth having, but it is additive on top of
    this machinery and ships better once the machinery has settled in real
    use.

## API surface

```python
db.add_table(
    "contracts",
    contracts_lf,
    primary_key="id",
    row_creation_time="signed_at",
    where={"enterprise": nw.col("plan") == "enterprise"},
    when={
        "current": lambda cutoff: (nw.col("starts_at") <= cutoff)
                                & (nw.col("cancelled_at") > cutoff),
    },
)

feature_matrix, features = tusk.deep_feature_synthesis(
    database=db,
    target_table="customers",
    where_primitives=("count", "sum"),
    max_depth=2,
    cutoff_time=datetime(2026, 1, 1),
)
```

`where` and `when` are stored on `TableSchema` as given. `where_primitives`
accepts names or instances, is `_require_kind`-checked as aggregations exactly
as `agg_primitives` is, and is threaded through `synthesize`. Passing `()`
disables clause features entirely.

Only aggregations take a clause. A transform is row-wise, so a masked
transform is `nw.when(...)`, which the user can already write directly.

A clause is an expression over the child table's **declared columns only**. It
cannot reference a computed feature.

## Feature model and naming

`AggregationFeature` gains:

```python
clause: tuple[Literal["where", "when"], str] | None
```

A clause masks the rows of the frame being grouped, so it is resolved against
the schema of the relationship's `child`. Aggregating `contracts` onto
`customers`, clauses declared on `contracts` apply; clauses on
`subscription_types` do not — those gate aggregations *over*
`subscription_types`. This follows from the definition and needs no rule.

Names mirror the parameter the key came from:

```
SUM__contracts__subscription_types__price__WHEN__current
SUM(contracts.subscription_types.price WHEN current)

COUNT__contracts__WHERE__enterprise
COUNT(contracts WHERE enterprise)
```

Zero-arity primitives take clauses too, so `COUNT(contracts WHEN current)` is
the current-contract count.

Generated feature count is
`base_aggs + (where_primitives x clauses x matching_inputs)`. It grows
linearly in the number of clauses declared, which the user controls directly.

## Compilation

One change, in `_add_aggregations`, where the expression is built:

```python
mask = _clause_expr(database.schema(relationship.child), feature.clause, cutoff_time)
inputs = [nw.col(b.name).filter(mask) for b in feature.base_features]
```

A zero-arity primitive has no input to mask, so its own expression is filtered
instead. `_clause_expr` returns `None` for an unclaused feature and the
existing path is taken unchanged, so features generated today compile exactly
as they do today.

`_clause_expr` looks the key up in `schema.where` and uses it as-is, or in
`schema.when` and calls it with `cutoff_time`.

`_table_frame` returns the table's own columns plus the needed features, so
mask columns are always present without adding them to `child_needed`.

`_require_cutoff_time` extends from primitives to clauses: today it scans for
`isinstance(feature.primitive, NeedsCutoffTime)`, and it gains a second
condition for a feature whose `clause` kind is `"when"`. Structural, with
nothing called.

### Empty masks

Verified on polars, for a group where no row passes the mask:

| primitive  | empty-mask result | `default_value` |
|------------|-------------------|-----------------|
| `sum`      | `0`               | `0`             |
| `n_unique` | `0`               | `0`             |
| `mean`     | `null`            | none → null     |
| `min`      | `null`            | none → null     |

Each already matches its `default_value`. On a backend returning NULL for an
empty `sum`, the existing post-join `fill_null(default_value)` corrects it to
the same `0`. No new mechanism is required; what is required is the
cross-backend test matrix below.

## Clause scope along a join path

**A clause scopes the level it is declared on, not the subtree beneath it.**
This must be documented in the user-facing docs, not only here, because the
wrong reading is silently harmful and the feature name looks innocent.

Consider `customers <- cars <- repairs`, where a car was owned by customer A
and then by customer B, and repairs happened under both owners. With a
`current` clause on `cars`:

```
SUM(cars.COUNT(cars.repairs) WHEN current)
```

reads as "over the cars this customer currently owns, the sum of each car's
**lifetime** repair count". Every repair counts toward B, including those
performed while A owned the car.

No clause can fix this. `repairs` has no `customer_id` and no knowledge of
ownership windows, so no predicate over its own columns can express "during
this customer's ownership". Splitting the repairs by owner requires an
interval join between `repaired_at` and the ownership window, which is a
different mechanism from masking.

Two things follow.

First, this attribution is **not introduced by this design**. A depth-2
aggregation today already rolls a car's whole history up to whichever customer
its foreign key points at. Clauses make the existing behaviour visible rather
than creating it.

Second, **normalizing the schema does not fix it either.** Modelling ownership
as its own table — `ownerships(car_id, customer_id, valid_from, valid_until)`
— decides *which customer* a car belongs to in a given window, but not *which
repairs*. Reaching repairs from customers still goes: aggregate `repairs` onto
`cars`, direct-feature onto `ownerships`, aggregate onto `customers`. That
middle rollup is the car's lifetime repair count, so every ownership row
inherits the car's whole history and A's repairs still land on B.

The split falls out of existing machinery in exactly one case: when `repairs`
carries an `ownership_id`, so repairs hang off `ownerships` directly. Deriving
that key from `repairs(car_id, repaired_at)` is itself the interval join, so
this only helps when the source data already materializes the link. Where it
does, the docs should show it.

Otherwise the missing mechanism is propagating an ancestor's validity window
down to filter descendant rows by their own timestamps — join the window onto
`repairs` and filter `repaired_at` within it. Feasible, but it makes a child's
aggregation depend on which parent it is rolling toward, which breaks the
current batch-per-relationship compilation. Tracked as issue #29; this spec's
masking seam in `_add_aggregations` is where it would attach.

## Validation

Three checks, all answerable from the declared schema, so all join
`DEFAULT_TABLE_CHECKS`:

- `where_clauses_are_expressions` — every `where` value is an `nw.Expr`.
- `when_clauses_are_callable` — every `when` value is callable.
- `clause_keys` — no key contains `__`.

Each message names the offending key and, for the first two, the parameter it
belongs in.

A malformed clause *body* still surfaces at compile time rather than at
`add_table` time. A static `nw.Expr`'s output dtype is not knowable without a
frame, and a `when` lambda cannot even be built without a cutoff. This is a
documented limit, not worth an eager scan on every `add_table`.

## Testing

The parts that carry real risk:

- **The motivating feature end to end.**
  `SUM(contracts.subscription_types.price WHEN current)` at `max_depth=2`,
  against hand-computed expected values. This is the join-path reach no
  primitive-based design could deliver, and it is the reason for the whole
  spec.
- **Empty mask falls back to `default_value`**, parametrized across polars,
  duckdb and pyarrow, for a `default_value=0` primitive and a null-default
  one. This is the only place the backends could still diverge.
- **Clauses see pre-update values.** The mask is built on the frame
  `base_frame` already returned, so `row_update_times` restoration applies to
  it. Correct by construction, which is exactly the kind of property that
  breaks silently later — it needs an explicit leakage test.
- **`features.apply(db_new)`** resolves clause keys against the new schema,
  and raises `SchemaError` naming the key when it is absent.
- **One `FeatureList` at two cutoffs** gives different `WHEN` results and
  identical `WHERE` results.
- **Clause scope along a join path**, asserting the documented semantics from
  the section above rather than an intuitive-but-wrong split.
- **Unclaused features are byte-identical** to what is generated today.

## Out of scope

- **`valid_from` / `valid_until` sugar** — decision 11. Follow-up issue,
  desugaring into a reserved `current` clause.
- **`row_deletion_time`** — the missing half of knowledge time. It is a hard
  filter at every depth, not an opt-in mask, so it does not belong to this
  mechanism. Separate issue.
- **Interval-scoped join paths** (#29) — propagating an ancestor's validity
  window down to descendants, per the join-path section. Required whenever a
  descendant is linked to the interval-bearing table only transitively, which
  normalization alone does not resolve.
- **Discovered interesting values** — decision 2 removes the need. If ever
  wanted, it would be a `db.add_interesting_values()` that queries once and
  writes clauses into the schema *before* synthesis, preserving the
  no-frames-in-phase-1 invariant.
