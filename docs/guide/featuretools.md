# Differences from featuretools

tusk borrows several concepts from [featuretools](https://featuretools.alteryx.com/),
including relationships, primitives and cutoff times. tusk differs where a lazy
frame gives a benefit, or where an SQL backend needs a different answer.

- **Renamed container and entry points.** tusk keeps concepts similar to
  featuretools but renames them. A collection of tables joined by primary and
  foreign keys is a database.

  | featuretools | tusk |
  | --- | --- |
  | `es = EntitySet(id=…)` | `db = Database(name=…)` |
  | `es.add_dataframe(dataframe_name=…, dataframe=…, index=…, time_index=…)` | `db.add_table(name, table, primary_key=…, row_creation_time=…)` |
  | `es.set_secondary_time_index(dataframe_name=…, secondary_time_index={col: [...]})` | `db.add_table(..., row_update_times={col: {...}})` |
  | `dfs(entityset=…, target_dataframe_name=…)` | `deep_feature_synthesis(database=…, target_table=…)` |
  | `calculate_feature_matrix(features, entityset)` | `apply_features(features, database)` |

- **Lazy output**, wherever possible. tusk builds a query plan for the feature
  matrix. You can pass eager or lazy frames as input. If your backend supports
  it, tusk returns an uncomputed lazy frame. Eager backends, like pandas,
  always compute directly.

- **Almost any backend you like.** Just keep it consistent within one
  database.

- **Feature names are SQL identifiers.** Featuretools writes
  `MEAN(orders.quantity)`. tusk writes `MEAN__orders__quantity`. A backend
  that produces SQL treats dots and parentheses as table qualifiers and
  function calls, not as parts of a column name.
  [`Feature.display_name`][tusk.features.Feature.display_name] keeps the
  conventional form for logs, docs and error messages.

- **`primary_key` and `row_creation_time`** rather than `index` and
  `time_index`. Narwhals has no index concept. `row_creation_time` names
  what the column means: when the row became knowable.

- **`row_update_times` keeps the column's earlier value.** featuretools'
  `set_secondary_time_index` always makes it null instead. See [row update
  times](databases.md#row-update-times).

- **Simplified relationships.** `add_relationship(parent=, child=,
  foreign_key=)`. On the parent side, the join column is always the parent's
  primary key.

- **Opt-in [validation](databases.md#validation).**
  You can use tusk to check whether your definitions match the real datasets.
  By default, tusk enables only the checks that need the table schema. If you
  can spend compute time on the full datasets, run `db.validate()`, or set
  `validate=True` on `db.add_table(...)` or `db.add_relationship(...)`.

- **One global `cutoff_time`** applies to the target table too. tusk leaves
  out a row that did not yet exist at the cutoff time. tusk treats a table
  with no `row_creation_time` as timeless and passes it through unfiltered.

- **Group and ordered transforms only run within foreign-key groups.**
  tusk has no `groupby_trans_primitives` argument. Primitives like `cum_sum`
  and `percentile`, passed in `trans_primitives`, run within each foreign-key
  group. A row only sees the rows that share its foreign key. featuretools
  needs `groupby_trans_primitives` for that behavior, and it also runs these
  primitives across the whole table from `trans_primitives`. See [how
  transforms are applied](primitives.md#how-transforms-are-applied).

- **`primary_key` is optional.** A table without one cannot be a relationship
  parent or a DFS target. On such a table,
  [ordered transform][tusk.primitives.OrderedTransformPrimitive] and
  [ordered aggregation][tusk.primitives.OrderedAggregationPrimitive]
  primitives might behave non-deterministically.


## Comparing primitives

[Primitive coverage](primitive-coverage.md) lists every featuretools primitive
beside its tusk counterpart.
