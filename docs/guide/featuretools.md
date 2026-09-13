# Differences from featuretools

tusk is inspired by [featuretools](https://featuretools.alteryx.com/) and keeps
some of its concepts, like relationships, primitives and cutoff times.
But it differs where it can benefit from lazy frames or SQL backends require
a different answer.

- **Renamed container and entry points.** Similar concepts to featuretools,
  but adapted naming:
  A collection of tables joined by primary and foreign keys is a database.

  | featuretools | tusk |
  | --- | --- |
  | `es = EntitySet(id=…)` | `db = Database(name=…)` |
  | `es.add_dataframe(dataframe_name=…, dataframe=…, index=…, time_index=…)` | `db.add_table(name, table, primary_key=…, row_creation_time=…)` |
  | `es.set_secondary_time_index(dataframe_name=…, secondary_time_index={col: [...]})` | `db.add_table(..., row_update_times={col: {...}})` |
  | `dfs(entityset=…, target_dataframe_name=…)` | `deep_feature_synthesis(database=…, target_table=…)` |
  | `calculate_feature_matrix(features, entityset)` | `apply_features(features, database)` |

- **Lazy output**, wherever possible. tusk builds a query plan for the matrix. Eager or
  lazy frames both go in and an uncomputed lazy frame is returned, if your backend
  supports it. Eager backends (like pandas) always compute directly.

- Almost **any dataframe backend** you like. Just keep it consistent within
  one database.

- **Feature names are SQL identifiers.** Featuretools writes
  `MEAN(orders.quantity)`; tusk writes `MEAN__orders__quantity`. This is, because
  on a backend that generates SQL, dots and parentheses are table qualifiers
  and function calls rather than as part of a column name.
  The conventional from is only kept on [`Feature.display_name`][tusk.features.Feature.display_name]
  for logs, docs and error messages.

- **`primary_key` and `row_creation_time`** rather than `index` and
  `time_index`. Narwhals has no index concept, and `row_creation_time` names
  what the column means: when the row became knowable.

- **`row_update_times` asks what the column held before** where featuretools
  `set_secondary_time_index` always nulls it. See [row update
  times](databases.md#row-update-times).

- **Simplified relationships.** `add_relationship(parent=, child=, foreign_key=)`
  The parent side join-column is always the parent's primary key.

- **Opt-in [validation](databases.md#validation).**
  tusk can be used to verify if your definitions match the real datasets.
  By default only checks that require only the tables schema are enabled.
  If you are willing to spend some compute time to on the full datasets you can run
  `db.validate()` or set `validate=True` for `db.add_table(...)` or `db.add_relationship(...)`

- **One global `cutoff_time`**, which applies to the target table too,
  so a row that did not exist yet at the cutoff will be left out.
  Tables with no `row_creation_time` are treated as timeless and pass through unfiltered.

- **`primary_key` is optional**, but a table without one cannot be a
  relationship parent or a DFS target. Be warned: Order-dependent primitives on
  such tables might behave non-deterministic.


## Comparing primitives

[Primitive coverage](primitive-coverage.md) lists every featuretools primitive
beside its tusk counterpart.
